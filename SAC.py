import glob
import os
import sys
import numpy as np
import cv2
import time
import torch
from collections import deque
from tqdm import tqdm
from torch.optim import Adam
import random
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from torch import nn
from torch import optim
from threading import Thread

from models import Actor, Critic
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse


IM_WIDTH = 640
IM_HEIGHT = 480
RESIZE_WIDTH = 64
RESIZE_HEIGHT = 48

ACTION_DIM = 4

SHOW_CAM = False
SHOW_LIDAR = False

EPISODES_LENGTH = 15
REPLAY_MEMORY_SIZE = 10000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5

MODEL_NAME = 'R25SAC'

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10

MIN_REWARD = 1500


class ModifiedTensorBoard:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 1  # Global step counter across episodes

    def update_stats(self, **stats):
        """Log custom metrics at the current step."""
        for key, value in stats.items():
            self.writer.add_scalar(key, value, self.step)

    def log_episode_end(self, episode_num, **logs):
        """Log metrics at the end of each episode."""
        for key, value in logs.items():
            self.writer.add_scalar(f"episode_{key}", value, episode_num)

    def close(self):
        """Close the writer when done."""
        self.writer.close()


class CarlaEnv:
    def __init__(self, client):
        self.client = client
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.bp_r25 = self.blueprint_library.filter('R25')[0]
        self.im_width = IM_WIDTH
        self.im_height = IM_HEIGHT
        self.show_cam = SHOW_CAM
        self.front_camera = None
        self.lidar_cast_ray = None

    def on_collision(self, event):
        self.collision_hist.append(event)

    def process(self, image):
        # show image
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        i3 = cv2.cvtColor(i3, cv2.COLOR_BGR2GRAY)
        _, i3 = cv2.threshold(i3, 50, 255, cv2.THRESH_BINARY)
        # resize image to 6x4
        i3 = cv2.resize(i3, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
        _, i3 = cv2.threshold(i3, 170, 255, cv2.THRESH_BINARY) # 170, 200, current options
        if self.show_cam:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        i3 = i3.reshape(1, RESIZE_HEIGHT, RESIZE_WIDTH).tolist() # PyTorch expects CHW instead of HWC
        self.front_camera = i3

    def process_lidar(self, data):
        raw_data = np.frombuffer(data.raw_data, dtype=np.float32)
        points = np.reshape(raw_data, (-1, 3))  # Each point is (x, y, z)
        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))  # angle in degrees

        min_angle = -180
        max_angle = 0
        valid_points = points[(angles >= min_angle) & (angles <= max_angle)]

        # Visualize or save the filtered points (e.g., convert to image coordinates and save)
        x_valid, y_valid = valid_points[:, 0], valid_points[:, 1]

        image_size = 500  # Image will be 500x500 pixels
        scale = 0.1       # Scaling factor to fit points into the image

        # Convert to image coordinates
        x_img = ((x_valid / scale) + (image_size / 2)).astype(np.int32)
        y_img = ((y_valid / scale) + (image_size / 2)).astype(np.int32)

        # Create an empty image to draw the points on
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        # Draw each point as a small white dot
        for j in range(len(x_img)):
            if 0 <= x_img[j] < image_size and 0 <= y_img[j] < image_size:
                cv2.circle(image, (x_img[j], y_img[j]), 1, (255, 255, 255), -1)

        # Target angles for filtering
        target_angles = [0, -30, -60, -90, -120, -150, -180]
        angle_tolerance = 2  # Tolerance in degrees for nearby points

        # Filter points for each target angle
        avg_points = []
        for target_angle in target_angles:
            # Find points within the tolerance of the target angle
            near_points = points[(angles >= target_angle - angle_tolerance) & (angles <= target_angle + angle_tolerance)]

            if len(near_points) > 0:
                # Compute the average position of the nearby points
                avg_x = np.mean(near_points[:, 0])
                avg_y = np.mean(near_points[:, 1])
                avg_points.append((avg_x, avg_y))
            else:
                # print(f"No points found near angle {target_angle+90}º.")
                avg_points.append(None)

        lidar_pos = (image_size // 2, image_size // 2)  # LiDAR position in image space

        distances = []

        for point, angle in zip(avg_points, target_angles):
            if point is not None:
                # Convert the average point to image coordinates
                x_img = int((point[0] / scale) + lidar_pos[0])
                y_img = int((point[1] / scale) + lidar_pos[1])

                # Compute distance and map to gradient color
                distance = np.sqrt(point[0]**2 + point[1]**2)  # Euclidean distance
                distances.append(distance)
                # print(f"Distance at {angle+90}º: {distance:.2f} m")
                distance_clamped = max(0, min(distance, 7))  # Clamp distance to [0, 10] meters
                green = int((distance_clamped / 7) * 255)    # Green decreases with distance
                red = 255 - green                              # Red increases with distance
                color = (0, green, red)  # Color based on distance

                # Draw the point as a small white dot
                cv2.circle(image, (x_img, y_img), 5, color, -1)

                if target_angle == -90:
                    color = (255,0,0)

                # Draw a line connecting the LiDAR position to the point
                cv2.line(image, lidar_pos, (x_img, y_img), color, 1)

                # Annotate the angle for reference
                cv2.putText(image, f"{distance:.2f}", (x_img + 5, y_img - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            else:
                distances.append(50)

        if SHOW_LIDAR:
            cv2.imshow("LiDAR", image)
            cv2.waitKey(1)
            #cv2.imwrite(f"./lidar/lidar_points_frame{i}.png", image)

        self.lidar_cast_ray = distances

    def reset(self):
        self.actor_list = []
        self.collision_hist = []
        self.spawn_point = np.random.choice(self.world.get_map().get_spawn_points())
        # self.spawn_point = self.world.get_map().get_spawn_points()[3]
        self.spawn_point.rotation.yaw += random.randint(-5, 5)
        self.r25 = self.world.spawn_actor(self.bp_r25, self.spawn_point)
        self.actor_list.append(self.r25)

        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        self.camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        self.camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=2.0, z=0.6))
        self.camera = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.r25)
        self.actor_list.append(self.camera)

        self.lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('range', '50')  # Range in meters
        self.lidar_bp.set_attribute('rotation_frequency', '10')  # Rotations per second
        self.lidar_bp.set_attribute('channels', '1')  # Single horizontal scan
        self.lidar_bp.set_attribute('upper_fov', '0')  # Horizontal scan at 0 degrees
        self.lidar_bp.set_attribute('lower_fov', '0')  # Horizontal scan at 0 degrees
        self.lidar_bp.set_attribute('points_per_second', '10000')  # Fewer points for simplicity
        self.lidar_bp.set_attribute('sensor_tick', '0.1')  # Sensor update rate
        lidar_transform = carla.Transform(carla.Location(x=2.5, z=0.6))
        self.lidar = self.world.spawn_actor(self.lidar_bp, lidar_transform, attach_to=self.r25)
        self.actor_list.append(self.lidar)

        self.camera.listen(lambda image: self.process(image))
        self.lidar.listen(lambda data: self.process_lidar(data))

        self.r25.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))

        self.colsensor_bp = self.blueprint_library.find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(self.colsensor_bp, carla.Transform(), attach_to=self.r25)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.on_collision(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.r25.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))

        return self.front_camera

    def step(self, action: np.ndarray):
        throttle = float(action[0])
        brake = 0
        steer_right = float(action[2])
        steer_left = float(action[3])
        steer = float(steer_right - steer_left)

        self.r25.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

        lidar = np.array(self.lidar_cast_ray)
        lidar_right = sum(lidar[0:3]) / 3
        lidar_left = sum(lidar[4:7]) / 3
        lidar_front = lidar[3]

        diff = abs(lidar_right - lidar_left)

        v = self.r25.get_velocity()
        kmh = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))

        reward = 0
        done = False

        if len(self.collision_hist) != 0:
            done = True
            reward -= 2000
        
        if brake > 0.001:
            reward -= 50
        elif diff < 1.0 and throttle > 0.8:
            reward += 15
        elif throttle > 0.4:
            reward += 4


        if diff < 0.6 and brake > 0.001:
            reward -= 20

        if lidar_front > 5.0 and diff < 1.0:
            if kmh < 1:
                reward = -30
            elif 1 < kmh < 20:
                reward = 12 - kmh
            else:
                reward += 1.5 * lidar_front
        elif diff <= 2:
            reward += -(diff - 2)/2 * 30
        elif diff < 10:
            reward += -((diff - 2)/10) * 90
        else:
            reward -= 100.0

        if (time.time() - self.episode_start) > EPISODES_LENGTH:
            done = True

        # print(f"Diff: {diff:.6f}")

        return self.front_camera, reward, done, None


class SACAgent:
    def __init__(self, state_shape=(1, 48, 64), action_dim=4, lr_actor=3e-4, lr_critic=3e-4, alpha=0.2, tau=0.005):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_shape = state_shape

        # Initialize networks
        self.actor = Actor(state_shape, action_dim).to(self.device)
        self.critic1 = Critic(state_shape, action_dim).to(self.device)
        self.critic2 = Critic(state_shape, action_dim).to(self.device)
        self.target_critic1 = Critic(state_shape, action_dim).to(self.device)
        self.target_critic2 = Critic(state_shape, action_dim).to(self.device)

        # Copy weights from critics to target critics
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # SAC parameters
        self.alpha = alpha  # Entropy coefficient
        self.tau = tau  # Soft update rate
        self.gamma = 0.99  # Discount factor

        # Replay buffer and logging
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/SAC-{int(time.time())}")
        self.target_update_counter = 0

        # Training flags
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def update_replay_memory(self, transition):
        """Stores (state, action, reward, next_state, done) in replay memory."""
        self.replay_memory.append(transition)

    def select_action(self, state, deterministic=False):
        """Selects an action using the current policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.actor(state)

        if deterministic:
            return mean.cpu().detach().numpy()[0]
        
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.sigmoid(z)  # Squash output to [0,1] range
        return action.cpu().detach().numpy()[0]

    def train(self):
        """Trains the SAC agent using sampled experiences."""
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        states = torch.tensor(np.array([t[0] for t in minibatch]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([t[1] for t in minibatch]), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array([t[2] for t in minibatch]), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array([t[3] for t in minibatch]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array([t[4] for t in minibatch]), dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute target Q-values using min(Q1, Q2) for stability
        with torch.no_grad():
            next_actions, log_probs = self.actor.sample_action(next_states)
            next_Q1 = self.target_critic1(next_states, next_actions)
            next_Q2 = self.target_critic2(next_states, next_actions)
            next_Q = torch.min(next_Q1, next_Q2) - self.alpha * log_probs
            target_Q = rewards + self.gamma * (1 - dones) * next_Q

        # Train Critic 1
        Q1 = self.critic1(states, actions)
        critic1_loss = nn.MSELoss()(Q1, target_Q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Train Critic 2
        Q2 = self.critic2(states, actions)
        critic2_loss = nn.MSELoss()(Q2, target_Q)

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Train Actor
        new_actions, log_probs = self.actor.sample_action(states)
        Q1_new = self.critic1(states, new_actions)
        Q2_new = self.critic2(states, new_actions)
        Q_min = torch.min(Q1_new, Q2_new)

        actor_loss = (self.alpha * log_probs - Q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Logging
        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        if log_this_step:
            self.tensorboard.update_stats(critic1_loss=critic1_loss.item(), critic2_loss=critic2_loss.item(), actor_loss=actor_loss.item())

    def train_in_loop(self):
        """Continuously trains the agent in a loop."""
        # Initialize network with dummy input
        dummy_input = np.random.rand(1, *self.state_shape).astype(np.float32)
        with torch.no_grad():
            self.actor(torch.tensor(dummy_input).to(self.device))
            self.critic1(torch.tensor(dummy_input).to(self.device), torch.zeros((1, ACTION_DIM)).to(self.device))
            self.critic2(torch.tensor(dummy_input).to(self.device), torch.zeros((1, ACTION_DIM)).to(self.device))

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    
    args = argparser.parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    
    FPS = 30
    ep_rewards = [-200]

    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)

    if not os.path.exists('models'):
        os.makedirs('models')

    agent = SACAgent(state_shape=(1, RESIZE_HEIGHT, RESIZE_WIDTH), action_dim=ACTION_DIM)
    env = CarlaEnv(client=client)

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()

    while not agent.training_initialized:
        time.sleep(0.01)

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        time.sleep(1)
        env.collision_hist = []
        agent.tensorboard.step = episode  # Track episode in TensorBoard
        episode_reward = 0
        step = 0
        current_state = env.reset()
        done = False
        episode_start = time.time()
        logs = []

        while not done:
            step += 1

            # **Select action using SAC agent**
            action = agent.select_action(current_state, deterministic=False)
            print(action)

            # **Take step in environment**
            new_state, reward, done, _ = env.step(action)
            print(reward)

            # **Store transition in replay buffer**
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            episode_reward += reward
            logs.append((episode, action.tolist(), reward, episode_reward))
            current_state = new_state
            time.sleep(1/FPS)

        # **Destroy actors at end of episode**
        for actor in env.actor_list:
            actor.destroy()
        
        # save logs in log file
        with open(f'logs/{MODEL_NAME}.txt', 'a') as f:
            for log in logs:
                f.write(f"{log}\n")

        # **Log episode rewards**
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            
            agent.tensorboard.log_episode_end(
                episode, avg_reward=average_reward, min_reward=min_reward, max_reward=max_reward
            )

            # **Save model if performance is good**
            if min_reward >= MIN_REWARD:
                torch.save(agent.actor.state_dict(), f'models/{MODEL_NAME}_actor_{average_reward:.2f}.pth')

    # **Terminate agent and close TensorBoard**
    agent.terminate = True
    trainer_thread.join()
    agent.tensorboard.close()

    # **Final save**
    torch.save(agent.actor.state_dict(), f'models/{MODEL_NAME}_final_actor.pth')
