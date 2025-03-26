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
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import MSELoss
from threading import Thread

from models import R25Tiny

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

SHOW_CAM = False
SHOW_LIDAR = False

EPISODES_LENGTH = 12
REPLAY_MEMORY_SIZE = 10000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5

MODEL_NAME = 'R25CamNetwLiDAR'

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10

MIN_REWARD = 150


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
        # print(points)
        # print('-'*20)

        # TODO: Check if this is necessary
        # rotation = lidar_transform.rotation

        # # Create a rotation matrix from the LiDAR's orientation (pitch, yaw, roll)
        # # LiDAR rotation: pitch, yaw, roll in degrees, convert to radians
        # pitch, yaw, roll = np.radians([rotation.pitch, rotation.yaw, rotation.roll])

        # # Construct rotation matrix (3x3) based on LiDAR's orientation
        # rotation_matrix = np.array([
        #     [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        #     [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        #     [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
        # ])

        # # Transform the points to the LiDAR's local coordinate system
        # points_local = np.dot(points, rotation_matrix.T)

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
        # view = np.array(self.front_camera) / 255.0
        throttle = float(action[0])
        brake = round(float(action[1]), 3) # TODO: see if rounding helps
        steer_right = float(action[2])
        steer_left = float(action[3])
        steer = float(steer_right - steer_left)

        self.r25.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

        lidar = np.array(self.lidar_cast_ray)
        lidar_right = sum(lidar[0:3]) / 3
        lidar_left = sum(lidar[4:7]) / 3

        diff = abs(lidar_right - lidar_left)

        v = self.r25.get_velocity()
        kmh = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))

        reward = 0
        done = False

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        # elif kmh < 10:
        #     done = False
        #     reward = -10
        # elif 10 < kmh < 50:
        #     done = False
        #     reward = -1
        # else:
        #     done = False
        #     reward = 10
        
        if brake > 0.001:
            reward -= 0.5

        if throttle > 0.4:
            reward += 0.4

        if diff < 1.0 and throttle > 0.8:
            reward += 0.8

        if diff < 0.6 and brake > 0.001:
            reward -= 0.8

        if diff < 2:
            reward += -(diff - 2)/3
        elif diff < 10:
            reward += -((diff - 2)/10)
        else:
            reward -= 1.0

        if (time.time() - self.episode_start) > EPISODES_LENGTH:
            done = True

        return self.front_camera, reward, done, None


class DQNAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.create_model().to(self.device)
        self.target_model = self.create_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

        self.criterion = MSELoss()  # Example loss function, change as needed
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def create_model(self):
        model = R25Tiny(input_shape=(1, RESIZE_HEIGHT, RESIZE_WIDTH))
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition) # transition is a tuple of (current_state, action, reward, new_state, done)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = torch.tensor(np.array([transition[0] for transition in minibatch]) / 255.0, dtype=torch.float32).to(self.device)
        new_current_states = torch.tensor(np.array([transition[3] for transition in minibatch]) / 255.0, dtype=torch.float32).to(self.device)

        # predecir solo qs para el futuro, se entrenará usando la accion real y no la predicha
        with torch.no_grad():
            predicted_actions_list = self.model(current_states)
            future_predicted_actions_list = self.target_model(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                # For the next state, use the future predicted actions from the target model
                future_predicted_actions = future_predicted_actions_list[index].cpu().numpy()
            else:
                # If done, we don't have future predictions, so we just rely on the current reward
                future_predicted_actions = action

            # Apply reward scaling (soft update)
            # Instead of multiplying by reward, we use reward to scale the difference between predicted and true actions
            # This helps in learning the real action values
            target_action = predicted_actions_list[index].cpu().numpy() + (reward * (action - predicted_actions_list[index].cpu().numpy()))

            X.append(current_state)
            y.append(target_action)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        X = torch.tensor(np.array(X) / 255.0, dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(y), dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=False)

        self.model.train()
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            y_pred = self.model(X_batch)

            loss = self.criterion(y_pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if log_this_step:
            self.tensorboard.update_stats(loss=loss.item())
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

    def get_qs(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device) / 255.0
            qs = self.model(state)
            return qs[0]
    
    def train_in_loop(self):
        X = np.random.randint(0, 1, size=(1, 1, RESIZE_HEIGHT, RESIZE_WIDTH)).astype(np.float32)

        with torch.no_grad():
            self.model(torch.tensor(X).to(self.device))
            self.target_model(torch.tensor(X).to(self.device))

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
    
    FPS = 60
    ep_rewards = [-200]

    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)

    if not os.path.exists('models'):
        os.makedirs('models')

    agent = DQNAgent()
    env = CarlaEnv(client=client)

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()

    while not agent.training_initialized:
        time.sleep(0.01)

    agent.get_qs(np.random.randint(0, 1, size=(1, 1, RESIZE_HEIGHT, RESIZE_WIDTH)).astype(np.float32))

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.collision_hist = []
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()

        while True:
            if np.random.random() > epsilon:
                action = agent.get_qs(current_state).cpu().numpy()
                action = np.array([action[0], action[1], action[2], action[3]])
                print(f'Action: {action}')
                time.sleep(1/FPS)
            else:
                action = np.random.rand(4)
                action[1] = 0 if np.random.rand() > 0.2 else action[1]
                time.sleep(1/FPS)
                # print(f'Random Action: {action}')


            new_state, reward, done, _ = env.step(action)
            print(f'Reward: {reward}')
            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            if done:
                break

        for actor in env.actor_list:
            actor.destroy()

        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.log_episode_end(episode, 
                                              avg_reward=average_reward, 
                                              min_reward=min_reward, 
                                              max_reward=max_reward)
            
            if min_reward >= MIN_REWARD:
                torch.save(agent.model, f'models/{MODEL_NAME}__{average_reward:_>7.2f}avg__{int(time.time())}.pth')

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    agent.terminate = True
    trainer_thread.join()
    agent.tensorboard.close()

    torch.save(agent.model, f'models/{MODEL_NAME}__{average_reward:_>7.2f}avg__{int(time.time())}.pth')