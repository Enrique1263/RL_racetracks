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
from PIL import Image

from torchvision import models, transforms
from models import R25MLP
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
SHOW_CAM = False

EPISODES_LENGTH = 10
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 2

MODEL_NAME = 'R25CamNet'

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10

MIN_REWARD = 10


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


class AdHocVGG19:
    def __init__(self):
        self.model = models.vgg19(pretrained=True).features
        self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.eval()

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        x = Image.fromarray(x)
        x = self.transforms(x).unsqueeze(0)
        x = x.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        with torch.no_grad():
            x = self.model(x)
        return x.squeeze(0).view(-1).cpu().numpy().tolist()


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
        self.VGG19 = AdHocVGG19()

    def on_collision(self, event):
        self.collision_hist.append(event)

    def process(self, image):
        # show image
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.show_cam:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def reset(self):
        self.actor_list = []
        self.collision_hist = []
        self.spawn_point = self.world.get_map().get_spawn_points()[0]
        self.r25 = self.world.spawn_actor(self.bp_r25, self.spawn_point)
        self.actor_list.append(self.r25)

        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        self.camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        self.camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=2.0, z=0.6))
        self.camera = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.r25)
        self.actor_list.append(self.camera)

        self.camera.listen(lambda image: self.process(image))

        self.r25.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))

        self.colsensor_bp = self.blueprint_library.find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(self.colsensor_bp, carla.Transform(), attach_to=self.r25)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.on_collision(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.r25.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))

        camera_features = self.VGG19.forward(self.front_camera)

        current_state = camera_features + [0, 0, 0, 0, 0] # Add speed, throttle, brake, steer_right, steer_left

        return current_state

    def step(self, action: np.ndarray):
        throttle = float(action[0])
        brake = float(action[1])
        steer_right = float(action[2])
        steer_left = float(action[3])
        steer = float(steer_right - steer_left)

        self.r25.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

        v = self.r25.get_velocity()
        kmh = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))

        reward = 0
        done = False

        if len(self.collision_hist) != 0:
            done = True
            reward = -20
        elif (throttle - brake) > 0.8:
            reward += 0.5
        elif (throttle - brake) > 0.5:
            reward += 0.1
        elif (throttle - brake) > 0.1:
            reward += 0.01
        elif (throttle - brake) <= 0 and (throttle - brake) > -0.5:
            reward -= 0.1
        elif (throttle - brake) <= -0.5:
            reward -= 0.5
        

        if (time.time() - self.episode_start) > EPISODES_LENGTH:
            done = True


        new_camera_features = self.VGG19.forward(self.front_camera)
        new_state = new_camera_features + [kmh ,throttle, brake, steer_right, steer_left] # Add speed, throttle, brake, steer_right, steer_left

        return new_state, reward, done, None


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
        model = R25MLP(input_shape=(25088 + 5,)) # 25088 is the number of features from VGG19, 5 is the number of additional features
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition) # transition is a tuple of (current_state, action, reward, new_state, done)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # print(f'Minibatch: {minibatch}')

        current_states = torch.tensor(np.array([transition[0] for transition in minibatch]), dtype=torch.float32).to(self.device)
        new_current_states = torch.tensor(np.array([transition[3] for transition in minibatch]), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            current_qs_list = self.model(current_states).cpu().numpy()
            future_qs_list = self.target_model(new_current_states).cpu().numpy()

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = future_qs_list[index]
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[:] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        X = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
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
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            qs = self.model(state)
            return qs
    
    def train_in_loop(self):
        X = np.random.uniform(size=(25088 + 5,)).astype(np.float32) # 25088 is the number of features from VGG19, 5 is the number of additional features

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
    
    FPS = 15
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

    agent.get_qs(np.random.uniform(size=(25088 + 5,)).astype(np.float32)) # 25088 is the number of features from VGG19, 5 is the number of additional features

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
                print(f'Action: {action}')
            else:
                action = np.random.rand(4)
                action[1] = 0
                time.sleep(1/FPS)
                print(f'Random Action: {action}')


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
                torch.save(agent.model.state_dict(), f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.pth')

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    agent.terminate = True
    trainer_thread.join()
    agent.tensorboard.close()

    torch.save(agent.model.state_dict(), f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.pth')