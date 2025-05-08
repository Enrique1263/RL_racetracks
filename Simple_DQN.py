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

from models import R25XY

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse

EPISODES_LENGTH = 30
REPLAY_MEMORY_SIZE = 10000
MIN_REPLAY_MEMORY_SIZE = 200
MINIBATCH_SIZE = 64
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5

MODEL_NAME = 'R25XY'

EPISODES = 1500

DISCOUNT = 0.99
# epsilon = 0.25
epsilon = 0.9
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.2

AGGREGATE_STATS_EVERY = 10

MIN_REWARD = 55000

NUM_TIMES_PASSED = 40
START_STATE_CHECKPOINT = 0
memory_flag = False


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
    def __init__(self, client, epsilon=1.0, memory_flag=False, state_checkpoint=0):
        self.client = client
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.bp_r25 = self.blueprint_library.filter('R25')[0]
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None
        self.checkpoints = [(22.40, 29.70), (-5.3, 52.8), (-14.50, 53.50), (-33.01, 48.57), (-51.75, 36.25), (-44.48, 22.41), (-30.92, 23.95), (-22.60, 21.53), (-21.39, 16.18), (-39.31, -9.51)]
        self.checkpoint = 0
        self.px = None
        self.py = None
        self.state_checkpoint = state_checkpoint
        self.times_passed = 0
        self.d = 0
        self.epsilon = epsilon
        self.memory_flag = memory_flag

    def on_collision(self, event):
        self.collision_hist.append(event)

    def reset(self):
        self.actor_list = []
        self.collision_hist = []
        self.checkpoint = 0
        # self.spawn_point = np.random.choice(self.world.get_map().get_spawn_points())
        self.spawn_point = self.world.get_map().get_spawn_points()[0]
        # self.spawn_point.rotation.yaw += random.randint(-5, 5)
        self.r25 = self.world.spawn_actor(self.bp_r25, self.spawn_point)
        self.actor_list.append(self.r25)

        self.r25.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))

        self.colsensor_bp = self.blueprint_library.find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(self.colsensor_bp, carla.Transform(), attach_to=self.r25)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.on_collision(event))

        self.episode_start = time.time()

        self.r25.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        v = self.r25.get_velocity()
        self.vx = round(v.x,2)
        self.vy = round(v.y,2)
        loc = self.r25.get_location()
        self.x = round(loc.x,2)
        self.y = round(loc.y,2)
        self.slow_reward = 0
        self.d = 0

        return (self.x, self.y, self.vx, self.vy)

    def step(self, action):
        self.px = self.x
        self.py = self.y
        if action is 0:
            # print("steer left")
            # steer left
            self.r25.apply_control(carla.VehicleControl(throttle=0.8, steer=-0.9))
        elif action is 1:
            # print("forward")
            # forward
            self.r25.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        elif action is 2:
            # print("steer right")
            # steer right
            self.r25.apply_control(carla.VehicleControl(throttle=0.8, steer=0.9))
        elif action is 3:
            # print("decelerate")
            # decelerate
            self.r25.apply_control(carla.VehicleControl(throttle=0.1))

        time.sleep(1/15)

        v = self.r25.get_velocity()
        kmh = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))
        self.vx = round(v.x,2)
        self.vy = round(v.y,2)
        loc = self.r25.get_location()
        self.x = round(loc.x,2)
        self.y = round(loc.y,2)

        if len(self.collision_hist) != 0:
            done = True
            reward = -8000 / self.d
        # elif np.sqrt((self.x - self.checkpoints[self.checkpoint][0]) ** 2 + (self.y - self.checkpoints[self.checkpoint][1]) ** 2) < 1:
        #     print(f"Checkpoint {self.checkpoint} reached")
        #     self.checkpoint += 1
        #     reward = 10000
        #     if self.checkpoint >= len(self.checkpoints):
        #         done = True
        #         reward = 2000
        #         print(f"Episode finished after {time.time() - self.episode_start} seconds")
        #     else:
        #         done = False
        # else:
        #     done = False
        #     reward = -1
        else:
            done = False
            d = np.sqrt((self.x - self.px) ** 2 + (self.y - self.py) ** 2)
            self.d += d
            if (time.time() - self.episode_start) > 3 and d < 0.2:
                reward = -20
                self.slow_reward -= 20
            elif d > 0.85:
                reward = d * 30
            else:
                reward = d * 10

        # if np.sqrt((self.x - 42.28) ** 2 + (self.y - 11.56) ** 2) < 1 and action != 1:
        #     reward -= 100
        # elif kmh < 10.0:
        #     reward += kmh -10.0
        if np.sqrt((self.x - self.checkpoints[self.checkpoint][0]) ** 2 + (self.y - self.checkpoints[self.checkpoint][1]) ** 2) < 1.05:
            print(f"Checkpoint {self.checkpoint} reached")
            if self.checkpoint == self.state_checkpoint:
                self.times_passed += 1
                if self.times_passed == NUM_TIMES_PASSED:
                    self.state_checkpoint += 1
                    self.times_passed = 0
                    self.epsilon = 0.95
                    self.memory_flag = True

            self.checkpoint += 1
            reward += 100

        if time.time() - self.episode_start > EPISODES_LENGTH:
            done = True

        return (self.x, self.y, self.vx, self.vy), reward, done, None


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

        self.model_list = []

    def create_model(self):
        model = R25XY(input_shape=(4))
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition) # transition is a tuple of (current_state, action, reward, new_state, done)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = torch.tensor(np.array([transition[0] for transition in minibatch]), dtype=torch.float32).to(self.device)
        new_current_states = torch.tensor(np.array([transition[3] for transition in minibatch]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            current_qs_list = self.model(current_states).cpu().numpy()
            future_qs_list = self.target_model(new_current_states).cpu()

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = reward + DISCOUNT * torch.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = max_future_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        X = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(y), dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        predictions = self.model(X)

        loss = self.criterion(predictions, y)
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
        X = np.random.randint(0, 1, size=(1, 4)).astype(np.float32)

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
    env = CarlaEnv(client=client, epsilon=epsilon, memory_flag=memory_flag)

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()

    while not agent.training_initialized:
        time.sleep(0.01)

    agent.get_qs(np.random.randint(0, 1, size=(1, 4)).astype(np.float32))

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        if env.memory_flag:
            # Save a frozen copy of the model into the model list
            model = agent.create_model()
            model.load_state_dict(agent.model.state_dict())
            torch.save(model, f'models/{MODEL_NAME}__sector{env.state_checkpoint - 1}.pth')
            agent.model_list.append(model)
            agent.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
            env.memory_flag = False
            print("Memory cleared")
        env.collision_hist = []
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()
        print(f"Waiting til {env.state_checkpoint - 1} checkpoint to explore")
        print(f"Checkpoint {env.state_checkpoint} has been passed {env.times_passed} times")
        print(f"Memory amount: {len(agent.replay_memory)}")
        while True:
            if env.state_checkpoint > env.checkpoint:
                # Load the frozen model from the model list (Not overwriting the current model)
                model = agent.model_list[env.checkpoint].to(agent.device)
                model.eval()
                with torch.no_grad():
                    action = model(torch.tensor(current_state, dtype=torch.float32).to(agent.device)).cpu().numpy()
                    action = int(np.argmax(action))
                    print(f'Model_{env.checkpoint}, Action: {action}')
            elif np.random.random() > env.epsilon:
                print("Using trainable model")
                action = agent.get_qs(current_state).cpu().numpy()
                # print(f'Q Values: {action}')
                action = int(np.argmax(action))
                # print(f'Action: {action}')
            else:
                action = np.random.randint(0, 4)
                # print(f'Random Action: {action}')


            new_state, reward, done, _ = env.step(action)
            # print(f'Reward: {reward}')
            episode_reward += reward
            if env.state_checkpoint <= env.checkpoint:
                agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            if done:
                break
        print(f"Episode {episode} finished after {step} steps with reward {episode_reward}")
        print(f"{env.slow_reward} slow reward")
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

        if env.epsilon > MIN_EPSILON:
            env.epsilon *= EPSILON_DECAY
            env.epsilon = max(MIN_EPSILON, env.epsilon)

    agent.terminate = True
    trainer_thread.join()
    agent.tensorboard.close()

    torch.save(agent.model, f'models/{MODEL_NAME}__{average_reward:_>7.2f}avg__{int(time.time())}.pth')