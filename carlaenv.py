import glob
import os
import sys
import numpy as np
import cv2
import time
import random
from tqdm import tqdm
import torch

from models import R25TinySimple

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

    def reset(self):
        self.actor_list = []
        self.collision_hist = []
        self.spawn_point = np.random.choice(self.world.get_map().get_spawn_points())
        # self.spawn_point.rotation.yaw += random.randint(-10, 10)
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

        return self.front_camera

    def step(self, action):
        # view = np.array(self.front_camera) / 255.0
        # throttle = float(action[0])
        # brake = round(float(action[1]), 3) # TODO: see if rounding helps
        # steer_right = float(action[2])
        # steer_left = float(action[3])
        # steer = float(steer_right - steer_left)

        # self.r25.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

        if action == 0:
            # steer left
            self.r25.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))
        elif action == 1:
            # forward
            self.r25.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        elif action == 2:
            # steer right
            self.r25.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0))
        elif action == 3:
            # brake
            self.r25.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        reward = 0
        done = False

        if len(self.collision_hist) != 0:
            reward = -200
        
        if (time.time() - self.episode_start) > EPISODES_LENGTH:
            done = True

        return self.front_camera, reward, done, None
    
if __name__ == '__main__':
    FPS = 120
    EPISODES_LENGTH = 15
    EPISODES = 10


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

    random.seed(5)
    np.random.seed(5)

    env = CarlaEnv(client=client)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = torch.load('models/R25Simple_third_low.pth')
    agent.eval()
    agent.to(device)


    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.collision_hist = []
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()

        while True:
            state = torch.tensor(current_state, dtype=torch.float32).to(device) / 255.0
            action = agent(state)[0].detach().cpu().numpy()
            # action = np.array([action[0], action[1], action[2], action[3]])
            action = np.argmax(action)
            print(action)
            time.sleep(1/FPS)

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward

            current_state = new_state
            step += 1

            if done:
                break

        print(f'Episode: {episode}, Reward: {episode_reward}, Steps: {step}, Time: {time.time() - episode_start}')

        for actor in env.actor_list:
            actor.destroy()