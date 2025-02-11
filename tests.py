import glob
import os
import sys
import numpy as np
import cv2
import time
import random
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

def process(image):
    # show image
    i0 = np.array(image.raw_data)
    i2 = i0.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    # convert to b/w with threshold
    i3 = cv2.cvtColor(i3, cv2.COLOR_BGR2GRAY)
    _, i3 = cv2.threshold(i3, 50, 255, cv2.THRESH_BINARY)
    # resize image to 6x4
    i3 = cv2.resize(i3, (64, 48), interpolation=cv2.INTER_AREA)
    _, i3 = cv2.threshold(i3, 170, 255, cv2.THRESH_BINARY) # 170, 200, current options
    if SHOW_CAM:
        # save image to disk
        cv2.imwrite(f'./images/frame{i}.png', i3)
    
    i3 = i3.reshape(1, 48, 64)/255
    # sumar los valores de las dos ultimas columnas de la imagen
    # suma[i] = np.sum(i3[:, :, 36:])

def process_lidar(data):
    raw_data = np.frombuffer(data.raw_data, dtype=np.float32)
    points = np.reshape(raw_data, (-1, 3))  # Each point is (x, y, z)
    # print(points)
    # print('-'*20)

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

    min_angle = -180  # Adjust the min and max angles for the desired FOV (here -90 to 90 degrees)
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

            if angle == -90:
                color = (255, 0, 0)

            # Draw a line connecting the LiDAR position to the point
            cv2.line(image, lidar_pos, (x_img, y_img), color, 1)

            # Annotate the angle for reference
            cv2.putText(image, f"{distance:.2f}", (x_img + 5, y_img - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        else:
            distances.append(50)

    # print(distances)
    print(f'Right side mean distance: {np.mean(distances[0:3]):.2f} m. Left side mean distance: {np.mean(distances[4:]):.2f} m. Difference: {np.mean(distances[0:3]) - np.mean(distances[4:]):.2f} m.')
    cv2.imwrite(f"./lidar/lidar_points_frame{i}.png", image)



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

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    bp_r25 = blueprint_library.filter('R25')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    spawn_point.rotation.yaw += random.randint(-10, 10)
    r25 = world.spawn_actor(bp_r25, spawn_point)
    actor_list = []
    actor_list.append(r25)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    camera_bp.set_attribute('fov', '110')
    camera_transform = carla.Transform(carla.Location(x=2.0, z=0.6))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=r25)
    actor_list.append(camera)

    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '50')  # Range in meters
    lidar_bp.set_attribute('rotation_frequency', '10')  # Rotations per second
    lidar_bp.set_attribute('channels', '1')  # Single horizontal scan
    lidar_bp.set_attribute('upper_fov', '0')  # Horizontal scan at 0 degrees
    lidar_bp.set_attribute('lower_fov', '0')  # Horizontal scan at 0 degrees
    lidar_bp.set_attribute('points_per_second', '100000')  # Fewer points for simplicity
    lidar_bp.set_attribute('sensor_tick', '0.1')  # Sensor update rate
    lidar_transform = carla.Transform(carla.Location(x=2.5, z=0.8))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=r25)
    actor_list.append(lidar)

    # suma = np.zeros(11, dtype=np.uint8)
    camera.listen(lambda data: process(data))
    lidar.listen(lambda data: process_lidar(data))

    i = 0
    r25.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
    time.sleep(0.1)
    r25.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

    time.sleep(2)
    while i < 10:
        print(f'frame {i}')
        action = np.random.rand(4)
        action[1] = 0
        throttle = float(action[0])
        brake = float(action[1])
        steer_right = float(action[2])
        steer_left = float(action[3])
        steer = float(steer_right - steer_left)
        r25.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        i += 1
        time.sleep(1)
    print('Done')

    for actor in actor_list:
        actor.destroy()
        # print('Actor destroyed')
    print('All cleaned up!')
