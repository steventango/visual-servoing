import gymnasium as gym
import cv2 as cv
import numpy as np


def main():
    """
    Renders the reward at image points
    """
    env = gym.make('WAMReachDense3DOF-v2', render_mode='human')

    observation, info = env.reset()
    image = np.zeros((480 * 2, 640, 3), dtype=np.uint8)
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        points = np.concatenate([
            observation['achieved_goal'],
            observation['desired_goal']
        ]).reshape(4, 2)
        points[:, 0] *= 480
        points[:, 0] += 640 // 2
        points[:, 1] *= 480
        points[:, 1] += 480 // 2
        points = points.astype(np.int64)
        points[1::2, 1] += 480
        RED = (0, 0, 255)
        color = (255 - max(-reward / 0.3, 0) * 255, 0, 0)
        image[tuple(points[0][::-1])] = np.maximum(image[tuple(points[0][::-1])], color)
        image[tuple(points[1][::-1])] = np.maximum(image[tuple(points[1][::-1])], color)
        image_copy = image.copy()
        image_copy = cv.circle(image_copy, points[2], 10, RED, -1)
        image_copy = cv.circle(image_copy, points[3], 10, RED, -1)

        cv.imshow('image', image_copy)
        cv.waitKey(1)
        env.render()

        if terminated or truncated:
            observation, info = env.reset()


if __name__ == "__main__":
    main()
