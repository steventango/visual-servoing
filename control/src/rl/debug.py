import gymnasium as gym
import cv2 as cv
import numpy as np


def main():
    env = gym.make('WAMVisualReachDense3DOF-v2', render_mode='human')
    # env = gym.make('FetchReach-v2', render_mode='human')
    # TODO initial joint positioning
    # TODO better code for baselines / comparisions / plots perhaps / video generation
    # env = gym.make('FetchReach-v2', render_mode='human')

    print(env.action_space)
    print(env.observation_space)

    observation, info = env.reset()
    while True:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        # print("Observation:")
        # print(observation, observation['observation'].shape, observation['desired_goal'].shape, observation['achieved_goal'].shape)
        # print("Action:")
        # print(action.shape, action)
        try:
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
            image = np.ones((480 * 2, 640, 3), dtype=np.uint8) * 255
            BLUE = (255, 0, 0)
            RED = (0, 0, 255)
            image = cv.circle(image, points[0], 10, BLUE, -1)
            image = cv.circle(image, points[1], 10, BLUE, -1)
            image = cv.circle(image, points[2], 10, RED, -1)
            image = cv.circle(image, points[3], 10, RED, -1)

            cv.imshow('image', image)
            cv.waitKey(1)
        except ValueError:
            pass

        env.render()

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
