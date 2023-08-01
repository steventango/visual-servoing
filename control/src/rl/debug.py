import cv2 as cv
import gymnasium as gym
import numpy as np
from uvs import UVS
from stable_baselines3 import TD3
import time


def main():
    env = gym.make('WAMVisualReachDense-v2', render_mode='human')
    # env = gym.make('FetchReach-v2', render_mode='human')
    # TODO initial joint positioning
    # TODO better code for baselines / comparisions / plots perhaps / video generation
    # env = gym.make('FetchReach-v2', render_mode='human')

    # model = TD3("CustomMultiInputPolicy", env)
    model = UVS(
        "MultiInputPolicy",
        env,
        learning_rate=1
    )
    model.learn(4 * env.action_space.shape[0])
    print(model.policy.J)

    print(env.action_space)
    print(env.observation_space)

    observation, info = env.reset()
    prev_observation = None
    while True:
        action, states_ = model.predict(observation)
        # action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        # print(f"Observation (observation: {observation['observation'].shape}, desired_goal: {observation['desired_goal'].shape}, achieved_goal: {observation['achieved_goal'].shape}: {observation}")
        # print(f"Action ({action.shape}): {action}")
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
            MAGENTA = (255, 0, 255)
            image = cv.circle(image, points[0], 10, BLUE, -1)
            image = cv.circle(image, points[1], 10, BLUE, -1)
            image = cv.circle(image, points[2], 10, RED, -1)
            image = cv.circle(image, points[3], 10, RED, -1)
            if np.linalg.norm(points[0] - points[2]) > np.finfo(float).eps:
                image = cv.arrowedLine(image, points[0], points[2], MAGENTA, 5, tipLength=20 / np.linalg.norm(points[0] - points[2]))
                image = cv.arrowedLine(image, points[1], points[3], MAGENTA, 5, tipLength=20 / np.linalg.norm(points[1] - points[3]))
            if prev_observation is not None:
                dq = observation['observation'] - prev_observation
                error = np.array(model.policy.J.squeeze()) @ dq
                error *= 480
                # error *= 10
                image = cv.arrowedLine(image, points[0], points[0] + error[:2].astype(np.int64), (0, 255, 0), 5, tipLength=20 / np.linalg.norm(error[:2]))
                image = cv.arrowedLine(image, points[1], points[1] + error[2:].astype(np.int64), (0, 255, 0), 5, tipLength=20 / np.linalg.norm(error[2:]))
            prev_observation = observation['observation'].copy()
            cv.imshow('image', image)
            cv.waitKey(1)
        except ValueError:
            pass

        env.render()

        if terminated or truncated:
            observation, info = env.reset()
            prev_observation = None
        else:
            # input("Press a key to continue...")
            pass

    env.close()


if __name__ == "__main__":
    main()
