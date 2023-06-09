import argparse

import cv2 as cv
import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    args = parser.parse_args()
    env_id = 'WAMVisualReachDense3DOF-v2'
    env = gym.make(env_id, render_mode='human')
    alg = TD3
    try:
        model = alg.load(args.model_path, env)
    except IsADirectoryError:
        model = alg.load(args.model_path + '/best_model.zip', env)
    rewards = []
    episode_reward = 0
    successes = 0

    observation, info = env.reset()
    while True:
        action, states_ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
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
        episode_reward += reward
        if terminated or truncated:
            rewards.append(episode_reward)
            print(f"Average Reward: {np.mean(rewards):.2f} +- {np.std(rewards):.2f}")
            episode_reward = 0
            if terminated:
                successes += 1
                print(f"Success ({successes}/{len(rewards)})")
                import time
                time.sleep(1)
            print(f"Success Rate: {successes / len(rewards):.2f}")
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
