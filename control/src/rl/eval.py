import argparse

import cv2 as cv
import gymnasium as gym
import numpy as np
from uvs import UVS
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from tqdm import tqdm
from train import ALGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, nargs='?', default=None)
    parser.add_argument(
        '--render_mode',
        type=str,
        default='human',
        choices=['human', 'rgb_array', 'none']
    )
    parser.add_argument(
        '--alg',
        type=str,
        default='TD3',
        choices=['TD3', 'UVS']
    )
    args = parser.parse_args()
    env_id = 'WAMVisualReachDense4DOF-v2'
    env = gym.make(env_id, render_mode=None if args.render_mode == 'none' else args.render_mode, max_episode_steps=200)
    # eval_callback = EvalCallback(
    #     env,
    #     n_eval_episodes=args.n_eval_episodes,
    #     eval_freq=args.eval_freq,
    #     log_path=args.eval_log_path,
    #     verbose=args.verbose,
    #     best_model_save_path=args.model_path,
    # )
    alg = ALGS[args.alg]
    if args.alg == 'UVS':
        model = UVS("MultiInputPolicy", env)
        model.learn(100)
    else:
        try:
            model = alg.load(args.model_path, env)
        except IsADirectoryError:
            model = alg.load(args.model_path + '/best_model.zip', env)
    rewards = np.zeros(100000)
    episode_reward = 0
    successes = 0
    n_episodes = 0

    observation, info = env.reset()
    terminated, truncated = False, False

    for i in tqdm(range(100000)):
        # if not uvs_model.initialized:
        #     action, states_ = uvs_model.predict(observation, terminated, truncated)
        # elif J is None:
        #     J = uvs_model.J
        #     print(J)
        # else:
        #     action, states_ = model.predict(observation)
        #     actorJ = model.actor.J
        # print(actorJ)
        action, states_ = model.predict(observation)

        observation, reward, terminated, truncated, info = env.step(action)

        if args.render_mode == 'human':
            render_image_observation(observation)

        episode_reward += reward
        if terminated or truncated:
            rewards[i] = episode_reward
            episode_reward = 0
            if terminated:
                successes += 1
                if args.render_mode == 'human':
                    import time
                    time.sleep(1)
            observation, info = env.reset()
            n_episodes += 1

        # if i > 0 and i % args.eval_freq == 0:
        #     eval_reward = eval(env, model, args.n_eval_episodes)
        #     print(f"Average Reward: {np.mean(rewards):.2f} +- {np.std(rewards):.2f}")
        #     print(f"Success ({successes}/{n_episodes})")
        #     if n_episodes > 0:
        #         print(f"Success Rate: {successes / n_episodes:.2f}")
    env.close()


def render_image_observation(observation):
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


if __name__ == "__main__":
    main()
