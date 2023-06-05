import argparse
import itertools
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from tqdm import tqdm

ONE_ENV_ALGS = {DDPG, TD3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Path to save model as')
    parser.add_argument('--total_timesteps', type=int, default=100_000, help='The total number of samples (env steps) to train on')
    parser.add_argument('--eval_freq', type=int, default=10000, help='Evaluate the agent every eval_freq.')
    parser.add_argument('--reward_type', type=str, choices=["Dense", "Sparse"], default="Dense", help='The reward type, i.e. sparse or dense')
    parser.add_argument('--dof', type=int, default=3, help='The number of degrees of freedom of the robot')
    parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat training')
    parser.add_argument('--std', type=float, default=0.1, help='Scale of the noise')
    args = parser.parse_args()
    reward_type = "Dense" if args.reward_type == "Dense" else ""
    dof = "3DOF" if args.dof == 3 else ""
    env_id = f'WAMVisualReach{reward_type}{dof}-v2'
    algs = [TD3]

    # vec_env = make_vec_env(env_id, n_envs=1, vec_env_cls=SubprocVecEnv)
    # algs = [PPO]  # , SAC, A2C]
    env = gym.make(env_id)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=args.std * np.ones(n_actions)
    )

    goal_selection_strategy = 'future'

    # for alg, hidden_size, depth in tqdm(itertools.product(algs, hidden_sizes, depths), total=len(hidden_sizes) * len(depths)):
    for _ in tqdm(range(args.repeat)):
        for alg in tqdm(algs):
            n_envs = 1 if alg in ONE_ENV_ALGS else os.cpu_count()
            vec_action_noise = action_noise if alg in ONE_ENV_ALGS else VectorizedActionNoise(action_noise, n_envs=n_envs)
            vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=DummyVecEnv)
            # if Path(f"./logs/{env_id}_{hidden_size}_{depth}/").exists():
            #     continue
            model = alg(
                "MultiInputPolicy",
                vec_env,
                # verbose=1,
                # batch_size=512,
                tensorboard_log=f"./logs/{args.model_path}/",
                # replay_buffer_class=HerReplayBuffer,
                # # Parameters for HER
                # replay_buffer_kwargs=dict(
                #     n_sampled_goal=4,
                #     goal_selection_strategy=goal_selection_strategy,
                # ),
                action_noise=vec_action_noise,
                # policy_kwargs=dict(
                #    net_arch=dict(
                #       pi=[1024, 1024],
                #       qf=[1024, 1024],
                #    )
                # ),
            )
            print(model.policy)
            print(f'{sum(param.numel() for param in model.policy.parameters()) / 1e6:.2f} M parameters')
            callback_on_best = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=3,
                verbose=1
            )
            eval_callback = EvalCallback(
                vec_env,
                callback_on_new_best=callback_on_best,
                eval_freq=args.eval_freq,
                verbose=1,
                best_model_save_path=args.model_path,
            )

            model.learn(
                total_timesteps=args.total_timesteps,
                callback=eval_callback,
                progress_bar=True
            )


if __name__ == "__main__":
    main()
