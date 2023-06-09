import argparse
import itertools
import os

import custom_policy
import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import (
    EvalCallback
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from tqdm import tqdm

ONE_ENV_ALGS = {DDPG, TD3, SAC}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to save model as")
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100_000,
        help="The total number of samples (env steps) to train on",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10000,
        help="Evaluate the agent every eval_freq.",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=1000,
        help="The number of episodes to test the agent",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        choices=["Dense", "Sparse"],
        default="Dense",
        help="The reward type, i.e. sparse or dense",
    )
    parser.add_argument(
        "--dof",
        type=int,
        default=3,
        help="The number of degrees of freedom of the robot",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat training",
    )
    parser.add_argument(
        "--std", type=float, default=0.1, help="Scale of the noise"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Number of hidden units in the policy network",
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbose level',
    )
    parser.add_argument(
        '--no_progress_bar',
        action='store_true',
        help='Do not display a progress bar using tqdm and rich',
    )
    args = parser.parse_args(argv)
    reward_type = "Dense" if args.reward_type == "Dense" else ""
    dof = "3DOF" if args.dof == 3 else ""
    args.env_id = f"WAMVisualReach{reward_type}{dof}-v2"
    return args


def train(args):
    env = gym.make(args.env_id)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=args.std * np.ones(n_actions)
    )
    alg = TD3
    for _ in range(args.repeat):
        n_envs = 1 if alg in ONE_ENV_ALGS else os.cpu_count()
        vec_action_noise = (
                action_noise
                if alg in ONE_ENV_ALGS
                else VectorizedActionNoise(action_noise, n_envs=n_envs)
            )
        vec_env = make_vec_env(
            args.env_id, n_envs=n_envs, vec_env_cls=DummyVecEnv
        )
        model = alg(
            "CustomMultiInputPolicy",
            vec_env,
            # verbose=1,
            # batch_size=512,
            tensorboard_log=f"./logs/{args.model_path}_{args.hidden_size}/",
            # replay_buffer_class=HerReplayBuffer,
            # # Parameters for HER
            # replay_buffer_kwargs=dict(
            #     n_sampled_goal=4,
            #     goal_selection_strategy=goal_selection_strategy,
            # ),
            action_noise=vec_action_noise,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[args.hidden_size],
                    qf=[args.hidden_size],
                ),
                # optimizer_class=torch.optim.AdamW,
                share_features_extractor=True,
            ),
        )
        if args.verbose >= 1:
            print(model.policy)
        num_params = sum(p.numel() for p in model.policy.parameters())
        param_str = ""
        if num_params > 1e6:
            param_str = f"{num_params / 1e6:.2f} M parameters"
        else:
            param_str = f"{num_params} parameters"
        if args.verbose >= 1:
            print(param_str)
        eval_callback = EvalCallback(
            vec_env,
            n_eval_episodes=args.n_eval_episodes,
            eval_freq=args.eval_freq,
            log_path=f"./data/{args.model_path}_{args.hidden_size}/",
            verbose=args.verbose,
            best_model_save_path=args.model_path,
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
            progress_bar=not args.no_progress_bar,
        )
    return num_params


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
