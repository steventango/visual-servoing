import argparse
import os

import custom_policy
import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from uvs import UVS

ONE_ENV_ALGS = {DDPG, TD3, SAC}

ALGS = {
    "A2C": A2C,
    "DDPG": DDPG,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "UVS": UVS,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to save model as")
    parser.add_argument("--eval_log_path", type=str, help="Path to save evaluation logs as")
    parser.add_argument("--tensorboard_log_path", type=str, help="Path to save tensorboard logs as")
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
        default=100,
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
        default=4,
        help="The number of degrees of freedom of the robot",
    )
    parser.add_argument("--std", type=float, default=0.1, help="Scale of the noise")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Number of hidden units in the policy network",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Number of hidden layers in the policy network",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="learning rate for adam optimizer, the same learning rate will be used for all networks (Q-Values, Actor and Value function) it can be a function of the current progress remaining (from 1 to 0)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbose level",
    )
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Do not display a progress bar using tqdm and rich",
    )
    parser.add_argument(
        '--policy',
        type=str,
        default='MultiInputPolicy',
        choices=[
            'MlpPolicy',
            'CnnPolicy',
            'MultiInputPolicy',
            'CustomPolicy',
            'CustomCnnPolicy',
            'CustomMultiInputPolicy',
        ],
    )
    parser.add_argument(
        '--alg',
        type=str,
        default='TD3',
        choices=['A2C', 'DDPG', 'PPO', 'SAC', 'TD3', 'UVS'],
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel environments",
    )
    args = parser.parse_args(argv)
    return args


def train(args):
    reward_type = "Dense" if args.reward_type == "Dense" else ""
    dof = f"{args.dof}DOF" if args.dof < 7 else ""
    env_id = f"WAMVisualReach{reward_type}{dof}-v2"

    env = gym.make(env_id)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.std * np.ones(n_actions))
    alg = ALGS[args.alg]
    n_envs = 1 if alg in ONE_ENV_ALGS else args.n_envs
    vec_action_noise = action_noise if alg in ONE_ENV_ALGS else VectorizedActionNoise(action_noise, n_envs=n_envs)
    vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=DummyVecEnv)
    model = alg(
        args.policy,
        vec_env,
        verbose=args.verbose,
        tensorboard_log=args.tensorboard_log_path,
        # replay_buffer_class=HerReplayBuffer,
        # # Parameters for HER
        # replay_buffer_kwargs=dict(
        #     n_sampled_goal=4,
        #     goal_selection_strategy=goal_selection_strategy,
        # ),
        action_noise=vec_action_noise,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[args.hidden_size] * args.depth,
                qf=[args.hidden_size] * args.depth,
            ),
            # optimizer_class=torch.optim.AdamW,
            share_features_extractor=True,
        ),
        learning_rate=args.learning_rate,
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
        log_path=args.eval_log_path,
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
