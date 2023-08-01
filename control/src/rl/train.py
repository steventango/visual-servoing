import argparse
import os

import gymnasium as gym
import nj
import numpy as np
from gymnasium.wrappers import RecordVideo
from jsrl.jsrl import get_jsrl_algorithm
from pyvirtualdisplay import Display
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.base_class import BaseAlgorithm
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
    "TD3-NJ": TD3,
    "UVS": UVS,
    "JSRL-UVS-TD3": (UVS, TD3),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to save model as")
    parser.add_argument("--eval_log_path", type=str, help="Path to save evaluation logs as")
    parser.add_argument("--tensorboard_log_path", type=str, help="Path to save tensorboard logs as")
    parser.add_argument("--video_path", type=str, help="Path to save videos as")
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
        help="Number of hidden units in the policy network",
    )
    parser.add_argument(
        "--depth",
        type=int,
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
        "--policy",
        type=str,
        default="MultiInputPolicy",
        choices=[
            "MlpPolicy",
            "CnnPolicy",
            "MultiInputPolicy",
            *nj.__all__,
        ],
    )
    parser.add_argument(
        "--alg",
        type=str,
        default="TD3",
        choices=ALGS.keys(),
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
    algs = ALGS[args.alg]
    n_envs = 1 if algs in ONE_ENV_ALGS else args.n_envs
    vec_action_noise = action_noise if algs in ONE_ENV_ALGS else VectorizedActionNoise(action_noise, n_envs=n_envs)
    vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=DummyVecEnv)
    eval_vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=DummyVecEnv)
    policy_kwargs = dict(
        # optimizer_class=torch.optim.AdamW,
        share_features_extractor=True,
    )
    if args.hidden_size and args.depth:
        policy_kwargs["net_arch"] = dict(
            pi=[args.hidden_size] * args.depth,
            qf=[args.hidden_size] * args.depth,
        )
    if args.alg.startswith("JSRL"):
        guide_model = algs[0]("MultiInputPolicy", env)
        if isinstance(guide_model, UVS):
            guide_model.learn(total_timesteps=100)
        alg = get_jsrl_algorithm(algs[1])
        max_horizon = 10
        n = 5
        policy_kwargs["guide_policy"] = guide_model.policy
        policy_kwargs["max_horizon"] = max_horizon
        policy_kwargs["horizons"] = np.arange(max_horizon, -1, -max_horizon // n)
        policy_kwargs["tolerance"] = 0.1
        policy_kwargs["window_size"] = 1
    else:
        alg = algs
    model: BaseAlgorithm = alg(
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
        policy_kwargs=policy_kwargs,
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
        eval_vec_env,
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

    display = Display(visible=0, size=(480, 480))
    display.start()
    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(
        env,
        args.video_path,
        video_length=600,
        disable_logger=args.verbose < 1
    )
    observation, info = env.reset()
    for _ in range(10):
        while True:
            action, states_ = model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()
                break
    env.close()

    return num_params


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
