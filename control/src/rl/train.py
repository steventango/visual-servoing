import argparse
import os
from pathlib import Path

import gymnasium as gym
import nj
import numpy as np
from gymnasium.wrappers import RecordVideo
from jsrl import get_jsrl_algorithm
from residualrl import get_rrl_algorithm
from pyvirtualdisplay import Display
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from uvs import UVS
from torch import nn

ONE_ENV_ALGS = {DDPG, TD3, SAC}

ALGS = {
    "A2C": A2C,
    "DDPG": DDPG,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "UVS": UVS,
    "JSRL-UVS-TD3": (UVS, TD3),
    "RRL-UVS-TD3": (UVS, TD3),
}

ACTIVATION_FNS = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to save model as")
    parser.add_argument("--eval_log_path", type=str, help="Path to save evaluation logs as")
    parser.add_argument("--tensorboard_log_path", type=str, help="Path to save tensorboard logs as")
    parser.add_argument("--final_video_path", type=str, help="Path to save videos as", default=None)
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
        choices=["Dense", "Sparse", "Timestep"],
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
            *nj.__all__
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
    parser.add_argument(
        "--activation_fn",
        type=str,
        default="ReLU",
        choices=ACTIVATION_FNS.keys()
    )
    parser.add_argument(
        "--learning_video_path",
        type=str,
        help="Path to save learning videos as",
        default=None,
    )
    parser.add_argument(
        "--learning_video_length",
        type=int,
        default=5000,
        help="Length of learning videos",
    )
    args = parser.parse_args(argv)
    return args


class LoggingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.logger.record("train/action_norm", np.linalg.norm(self.locals['actions']))
        self.logger.record("train/action_min", np.min(self.locals['actions']))
        self.logger.record("train/action_max", np.max(self.locals['actions']))
        try:
            if hasattr(self.model, "actor") and hasattr(self.model.actor, "J"):
                J = self.model.actor.J.detach().cpu().numpy()
                # self.logger.record("train/actor_a", self.model.actor.a.detach().cpu().numpy().item())
                self.logger.record("train/actor_J_norm", np.linalg.norm(J))
                self.logger.record("train/actor_J_cond", np.linalg.cond(J).item())
                self.logger.record("train/actor_J_rank", np.linalg.matrix_rank(J).item())
                self.logger.record("train/actor_J_min", np.min(J).item())
                self.logger.record("train/actor_J_max", np.max(J).item())
                U, S, Vh = np.linalg.svd(J)
                self.logger.record("train/actor_J_max_singular_value", np.max(S).item())
                self.logger.record("train/actor_J_min_singular_value", np.min(S).item())
                try:
                    self.logger.record("train/actor_lstsq_solution_norm", np.linalg.norm(self.model.actor.lstsq_solution.cpu()))
                except AttributeError:
                    pass
        except:
            pass

        if type(self.model.policy).__name__ == 'RRLPolicy':
            for i in range(self.model.policy.residual_actions.shape[1]):
                self.logger.record(f"train/rrl/residual_action_{i}", self.model.policy.residual_actions[0, i])
                self.logger.record(f"train/rrl/control_action_{i}", self.model.policy.control_actions[0, i])
                self.logger.record(f"train/rrl/action_{i}", self.model.policy.actions[0, i])
            self.logger.record("train/rrl/residual_action_norm", np.linalg.norm(self.model.policy.residual_actions))
            self.logger.record("train/rrl/control_action_norm", np.linalg.norm(self.model.policy.control_actions))
            self.logger.record("train/rrl/action_norm", np.linalg.norm(self.model.policy.actions))
            # self.logger.record("train/rrl/alpha", self.model.policy.alpha)

        return True


def train(args):
    reward_type = "" if args.reward_type == "Sparse" else args.reward_type
    dof = f"{args.dof}DOF" if args.dof < 7 else ""
    env_id = f"WAMVisualReach{reward_type}{dof}-v2"

    env = gym.make(env_id)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.std * np.ones(n_actions))
    algs = ALGS[args.alg]
    n_envs = 1 if algs in ONE_ENV_ALGS else args.n_envs
    vec_action_noise = action_noise if algs in ONE_ENV_ALGS else VectorizedActionNoise(action_noise, n_envs=n_envs)
    env_kwargs = None
    display = None
    if args.learning_video_path:
        display = Display(visible=0, size=(480, 480))
        display.start()
        env_kwargs["render_mode"] = "rgb_array"
    vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs)
    if args.learning_video_path:
        vec_env = VecVideoRecorder(
            vec_env,
            str(Path(args.learning_video_path) / "learning/"),
            record_video_trigger=lambda x: x == 0,
            video_length=args.learning_video_length,
        )
    eval_vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=DummyVecEnv)
    policy_kwargs = dict(
        # optimizer_class=torch.optim.AdamW,
        share_features_extractor=True,
        activation_fn=ACTIVATION_FNS[args.activation_fn],
    )
    if args.hidden_size and args.depth:
        policy_kwargs["net_arch"] = dict(
            pi=[args.hidden_size] * args.depth,
            qf=[args.hidden_size] * args.depth,
        )
    if args.alg.startswith("JSRL"):
        guide_model = algs[0]("MultiInputPolicy", env, learning_rate=0)
        if isinstance(guide_model, UVS):
            guide_model.learn(total_timesteps=1000)
        alg = get_jsrl_algorithm(algs[1])
        max_horizon = 10
        n = 5
        policy_kwargs["guide_policy"] = guide_model.policy
        policy_kwargs["max_horizon"] = max_horizon
        policy_kwargs["horizons"] = np.arange(max_horizon, -1, -max_horizon // n)
        policy_kwargs["tolerance"] = 0.1
        policy_kwargs["window_size"] = 1
    elif args.alg.startswith("RRL"):
        control_model = algs[0]("MultiInputPolicy", env, learning_rate=0)
        if isinstance(control_model, UVS):
            control_model.learn(total_timesteps=1000)
        policy_kwargs["control_policy"] = control_model.policy
        alg = get_rrl_algorithm(algs[1])
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
        learning_starts=0,
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
    logging_callback = LoggingCallback()
    callbacks = CallbackList([eval_callback, logging_callback])
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=not args.no_progress_bar
    )
    vec_env.close()
    eval_vec_env.close()

    if args.final_video_path:
        record_final_video(args, env_id, model)

    return num_params


def record_final_video(args, env_id, model, episodes=10):
    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(
        env,
        str(Path(args.final_video_path) / "final/"),
        video_length=600,
        disable_logger=args.verbose < 1
    )
    observation, info = env.reset()
    for _ in range(episodes):
        while True:
            action, _state = model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()
                break
    env.close()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
