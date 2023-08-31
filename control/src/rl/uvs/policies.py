import logging
from typing import Any, Callable, Dict, Optional, Type

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from torch import Tensor

Schedule = Callable[[float], float]


class UVSPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        J: Tensor = None,
        alpha: float = 1.0,
        gamma: float = 0.02,
        steps: int = 5,
        lr_schedule: Schedule = None,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
            normalize_images=normalize_images,
        )
        """
        ## Parameters
        env: gym environment
        J: Initial jacobian (obs_shape * action_shape)
        alpha: Newton step size
        learning_rate: Broyden's update step size
        gamma: Action velocity for central differences
        steps: Number of steps to take for central differences
        """
        self.observation_shape = observation_space["observation"].shape[0]
        self.goal_shape = observation_space["achieved_goal"].shape[0]
        self.action_shape = action_space.shape[0]

        if J is not None:
            assert J.shape == (self.goal_shape, self.action_shape)
            self.J = J.clone()
            self.initialized = True
        else:
            self.J = None
            self.initialized = False
        self.B = None

        self.alpha = alpha
        self.lr_schedule = lr_schedule
        self.gamma = gamma
        self.steps = steps

        self.central_differences_trajectory = self.generate_central_differences_trajectory()

        self.first_step = True
        self.errors = None
        self.observations = None
        self.i = 0
        self.j = 0
        self.k = 0
        self.prev_gamma = 0

        self.prev_error = None
        self.prev_action = None
        self.prev_obs = None

    def generate_central_differences_trajectory(self):
        """
        Generate a trajectory using central differences

        ## Returns
        trajectory: trajectory to use for central differences
        """
        trajectory = th.full((4 * self.steps,), self.gamma, dtype=th.float64, device=self.device)
        trajectory[self.steps : 3 * self.steps] *= -1
        return trajectory

    def calculate_error(self, observation: dict):
        """
        Calculate the error between the achieved and desired goal

        ## Parameters
        observation: observation from the environment

        ## Returns
        error: error between the achieved and desired goal
        """
        return observation["achieved_goal"] - observation["desired_goal"]

    def calculate_central_differences(self, observation: dict):
        """
        Initialize the Jacobian using central differences

        ## Parameters
        observation: observation from the environment

        ## Returns
        action: action to take
        """
        # estimate one column of the jacobian at a time
        while self.i < self.action_shape:
            action = th.zeros((self.batch_size, self.action_shape), device=self.device)
            if self.j < len(self.central_differences_trajectory):
                gamma = self.central_differences_trajectory[self.j]
                if not self.first_step and self.prev_action is not None and gamma * self.prev_gamma < 0:
                    self.errors[:, self.k] = self.calculate_error(observation)
                    self.observations[:, self.k] = observation["observation"].clone()
                    self.k += 1
                action[:, self.i] = gamma
                self.prev_gamma = gamma
                self.j += 1
                return action
            else:
                de = self.errors[:, 0] - self.errors[:, 1]
                ds = self.observations[:, 0, self.i] - self.observations[:, 1, self.i]
                self.J[:, :, self.i] = de / ds
                self.i += 1
                self.j = 0
                self.k = 0
        self.B = self.J.clone()
        self.initialized = True
        return None

    def forward(self, obs: Dict[str, th.Tensor], deterministic: bool = False) -> th.Tensor:
        obs = {k: v.to(self.device) for k, v in obs.items()}
        self.batch_size = obs['observation'].shape[0]
        if self.errors is None:
            self.errors = th.zeros((self.batch_size, 2, self.goal_shape), dtype=th.float64, device=self.device)
            self.observations = th.zeros((self.batch_size, 2, self.observation_shape), dtype=th.float64, device=self.device)
            self.J = th.zeros((self.batch_size, self.goal_shape, self.action_shape), dtype=th.float64, device=self.device)
        if not self.initialized:
            obs['desired_goal'] = th.zeros_like(obs['desired_goal'], device=self.device)
            action = self.calculate_central_differences(obs)
        if self.initialized or action is None:
            action = self._predict(obs, deterministic)

        self.first_step = False
        self.prev_obs = {k: v.clone() for k, v in obs.items()}
        self.prev_action = action.clone()
        return action

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        observation = {k: v.to(self.device) for k, v in observation.items()}
        if self.is_new_episode(observation):
            self.B = self.J.clone()

        error = self.calculate_error(observation)
        self.broydens_step(observation, error)
        action = self.visual_servo(error)

        self.prev_obs = {k: v.clone() for k, v in observation.items()}
        self.prev_error = error.clone()
        self.prev_action = action.clone()

        return action

    def is_new_episode(self, observation):
        if self.prev_obs is None:
            return False
        delta_desired_goal_norm = th.linalg.norm(observation['desired_goal'] - self.prev_obs['desired_goal'])
        return delta_desired_goal_norm > 1e-3

    def visual_servo(self, error):
        try:
            update = -th.linalg.pinv(self.B) @ error.squeeze()
            self.prev_update = update.clone()
            action = self.alpha * update
        except th.linalg.LinAlgError as e:
            logging.error(e)
            logging.error(self.B)
            action = th.zeros((self.batch_size, self.action_shape), device=self.device)

        return action

    def broydens_step(self, obs, error):
        if self.prev_obs is None or self.prev_error is None:
            return
        s = obs['observation'] - self.prev_obs['observation']
        s_norm = th.linalg.norm(s, dim=1)
        s = s[:, :, None]
        sT = th.transpose(s, 1, 2)
        y = error - self.prev_error
        y = th.unsqueeze(y, 2)
        y_norm = th.linalg.norm(y, dim=1)
        y_norm = th.squeeze(y_norm, 1)
        y_estimate = self.B @ s
        B = ((y - y_estimate) @ sT) / (sT @ s)
        B[s_norm < 1e-2] = 0
        B[y_norm < 1e-2] = 0
        self.B += self.lr_schedule(0) * B
