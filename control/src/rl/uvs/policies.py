import logging
from typing import Dict, Optional, Type, Any
from gymnasium import spaces
import torch as th
from torch import Tensor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor


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
        beta: float = 0.,
        gamma: float = 1.0,
        steps: int = 3,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )
        """
        ## Parameters
        env: gym environment
        J: Initial jacobian (obs_shape * action_shape)
        alpha: Newton step size
        beta: Broyden's update step size
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

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.steps = steps

        self.central_differences_trajectory = self.generate_central_differences_trajectory()

        self.first_step = True
        self.errors = None
        self.observations = None
        self.i = 0
        self.j = 0
        self.k = 0

        self.prev_error = None
        self.prev_action = None

    def generate_central_differences_trajectory(self):
        """
        Generate a trajectory using central differences

        ## Returns
        trajectory: trajectory to use for central differences
        """
        trajectory = th.full((4 * self.steps,), self.gamma, dtype=th.float64)
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
        if self.i < self.action_shape:
            actions = th.zeros(self.batch_size, self.action_shape)
            if self.j < len(self.central_differences_trajectory):
                gamma = self.central_differences_trajectory[self.j]
                if not self.first_step and self.prev_action is not None and gamma * th.sum(self.prev_action) < 0:
                    self.errors[:, self.k] = self.calculate_error(observation)
                    self.observations[:, self.k] = observation["observation"].clone()
                    self.k += 1
                actions[:, self.i] = gamma
                self.j += 1
            else:
                de = self.errors[:, 0] - self.errors[:, 1]
                ds = self.observations[:, 0, self.i] - self.observations[:, 1, self.i]
                self.J[:, :, self.i] = de / (2 * ds * self.steps)
                self.i += 1
                self.j = 0
                self.k = 0
            return actions
        else:
            self.initialized = True
            return None

    def forward(self, obs: Dict[str, th.Tensor], deterministic: bool = False) -> th.Tensor:
        self.batch_size = obs['observation'].shape[0]
        if self.errors is None:
            self.errors = th.zeros((self.batch_size, 2, self.goal_shape), dtype=th.float64)
            self.observations = th.zeros((self.batch_size, 2, self.observation_shape), dtype=th.float64)
            self.J = th.zeros((self.batch_size, self.goal_shape, self.action_shape), dtype=th.float64)
        if not self.initialized:
            action = self.calculate_central_differences(obs)
        if self.initialized or action is None:
            action = self._predict(obs, deterministic)

        self.prev_action = action
        self.first_step = False
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
        error = self.calculate_error(observation)
        action = self.visual_servo(error)

        self.broydens_step(error)
        self.prev_error = error

        return action

    def visual_servo(self, error):
        try:
            action, *_ = th.linalg.lstsq(self.J, -error, rcond=-1)
        except th.linalg.LinAlgError as e:
            action = th.zeros(self.action_shape)
            logging.error(e)
        action *= self.alpha
        return action

    def broydens_step(self, error):
        if self.prev_error is None:
            return
        y = (error - self.prev_error).unsqueeze(2)
        prev_action = self.prev_action[:, :, None]
        prev_action_T = th.transpose(prev_action, 1, 2)
        B = (y - self.J @ prev_action) @ prev_action_T / (prev_action_T @ prev_action)
        self.J += self.beta * B
