import logging

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class UncalibratedVisualServoingModel:
    def __init__(
        self,
        env: gym.Env,
        J: NDArray[np.float64] = None,
        alpha: float = 0.5,
        beta: float = 0.001,
        gamma: float = 1.0,
        steps: int = 3,
    ):
        """
        ## Parameters
        env: gym environment
        J: Initial jacobian (obs_shape * action_shape)
        alpha: Newton step size
        beta: Broyden's update step size
        gamma: Action velocity for central differences
        steps: Number of steps to take for central differences
        """
        self.obs_shape = env.observation_space["achieved_goal"].shape[0]
        self.action_shape = env.action_space.shape[0]

        if J is not None:
            assert J.shape == (self.obs_shape, self.action_shape)
            self.J = J.copy()
            self.initialized = True
        else:
            self.J = np.zeros((self.obs_shape, self.action_shape))
            self.initialized = False

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.steps = steps

        self.central_differences_trajectory = self.generate_central_differences_trajectory()

        self.reset()

    def reset(self):
        self.errors = np.array([None, None])
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
        trajectory = np.full((4 * self.steps), self.gamma)
        trajectory[self.steps: 3 * self.steps] *= -1
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
            action = np.zeros(self.action_shape)
            if self.j < len(self.central_differences_trajectory):
                gamma = self.central_differences_trajectory[self.j]
                if self.prev_action is not None and gamma * np.sum(self.prev_action) < 0:
                    self.errors[self.k] = self.calculate_error(observation)
                    self.k += 1
                action[self.i] = gamma
                self.j += 1
            else:
                self.J[:, self.i] = (self.errors[0] - self.errors[1]) / (2 * self.gamma * self.steps)
                self.i += 1
                self.j = 0
                self.k = 0
            return action
        else:
            self.initialized = True
            return None

    def predict(self, observation):
        if not self.initialized:
            action = self.calculate_central_differences(observation)

        if self.initialized or action is None:
            error = self.calculate_error(observation)
            action = self.visual_servo(error)

            self.broydens_step(error)
            self.prev_error = error

        self.prev_action = action
        return action, None

    def visual_servo(self, error):
        try:
            action, *_ = np.linalg.lstsq(self.J, -error, rcond=-1)
        except np.linalg.LinAlgError as e:
            action = np.zeros(self.action_shape)
            logging.error(e)
        action *= self.alpha
        action = np.clip(action, -1, 1)
        return action

    def broydens_step(self, error):
        if self.prev_error is None:
            return
        y = error - self.prev_error
        B = np.outer(y - self.J @ self.prev_action, self.prev_action.T) / (self.prev_action.T @ self.prev_action)
        self.J += self.beta * B
