from typing import List
import argparse
import numpy as np
import numpy.linalg as la
import rospy
from stable_baselines3 import TD3
from wam import WAM

from control import ControlMethod


class RLControl(ControlMethod):
    def __init__(self, args: List[str], subparsers=None):
        super().__init__(args)
        self.subparsers = subparsers
        self.rl_parser = None
        self.alg = TD3
        self.epsilon = 30
        self.model = self.alg.load("models/best_model.zip")
        self.active_joints = None
        self.observation_joint_indices = None
        # TODO: remove hard coding
        self.width = 640
        self.height = 480

        if self.subparsers is not None:
            self.rl_parser = self.subparsers.add_parser(
                'rl',
                help='Reinforcement Learning Control Method'
            )
            self.rl_parser.set_defaults(func=self.handle_args)
            self.visual_servoing_parser.add_argument(
                '--epsilon',
                type=float,
                default=None,
                help='Error threshold'
            )

    def initialize(self, wam: WAM, control_node):
        super().initialize(wam, control_node)
        self.active_joints = wam.active_joints
        dof = len(self.active_joints)
        if dof == 3:
            self.observation_joint_indices = self.active_joint_indices + [4]
        else:
            self.observation_joint_indices = self.active_joint_indices

    def get_error(self, state: np.ndarray):
        """
        :param state: (image, point, 3) homogeneous coordinate state

        :return: (2,) error vector
        """
        error = state[:, 0, :2] - state[:, 1, :2]
        error = error.flatten()
        return error

    def map_state_to_observation(self, state: np.ndarray, wam: WAM):
        """
        :param state: (image, point, 3) homogeneous coordinate state
        :param wam: WAM object

        :return: dict {
            "observation": (8 + 2 * len(self.observation_joint_indices),) observation vector
        }
        """
        state_img = state.copy()[:, :, :2]
        ee_img = state_img[:, 0]
        target_img = state_img[:, 1]
        ee_img = self.normalize_image(ee_img)
        target_img = self.normalize_image(target_img)

        robot_qpos = self.wam.position[self.observation_joint_indices]
        robot_qvel = self.wam.velocity[self.observation_joint_indices]

        obs = np.concatenate([
            ee_img.ravel(),
            target_img.ravel(),
            robot_qpos,
            robot_qvel
        ])
        obs = {
            "observation": obs
        }
        return obs

    def normalize_image(self, img: np.ndarray):
        """
        :param img: (n, 2) image points to normalize

        :return: (n, 2) normalized image points
        """
        min_shape = np.min([self.width, self.height])
        img[:, 0] -= self.width / 2
        img[:, 1] -= self.height / 2
        img /= min_shape
        return img

    def get_action(self, state: np.ndarray, wam: WAM):
        """
        :param state: (image, point, 3) homogeneous coordinate state
        :param wam: WAM object

        :return: ((wam.dof,), bool) action, done
        """
        if not self.initialized:
            rospy.logerror("RL not initializied!")

        done = False
        if la.norm(self.get_error(state)) < self.epsilon:
            done = True

        obs = self.map_state_to_observation(state, wam)
        rospy.loginfo(f"Obs: {np.array2string(obs['observation'], precision=4, floatmode='fixed')}")
        action = np.zeros_like(self.wam.velocity)
        action[self.active_joints], _ = self.model.predict(obs)
        rospy.loginfo(f"Action: {np.array2string(action, precision=4, floatmode='fixed')}")
        return action, done

    def handle_args(self, args: argparse.Namespace):
        """
        :param args: Argument paraser name space
        """
        if args.epsilon is not None:
            self.epsilon = args.epsilon
