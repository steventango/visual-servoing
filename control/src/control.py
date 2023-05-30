import argparse
import numpy as np
from wam import WAM


class ControlMethod:
    def __init__(self, args: argparse.Namespace):
        """
        :param args: arguments
        """
        pass

    def initialize(self, wam: WAM, control_node):
        """
        :param wam: WAM object
        "param control_node: control_node object
        """
        if self.initialized:
            return

    def get_action(self, state: np.ndarray, wam: WAM):
        """
        :param state: (image, point, 3) homogeneous coordinate state
        :param wam: WAM object

        :return: ((wam.dof,), bool) action, done
        """
        raise NotImplementedError()
