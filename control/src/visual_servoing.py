from typing import List
from control import ControlMethod
import numpy as np
import rospy
from wam import WAM


class VisualServoing(ControlMethod):
    def __init__(self, args: List[str]):
        super().__init__(args)
        self.initialized = False
        self.B = None

    def initialize(self, wam: WAM, control_node):
        if self.initialized:
            return
        self.B = self.init_jacobian_central_differences(wam, control_node)
        self.initialized = True

    def init_jacobian_central_differences(self, wam: WAM, control_node, delta=0.15):
        rospy.loginfo("Initializing Jacobian with Central Differences")
        J = np.zeros((np.sum(control_node.state.shape[:2]), wam.dof))
        
        action = wam.position
        for i in range(wam.dof):   
            action[i] += delta
            wam.joint_move(action)
            rospy.sleep(5)
            state_one = control_node.state.copy()
            error_one = self.get_error(state_one)
            action[i] -= 2 * delta
            wam.joint_move(action)
            rospy.sleep(5)
            state_two = control_node.state.copy()
            error_two = self.get_error(state_two)
            J[:, i] = (error_one - error_two) / (2 * delta)
            action[i] += delta
            wam.joint_move(action)
        return J

    def get_error(self, state: np.ndarray):
        error = state[:, 0, :2] - state[:, 1, :2]
        error = error.flatten()
        return error


    def get_action(self, state: np.ndarray):
        if not self.initialized:
            rospy.logerror("VS not initializied!")
        
        
