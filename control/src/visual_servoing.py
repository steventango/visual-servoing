from typing import List
from control import ControlMethod
import numpy as np
import numpy.linalg as la
import rospy
from wam import WAM


class VisualServoing(ControlMethod):
    def __init__(self, args: List[str], subparsers=None):
        super().__init__(args)
        self.initialized = False
        self.B = None
        self.epsilon = 30
        self.alpha = 0.1
        self.beta = 0.5
        self.error_prev = None
        self.update_prev = None
        self.active_joints = None
        self.subparsers = subparsers
        self.visual_servoing_parser = None
        if self.subparsers is not None:
            self.visual_servoing_parser = self.subparsers.add_parser(
                'visual_servoing',
                help='Visual Servoing'
            )
            self.visual_servoing_parser.add_argument(
                '--alpha',
                type=float,
                help='Newton step size'
            )
            self.visual_servoing_parser.add_argument(
                '--beta',
                type=float,
                help='Brodyen\'s update step size'
            )
            self.visual_servoing_parser.add_argument(
                '--epsilon',
                type=float,
                help='Error threshold'
            )


    def initialize(self, wam: WAM, control_node):
        if self.initialized:
            return
        self.B = self.init_jacobian_central_differences(wam, control_node)
        self.active_joints = wam.active_joints
        self.initialized = True

    def init_jacobian_central_differences(self, wam: WAM, control_node, delta=0.15):
        rospy.loginfo("Initializing Jacobian with Central Differences")
        J = np.zeros((np.sum(control_node.state.shape[:2]), len(wam.active_joints)))

        action = wam.position
        for i, joint in enumerate(wam.active_joints):
            action[joint] += delta
            wam.joint_move(action)
            rospy.sleep(1)
            state_one = control_node.state.copy()
            error_one = self.get_error(state_one)
            action[joint] -= 2 * delta
            wam.joint_move(action)
            rospy.sleep(1)
            state_two = control_node.state.copy()
            error_two = self.get_error(state_two)
            J[:, i] = (error_one - error_two) / (2 * delta)
            action[joint] += delta
            wam.joint_move(action)
        return J

    def get_error(self, state: np.ndarray):
        # (image, point (ee/t), xy1)
        # TODO: better comment
        error = state[:, 0, :2] - state[:, 1, :2]
        error = error.flatten()
        return error


    def get_action(self, state: np.ndarray, position: np.ndarray):
        if not self.initialized:
            rospy.logerror("VS not initializied!")
        action, done = self.broydens_step(state, position)
        return action, done

    def handle_args(self, args):
        if args.alpha is not None:
            self.alpha = args.alpha
        if args.beta is not None:
            self.beta = args.beta
        if args.epsilon is not None:
            self.epsilon = args.epsilon

    def broydens_step(self, state: np.ndarray, action: np.ndarray):
        rospy.loginfo(f"B: {np.array2string(self.B, precision=4, floatmode='fixed')}")
        rospy.loginfo(f"cond: {la.cond(self.B):.4f}")

        error = self.get_error(state)
        rospy.loginfo(f"Error: {la.norm(error):.2f}")

        if la.norm(error) < self.epsilon:
            return None, True

        try:
            update = self.alpha * la.lstsq(self.B, -error, rcond=None)[0]
        except la.LinAlgError:
            rospy.logwarn("LinAlgError")
            return None, False
        rospy.loginfo(f"Update: {np.array2string(update, precision=4, floatmode='fixed')}")
        if la.norm(update) > 1:
            rospy.logwarn("Big update")
            return None, False

        action[self.active_joints] += update
        rospy.loginfo(f"Action: {np.array2string(action, precision=4, floatmode='fixed')}")

        # Update B
        if self.error_prev is not None:
            y = error - self.error_prev
            self.B += self.beta * np.outer(y - self.B @ self.update_prev, self.update_prev.T) / (self.update_prev.T @ self.update_prev)

        self.error_prev = error
        self.update_prev = update

        return action, False
