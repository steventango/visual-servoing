#!/usr/bin/python3
import rospy
from std_msgs.msg import Float32MultiArray
from rospy.numpy_msg import numpy_msg
import numpy as np
from control import ControlMethod
from visual_servoing import VisualServoing
from wam_srvs.srv import JointMove, JointMoveRequest
from sensor_msgs.msg import JointState
from typing import Dict

CONTROL_METHODS: Dict[str, ControlMethod] = {
    'visual_servoing': VisualServoing
}

class ControlNode:
    def __init__(self):
        self.rate = rospy.get_param('~rate')
        self.type = rospy.get_param('~type')
        self.args = rospy.get_param('~args', [])

        self.state = None

        self.control_method = CONTROL_METHODS[self.type](self.args)

        self.sub_wam_joint_states = rospy.Subscriber(
            '/wam/joint_states',
            JointState,
            self.cb_joint_state
        )
        rospy.wait_for_service('/wam/joint_move')
        self.joint_move = rospy.ServiceProxy(
            '/wam/joint_move',
            JointMove
        )
        self.joint_state_position = None
        self.sub = rospy.Subscriber(
            '/perception_node/state',
            numpy_msg(Float32MultiArray),
            self.callback
        )
        print("initialized!")

    def callback(self, message):
        state = message.data
        state = state.reshape([dim.size for dim in message.layout.dim])
        self.state = state

    def cb_joint_state(self, message):
        self.joint_state_position = message.position

    def run(self):
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            rate.sleep()
            # if self.state is None:
            #     continue
            if self.joint_state_position is None:
                continue
            action = list(self.joint_state_position)
            action[1] += 0.1
            action[1] = np.clip(action[1], -2.6, 2.6)
            action[1] = min(action[1], 0)
            rospy.loginfo(f"action: {action}")
            self.joint_move(action)
            # rospy.loginfo_throttle(10, self.state)
            # action = self.control_method(self.state)
            # self.joint_move(action)

def main():
    rospy.init_node('control_node')
    node = ControlNode()
    node.run()
    rospy.spin()


if __name__ == '__main__':
  main()