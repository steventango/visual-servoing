#!/usr/bin/python3
import rospy
from std_msgs.msg import Float32MultiArray
from rospy.numpy_msg import numpy_msg
import numpy as np
import numpy.linalg as la
from control import ControlMethod
from visual_servoing import VisualServoing
from typing import Dict
from wam import WAM


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

        self.wam = WAM(rospy.get_param('~wam_namespace'))
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

    def wait_initialization(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()
            if self.state is None:
                rospy.loginfo("Waiting for state...")
            elif np.any(self.state < 0):
                rospy.loginfo("Lost tracking...")
            elif not self.wam.ready:
                rospy.loginfo("Waiting for WAM...")
            else:
                break
            self.wam.emergency = True
        rospy.loginfo("Ready...")
        self.wam.emergency = False

    def move_wam_to_ready_position(self):
        rospy.loginfo("Moving WAM to ready position...")
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            rate.sleep()
            if not self.wam.ready:
                rospy.loginfo("Waiting for WAM...")
                continue
            break

        steps = 8
        trajectory = np.linspace(self.wam.position, self.wam.ready_position, steps)
        for action in trajectory:
            if la.norm(self.wam.position - self.wam.ready_position) < 1e-2:
                break
            rospy.loginfo(f"Action: {action}")
            self.wam.joint_move(self.wam.ready_position)
            rate.sleep()
        rospy.loginfo("WAM in ready position!")


    def run(self):
        self.move_wam_to_ready_position()

        rate = rospy.Rate(self.rate)
        done = False
        while not rospy.is_shutdown():
            rate.sleep()
            self.wait_initialization()
            self.control_method.initialize(self.wam, self)
            # action = self.control_method.get_action(self.state)
            # self.wam.joint_move(action)
            done = True
            break
        if done:
            self.wam.go_home()
            rospy.loginfo("Done!")


def main():
    rospy.init_node('control_node')
    node = ControlNode()
    node.run()
    rospy.spin()


if __name__ == '__main__':
    main()
