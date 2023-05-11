#!/usr/bin/python3
import argparse
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

        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.parser.add_argument('--quit', action='store_true')
        self.parser.add_argument('--exit', action='store_true')
        self.parser.add_argument('--shutdown', action='store_true')
        self.parser.add_argument(
            '--mode',
            choices={'step', 'auto'}
        )

        self.wam = WAM(
            rospy.get_param('~wam_namespace'),
            subparsers=self.subparsers
        )

        self.control_method = CONTROL_METHODS[self.type](
            self.args,
            subparsers=self.subparsers
        )

        self.sub = rospy.Subscriber(
            '/perception_node/state',
            numpy_msg(Float32MultiArray),
            self.callback
        )

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

    def handle_args(self):
        while not rospy.is_shutdown():
            try:
                args = self.parser.parse_args(input(">> ").split())
            except SystemExit:
                continue
            if args.quit or args.exit or args.shutdown:
                rospy.signal_shutdown("Quit")
                break
            if args.mode is not None:
                self.mode = args.mode
            if args.help:
                continue
            self.wam.handle_args(args)
            self.control_method.handle_args(args)
        return args

    def run(self):
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            rate.sleep()
            if self.wam.ready:
                break
            rospy.loginfo("Waiting for WAM...")

        self.handle_args()

        self.wam.go_start()
        rospy.loginfo("WAM in start position!")

        done = False
        while not rospy.is_shutdown() and not self.wam.emergency:
            rate.sleep()
            self.wait_initialization()
            self.control_method.initialize(self.wam, self)
            rospy.loginfo(f"Position: {self.wam.position}")
            action, done = self.control_method.get_action(self.state, self.wam.position)
            if done:
                break
            if action is None:
                continue
            if self.mode == 'step':
                rospy.loginfo("Press key to move...")
                self.handle_args()
            self.wam.joint_move(action)
        if done:
            rospy.loginfo("Done!")


def main():
    rospy.init_node('control_node')
    node = ControlNode()
    node.run()
    rospy.spin()


if __name__ == '__main__':
    main()
