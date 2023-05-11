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

        self.mode = 'step'
        self.parser = argparse.ArgumentParser()
        self.parser.set_defaults(func=lambda _: None)
        self.subparsers = self.parser.add_subparsers()
        self.parser.add_argument(
            '-q', '--quit', action='store_true',
            help='quit program'
        )
        self.parser.add_argument(
            '-s', '--step', action='store_true',
            help='step one iteration of the control loop'
        )
        self.parser.add_argument(
            '--mode',
            choices={'step', 'normal'},
            help='Select mode, step: run one step of the control loop, then wait for input; normal: run control loop without input'
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
                continue
            rospy.loginfo(f"State: {self.state}")
            if np.any(self.state < 0):
                rospy.loginfo("Lost tracking...")
            elif not self.wam.ready:
                rospy.loginfo("Waiting for WAM...")
            else:
                break
            self.wam.emergency = True
        rospy.loginfo("Ready...")
        self.wam.emergency = False

    def handle_args(self):
        while True:
            try:
                args = self.parser.parse_args(input(">>> ").split())
            except SystemExit:
                continue
            if args.quit:
                exit()
            elif args.mode is not None:
                self.mode = args.mode
            else:
                args.func(args)
            if args.step:
                break
        return args

    def loop(self):
        rate = rospy.Rate(self.rate)
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
                rospy.loginfo("Input [-s] to move one step...")
                self.handle_args()
            self.wam.joint_move(action)

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

        while True:
            self.loop()
            rospy.loginfo("Done!")
            self.handle_args()


def main():
    rospy.init_node('control_node')
    node = ControlNode()
    node.run()
    rospy.spin()


if __name__ == '__main__':
    main()
