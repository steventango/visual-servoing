#!/usr/bin/python3
import rospy
from std_msgs.msg import Float32MultiArray
from rospy.numpy_msg import numpy_msg
import numpy as np
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

        self.wam = WAM()
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

    def run(self):
        wait_rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self.state is not None and self.wam.ready:
                break
            wait_rate.sleep()

        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            rate.sleep()
            if np.any(self.state < 0):
                rospy.loginfo("Lost tracking...")
                self.wam.emergency_stop()
                continue
            action = self.wam.position
            action[1] += 0.1
            action[1] = np.clip(action[1], -2.6, 2.6)
            action[1] = min(action[1], 0)
            if action[1] == 0:
                break
            rospy.loginfo(f"action: {action}")
            self.wam.joint_move(action)
        self.wam.go_home()

def main():
    rospy.init_node('control_node')
    node = ControlNode()
    node.run()
    rospy.spin()


if __name__ == '__main__':
    main()
