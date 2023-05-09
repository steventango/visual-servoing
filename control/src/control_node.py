#!/usr/bin/python3
import rospy
from std_msgs.msg import Float32MultiArray
from rospy.numpy_msg import numpy_msg
import numpy as np
from control import ControlMethod
from visual_servoing import VisualServoing
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

        self.control = CONTROL_METHODS[self.type](self.args)

        # self.pub = rospy.Publisher(
        #     f"wam_common/?",
        #     numpy_msg(Float32MultiArray),
        #     queue_size=1
        # )
        self.sub = rospy.Subscriber(
            '/perception_node/state',
            numpy_msg(Float32MultiArray),
            self.callback
        )

    def callback(self, message):
        state = message.data
        state = state.reshape([dim.size for dim in message.layout.dim])
        self.state = state

    def run(self):
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            rate.sleep()
            if self.state is None:
                continue
            rospy.loginfo_throttle(10, self.state)
            

def main():
    rospy.init_node('control_node')
    node = ControlNode()
    node.run()
    rospy.spin()


if __name__ == '__main__':
  main()