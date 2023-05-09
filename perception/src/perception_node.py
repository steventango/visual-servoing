#!/usr/bin/python3
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np
import message_filters
from perception import Perception
from tracking import Tracking

PERCEPTION_METHODS = {
    'tracking': Tracking
}

class PerceptionNode:
    def __init__(self):
        self.rate = rospy.get_param('~rate')
        self.type = rospy.get_param('~type')
        self.args = rospy.get_param('~args', [])

        self.bridge = CvBridge()
        self.raw_images = None

        self.perception = PERCEPTION_METHODS[self.type](self.args)

        self.pub = rospy.Publisher(
            f"~state",
           Float32MultiArray,
            queue_size=1
        )
        self.subs = (message_filters.Subscriber(
            f"/cam0/image_raw/compressed",
            CompressedImage
        ), message_filters.Subscriber(
            f"/cam1/image_raw/compressed",
            CompressedImage
        ))
        self.ts = message_filters.ApproximateTimeSynchronizer(self.subs, 1, 1)
        self.ts.registerCallback(self.callback)

    def callback(self, *compresseds):
        self.raw_images = [
            self.bridge.compressed_imgmsg_to_cv2(compressed)
            for compressed in compresseds
        ]

    def publish_state(self, state):
        message = Float32MultiArray()
        message.data = state.flatten()
        message.layout.dim = []
        stride = state.size
        for i, s in enumerate(state.shape):
            dim = MultiArrayDimension()
            dim.label = str(i)
            dim.size = s
            dim.stride = stride
            message.layout.dim.append(dim)
            stride //= s
        self.pub.publish(message)

    def run(self):
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            rate.sleep()
            if self.raw_images is None:
                continue
            state = self.perception.get_state(self.raw_images)
            self.publish_state(state)
            


def main():
    rospy.init_node('perception_node')
    node = PerceptionNode()
    node.run()
    rospy.spin()


if __name__ == '__main__':
  main()