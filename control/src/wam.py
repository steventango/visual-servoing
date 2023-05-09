import rospy
import numpy as np

from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from wam_srvs.srv import JointMove

class WAM:
    def __init__(self, namespace):
        self.sub_wam_joint_states = rospy.Subscriber(
            f'/{namespace}/joint_states',
            JointState,
            self.callback_joint_states
        )
        rospy.wait_for_service(f'/{namespace}/joint_move')
        self._joint_move = rospy.ServiceProxy(
            f'/{namespace}/joint_move',
            JointMove
        )
        rospy.wait_for_service(f'/{namespace}/go_home')
        self._go_home = rospy.ServiceProxy(
            f'/{namespace}/go_home',
            Empty
        )
        self.join_limits = np.array((
            (-2.6, 2.6),
            (-2.0, 2.0),
            (-2.8, 2.8),
            (-0.9, 3.1),
            (-4.76, 1.24),
            (-1.6, 1.6),
            (-3.0, 3.0)
        ))
        self._position = None
        self._velocity = None
        self.dof = None
        self.ready = False
        rospy.on_shutdown(self.on_shutdown)

    @property
    def position(self):
        return self._position.copy()

    @property
    def velocity(self):
        return self._velocity.copy()

    def on_shutdown(self):
        rospy.loginfo("On shutdown...")
        self.go_home()
        self.ready = False

    def emergency_stop(self):
        rospy.logwarn("Emergency stop...")
        self.ready = False

    def callback_joint_states(self, message: JointState):
        self.joint_state = message
        self.dof = len(message.position)
        self.position = np.array(message.position)
        self.velocity = np.array(message.velocity)
        self.ready = True

    def joint_move(self, action):
        if not self.ready:
            rospy.logwarn("WAM not ready!")
            return
        if len(action) != self.dof:
            rospy.logwarn(f"Action must have length {self.dof}!")
            return
        violated_joint_limits = (
            action < self.join_limits[:self.dof, 0] |
            self.join_limits[:self.dof, 1] > action
        )
        if np.any(violated_joint_limits):
            rospy.logwarn("Joint limits violated!")
            rospy.logwarn("Violated joint limits: ")
            rospy.logwarn(
                violated_joint_limits,
                self.join_limits[violated_joint_limits]
            )
            return
        rospy.loginfo("Joint move...")
        self._joint_move(action)

    def go_home(self):
        rospy.loginfo("Go home...")
        self._go_home(Empty())
