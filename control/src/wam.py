import rospy
import numpy as np

from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from wam_srvs.srv import JointMove

class WAM:
    def __init__(self, namespace, constraints={'table'}):
        self.sub_joint_states = rospy.Subscriber(
            f'{namespace}/joint_states',
            JointState,
            self.callback_joint_states
        )
        self.sub_pose = rospy.Subscriber(
            f'{namespace}/pose',
            PoseStamped,
            self.callback_pose
        )
        rospy.wait_for_service(f'{namespace}/joint_move')
        self._joint_move = rospy.ServiceProxy(
            f'/{namespace}/joint_move',
            JointMove
        )
        rospy.wait_for_service(f'{namespace}/go_home')
        self._go_home = rospy.ServiceProxy(
            f'/{namespace}/go_home',
            Empty
        )
        self.joint_limits = np.array((
            (-2.6, 2.6),
            (-2.0, 2.0),
            (-2.8, 2.8),
            (-0.9, np.pi),
            (-4.76, 1.24),
            (-np.pi/2, np.pi/2),
            (-3.0, 3.0)
        ))
        self._ready_position = np.array([
            0.002227924477643431,
            -0.1490540623980915,
            -0.04214558734519736,
            1.6803055108189549,
            0.06452207850075688,
            -0.06341508205589094,
            0.01366506663019359,
        ])
        self._position = None
        self._velocity = None
        self.dof = None
        self.ready = False
        self.constraints = constraints
        self.emergency = False
        rospy.on_shutdown(self.on_shutdown)

    @property
    def position(self):
        return self._position.copy()

    @property
    def velocity(self):
        return self._velocity.copy()

    @property
    def ready_position(self):
        return self._ready_position[:self.dof].copy()

    def on_shutdown(self):
        rospy.loginfo("On shutdown...")
        self.ready = False

    def emergency_stop(self):
        rospy.logwarn("Emergency stop...")
        self.emergency = True

    def callback_joint_states(self, message: JointState):
        self.joint_state = message
        self.dof = len(message.position)
        self._position = np.array(message.position)
        self._velocity = np.array(message.velocity)
        self.ready = True

    def callback_pose(self, message: PoseStamped):
        self.pose = message
        rospy.loginfo("Pose:")
        rospy.loginfo(self.pose)
        self.enforce_constraints()

    def enforce_constraints(self):
        if 'table' in self.constraints and self.pose.position.z < 0.3:
            rospy.logerr("Table constraint violated!")
            self.emergency_stop()
            return

    def joint_move(self, action: np.ndarray):
        if not self.ready:
            rospy.logwarn("WAM not ready!")
            return
        if self.emergency:
            rospy.logwarn("WAM in emergency")
            return
        if len(action) != self.dof:
            rospy.logwarn(f"Action must have length {self.dof}!")
            return
        violated_joint_limit_l = action < self.joint_limits[:self.dof, 0]
        violated_joint_limit_u = action > self.joint_limits[:self.dof, 1]
        violated_joint_limits = violated_joint_limit_l | violated_joint_limit_u
        if np.any(violated_joint_limits):
            rospy.logwarn("Joint limits violated!")
            rospy.logwarn(action)
            rospy.logwarn("Violated joint limits: ")
            rospy.logwarn(violated_joint_limits)
            violated_joint_limits = np.pad(violated_joint_limits, (0, len(self.joint_limits) - self.dof), 'constant', constant_values=(0, 0))
            rospy.logwarn(self.joint_limits[violated_joint_limits])
            return
        rospy.loginfo("Joint move...")
        self._joint_move(action)

    def go_home(self):
        rospy.loginfo("Go home...")
        self._go_home()
