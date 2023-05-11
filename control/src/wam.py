import rospy
import numpy as np

from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from wam_srvs.srv import JointMoveBlock


class WAM:
    def __init__(self, namespace, constraints={'table'}, subparsers=None):
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
        rospy.wait_for_service(f'{namespace}/joint_move_block')
        self._joint_move = rospy.ServiceProxy(
            f'/{namespace}/joint_move_block',
            JointMoveBlock
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
        self._start_position = np.array([
            0.004784559124119499,
            0.4490540623980915,
            -0.0726144751661842,
            1.6803055108189549,
            0.06452207850075688,
            -0.06341508205589094,
            0.01366506663019359,
        ])
        self.active_joints = [0, 1, 3]
        # TODO: constrain so only active joints are useds
        self.table_constraint_z = -0.15
        self._position = None
        self._velocity = None
        self.dof = None
        self.ready = False
        self.constraints = constraints
        self.emergency = False

        self.subparsers = subparsers
        if self.subparsers is not None:
            self.parser = self.subparsers.add_parser('wam', help='WAM')
            self.parser.set_defaults(func=self.handle_args)
            self.parser.add_argument(
                '--position',
                default=None,
                choices=['home', 'start'],
                help="Move robot to position"
            )
            self.parser.add_argument(
                '--joints',
                type=float,
                default=None,
                nargs="*",
                help='Move robot to joint angles'
            )
            self.parser.add_argument(
                '--info',
                action='store_true',
                help='Get information on WAM state'
            )
        rospy.on_shutdown(self.on_shutdown)

    @property
    def position(self):
        return self._position.copy()

    @property
    def velocity(self):
        return self._velocity.copy()

    @property
    def start_position(self):
        return self._start_position[:self.dof].copy()

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
        self.pose = message.pose
        self.enforce_constraints()

    def enforce_constraints(self):
        if 'table' in self.constraints and self.pose.position.z < self.table_constraint_z:
            rospy.logerr("Table constrains violated!")
            self.emergency_stop()
            return

    def joint_move(self, action: np.ndarray, block: bool = True):
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
        self._joint_move(action, block)

    def go_home(self):
        rospy.loginfo("Go home...")
        self._go_home()

    def go_start(self):
        rospy.loginfo("Go start...")
        self.joint_move(self.start_position)

    def handle_args(self, args):
        if args.position == 'home':
            self.go_home()
        elif args.position == 'start':
            self.go_start()
        elif args.joints is not None:
            if len(args.joints) != self.dof:
                rospy.logwarn(f"--joints requires {self.dof} floats")
                return
            self.joint_move(args.joints)
        if args.info:
            rospy.loginfo("WAM Info")
            rospy.loginfo(f"  Position: {self.position}")
            rospy.loginfo(f"  Velocity: {self.velocity}")
            rospy.loginfo(f"  Pose: ")
            rospy.loginfo(self.pose)
