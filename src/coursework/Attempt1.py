import numpy as np
import time
import trap_trajectory as trap
import viz_trajectory as viz

from sympy import Matrix
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from play_motion2_msgs.action import PlayMotion2

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

# import our own modules
from transformations import HT, HR

# max speeds for each joint
max_speed = np.array([
    18.0,
    18.0,
    22.0,
    22.0,
    17.0,
    17.0,
    17.0
])
# rpm to rad/s
max_speed = max_speed / 60.0 * 2 * np.pi

# The position our arm should reach
target_position = [
    0.6496184220867192,    #arm_1_joint
    -1.1910595388819816,  #arm_2_joint
    -3.141592653589793,  #arm_3_joint
    1.7904987228240348,   #arm_4_joint
    -0.,  #arm_5_joint
    -0.,  #arm_6_joint
    -0.   #arm_7_joint
]
class InverseKinematics(Node):
    def __init__(self, base_frame, target_frame):
        super().__init__('inverse_kinematics_node')

        self.base_frame = base_frame

        # Declare and acquire `target_frame` parameter
        self.from_frame_rel = self.declare_parameter(
          'target_frame', target_frame).get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # subscriber to joint states
        self.joint_idx = {}
        self.js_pub = self.create_subscription(
            JointState,
            "/joint_states",
            self.js_cb,
            10
        )

        self.current_configuration = None
        self.target_configuration = None

        # to open/close the gripper
        self.gripper_action = ActionClient(self, PlayMotion2, 'play_motion2')

        period = 1
        self.timer = self.create_timer(period, self.get_transform)
                # publisher for joint command
        self.publisher_ = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)

        self.last_state = None
        self.joint_idx = {}

        # subscriber to joint states
        self.js_pub = self.create_subscription(
            JointState,
            "/joint_states",
            self.js_cb,
            10
        )
        
        # initialise visualisation module
        viz.init(self)

        # period = 0.5
        self.i = 0
        # self.timer = self.create_timer(period, self.timer_cb)

    # joints state callback
    def js_cb(self, msg):
        if len(self.joint_idx) == 0:
            self.joint_idx = dict(zip(
                msg.name,
                [i for i in range(len(msg.name))]
            ))

        self.current_configuration = msg
        # self.get_logger().info('Got: {}'.format(msg))

    def get_transform(self):
        if self.current_configuration is None:
            return 

        # Look up for the transformation between base_frame and target_frame 
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.from_frame_rel,
                rclpy.time.Time())
            self.get_logger().info(
                f'Transform {self.base_frame} to {self.from_frame_rel} is: {t}')
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.base_frame} to {self.from_frame_rel}: {ex}')
        else:
            target = t.transform.translation
            # TODO compute a valid configuration
            # 1. set q1 to point towards the target (w.r.t. current q1 value)
            q1_ = np.arctan2(target.y, target.x)
            #  we need to add the current value of q1 (to make it abs)
            q1 = q1_ + self.current_configuration.position[self.joint_idx["arm_1_joint"]]

            # rotate the frame around q1
            HRq1 = HR(axis='z', q='q1').subs({'q1': q1_})
            # translate to arm_2_link frame (get the numbers from the xacro file)
            HT2 = HT(x="x", y="y", z="z").subs({
                "x": 0.125,
                "y": 0.0195,
                "z": -0.031
            })
            # change the axis of rotation for frame 2
            HR2 = HR(axis='x', q='a1').subs({'a1': np.pi/2.})

            # compute the total transformation and get the target position in the new frame
            T = HRq1 * HT2 * HR2 
            target_2 = T.inv() * Matrix([target.x, target.y, target.z, 1])
            target_2 = np.array(target_2).astype(np.float64).flatten() # convert to np array

            # comput shoulder and elbow as a planar 2R robot, elbow-down configuration
            l1 = 0.0895 + 0.222 # from arm_2_link to arm_4_link
            l2 = 0.162 + 0.15 + 0.22# from arm_4_link to centre of the fingers 
            print("l1: {}, l2: {}".format(l1,l2))

            # cos q4 and sin q4
            c4 = (target_2[0]**2 + target_2[1]**2 - l1**2 - l2**2)/(2*l1 * l2)
            s4 = np.sqrt(1 - c4) #set minus to change elbow-up
            q4 = np.arctan2(s4, c4) # elbow

            q2 = np.arctan2(target_2[1], target_2[0]) - np.arctan2(l2*s4, l1+l2*c4) # shoulder

            # keep q3 zero
            q3 = -np.pi

            # 3. keep the wrist fixed to the zero configuration
            q5 = 0.
            q6 = 0.
            q7 = 0.
            self.target_configuration = [q1,q2,q3,q4,q5,q6,q7]

            print(self.target_configuration)

    def gripper_motion(self, motion="open"):
        goal_msg = PlayMotion2.Goal()
        goal_msg.motion_name = motion
        self.gripper_action.wait_for_server()

        return self.gripper_action.send_goal_async(goal_msg)

def main():
    rclpy.init()
    node = InverseKinematics("arm_1_link", "B")

    while node.target_configuration is None:
        try:
            rclpy.spin_once(node)
        except KeyboardInterrupt:
            break

    # open the gripper
    future = node.gripper_motion("open")
    rclpy.spin_until_future_complete(node, future)
    time.sleep(5)
    
    # TODO execute the trajectory to target point

    # close the gripper
    future = node.gripper_motion("close")
    rclpy.spin_until_future_complete(node, future)
    time.sleep(5)

    # TODO execute other trajectories


    rclpy.shutdown()

if __name__== "__main__":
    main()