import numpy as np
import time
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
import trap_trajectory as trap
import viz_trajectory as viz

# import our own modules
from transformations import HT, HR
from reach import MinimalPublisher

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
            s4 = np.sqrt(1 - c4) # make minus for elbow-up
            q4 = np.arctan2(s4, c4) # elbow

            q2 = np.arctan2(target_2[1], target_2[0]) - np.arctan2(l2*s4, l1+l2*c4) # shoulder

            # keep q3 zero
            q3 = -np.pi

            # 3. keep the wrist fixed to the zero configuration
            q5 = 0.
            q6 = 0.
            q7 = 0.
            self.target_configuration = [q1,q2,q3,q4,q5,q6,q7]
            # get target position of desired object and set positional array as global
            print(self.target_configuration)
            global target_position
            target_position = self.target_configuration
            
    #control gripper state
    def gripper_motion(self, motion="close"):
        goal_msg = PlayMotion2.Goal()
        goal_msg.motion_name = motion
        self.gripper_action.wait_for_server()

        return self.gripper_action.send_goal_async(goal_msg)

def main():
    target_positions = []
    rclpy.init()

    #finding target point A
    node = InverseKinematics("arm_1_link", "A")
    #Loops until target point is found
    while node.target_configuration is None:
        try:
            rclpy.spin_once(node)
        except KeyboardInterrupt:
            break
    #add new target point to array
    target_positions.append(target_position)
    #finding target point B
    node = InverseKinematics("arm_1_link", "B")

    while node.target_configuration is None:
        try:
            rclpy.spin_once(node)
        except KeyboardInterrupt:
            break

    target_positions.append(target_position)
    #finding target point C
    node.current_configuration = target_positions[-1][::]
    node = InverseKinematics("arm_1_link", "C")

    while node.target_configuration is None:
        try:
            rclpy.spin_once(node)
        except KeyboardInterrupt:
            break

    target_positions.append(target_position)
    #set gripper state to closed
    future = node.gripper_motion("open")
    rclpy.spin_until_future_complete(node, future)
    time.sleep(2)
    #call the class to plan trajectory and move joints
    node2 = MinimalPublisher(target_positions)
    rclpy.spin(node2)

    node.destroy_node()

    rclpy.shutdown()

if __name__== "__main__":
    main()
