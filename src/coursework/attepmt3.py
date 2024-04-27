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

            print(self.target_configuration)
            global target_position
            target_position = self.target_configuration

    def gripper_motion(self, motion="close"):
        goal_msg = PlayMotion2.Goal()
        goal_msg.motion_name = motion
        self.gripper_action.wait_for_server()

        return self.gripper_action.send_goal_async(goal_msg)

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('joint_commander')

        # publisher for joint command
        self.publisher_ = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)

        self.last_state = None
        self.joint_idx = {}
        #self.target_position = target_position
        # subscriber to joint states
        self.js_pub = self.create_subscription(
            JointState,
            "/joint_states",
            self.js_cb,
            10
        )
        
        # initialise visualisation module
        viz.init(self)

        period = 0.5
        self.i = 0
        self.timer = self.create_timer(period, self.timer_cb)


    def timer_cb(self):
        # wait until you have the current joint configuration
        while self.last_state is None:
            return 

        if self.i == 0:
            trajectory, times = self.compute_joint_trajectory()
            traj_msg = self.to_JointTrajectory(trajectory, times)
            viz.display(self, traj_msg)
            viz.display(self, traj_msg)
            # time.sleep(5)
            self.plot(trajectory, times)
            self.send_commands(traj_msg)
            self.i += 1

    def compute_joint_trajectory(self):
        #target_position = self.target_position
        # get initial position
        initial_state = self.last_state
        initial_position = [
            initial_state.position[self.joint_idx["arm_1_joint"]],
            initial_state.position[self.joint_idx["arm_2_joint"]],
            initial_state.position[self.joint_idx["arm_3_joint"]],
            initial_state.position[self.joint_idx["arm_4_joint"]],
            initial_state.position[self.joint_idx["arm_5_joint"]],
            initial_state.position[self.joint_idx["arm_6_joint"]],
            initial_state.position[self.joint_idx["arm_7_joint"]]
        ]
        
        # assume that all the joints move with speed equal to the lowest maximum speed of all joints
        qdmax = max_speed[-1]
        print("Max speed: {}".format(qdmax))

        # find the joint with the largest distance to target
        dists = [q[1] - q[0] for q in zip(initial_position, target_position)]
        abs_dists = [abs(d) for d in dists]
        max_dist_idx = np.argmax(abs_dists) # get the index

        # find the time needed by the joint with max dist
        total_time = abs_dists[max_dist_idx] / qdmax  + 2 # add 1 second as heuristic
        ticks = 50
        times = [float(i+1)*float(total_time)/ticks for i in range(ticks)]

        print("Joint with largest distance: {}".format(max_dist_idx))
        print("abs distances: {}".format(abs_dists))
        
        # compute trapezoidal profile for all joints
        trajectory = {}
        for i, (q0, qf) in enumerate(zip(initial_position, target_position)):

            # here use the lspb function
            ts, q, qd, ok = trap.lspb(total_time, q0, qf, qdmax, ticks)
            self.get_logger().info('{}: {}'.format(i, ok))
            if ok =="error 2":
                new_qdmax = abs(qf - q0) / (total_time - 1) # removing 1 sec as heuristic

                ts, q, qd, ok = trap.lspb(total_time, q0, qf, new_qdmax, ticks)
                self.get_logger().info('{}: {}'.format(i, ok))
                # # hack continuing at max speed
                # new_ticks = int(abs(qf - q0) / ((total_time / ticks) * abs(qdmax))) + 1
                # if new_ticks > 1:
                #     self.get_logger().info('{}_______'.format("Second attempt"))
                #     new_time = total_time / ticks * new_ticks
                #     ts, q, qd, ok = trap.lspb(new_time, q0, qf, qdmax, new_ticks)
                #     self.get_logger().info('{}: {}'.format(i, ok))
                #     # add the missing ticks by keeping the joint still 
                #     for j in range(new_ticks, ticks):
                #         ts.append(times[j])
                #         q.append(q[j-1])
                #         qd.append(0.0)

            if ts is not None:
                trajectory.update({
                    "joint_{}".format(i+1): {
                        "times": ts,
                        "positions": q,
                        "velocities": qd
                    }
                })
            else: #this case is when the joint is already at target position
                trajectory.update({
                    "joint_{}".format(i+1): {
                        "times": times,
                        "positions": [q0 for i in range(ticks)], 
                        "velocities": [0.0 for i in range(ticks)]
                    }
                })

        return trajectory, times

    def plot(self, traj, times):
        trap.plot(
            [times] * 7, 
            [traj[tjoint]["positions"] for tjoint in traj],
            [traj[tjoint]["velocities"] for tjoint in traj],
            ["q{}".format(i) for i in range(7)])


    def to_JointTrajectory(self, trajectory, times):
        # create joint msg
        msg = JointTrajectory()
        msg.joint_names = [
            "arm_1_joint",
            "arm_2_joint",
            "arm_3_joint",
            "arm_4_joint",
            "arm_5_joint",
            "arm_6_joint",
            "arm_7_joint"
        ]
        msg.points = []

        for i, t in enumerate(times):
            # define our  point message
            msg_point = JointTrajectoryPoint()
            msg_point.positions = [
                trajectory["joint_1"]["positions"][i],
                trajectory["joint_2"]["positions"][i],
                trajectory["joint_3"]["positions"][i],
                trajectory["joint_4"]["positions"][i],
                trajectory["joint_5"]["positions"][i],
                trajectory["joint_6"]["positions"][i],
                trajectory["joint_7"]["positions"][i]
            ]
            msg_point.velocities = [
                trajectory["joint_1"]["velocities"][i],
                trajectory["joint_2"]["velocities"][i],
                trajectory["joint_3"]["velocities"][i],
                trajectory["joint_4"]["velocities"][i],
                trajectory["joint_5"]["velocities"][i],
                trajectory["joint_6"]["velocities"][i],
                trajectory["joint_7"]["velocities"][i]
            ]
            msg_point.accelerations = [0.0] * 7
            secs = int(t)
            msg_point.time_from_start.sec = secs
            msg_point.time_from_start.nanosec = int((t -secs) * 1e9) 

            msg.points.append(msg_point)
        return msg

    def send_commands(self, joint_traj_msg):

        self.publisher_.publish(joint_traj_msg)
        self.get_logger().info('Publishing commands')


    # joints state callback
    def js_cb(self, msg):

        if len(self.joint_idx) == 0:
            self.joint_idx = dict(zip(
                msg.name,
                [i for i in range(len(msg.name))]
            ))

        self.last_state = msg

def main():
    rclpy.init()
    node = InverseKinematics("arm_1_link", "B")
    # node2 = MinimalPublisher(target_position)

    while node.target_configuration is None:
        try:
            rclpy.spin_once(node)
        except KeyboardInterrupt:
            break

    # open the gripper
    future = node.gripper_motion("open")
    rclpy.spin_until_future_complete(node, future)
    time.sleep(5)
    
    #target_position = node.target_configuration
    # move = node2.compute_joint_trajectory()
    node2 = MinimalPublisher()
    rclpy.spin(node2)
    #rclpy.spin_once(node2)
    # move = node2.compute_joint_trajectory()
    # rclpy.spin_until_future_complete(node2, move)

    # close the gripper
    future = node.gripper_motion("close")
    rclpy.spin_until_future_complete(node, future)
    time.sleep(5)

    # TODO execute other trajectories


    rclpy.shutdown()

if __name__== "__main__":
    main()