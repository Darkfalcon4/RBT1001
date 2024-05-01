# importing our custom made modules
import trap_trajectory as trap
import viz_trajectory as viz

import numpy as np
import time
import rclpy
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

# max speeds and accelerations for each joint
max_speed = np.array([18.0, 18.0, 22.0, 22.0, 17.0, 17.0, 17.0]) / 60.0 * 2 * np.pi
max_acceleration = max_speed / 2  # Example acceleration limit

class MinimalPublisher(Node):

    def __init__(self, target_positions):
        super().__init__('joint_commander')
        self.target_positions = target_positions
        self.publisher_ = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.last_state = None
        self.joint_idx = {}
        self.js_pub = self.create_subscription(JointState, "/joint_states", self.js_cb, 10)
        viz.init(self)
        self.timer = self.create_timer(0.5, self.timer_cb)

    def timer_cb(self):
        if self.last_state is None:
            return
        
        trajectory, times = self.compute_joint_trajectory(self.target_positions)
        if trajectory:
            traj_msg = self.to_JointTrajectory(trajectory, times)
            self.send_commands(traj_msg)
            viz.display(self, traj_msg)

    def compute_joint_trajectory(self, target_positions):
        initial_position = [self.last_state.position[self.joint_idx[joint]] for joint in self.last_state.name]
        qdmax = min(max_speed)
        qamax = min(max_acceleration)
        trajectory = trap.compute_trajectory(initial_position, target_positions, qdmax, qamax)
        
        if not trajectory:
            self.get_logger().error("Trajectory computation failed")
            return None, None

        return trajectory

    def to_JointTrajectory(self, trajectory, times):
        msg = JointTrajectory()
        msg.joint_names = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"]
        for i, t in enumerate(times):
            msg_point = JointTrajectoryPoint()
            msg_point.positions = [trajectory[j]["positions"][i] for j in range(7)]
            msg_point.velocities = [trajectory[j]["velocities"][i] for j in range(7)]
            msg_point.accelerations = [trajectory[j]["accelerations"][i] for j in range(7)]
            secs, nsecs = divmod(t, 1)
            msg_point.time_from_start.sec = int(secs)
            msg_point.time_from_start.nanosec = int(nsecs * 1e9)
            msg.points.append(msg_point)
        return msg

    def send_commands(self, joint_traj_msg):
        self.publisher_.publish(joint_traj_msg)
        self.get_logger().info('Commands published successfully.')

    def js_cb(self, msg):
        if not self.joint_idx:
            self.joint_idx = {name: idx for idx, name in enumerate(msg.name)}
        self.last_state = msg

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher(target_positions=[[...]])  # Define target positions
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
