# importing necessary modules
import trap_trajectory as trap
import viz_trajectory as viz
import numpy as np
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

# max speeds and accelerations for each joint in rad/s and rad/s^2
max_speed = np.array([18.0, 18.0, 22.0, 22.0, 17.0, 17.0, 17.0]) / 60 * 2 * np.pi
max_acceleration = max_speed / 2  # Assuming a reasonable acceleration limit

class MinimalPublisher(Node):

    def __init__(self, target_positions):
        super().__init__('joint_commander')
        self.target_positions = target_positions
        self.publisher_ = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.last_state = None
        self.joint_idx = {}
        self.js_pub = self.create_subscription(JointState, "/joint_states", self.js_cb, 10)
        viz.init(self)
        self.timer = self.create_timer(0.1, self.timer_cb)  # Increased frequency for higher responsiveness

    def timer_cb(self):
        if self.last_state is None:
            return

        for target_position in self.target_positions:
            trajectory, times = self.compute_joint_trajectory(target_position)
            if trajectory:
                traj_msg = self.to_JointTrajectory(trajectory, times)
                self.send_commands(traj_msg)
                viz.display(self, traj_msg)

    def compute_joint_trajectory(self, target_position):
        if not self.last_state:
            return None, None
        
        initial_position = [self.last_state.position[idx] for idx in self.joint_idx.values()]
        trajectory, times = trap.compute_trajectory(initial_position, target_position, max_speed, max_acceleration)
        
        if trajectory is None:
            self.get_logger().error("Failed to compute trajectory")
            return None, None

        return trajectory, times

    def to_JointTrajectory(self, trajectory, times):
        msg = JointTrajectory()
        msg.joint_names = self.last_state.name
        for time, positions, velocities in zip(times, trajectory['positions'], trajectory['velocities']):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = velocities
            point.time_from_start.sec = int(time)
            point.time_from_start.nanosec = int((time % 1) * 1e9)
            msg.points.append(point)
        return msg

    def send_commands(self, traj_msg):
        self.publisher_.publish(traj_msg)
        self.get_logger().info('Joint trajectory published.')

    def js_cb(self, msg):
        if len(self.joint_idx) == 0:
            self.joint_idx = {name: i for i, name in enumerate(msg.name)}
        self.last_state = msg

def main(args=None):
    rclpy.init(args=args)
    target_positions = [[1, 1.5, 2, 1, 1, 0.5, 0], ...]  # Target positions should be defined here
    minimal_publisher = MinimalPublisher(target_positions)
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
