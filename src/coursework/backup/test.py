import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import time

class SpeedOdom(Node):
    def __init__(self):
        super().__init__('speed_odom')
        super().__init__('lidar_closest')

        # Subscribe to the odom topic
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.movement_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

        # Create a publisher for float topic
        self.publisher_scan_closest = self.create_publisher(Float32, 'scan_closest', 10)

        # Create publishers for float topics
        self.linear_speed_publisher = self.create_publisher(Float32, 'speed_linear', 10)
        self.angular_speed_publisher = self.create_publisher(Float32, 'speed_angular', 10)

    def odom_callback(self, msg):

        linear_speed = msg.twist.twist.linear.x
        angular_speed = msg.twist.twist.angular.z
        
        # Publish the speeds to the float topics
        linear_speed_msg = Float32()
        linear_speed_msg.data = linear_speed
        self.linear_speed_publisher.publish(linear_speed_msg)

        angular_speed_msg = Float32()
        angular_speed_msg.data = angular_speed
        self.angular_speed_publisher.publish(angular_speed_msg)

        

    def lidar_callback(self, msg):
        danger_dist = 0.4
        twist_msg = Twist()
        twist_msg.linear.x = 0.2
        
        if not msg.ranges:
            return
        
        # Filter out distances less than or equal to 0
        valid_ranges = [x for x in msg.ranges if x > 0]

        # If all distances are 0 or less, return
        if not valid_ranges:
            return

        # Find the closest distance from the laser scan data
        closest_distance = min(valid_ranges)

        if closest_distance <= danger_dist:
            twist_msg.linear.x = 0.0
            user_input = input("should i move? (y if is)")
            if user_input == "y" and not (closest_distance <= danger_dist):
                twist_msg.linear.x = 0.2

        # Print the closest distance to the terminal
        #print('Closest Distance (m): {}'.format(closest_distance))

        # Publish the closest distance on the "scan_closest" topic
        scan_closest_msg = Float32()
        scan_closest_msg.data = closest_distance
        self.publisher_scan_closest.publish(scan_closest_msg)
        self.movement_publisher.publish(twist_msg)

def main(args=None):

    rclpy.init(args=args)

    speed_odom = SpeedOdom()

    rclpy.spin(speed_odom)

    speed_odom.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()