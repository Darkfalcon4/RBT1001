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

import attepmt3 as point1
import attepmt2 as point2
import attempt3_reach as pointA

A = []
while True:
    point1.main()

    time.sleep(3)
    point2.main()