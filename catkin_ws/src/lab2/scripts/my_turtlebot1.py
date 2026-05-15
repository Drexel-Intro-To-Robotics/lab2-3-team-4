#!/usr/bin/env python3

import sys
import math
import rospy

from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


class myTurtle:
    def __init__(self):
        """
        Create publishers, subscribers, and internal robot state.
        """
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_cb)

        # Subscribes to RViz 2D Nav Goal topic
        self.goal_sub = rospy.Subscriber(
            '/move_base_simple/goal',
            PoseStamped,
            self.nav_to_pose
        )

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.current_odom = None
        self.rate = rospy.Rate(10)

    def nav_to_pose(self, goal):
        """
        Callback for /move_base_simple/goal.

        The robot:
        1. Turns toward the goal position.
        2. Drives straight to the goal.
        3. Rotates to match the final goal orientation.
        """
        goal_x = goal.pose.position.x
        goal_y = goal.pose.position.y

        dx = goal_x - self.x
        dy = goal_y - self.y

        distance = math.sqrt(dx ** 2 + dy ** 2)
        heading = math.atan2(dy, dx)

        rospy.loginfo("Received nav goal: x=%.2f, y=%.2f", goal_x, goal_y)
        rospy.loginfo("Distance to goal: %.2f", distance)

        self.rotate(heading - self.theta)
        self.drive_straight(distance, 0.2)

        q = goal.pose.orientation
        final_yaw = self.convert_to_euler(q)
        self.rotate(final_yaw - self.theta)

        self.stop()

    def odom_cb(self, msg):
        """
        Odometry callback.

        Updates the robot's current x, y, and theta values from /odom.
        """
        self.current_odom = msg

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.theta = self.convert_to_euler(msg.pose.pose.orientation)

    def stop(self):
        """
        Stop the robot.
        """
        vel_msg = Twist()
        self.cmd_vel_pub.publish(vel_msg)

    def drive_straight(self, dist, vel):
        """
        Drive straight for a requested distance.

        Args:
            dist: Distance in meters.
            vel: Linear velocity in m/s.
        """
        start_x = self.x
        start_y = self.y

        vel_msg = Twist()
        vel_msg.linear.x = abs(vel)
        vel_msg.angular.z = 0.0

        rospy.loginfo("Driving straight %.2f meters", dist)

        while not rospy.is_shutdown():
            dx = self.x - start_x
            dy = self.y - start_y
            traveled = math.sqrt(dx ** 2 + dy ** 2)

            if traveled >= dist:
                break

            self.cmd_vel_pub.publish(vel_msg)
            self.rate.sleep()

        self.stop()
        rospy.sleep(0.3)

    def spin_wheels(self, u1, u2, duration):
        """
        Spin the two wheels using a simple differential-drive approximation.

        Args:
            u1: Left wheel speed approximation.
            u2: Right wheel speed approximation.
            duration: Time in seconds.
        """
        vel_msg = Twist()
        vel_msg.linear.x = (u1 + u2) / 2.0
        vel_msg.angular.z = (u2 - u1)

        start_time = rospy.Time.now().to_sec()

        rospy.loginfo("Spinning wheels for %.2f seconds", duration)

        while not rospy.is_shutdown():
            elapsed = rospy.Time.now().to_sec() - start_time

            if elapsed >= duration:
                break

            self.cmd_vel_pub.publish(vel_msg)
            self.rate.sleep()

        self.stop()
        rospy.sleep(0.3)

    def rotate(self, angle):
        """
        Rotate in place by a requested angle in radians.

        Args:
            angle: Rotation angle in radians.
        """
        target_theta = self.theta + angle
        target_theta = math.atan2(math.sin(target_theta), math.cos(target_theta))

        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.3 if angle >= 0 else -0.3

        rospy.loginfo("Rotating %.2f radians", angle)

        while not rospy.is_shutdown():
            error = target_theta - self.theta
            error = math.atan2(math.sin(error), math.cos(error))

            if abs(error) < 0.02:
                break

            self.cmd_vel_pub.publish(vel_msg)
            self.rate.sleep()

        self.stop()
        rospy.sleep(0.3)

    def convert_to_euler(self, quat):
        """
        Convert quaternion orientation to yaw angle.

        Args:
            quat: geometry_msgs/Quaternion

        Returns:
            yaw angle in radians.
        """
        q = [quat.x, quat.y, quat.z, quat.w]
        _, _, yaw = euler_from_quaternion(q)
        return yaw

    def drive_circle(self, radius):
        """
        Drive in a circle with a given radius.

        Args:
            radius: Circle radius in meters.
        """
        vel_msg = Twist()
        vel_msg.linear.x = 0.2
        vel_msg.angular.z = vel_msg.linear.x / radius

        start_time = rospy.Time.now().to_sec()
        duration = (2.0 * math.pi * radius) / vel_msg.linear.x

        rospy.loginfo("Driving circle with radius %.2f meters", radius)

        while not rospy.is_shutdown():
            elapsed = rospy.Time.now().to_sec() - start_time

            if elapsed >= duration:
                break

            self.cmd_vel_pub.publish(vel_msg)
            self.rate.sleep()

        self.stop()
        rospy.sleep(0.3)

    def drive_square(self, side_length):
        """
        Drive in a square.

        Args:
            side_length: Side length in meters.
        """
        rospy.loginfo("Driving square with side length %.2f meters", side_length)

        for i in range(4):
            rospy.loginfo("Square side %d of 4", i + 1)
            self.drive_straight(side_length, 0.2)
            self.rotate(math.pi / 2.0)

        self.stop()

    def random_dance(self):
        """
        Run a simple random dance motion.
        """
        rospy.loginfo("Starting random dance")

        self.drive_straight(0.25, 0.2)

        self.rotate(math.pi)
        self.drive_straight(0.25, 0.2)

        self.rotate(math.pi)
        self.rotate(math.pi / 2.0)

        self.drive_straight(0.25, 0.2)

        self.rotate(math.pi)
        self.drive_straight(0.25, 0.2)

        self.rotate(math.pi)

        # Mini oval / wheel-spin pattern
        self.spin_wheels(0.10, 0.22, 2.0)
        self.spin_wheels(0.22, 0.10, 2.0)

        self.stop()
        rospy.loginfo("Random dance complete")


def main():
    """
    Start the ROS node and choose which Lab 2 Task 1 motion to run.

    Usage:
        rosrun lab2 my_turtlebot1.py circle
        rosrun lab2 my_turtlebot1.py square
        rosrun lab2 my_turtlebot1.py nav
        rosrun lab2 my_turtlebot1.py dance
    """
    rospy.init_node('my_turtlebot')

    turtle = myTurtle()

    rospy.sleep(2.0)

    if len(sys.argv) < 2:
        rospy.loginfo("No mode selected.")
        rospy.loginfo("Use one of: circle, square, nav, dance")
        turtle.stop()
        return

    mode = sys.argv[1].lower()

    if mode == "circle":
        turtle.drive_circle(0.5)

    elif mode == "square":
        turtle.drive_square(0.5)

    elif mode == "dance":
        turtle.random_dance()

    elif mode == "nav":
        rospy.loginfo("Nav mode active.")
        rospy.loginfo("Waiting for /move_base_simple/goal from RViz or rostopic...")
        rospy.spin()

    else:
        rospy.logerr("Unknown mode: %s", mode)
        rospy.loginfo("Use one of: circle, square, nav, dance")
        turtle.stop()


if __name__ == '__main__':
    main()