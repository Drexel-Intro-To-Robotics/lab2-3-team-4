#!/usr/bin/env python3

import math
import rospy
import tf
import tf2_ros

from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion


class PathFollowerNode(object):
    def __init__(self):
        rospy.init_node("path_follower_node")

        # Publisher used to command the robot's linear and angular velocity.
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Path topic can be /astar/path or /astar/waypoints.
        self.path_topic = rospy.get_param("~path_topic", "/astar/path")
        self.path_sub = rospy.Subscriber(self.path_topic, Path, self.path_callback)

        # Final goal orientation from RViz 2D Nav Goal.
        self.goal_sub = rospy.Subscriber(
            "/move_base_simple/goal",
            PoseStamped,
            self.goal_pose_callback
        )

        # Stop/cancel current path when new 2D Pose Estimate is clicked.
        self.initialpose_sub = rospy.Subscriber(
            "/initialpose",
            PoseWithCovarianceStamped,
            self.initialpose_callback
        )

        self.final_goal_yaw = None

        # TF listener so robot pose is read in the same frame as the path.
        self.tf_listener = tf.TransformListener()
        self.global_frame = rospy.get_param("~global_frame", "map")
        self.robot_frame = rospy.get_param("~robot_frame", "base_footprint")

        # Current robot pose in map frame.
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.have_pose = False

        # List of waypoint/path coordinates as [(x1, y1), (x2, y2), ...].
        self.waypoints = []
        self.current_index = 0
        self.following_path = False

        # Controller parameters.
        self.distance_tolerance = rospy.get_param("~distance_tolerance", 0.15)
        self.angle_tolerance = rospy.get_param("~angle_tolerance", 0.15)

        self.linear_gain = rospy.get_param("~linear_gain", 0.20)
        self.angular_gain = rospy.get_param("~angular_gain", 1.0)

        self.max_linear_speed = rospy.get_param("~max_linear_speed", 0.05)
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 0.30)

        # Hold zero velocity after reaching final pose.
        self.goal_stop_hold_time = rospy.get_param("~goal_stop_hold_time", 1.0)
        self.path_complete = False

        self.control_rate = rospy.Rate(10)

        rospy.loginfo("Path follower node started.")
        rospy.loginfo("Following path topic: %s", self.path_topic)
        rospy.loginfo("Using TF pose: %s -> %s", self.global_frame, self.robot_frame)

    def path_callback(self, msg):
        if len(msg.poses) == 0:
            rospy.logwarn("Received empty path.")
            self.following_path = False
            self.path_complete = False
            self.waypoints = []
            self.current_index = 0
            self.stop_robot()
            return

        path_frame = msg.header.frame_id

        if path_frame != "" and path_frame != self.global_frame:
            rospy.logwarn(
                "Path frame is '%s', but follower global_frame is '%s'. "
                "This may cause frame mismatch.",
                path_frame,
                self.global_frame
            )

        self.waypoints = []

        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            self.waypoints.append((x, y))

        self.current_index = 0
        self.following_path = True
        self.path_complete = False

        rospy.loginfo("Received path with %d points.", len(self.waypoints))
        rospy.loginfo("Starting path following.")

    def initialpose_callback(self, msg):
        """
        Stop the robot and cancel the current path when a new 2D Pose Estimate
        is clicked in RViz.
        """

        rospy.logwarn("New 2D Pose Estimate received. Stopping path follower.")

        self.following_path = False
        self.path_complete = False
        self.waypoints = []
        self.current_index = 0
        self.final_goal_yaw = None

        self.stop_robot()

    def goal_pose_callback(self, msg):
        """
        Saves the final goal yaw from RViz 2D Nav Goal.
        """

        q = msg.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)

        self.final_goal_yaw = yaw

        rospy.loginfo("Final goal heading set to %.3f rad.", self.final_goal_yaw)

    def update_robot_pose_from_tf(self):
        try:
            self.tf_listener.waitForTransform(
                self.global_frame,
                self.robot_frame,
                rospy.Time(0),
                rospy.Duration(0.2)
            )

            trans, rot = self.tf_listener.lookupTransform(
                self.global_frame,
                self.robot_frame,
                rospy.Time(0)
            )

            self.x = trans[0]
            self.y = trans[1]

            _, _, yaw = euler_from_quaternion(rot)
            self.theta = yaw

            self.have_pose = True
            return True

        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
            tf2_ros.TransformException
        ):
            rospy.logwarn_throttle(
                2.0,
                "Waiting for TF transform %s -> %s",
                self.global_frame,
                self.robot_frame
            )
            self.have_pose = False
            return False

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def clamp(self, value, min_value, max_value):
        return max(min(value, max_value), min_value)

    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def hold_stop(self):
        """
        Publish zero velocity repeatedly for a short time so the real TurtleBot
        fully receives the stop command.
        """

        stop_start = rospy.Time.now()

        while (
            not rospy.is_shutdown()
            and (rospy.Time.now() - stop_start).to_sec() < self.goal_stop_hold_time
        ):
            self.stop_robot()
            self.control_rate.sleep()

    def run(self):
        while not rospy.is_shutdown():

            # Update robot pose in map frame.
            if not self.update_robot_pose_from_tf():
                self.stop_robot()
                self.control_rate.sleep()
                continue

            # Do nothing until a path has been received.
            if not self.following_path:
                self.control_rate.sleep()
                continue

            # Finished all path points: rotate to final goal heading if available.
            if self.current_index >= len(self.waypoints):

                if self.final_goal_yaw is not None:
                    final_angle_error = self.normalize_angle(
                        self.final_goal_yaw - self.theta
                    )

                    if abs(final_angle_error) > self.angle_tolerance:
                        cmd = Twist()
                        cmd.linear.x = 0.0
                        cmd.angular.z = self.clamp(
                            self.angular_gain * final_angle_error,
                            -self.max_angular_speed,
                            self.max_angular_speed
                        )

                        self.cmd_vel_pub.publish(cmd)

                        rospy.loginfo_throttle(
                            1.0,
                            "Rotating to final goal heading. Error: %.3f",
                            final_angle_error
                        )

                        self.control_rate.sleep()
                        continue

                rospy.loginfo("Reached final goal pose. Stopping robot.")

                self.following_path = False
                self.path_complete = True
                self.waypoints = []
                self.current_index = 0

                self.hold_stop()
                continue

            # Current target point.
            goal_x, goal_y = self.waypoints[self.current_index]

            dx = goal_x - self.x
            dy = goal_y - self.y

            distance = math.sqrt(dx * dx + dy * dy)
            desired_theta = math.atan2(dy, dx)
            angle_error = self.normalize_angle(desired_theta - self.theta)

            if distance < self.distance_tolerance:
                rospy.loginfo(
                    "Reached path point %d/%d: x=%.3f y=%.3f",
                    self.current_index + 1,
                    len(self.waypoints),
                    goal_x,
                    goal_y
                )

                self.current_index += 1
                self.stop_robot()
                self.control_rate.sleep()
                continue

            cmd = Twist()

            # Rotate first if not facing the point.
            if abs(angle_error) > self.angle_tolerance:
                cmd.linear.x = 0.0
                cmd.angular.z = self.clamp(
                    self.angular_gain * angle_error,
                    -self.max_angular_speed,
                    self.max_angular_speed
                )

            # Drive forward only when mostly facing the point.
            else:
                cmd.linear.x = self.clamp(
                    self.linear_gain * distance,
                    0.0,
                    self.max_linear_speed
                )

                cmd.angular.z = self.clamp(
                    self.angular_gain * angle_error,
                    -self.max_angular_speed,
                    self.max_angular_speed
                )

            self.cmd_vel_pub.publish(cmd)
            self.control_rate.sleep()


if __name__ == "__main__":
    try:
        node = PathFollowerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass