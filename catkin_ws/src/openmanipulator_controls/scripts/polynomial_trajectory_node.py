#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import actionlib
import rospy

from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ik_solver import solve_ik


ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
]

SAMPLE_RATE_HZ = 50.0
DT = 1.0 / SAMPLE_RATE_HZ

# OpenManipulator-X effective link lengths in meters.
L1 = 0.0963
L2 = 0.1302
L3 = 0.1240
L4 = 0.1334


class PolynomialTrajectoryNode:
    """
    Lab 3 Tasks 9 and 10.

    Task 9:
        Generate a cubic polynomial trajectory in task space.
        Sample Cartesian points at 50 Hz and solve IK at every point.

    Task 10:
        Generate a cubic polynomial trajectory directly in joint space.
    """

    def __init__(self):
        rospy.init_node("polynomial_trajectory_node")

        self.current_joints: Dict[str, float] = {}

        rospy.Subscriber(
            "/joint_states",
            JointState,
            self.joint_state_callback,
            queue_size=1,
        )

        self.desired_pose_pub = rospy.Publisher(
            "/task9/desired_pose",
            PoseStamped,
            queue_size=10,
        )

        self.desired_velocity_pub = rospy.Publisher(
            "/task9/desired_velocity",
            TwistStamped,
            queue_size=10,
        )

        self.arm_client = actionlib.SimpleActionClient(
            "/arm_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )

        rospy.loginfo("Waiting for arm trajectory controller...")

        if not self.arm_client.wait_for_server(rospy.Duration(10.0)):
            raise RuntimeError(
                "Could not connect to "
                "/arm_controller/follow_joint_trajectory"
            )

        rospy.loginfo("Connected to arm trajectory controller.")

    def joint_state_callback(self, msg: JointState):
        measured = dict(zip(msg.name, msg.position))

        for joint_name in ARM_JOINTS:
            if joint_name in measured:
                self.current_joints[joint_name] = measured[joint_name]

    def wait_for_joint_states(self):
        timeout_time = time.time() + 5.0

        while not rospy.is_shutdown():
            if all(
                joint_name in self.current_joints
                for joint_name in ARM_JOINTS
            ):
                return

            if time.time() > timeout_time:
                raise RuntimeError(
                    "Timed out waiting for /joint_states."
                )

            rospy.sleep(0.05)

    def get_current_joint_values(self) -> List[float]:
        self.wait_for_joint_states()

        return [
            self.current_joints[joint_name]
            for joint_name in ARM_JOINTS
        ]

    @staticmethod
    def cubic_sample(
        start: float,
        goal: float,
        duration: float,
        time_value: float,
    ) -> Tuple[float, float]:
        """
        Cubic interpolation with zero velocity at the endpoints.

        q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        """

        delta = goal - start

        a0 = start
        a1 = 0.0
        a2 = 3.0 * delta / (duration ** 2)
        a3 = -2.0 * delta / (duration ** 3)

        position = (
            a0
            + a1 * time_value
            + a2 * (time_value ** 2)
            + a3 * (time_value ** 3)
        )

        velocity = (
            a1
            + 2.0 * a2 * time_value
            + 3.0 * a3 * (time_value ** 2)
        )

        return position, velocity

    @staticmethod
    def solve_fk(joints: List[float]) -> Tuple[float, float, float, float]:
        """
        Forward kinematics matching the Task 2 analytical model.
        """

        theta1, theta2, theta3, theta4 = joints

        pitch = theta2 + theta3 + theta4

        radial_distance = (
            L2 * math.cos(theta2)
            + L3 * math.cos(theta2 + theta3)
            + L4 * math.cos(pitch)
        )

        x = radial_distance * math.cos(theta1)
        y = radial_distance * math.sin(theta1)

        z = (
            L1
            + L2 * math.sin(theta2)
            + L3 * math.sin(theta2 + theta3)
            + L4 * math.sin(pitch)
        )

        return x, y, z, pitch

    @staticmethod
    def pitch_to_quaternion(pitch: float) -> Tuple[float, float, float, float]:
        """
        Convert a pitch-only orientation into a quaternion.
        """

        half_pitch = pitch / 2.0

        return (
            0.0,
            math.sin(half_pitch),
            0.0,
            math.cos(half_pitch),
        )

    def publish_desired_state(
        self,
        x: float,
        y: float,
        z: float,
        pitch: float,
        vx: float,
        vy: float,
        vz: float,
    ):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "base_footprint"

        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z

        qx, qy, qz, qw = self.pitch_to_quaternion(pitch)

        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        velocity_msg = TwistStamped()
        velocity_msg.header = pose_msg.header

        velocity_msg.twist.linear.x = vx
        velocity_msg.twist.linear.y = vy
        velocity_msg.twist.linear.z = vz

        self.desired_pose_pub.publish(pose_msg)
        self.desired_velocity_pub.publish(velocity_msg)

    def build_task_space_trajectory(
        self,
        goal_pose: List[float],
        duration: float,
        desired_csv: str,
    ) -> JointTrajectory:
        """
        Task 9:
        Cubically interpolate x, y, z, and pitch.
        Run IK for every Cartesian sample.
        """

        start_joints = self.get_current_joint_values()
        start_x, start_y, start_z, start_pitch = self.solve_fk(start_joints)

        goal_x, goal_y, goal_z, goal_pitch = goal_pose

        sample_count = int(math.ceil(duration * SAMPLE_RATE_HZ))

        trajectory = JointTrajectory()
        trajectory.joint_names = ARM_JOINTS
        trajectory.header.stamp = (
            rospy.Time.now() + rospy.Duration(0.25)
        )

        csv_rows = []

        for sample_index in range(1, sample_count + 1):
            t = min(sample_index * DT, duration)

            x, vx = self.cubic_sample(start_x, goal_x, duration, t)
            y, vy = self.cubic_sample(start_y, goal_y, duration, t)
            z, vz = self.cubic_sample(start_z, goal_z, duration, t)
            pitch, pitch_rate = self.cubic_sample(
                start_pitch,
                goal_pitch,
                duration,
                t,
            )

            joints = solve_ik(x, y, z, pitch)

            if joints is None:
                raise RuntimeError(
                    "IK failed during task-space interpolation at "
                    f"t={t:.3f} s, pose=[{x:.4f}, {y:.4f}, {z:.4f}, "
                    f"{pitch:.4f}]"
                )

            point = JointTrajectoryPoint()
            point.positions = joints
            point.time_from_start = rospy.Duration.from_sec(t)

            trajectory.points.append(point)

            csv_rows.append([
                t,
                x,
                y,
                z,
                pitch,
                vx,
                vy,
                vz,
                pitch_rate,
            ])

        with open(desired_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow([
                "time",
                "x",
                "y",
                "z",
                "pitch",
                "vx",
                "vy",
                "vz",
                "pitch_rate",
            ])

            writer.writerows(csv_rows)

        return trajectory

    def build_joint_space_trajectory(
        self,
        goal_joints: List[float],
        duration: float,
        desired_csv: str,
    ) -> JointTrajectory:
        """
        Task 10:
        Cubically interpolate joint1 through joint4 directly.
        """

        start_joints = self.get_current_joint_values()

        sample_count = int(math.ceil(duration * SAMPLE_RATE_HZ))

        trajectory = JointTrajectory()
        trajectory.joint_names = ARM_JOINTS
        trajectory.header.stamp = (
            rospy.Time.now() + rospy.Duration(0.25)
        )

        csv_rows = []

        for sample_index in range(1, sample_count + 1):
            t = min(sample_index * DT, duration)

            positions = []
            velocities = []

            for start, goal in zip(start_joints, goal_joints):
                position, velocity = self.cubic_sample(
                    start,
                    goal,
                    duration,
                    t,
                )

                positions.append(position)
                velocities.append(velocity)

            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = velocities
            point.time_from_start = rospy.Duration.from_sec(t)

            trajectory.points.append(point)

            x, y, z, pitch = self.solve_fk(positions)

            csv_rows.append([
                t,
                positions[0],
                positions[1],
                positions[2],
                positions[3],
                velocities[0],
                velocities[1],
                velocities[2],
                velocities[3],
                x,
                y,
                z,
                pitch,
            ])

        with open(desired_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow([
                "time",
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint1_velocity",
                "joint2_velocity",
                "joint3_velocity",
                "joint4_velocity",
                "x",
                "y",
                "z",
                "pitch",
            ])

            writer.writerows(csv_rows)

        return trajectory

    def execute_trajectory(
        self,
        trajectory: JointTrajectory,
        duration: float,
        task_space_mode: bool,
    ):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = rospy.Duration(2.0)

        rospy.loginfo(
            "Sending trajectory with %d points.",
            len(trajectory.points),
        )

        self.arm_client.send_goal(goal)

        start_time = rospy.Time.now()

        rate = rospy.Rate(SAMPLE_RATE_HZ)

        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()

            if elapsed >= duration:
                break

            if task_space_mode:
                trajectory_index = min(
                    int(elapsed * SAMPLE_RATE_HZ),
                    len(trajectory.points) - 1,
                )

                point = trajectory.points[trajectory_index]
                x, y, z, pitch = self.solve_fk(
                    list(point.positions)
                )

                self.publish_desired_state(
                    x=x,
                    y=y,
                    z=z,
                    pitch=pitch,
                    vx=0.0,
                    vy=0.0,
                    vz=0.0,
                )

            rate.sleep()

        completed = self.arm_client.wait_for_result(
            rospy.Duration(5.0)
        )

        if not completed:
            self.arm_client.cancel_goal()

            raise RuntimeError(
                "Trajectory execution timed out and was cancelled."
            )

        if self.arm_client.get_state() != GoalStatus.SUCCEEDED:
            raise RuntimeError(
                "Controller reported failure: "
                + self.arm_client.get_goal_status_text()
            )

        rospy.loginfo("Trajectory completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Lab 3 Tasks 9 and 10 polynomial trajectory node."
    )

    parser.add_argument(
        "--mode",
        choices=["task", "joint"],
        required=True,
        help="'task' for Task 9 or 'joint' for Task 10.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Trajectory duration in seconds.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Desired trajectory CSV output filename.",
    )

    parser.add_argument(
        "--task-goal",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "PITCH"),
        help="Task-space goal: x y z pitch.",
    )

    parser.add_argument(
        "--joint-goal",
        nargs=4,
        type=float,
        metavar=("J1", "J2", "J3", "J4"),
        help="Joint-space goal: joint1 joint2 joint3 joint4.",
    )

    args = parser.parse_args()

    node = PolynomialTrajectoryNode()

    if args.mode == "task":
        if args.task_goal is None:
            parser.error("--task-goal is required for --mode task.")

        trajectory = node.build_task_space_trajectory(
            goal_pose=list(args.task_goal),
            duration=args.duration,
            desired_csv=args.output,
        )

        node.execute_trajectory(
            trajectory=trajectory,
            duration=args.duration,
            task_space_mode=True,
        )

    else:
        if args.joint_goal is None:
            parser.error("--joint-goal is required for --mode joint.")

        trajectory = node.build_joint_space_trajectory(
            goal_joints=list(args.joint_goal),
            duration=args.duration,
            desired_csv=args.output,
        )

        node.execute_trajectory(
            trajectory=trajectory,
            duration=args.duration,
            task_space_mode=False,
        )


if __name__ == "__main__":
    main()