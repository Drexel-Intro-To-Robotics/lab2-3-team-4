#!/usr/bin/env python3

import math

import actionlib
import rospy

from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from openmanipulator_controls.srv import SetGoal, SetGoalResponse
from ik_solver import solve_ik


ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
]

SAMPLE_RATE_HZ = 50.0
DEFAULT_DURATION = 5.0


class TrajectoryCommandNode:
    """
    Lab 3 Task 8 trajectory-control node.

    Accepts:
      1. Joint-space goals
      2. Task-space goals
      3. Lists of task-space waypoints

    Generates smooth cubic joint trajectories and sends them to the
    physical OpenManipulator arm controller.
    """

    def __init__(self):
        rospy.init_node("trajectory_command_node")

        self.current_joints = {}

        rospy.Subscriber(
            "/joint_states",
            JointState,
            self.joint_state_callback,
            queue_size=1,
        )

        self.arm_client = actionlib.SimpleActionClient(
            "/arm_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )

        rospy.loginfo("Waiting for the arm trajectory controller...")

        if not self.arm_client.wait_for_server(rospy.Duration(10.0)):
            raise RuntimeError(
                "Could not connect to "
                "/arm_controller/follow_joint_trajectory"
            )

        rospy.Service(
            "/set_manipulator_goal",
            SetGoal,
            self.handle_goal,
        )

        rospy.loginfo("Connected to the arm trajectory controller.")
        rospy.loginfo("Task 8 trajectory-command node is ready.")

    def joint_state_callback(self, msg):
        """Store the latest measured positions for joint1 through joint4."""

        measured = dict(zip(msg.name, msg.position))

        for joint_name in ARM_JOINTS:
            if joint_name in measured:
                self.current_joints[joint_name] = measured[joint_name]

    def get_current_joint_values(self):
        """Return current arm position in joint1-to-joint4 order."""

        missing = [
            joint_name
            for joint_name in ARM_JOINTS
            if joint_name not in self.current_joints
        ]

        if missing:
            raise RuntimeError(
                "Missing /joint_states feedback for: "
                + ", ".join(missing)
            )

        return [
            self.current_joints[joint_name]
            for joint_name in ARM_JOINTS
        ]

    @staticmethod
    def cubic_sample(start, goal, duration, time_value):
        """
        Cubic interpolation with zero velocity at both endpoints.
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

    def append_segment(
        self,
        trajectory,
        start_joints,
        goal_joints,
        duration,
        time_offset,
    ):
        """Append one cubic trajectory segment sampled at 50 Hz."""

        sample_count = int(math.ceil(duration * SAMPLE_RATE_HZ))

        for sample_index in range(1, sample_count + 1):
            time_value = min(
                sample_index / SAMPLE_RATE_HZ,
                duration,
            )

            point = JointTrajectoryPoint()

            positions = []
            velocities = []

            for start, goal in zip(start_joints, goal_joints):
                position, velocity = self.cubic_sample(
                    start=start,
                    goal=goal,
                    duration=duration,
                    time_value=time_value,
                )

                positions.append(position)
                velocities.append(velocity)

            point.positions = positions
            point.velocities = velocities

            point.time_from_start = rospy.Duration.from_sec(
                time_offset + time_value
            )

            trajectory.points.append(point)

    def build_trajectory(self, joint_targets, total_duration):
        """Build a trajectory through one or more joint-space targets."""

        if total_duration <= 0.0:
            raise ValueError("Duration must be greater than zero.")

        if not joint_targets:
            raise ValueError("No trajectory targets were provided.")

        current_joints = self.get_current_joint_values()

        trajectory = JointTrajectory()
        trajectory.joint_names = ARM_JOINTS

        segment_duration = total_duration / len(joint_targets)
        time_offset = 0.0

        for target in joint_targets:
            if len(target) != 4:
                raise ValueError(
                    "Every joint target must contain four values."
                )

            self.append_segment(
                trajectory=trajectory,
                start_joints=current_joints,
                goal_joints=target,
                duration=segment_duration,
                time_offset=time_offset,
            )

            current_joints = target
            time_offset += segment_duration

        trajectory.header.stamp = (
            rospy.Time.now() + rospy.Duration(0.25)
        )

        return trajectory

    @staticmethod
    def pose_to_joints(pose):
        """
        Convert a Cartesian pose into joint angles using the Task 2 IK solver.

        The pitch angle is extracted from the requested quaternion.
        """

        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w

        sin_pitch = 2.0 * ((qw * qy) - (qz * qx))

        sin_pitch = max(-1.0, min(1.0, sin_pitch))

        pitch = math.asin(sin_pitch)

        joints = solve_ik(
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pitch,
        )

        if joints is None:
            raise ValueError(
                "The IK solver could not reach a requested Cartesian pose."
            )

        return joints

    def execute_trajectory(self, trajectory, total_duration):
        """Send the generated trajectory to the physical arm controller."""

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = rospy.Duration(2.0)

        rospy.loginfo(
            "Sending trajectory with %d points.",
            len(trajectory.points),
        )

        self.arm_client.send_goal(goal)

        completed = self.arm_client.wait_for_result(
            rospy.Duration(total_duration + 5.0)
        )

        if not completed:
            self.arm_client.cancel_goal()

            raise RuntimeError(
                "Trajectory execution timed out and was cancelled."
            )

        if self.arm_client.get_state() != GoalStatus.SUCCEEDED:
            raise RuntimeError(
                "Arm controller reported failure: "
                + self.arm_client.get_goal_status_text()
            )

    def handle_goal(self, request):
        """Route the request according to its goal type."""

        try:
            duration = request.duration

            if duration <= 0.0:
                duration = DEFAULT_DURATION

            if request.goal_type == request.GOAL_TYPE_JOINT:
                if len(request.joint_goals) != 4:
                    raise ValueError(
                        "Joint-space mode requires four joint values."
                    )

                targets = [
                    list(request.joint_goals)
                ]

                mode_name = "joint-space"

            elif request.goal_type == request.GOAL_TYPE_TASK:
                targets = [
                    self.pose_to_joints(request.task_goal)
                ]

                mode_name = "task-space"

            elif request.goal_type == request.GOAL_TYPE_WAYPOINTS:
                if not request.waypoints:
                    raise ValueError(
                        "Waypoint mode requires at least one waypoint."
                    )

                targets = [
                    self.pose_to_joints(waypoint)
                    for waypoint in request.waypoints
                ]

                mode_name = "waypoint"

            else:
                raise ValueError(
                    f"Unknown goal_type: {request.goal_type}"
                )

            trajectory = self.build_trajectory(
                joint_targets=targets,
                total_duration=duration,
            )

            self.execute_trajectory(
                trajectory=trajectory,
                total_duration=duration,
            )

            message = (
                f"Completed {mode_name} trajectory with "
                f"{len(trajectory.points)} samples over "
                f"{duration:.2f} seconds."
            )

            rospy.loginfo(message)

            return SetGoalResponse(
                success=True,
                message=message,
            )

        except Exception as error:
            rospy.logerr("Task 8 request failed: %s", error)

            return SetGoalResponse(
                success=False,
                message=str(error),
            )


def main():
    TrajectoryCommandNode()
    rospy.spin()


if __name__ == "__main__":
    main()