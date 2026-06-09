#!/usr/bin/env python3

import argparse
import sys

import rospy
import moveit_commander

from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler


POINTS = {
    "a": {
        "name": "Point A: Forward Center",
        "x": 0.32,
        "y": 0.00,
        "z": 0.18,
        "pitch": -0.50,
    },
    "b": {
        "name": "Point B: Forward Left",
        "x": 0.32,
        "y": 0.10,
        "z": 0.22,
        "pitch": -0.20,
    },
    "c": {
        "name": "Point C: Forward Right",
        "x": 0.32,
        "y": -0.10,
        "z": 0.22,
        "pitch": -0.20,
    },
}


PLANNERS = {
    "rrtconnect": "RRTConnectkConfigDefault",
    "prm": "PRMkConfigDefault",
    "kpiece": "KPIECEkConfigDefault",
}


def extract_plan(plan_output):
    """
    Support MoveIt versions that return either:
      RobotTrajectory
    or:
      success, trajectory, planning_time, error_code
    """

    if isinstance(plan_output, tuple):
        success = bool(plan_output[0])
        trajectory = plan_output[1]
        planning_time = float(plan_output[2])
    else:
        success = True
        trajectory = plan_output
        planning_time = -1.0

    has_points = bool(trajectory.joint_trajectory.points)

    return success and has_points, trajectory, planning_time


def main():
    parser = argparse.ArgumentParser(
        description="Plan and execute an exact Cartesian OpenManipulator goal."
    )

    parser.add_argument(
        "--point",
        choices=["a", "b", "c"],
        required=True,
        help="Cartesian goal point."
    )

    parser.add_argument(
        "--planner",
        choices=["rrtconnect", "prm", "kpiece"],
        required=True,
        help="MoveIt OMPL planner."
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute after planning and manual confirmation."
    )

    args = parser.parse_args()

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("task4_cartesian_goal", anonymous=True)

    arm = moveit_commander.MoveGroupCommander("arm")

    # Slow physical motion for safe testing.
    arm.set_max_velocity_scaling_factor(0.10)
    arm.set_max_acceleration_scaling_factor(0.10)

    point = POINTS[args.point]
    planner_id = PLANNERS[args.planner]

    planning_frame = arm.get_planning_frame()
    end_effector_link = arm.get_end_effector_link()

    print("")
    print("=" * 70)
    print("Task 4 Cartesian Motion-Planning Trial")
    print("=" * 70)
    print(f"Goal:             {point['name']}")
    print(f"Planner:          {planner_id}")
    print(f"Planning frame:   {planning_frame}")
    print(f"End-effector:     {end_effector_link}")

    # Pose orientation: roll = 0, pitch = selected value, yaw = 0.
    qx, qy, qz, qw = quaternion_from_euler(
        0.0,
        point["pitch"],
        0.0,
    )

    target = PoseStamped()
    target.header.frame_id = planning_frame
    target.header.stamp = rospy.Time.now()

    target.pose.position.x = point["x"]
    target.pose.position.y = point["y"]
    target.pose.position.z = point["z"]

    target.pose.orientation.x = qx
    target.pose.orientation.y = qy
    target.pose.orientation.z = qz
    target.pose.orientation.w = qw

    print(
        "Requested pose [x, y, z, pitch]: "
        f"[{point['x']:.4f}, {point['y']:.4f}, "
        f"{point['z']:.4f}, {point['pitch']:.4f}]"
    )

    arm.set_planner_id(planner_id)
    arm.set_pose_target(target, end_effector_link)

    valid_plan, trajectory, planning_time = extract_plan(
        arm.plan()
    )

    print(f"Planning time:    {planning_time:.6f} s")

    if not valid_plan:
        print("[ERROR] MoveIt could not produce a valid trajectory.")
        arm.clear_pose_targets()
        moveit_commander.roscpp_shutdown()
        sys.exit(1)

    print(
        "Trajectory points: "
        f"{len(trajectory.joint_trajectory.points)}"
    )

    if not args.execute:
        print("[INFO] Plan-only mode. The physical arm was not moved.")
        print("[INFO] Check the planned path in RViz.")
        arm.clear_pose_targets()
        moveit_commander.roscpp_shutdown()
        return

    input(
        "\nInspect the RViz plan and clear the physical workspace. "
        "Press ENTER to execute, or press Ctrl+C to stop: "
    )

    print("[INFO] Executing trajectory.")
    success = arm.execute(trajectory, wait=True)
    arm.stop()
    arm.clear_pose_targets()

    if success:
        print("[INFO] Motion completed successfully.")
    else:
        print("[WARN] MoveIt did not confirm successful completion.")

    current_pose = arm.get_current_pose(end_effector_link).pose

    print("")
    print("Measured final end-effector position:")
    print(f"x = {current_pose.position.x:.6f}")
    print(f"y = {current_pose.position.y:.6f}")
    print(f"z = {current_pose.position.z:.6f}")

    x_error = current_pose.position.x - point["x"]
    y_error = current_pose.position.y - point["y"]
    z_error = current_pose.position.z - point["z"]

    position_error = (
        x_error ** 2
        + y_error ** 2
        + z_error ** 2
    ) ** 0.5

    print(f"Final position error = {position_error:.6f} m")

    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()