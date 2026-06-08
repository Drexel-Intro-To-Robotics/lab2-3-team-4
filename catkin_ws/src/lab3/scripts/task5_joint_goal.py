#!/usr/bin/env python3

import argparse
import sys

import rospy
import moveit_commander


POINTS = {
    "a": {
        "name": "Point A: Forward Center",
        "joints": [0.0000, 0.7849, -0.3196, -0.9653],
    },
    "b": {
        "name": "Point B: Forward Left",
        "joints": [0.3029, 0.6915, -0.1190, -0.7725],
    },
    "c": {
        "name": "Point C: Forward Right",
        "joints": [-0.3029, 0.6915, -0.1190, -0.7725],
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
        description="Plan and execute an OpenManipulator joint-space goal."
    )

    parser.add_argument(
        "--point",
        choices=["a", "b", "c"],
        required=True,
        help="Select Point A, B, or C."
    )

    parser.add_argument(
        "--planner",
        choices=["rrtconnect", "prm", "kpiece"],
        required=True,
        help="Select the MoveIt OMPL planner."
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute after planning and manual confirmation."
    )

    args = parser.parse_args()

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("task5_joint_goal", anonymous=True)

    arm = moveit_commander.MoveGroupCommander("arm")

    # Reduced speed for safe physical testing.
    arm.set_max_velocity_scaling_factor(0.10)
    arm.set_max_acceleration_scaling_factor(0.10)

    arm.set_planning_time(10.0)
    arm.set_num_planning_attempts(10)

    point = POINTS[args.point]
    planner_id = PLANNERS[args.planner]
    target_joints = point["joints"]

    print("")
    print("=" * 70)
    print("Task 5 Joint-Space Motion-Planning Trial")
    print("=" * 70)
    print(f"Goal:             {point['name']}")
    print(f"Planner:          {planner_id}")
    print(f"Target joints:    {target_joints}")

    arm.set_planner_id(planner_id)
    arm.set_joint_value_target(target_joints)

    valid_plan, trajectory, planning_time = extract_plan(
        arm.plan()
    )

    print(f"Planning time:    {planning_time:.6f} s")

    if not valid_plan:
        print("[ERROR] MoveIt could not produce a valid trajectory.")
        moveit_commander.roscpp_shutdown()
        sys.exit(1)

    print(
        "Trajectory points: "
        f"{len(trajectory.joint_trajectory.points)}"
    )

    if not args.execute:
        print("[INFO] Plan-only mode. The physical arm was not moved.")
        print("[INFO] Inspect the displayed path in RViz.")
        moveit_commander.roscpp_shutdown()
        return

    input(
        "\nInspect the RViz plan and clear the workspace. "
        "Press ENTER to execute, or press Ctrl+C to stop: "
    )

    print("[INFO] Executing trajectory.")
    success = arm.execute(trajectory, wait=True)
    arm.stop()

    if success:
        print("[INFO] Motion completed successfully.")
    else:
        print("[WARN] MoveIt did not confirm successful completion.")

    measured_joints = arm.get_current_joint_values()

    print("")
    print("Measured final joint positions:")
    for index, measured_value in enumerate(measured_joints[:4], start=1):
        print(f"joint{index} = {measured_value:.6f}")

    joint_errors = [
        measured - desired
        for measured, desired
        in zip(measured_joints[:4], target_joints)
    ]

    print("")
    print("Final joint errors:")
    for index, error in enumerate(joint_errors, start=1):
        print(f"joint{index} error = {error:.6f} rad")

    current_pose = arm.get_current_pose().pose

    print("")
    print("Measured final end-effector position:")
    print(f"x = {current_pose.position.x:.6f}")
    print(f"y = {current_pose.position.y:.6f}")
    print(f"z = {current_pose.position.z:.6f}")

    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()