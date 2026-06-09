#!/usr/bin/env python3

import sys
import rospy
import moveit_commander

from ik_solver import solve_ik


# Official OpenManipulator-X controller joint limits in radians.
JOINT_LIMITS = [
    (-3.142,  3.142),   # joint1
    (-2.050,  1.571),   # joint2
    (-1.571,  1.530),   # joint3
    (-1.800,  2.000),   # joint4
]


# Conservative Cartesian targets for an initial physical test.
# Format: [x, y, z, pitch]
TARGETS = [
    #("Point A: Forward Center", [0.32,  0.00, 0.18, -0.50]),
    #("Point B: Forward Left",   [0.32,  0.10, 0.22, -0.20]),
    ("Point C: Forward Right",  [0.32, -0.10, 0.22, -0.20]),
]


def within_joint_limits(joints):
    """Return True only when every joint remains inside its controller limits."""
    for index, (joint, limits) in enumerate(zip(joints, JOINT_LIMITS), start=1):
        minimum, maximum = limits

        if not minimum <= joint <= maximum:
            print(
                f"[ERROR] joint{index} = {joint:.4f} rad is outside "
                f"[{minimum:.4f}, {maximum:.4f}] rad."
            )
            return False

    return True


def extract_plan(plan_output):
    """
    Handle MoveIt versions that return either:
    - RobotTrajectory
    - tuple(success, trajectory, planning_time, error_code)
    """
    if isinstance(plan_output, tuple):
        success = plan_output[0]
        trajectory = plan_output[1]
    else:
        trajectory = plan_output
        success = True

    has_points = bool(trajectory.joint_trajectory.points)
    return success and has_points, trajectory


def return_home(arm):
    """Move to the predefined home configuration."""
    print("[INFO] Returning to home position.")
    arm.set_named_target("home")
    success = arm.go(wait=True)
    arm.stop()

    if success:
        print("[INFO] Home position reached.")
    else:
        print("[WARN] MoveIt could not confirm the home motion.")


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("task2_move_points", anonymous=True)

    arm = moveit_commander.MoveGroupCommander("arm")

    # Slow motion for physical testing.
    arm.set_max_velocity_scaling_factor(0.10)
    arm.set_max_acceleration_scaling_factor(0.10)
    arm.set_goal_joint_tolerance(0.01)

    #return_home(arm)

    for name, target_pose in TARGETS:
        print("\n" + "=" * 60)
        print(name)
        print(f"Target Cartesian pose [x, y, z, pitch]: {target_pose}")

        joints = solve_ik(*target_pose)

        if joints is None:
            print("[ERROR] IK solver could not reach this Cartesian target.")
            continue

        rounded_joints = [round(joint, 4) for joint in joints]
        print(f"Calculated joints [th1, th2, th3, th4]: {rounded_joints}")

        if not within_joint_limits(joints):
            print("[ERROR] Skipping this point because it violates joint limits.")
            continue

        arm.set_joint_value_target(joints)

        valid_plan, trajectory = extract_plan(arm.plan())

        if not valid_plan:
            print("[ERROR] MoveIt could not generate a valid trajectory.")
            print("[ERROR] Skipping this point.")
            continue

        input(
            "\nInspect the arm and clear the workspace. "
            "Press ENTER to execute this motion, or press Ctrl+C to stop: "
        )

        print("[INFO] Executing planned movement.")
        success = arm.execute(trajectory, wait=True)
        arm.stop()

        if success:
            print("[INFO] Motion completed successfully.")
        else:
            print("[WARN] MoveIt did not confirm successful completion.")

        #input("Press ENTER to return the arm to home: ")
        #return_home(arm)

    print("\n[INFO] Task 2 motion sequence finished.")
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()