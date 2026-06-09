#!/usr/bin/env python3

import json
import os
import sys

import rospy
import moveit_commander


def main():
    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node(
        "reset_arm_to_saved_home",
        anonymous=True,
    )

    script_folder = os.path.dirname(
        os.path.abspath(__file__)
    )

    home_file = os.path.join(
        script_folder,
        "saved_home.json",
    )

    if not os.path.exists(home_file):
        raise RuntimeError(
            "saved_home.json does not exist. "
            "Run save_current_home.py first."
        )

    with open(home_file, "r", encoding="utf-8") as file:
        home_pose = json.load(file)

    arm = moveit_commander.MoveGroupCommander("arm")

    # Use reduced speed for safe physical testing.
    arm.set_max_velocity_scaling_factor(0.10)
    arm.set_max_acceleration_scaling_factor(0.10)

    print("[INFO] Moving the arm to the saved lab home position:")
    print(json.dumps(home_pose, indent=4))

    arm.set_joint_value_target(home_pose)

    success = arm.go(wait=True)

    arm.stop()

    if success:
        print("[INFO] Arm returned to the saved lab home position successfully.")
    else:
        print("[ERROR] MoveIt could not return the arm to the saved home pose.")

    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()