#!/usr/bin/env python3

import json
import os

import rospy
from sensor_msgs.msg import JointState


ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
]


def main():
    rospy.init_node("save_current_home_pose", anonymous=True)

    print("[INFO] Waiting for the current /joint_states message...")

    message = rospy.wait_for_message(
        "/joint_states",
        JointState,
        timeout=10.0,
    )

    measured_positions = dict(
        zip(message.name, message.position)
    )

    missing_joints = [
        joint_name
        for joint_name in ARM_JOINTS
        if joint_name not in measured_positions
    ]

    if missing_joints:
        raise RuntimeError(
            "Missing expected arm joints: "
            + ", ".join(missing_joints)
        )

    home_pose = {
        joint_name: measured_positions[joint_name]
        for joint_name in ARM_JOINTS
    }

    script_folder = os.path.dirname(
        os.path.abspath(__file__)
    )

    output_file = os.path.join(
        script_folder,
        "saved_home.json",
    )

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(home_pose, file, indent=4)

    print("")
    print("[INFO] Saved the current arm configuration as the lab home pose:")
    print(json.dumps(home_pose, indent=4))
    print("")
    print(f"[INFO] Saved file: {output_file}")


if __name__ == "__main__":
    main()