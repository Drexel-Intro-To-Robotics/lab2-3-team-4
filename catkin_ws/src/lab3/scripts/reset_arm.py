#!/usr/bin/env python3

import sys
import rospy
import moveit_commander


def main():
    """
    Return the OpenManipulator arm to the predefined MoveIt 'home' pose.
    """

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("return_arm_home", anonymous=True)

    group = moveit_commander.MoveGroupCommander("arm")

    # Reduced speed for safety.
    group.set_max_velocity_scaling_factor(0.10)
    group.set_max_acceleration_scaling_factor(0.10)

    # Give MoveIt more time and more attempts.
    group.set_planning_time(15.0)
    group.set_num_planning_attempts(10)

    # Start from the robot's actual current joint state.
    group.set_start_state_to_current_state()

    rospy.loginfo("Planning motion to the predefined home pose.")

    group.set_named_target("home")

    success = group.go(wait=True)

    group.stop()
    group.clear_pose_targets()

    if success:
        rospy.loginfo("Arm returned to the home position successfully.")
    else:
        rospy.logwarn("Failed to return the arm to the home position.")

    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()