#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd


# OpenManipulator-X effective link lengths in meters.
L1 = 0.0963
L2 = 0.1302
L3 = 0.1240
L4 = 0.1334


def solve_fk(theta1, theta2, theta3, theta4):
    """
    Compute the end-effector position using the same analytical model
    used for the Task 2 FK validation.
    """

    pitch = theta2 + theta3 + theta4

    radial_distance = (
        L2 * np.cos(theta2)
        + L3 * np.cos(theta2 + theta3)
        + L4 * np.cos(pitch)
    )

    x = radial_distance * np.cos(theta1)
    y = radial_distance * np.sin(theta1)

    z = (
        L1
        + L2 * np.sin(theta2)
        + L3 * np.sin(theta2 + theta3)
        + L4 * np.sin(pitch)
    )

    return x, y, z, pitch


def main():
    parser = argparse.ArgumentParser(
        description="Prepare clean Task 10 CSV files from raw ROS joint-state data."
    )

    parser.add_argument(
        "raw_joint_csv",
        help="Raw CSV extracted from /joint_states."
    )

    parser.add_argument(
        "output_folder",
        help="Folder for the cleaned CSV files."
    )

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    df = pd.read_csv(args.raw_joint_csv)

    required_columns = [
        "%time",
        "field.position2",
        "field.position3",
        "field.position4",
        "field.position5",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing expected columns: "
            + ", ".join(missing)
        )

    clean = pd.DataFrame()

    clean["time"] = (
        df["%time"] - df["%time"].iloc[0]
    ) / 1e9

    clean["joint1"] = df["field.position2"]
    clean["joint2"] = df["field.position3"]
    clean["joint3"] = df["field.position4"]
    clean["joint4"] = df["field.position5"]

    joint_output = os.path.join(
        args.output_folder,
        "task10_actual_joint_positions.csv",
    )

    clean.to_csv(joint_output, index=False)

    x, y, z, pitch = solve_fk(
        clean["joint1"].to_numpy(),
        clean["joint2"].to_numpy(),
        clean["joint3"].to_numpy(),
        clean["joint4"].to_numpy(),
    )

    ee = pd.DataFrame()

    ee["time"] = clean["time"]
    ee["x"] = x
    ee["y"] = y
    ee["z"] = z
    ee["pitch"] = pitch

    ee_output = os.path.join(
        args.output_folder,
        "task10_actual_end_effector.csv",
    )

    ee.to_csv(ee_output, index=False)

    print(f"[SUCCESS] Saved: {joint_output}")
    print(f"[SUCCESS] Saved: {ee_output}")


if __name__ == "__main__":
    main()