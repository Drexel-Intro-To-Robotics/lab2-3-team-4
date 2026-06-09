#!/usr/bin/env python3

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D  # Required to register the 3D projection


def plot_joint_positions(csv_file):
    """
    Plot measured OpenManipulator-X joint positions over time.

    Expected columns:
        time, joint1, joint2, joint3, joint4
    """

    df = pd.read_csv(csv_file)

    time_values = (
        df["time"] - df["time"].iloc[0]
    ).to_numpy()

    joint1 = df["joint1"].to_numpy()
    joint2 = df["joint2"].to_numpy()
    joint3 = df["joint3"].to_numpy()
    joint4 = df["joint4"].to_numpy()

    plt.figure(figsize=(10, 6))

    plt.plot(
        time_values,
        joint1,
        label="Joint 1: Base Pan",
        linewidth=2,
    )

    plt.plot(
        time_values,
        joint2,
        label="Joint 2: Shoulder",
        linewidth=2,
    )

    plt.plot(
        time_values,
        joint3,
        label="Joint 3: Elbow",
        linewidth=2,
    )

    plt.plot(
        time_values,
        joint4,
        label="Joint 4: Wrist",
        linewidth=2,
    )

    plt.title(
        "Task 10: OpenManipulator-X Joint Positions Over Time"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (rad)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "task10_joint_positions.png",
        dpi=300,
    )

    plt.close()

    print(
        "[SUCCESS] Saved: "
        + os.path.abspath("task10_joint_positions.png")
    )


def plot_end_effector_3d(csv_file):
    """
    Plot reconstructed measured end-effector path.

    Expected columns:
        time, x, y, z
    """

    df = pd.read_csv(csv_file)

    x_values = df["x"].to_numpy()
    y_values = df["y"].to_numpy()
    z_values = df["z"].to_numpy()

    figure = plt.figure(figsize=(8, 8))

    axes = figure.add_subplot(
        111,
        projection="3d",
    )

    axes.plot(
        x_values,
        y_values,
        z_values,
        label="Measured End-Effector Path",
        linewidth=2,
    )

    axes.scatter(
        x_values[0],
        y_values[0],
        z_values[0],
        s=100,
        label="Start",
    )

    axes.scatter(
        x_values[-1],
        y_values[-1],
        z_values[-1],
        s=100,
        label="Goal",
    )

    axes.set_title(
        "Task 10: Reconstructed End-Effector Path"
    )

    axes.set_xlabel("X Position (m)")
    axes.set_ylabel("Y Position (m)")
    axes.set_zlabel("Z Position (m)")
    axes.legend()

    plt.tight_layout()

    plt.savefig(
        "task10_end_effector_3d_path.png",
        dpi=300,
    )

    plt.close()

    print(
        "[SUCCESS] Saved: "
        + os.path.abspath(
            "task10_end_effector_3d_path.png"
        )
    )


def plot_tracking_error(actual_csv, desired_csv):
    """
    Plot Cartesian tracking error using normalized-time interpolation.

    Expected columns:
        time, x, y, z
    """

    actual = pd.read_csv(actual_csv)
    desired = pd.read_csv(desired_csv)

    actual_time = (
        actual["time"] - actual["time"].iloc[0]
    ).to_numpy()

    desired_time = (
        desired["time"] - desired["time"].iloc[0]
    ).to_numpy()

    actual_x = actual["x"].to_numpy()
    actual_y = actual["y"].to_numpy()
    actual_z = actual["z"].to_numpy()

    desired_x = desired["x"].to_numpy()
    desired_y = desired["y"].to_numpy()
    desired_z = desired["z"].to_numpy()

    comparison_end_time = min(
        actual_time[-1],
        desired_time[-1],
    )

    valid_samples = (
        desired_time <= comparison_end_time
    )

    comparison_time = desired_time[
        valid_samples
    ]

    desired_x = desired_x[
        valid_samples
    ]

    desired_y = desired_y[
        valid_samples
    ]

    desired_z = desired_z[
        valid_samples
    ]

    actual_x_interpolated = np.interp(
        comparison_time,
        actual_time,
        actual_x,
    )

    actual_y_interpolated = np.interp(
        comparison_time,
        actual_time,
        actual_y,
    )

    actual_z_interpolated = np.interp(
        comparison_time,
        actual_time,
        actual_z,
    )

    error_x = desired_x - actual_x_interpolated
    error_y = desired_y - actual_y_interpolated
    error_z = desired_z - actual_z_interpolated

    total_error = np.sqrt(
        (error_x ** 2)
        + (error_y ** 2)
        + (error_z ** 2)
    )

    rms_error = np.sqrt(
        np.mean(total_error ** 2)
    )

    maximum_error = np.max(
        total_error
    )

    final_error = total_error[-1]

    plt.figure(figsize=(10, 6))

    plt.plot(
        comparison_time,
        total_error,
        label="Cartesian Tracking Error",
        linewidth=2,
    )

    plt.title(
        "Task 10: Cartesian Trajectory Tracking Error"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "task10_tracking_error.png",
        dpi=300,
    )

    plt.close()

    print(
        "[SUCCESS] Saved: "
        + os.path.abspath("task10_tracking_error.png")
    )

    print("")
    print("--- Task 10 Tracking Error Metrics ---")
    print(f"RMS error:     {rms_error:.6f} m")
    print(f"Maximum error: {maximum_error:.6f} m")
    print(f"Final error:   {final_error:.6f} m")


if __name__ == "__main__":
    plot_joint_positions(
        "../data/task10_actual_joint_positions.csv"
    )

    plot_end_effector_3d(
        "../data/task10_actual_end_effector.csv"
    )

    plot_tracking_error(
        "../data/task10_actual_end_effector.csv",
        "../data/task10_desired_jointspace.csv",
    )