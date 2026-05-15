#!/usr/bin/env python3

import os
import matplotlib
matplotlib.use("Agg")


import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = "csv"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def get_time(df):
    t = df["%time"].astype(float)
    return (t - t.iloc[0]) / 1e9

def safe_plot(csv_name, columns, labels, title, ylabel, output_name):
    path = os.path.join(CSV_DIR, csv_name)

    if not os.path.exists(path):
        print(f"Missing: {path}")
        return

    df = pd.read_csv(path)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"Skipping {csv_name}; missing columns: {missing}")
        print("Available columns:")
        print(df.columns.tolist())
        return

    t = get_time(df).to_numpy()

    plt.figure()
    for col, label in zip(columns, labels):
        y = df[col].to_numpy()
        plt.plot(t, y, label=label)

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, output_name), dpi=200)
    plt.close()

def plot_trial(name):
    safe_plot(
        f"{name}_odom.csv",
        ["field.pose.pose.position.x", "field.pose.pose.position.y"],
        ["x position", "y position"],
        f"{name}: Odom Position",
        "Position (m)",
        f"{name}_odom_position.png"
    )

    safe_plot(
        f"{name}_odom.csv",
        ["field.twist.twist.linear.x", "field.twist.twist.angular.z"],
        ["linear x", "angular z"],
        f"{name}: Odom Velocity",
        "Velocity",
        f"{name}_odom_velocity.png"
    )

    safe_plot(
        f"{name}_imu.csv",
        ["field.angular_velocity.x", "field.angular_velocity.y", "field.angular_velocity.z"],
        ["angular velocity x", "angular velocity y", "angular velocity z"],
        f"{name}: IMU Angular Velocity",
        "Angular Velocity (rad/s)",
        f"{name}_imu_angular_velocity.png"
    )

    safe_plot(
        f"{name}_imu.csv",
        ["field.linear_acceleration.x", "field.linear_acceleration.y", "field.linear_acceleration.z"],
        ["linear acceleration x", "linear acceleration y", "linear acceleration z"],
        f"{name}: IMU Linear Acceleration",
        "Linear Acceleration (m/s²)",
        f"{name}_imu_linear_acceleration.png"
    )

def plot_cmd_vel(name):
    safe_plot(
        f"{name}_cmd_vel.csv",
        ["field.linear.x", "field.angular.z"],
        ["commanded linear x", "commanded angular z"],
        f"{name}: Commanded Velocity",
        "Commanded Velocity",
        f"{name}_cmd_vel.png"
    )

trials = [
    "circle_real",
    "square_real",
    "dance_real",
    "nav_goal_real"
]

for trial in trials:
    plot_trial(trial)

plot_cmd_vel("nav_goal_real")

print(f"Done. Plots saved in: {PLOT_DIR}")
