import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def plot_joint_data(csv_file, output_file, title):
    """
    Plot the four OpenManipulator-X arm-joint angles from a CSV file
    extracted from the ROS /joint_states topic.
    """

    df = pd.read_csv(csv_file)

    # Normalize ROS nanosecond timestamps so the plot begins at 0 seconds.
    df["time_sec"] = (df["%time"] - df["%time"].iloc[0]) / 1e9

    # /joint_states ordering on the TurtleBot3-mounted manipulator:
    # position0 = wheel_left_joint
    # position1 = wheel_right_joint
    # position2 = joint1
    # position3 = joint2
    # position4 = joint3
    # position5 = joint4
    required_columns = [
        "field.position2",
        "field.position3",
        "field.position4",
        "field.position5",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing expected CSV columns: "
            + ", ".join(missing_columns)
            + "\nCheck the CSV header with: head -n 2 <csv_file>"
        )

    plt.figure(figsize=(10, 6))

    plt.plot(df["time_sec"], df["field.position2"],
             label="Joint 1: Base Pan", linewidth=2)

    plt.plot(df["time_sec"], df["field.position3"],
             label="Joint 2: Shoulder", linewidth=2)

    plt.plot(df["time_sec"], df["field.position4"],
             label="Joint 3: Elbow", linewidth=2)

    plt.plot(df["time_sec"], df["field.position5"],
             label="Joint 4: Wrist", linewidth=2)

    plt.title(title, fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Joint Angle (radians)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    print(f"Saved plot: {os.path.abspath(output_file)}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot OpenManipulator-X joint positions from ROS CSV data."
    )

    parser.add_argument(
        "csv_file",
        help="CSV file extracted from the /joint_states rosbag topic."
    )

    parser.add_argument(
        "output_file",
        help="Output PNG filename."
    )

    parser.add_argument(
        "--title",
        default="OpenManipulator-X Joint Positions over Time",
        help="Plot title."
    )

    args = parser.parse_args()

    plot_joint_data(
        csv_file=args.csv_file,
        output_file=args.output_file,
        title=args.title,
    )


if __name__ == "__main__":
    main()