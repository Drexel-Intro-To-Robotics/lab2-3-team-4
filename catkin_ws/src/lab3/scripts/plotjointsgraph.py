import argparse
import os

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt


def plot_joint_data(csv_file, output_file, title):
    """
    Plot the four OpenManipulator-X arm-joint positions from a CSV file
    extracted from the ROS /joint_states topic.
    """

    # Load the CSV file extracted from the rosbag.
    df = pd.read_csv(csv_file)

    # Convert ROS timestamps from nanoseconds and begin the plot at t = 0 s.
    df["time_sec"] = (df["%time"] - df["%time"].iloc[0]) / 1e9

    # Convert the pandas Series into a NumPy array.
    # This avoids a compatibility issue between pandas 2.x and the
    # older Matplotlib version installed in the Dev Container.
    time_values = df["time_sec"].to_numpy()

    # TurtleBot3-mounted OpenManipulator /joint_states ordering:
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

    # Stop with a useful message if the CSV does not match the expected format.
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing expected CSV columns: "
            + ", ".join(missing_columns)
            + "\nInspect the header with: head -n 2 <csv_file>"
        )

    # Create the figure.
    plt.figure(figsize=(10, 6))

    # Plot each OpenManipulator arm joint.
    plt.plot(
        time_values,
        df["field.position2"].to_numpy(),
        label="Joint 1: Base Pan",
        linewidth=2,
    )

    plt.plot(
        time_values,
        df["field.position3"].to_numpy(),
        label="Joint 2: Shoulder",
        linewidth=2,
    )

    plt.plot(
        time_values,
        df["field.position4"].to_numpy(),
        label="Joint 3: Elbow",
        linewidth=2,
    )

    plt.plot(
        time_values,
        df["field.position5"].to_numpy(),
        label="Joint 4: Wrist",
        linewidth=2,
    )

    # Format the figure for the lab report.
    plt.title(title, fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Joint Angle (radians)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")
    plt.tight_layout()

    # Save the figure without opening a GUI window.
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved plot: {os.path.abspath(output_file)}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot OpenManipulator-X joint positions from ROS CSV data."
    )

    parser.add_argument(
        "csv_file",
        help="CSV file extracted from /joint_states."
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