import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_joint_positions(csv_file):
    """
    Plots the joint positions over time.
    Assumes CSV has columns: time, joint1, joint2, joint3, joint4
    """
    try:
        df = pd.read_csv(csv_file)
        # Normalize time to start at 0
        df['time'] = df['time'] - df['time'].iloc[0] 

        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['joint1'], label='Joint 1 (Pan)')
        plt.plot(df['time'], df['joint2'], label='Joint 2 (Tilt 1)')
        plt.plot(df['time'], df['joint3'], label='Joint 3 (Tilt 2)')
        plt.plot(df['time'], df['joint4'], label='Joint 4 (Tilt 3)')

        plt.title('OpenManipulator-X: Joint Positions Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('joint_positions.png', dpi=300)
        plt.show()
        print("[SUCCESS] Joint positions plotted and saved.")
    except Exception as e:
        print(f"[ERROR] Could not plot joint positions: {e}")

def plot_end_effector_3d(csv_file):
    """
    Plots the 3D path of the end-effector.
    Assumes CSV has columns: x, y, z
    """
    try:
        df = pd.read_csv(csv_file)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(df['x'], df['y'], df['z'], label='End-Effector Path', color='b', linewidth=2)
        ax.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], color='g', s=100, label='Start')
        ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], color='r', s=100, label='Goal')

        ax.set_title('Task-Space Trajectory: 3D Path')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.legend()
        plt.savefig('end_effector_3d_path.png', dpi=300)
        plt.show()
        print("[SUCCESS] 3D End-effector path plotted and saved.")
    except Exception as e:
        print(f"[ERROR] Could not plot 3D path: {e}")

def plot_tracking_error(actual_csv, desired_csv):
    """
    Calculates and plots the tracking error between desired and actual trajectories.
    Assumes both CSVs have columns: time, x, y, z
    """
    try:
        df_act = pd.read_csv(actual_csv)
        df_des = pd.read_csv(desired_csv)

        # Ensure both dataframes align in time for direct subtraction 
        # (In a real scenario, you may need to interpolate if time steps differ)
        min_len = min(len(df_act), len(df_des))
        df_act = df_act.iloc[:min_len]
        df_des = df_des.iloc[:min_len]

        df_act['time'] = df_act['time'] - df_act['time'].iloc[0]

        # Calculate Euclidean error
        error_x = df_des['x'] - df_act['x']
        error_y = df_des['y'] - df_act['y']
        error_z = df_des['z'] - df_act['z']
        total_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)

        plt.figure(figsize=(10, 6))
        plt.plot(df_act['time'], total_error, label='Total Euclidean Error', color='r')
        
        plt.title('Controller Performance: Trajectory Tracking Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error (m)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('tracking_error.png', dpi=300)
        plt.show()
        print("[SUCCESS] Tracking error plotted and saved.")
    except Exception as e:
        print(f"[ERROR] Could not plot tracking error: {e}")

if __name__ == "__main__":
    # Update these filenames to match what Julian sends you
    # plot_joint_positions('sample_joint_data.csv')
    # plot_end_effector_3d('sample_ee_data.csv')
    # plot_tracking_error('actual_ee_data.csv', 'desired_ee_data.csv')
    print("Pipeline ready. Waiting for CSV data...")
