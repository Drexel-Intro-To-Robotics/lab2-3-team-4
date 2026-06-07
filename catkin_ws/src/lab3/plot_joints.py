import pandas as pd
import matplotlib.pyplot as plt

def plot_joint_data(csv_file):
    # Load the CSV generated from the rosbag
    df = pd.read_csv(csv_file)
    
    # Normalize the time column so it starts at 0 seconds
    # (The '%time' column is standard from the rostopic echo -p command)
    df['time_sec'] = (df['%time'] - df['%time'].iloc[0]) / 1e9
    
    # Initialize the plot
    plt.figure(figsize=(10, 6))
    
    # Plot each joint. Note: field names might vary slightly depending on your exact ROS setup,
    # but they typically export as field.position0, field.position1, etc.
    plt.plot(df['time_sec'], df['field.position0'], label='Joint 1 (Pan)', linewidth=2)
    plt.plot(df['time_sec'], df['field.position1'], label='Joint 2 (Tilt 1)', linewidth=2)
    plt.plot(df['time_sec'], df['field.position2'], label='Joint 3 (Tilt 2)', linewidth=2)
    plt.plot(df['time_sec'], df['field.position3'], label='Joint 4 (Tilt 3)', linewidth=2)
    
    # Formatting the plot
    plt.title('OpenManipulator-X Joint Positions over Time (Target 2)', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Joint Angle (radians)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Save the figure cleanly for your report
    plt.tight_layout()
    plt.savefig('target2_joint_plot.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    # Test the script with the CSV you just extracted
    plot_joint_data('target2_data.csv')
