import matplotlib.pyplot as plt

# Data from your report
planners = ['KPIECE', 'PRM', 'RRTConnect']
times = [0.0228, 0.0176, 0.0172]

# Create the bar plot
plt.figure(figsize=(8, 6))
plt.bar(planners, times, color=['skyblue', 'salmon', 'lightgreen'])

# Add labels
plt.xlabel('Motion Planner')
plt.ylabel('Planning Time (s)')
plt.title('Comparison of Motion Planner Execution Times')

# Save the plot
plt.savefig('planning_time_comparison.png')
print("Plot saved as planning_time_comparison.png")
