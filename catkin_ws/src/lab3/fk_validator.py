import numpy as np

def dh_matrix(theta, d, a, alpha):
    """Generates standard DH transformation matrix"""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,             np.sin(alpha),                np.cos(alpha),               d],
        [0,             0,                            0,                           1]
    ])

def solve_fk(theta1, theta2, theta3, theta4):
    """
    Validates Forward Kinematics for OpenManipulator-X
    """
    L1, L2, L3, L4 = 0.0963, 0.1302, 0.1240, 0.1334
    
    # Frame transformations based on planar arm structure
    T01 = dh_matrix(theta1, L1, 0, np.pi/2)
    T12 = dh_matrix(theta2, 0, L2, 0)
    T23 = dh_matrix(theta3, 0, L3, 0)
    T34 = dh_matrix(theta4, 0, L4, 0)
    
    # End-effector pose relative to base
    T04 = T01 @ T12 @ T23 @ T34
    
    x = round(T04[0, 3], 4)
    y = round(T04[1, 3], 4)
    z = round(T04[2, 3], 4)
    
    # Pitch is the sum of the planar joints
    pitch = round(theta2 + theta3 + theta4, 4)
    
    return [x, y, z, pitch]

if __name__ == "__main__":
    # Test values from running IK on Point 2 [0.0, 0.20, 0.20, 0.0]
    test_joints = [1.5708, -0.6698, -0.2185, 0.8883] 
    
    print("--- FK Validation Check ---")
    print(f"Input Joints: {test_joints}")
    result = solve_fk(*test_joints)
    print(f"Resulting FK Pose [x, y, z, pitch]: {result}")
    print("Verification: If [0.0, 0.2, 0.2, 0.0] -> Math is rock solid.")
