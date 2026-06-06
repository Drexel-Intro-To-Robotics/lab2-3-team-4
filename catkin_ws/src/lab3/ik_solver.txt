import numpy as np

# OpenManipulator-X effective link lengths (meters)
L1 = 0.0963  # Base to Joint 2
L2 = 0.1302  # Joint 2 to Joint 3
L3 = 0.1240  # Joint 3 to Joint 4
L4 = 0.1334  # Joint 4 to End Effector

def solve_ik(x, y, z, pitch):
    """
    Solves inverse kinematics for OpenManipulator-X.
    Inputs in meters and radians.
    """
    # 1. Base joint (Pan)
    theta1 = np.arctan2(y, x)
    
    # 2. Planar projection
    r = np.sqrt(x**2 + y**2)
    
    # 3. Wrist center position
    r_w = r - L4 * np.cos(pitch)
    z_w = z - L1 - L4 * np.sin(pitch)
    
    # 4. Law of Cosines for theta3
    D = (r_w**2 + z_w**2 - L2**2 - L3**2) / (2 * L2 * L3)
    
    # Check reachability to prevent math domain errors
    if abs(D) > 1.0:
        print(f"Bruh. Pose [{x}, {y}, {z}] is out of reach.")
        return None
        
    # Elbow up configuration
    theta3 = np.arctan2(-np.sqrt(1 - D**2), D)
    
    # 5. Solve for theta2
    beta = np.arctan2(z_w, r_w)
    psi = np.arctan2(L3 * np.sin(theta3), L2 + L3 * np.cos(theta3))
    theta2 = beta - psi
    
    # 6. Solve for theta4 (maintaining desired pitch)
    theta4 = pitch - (theta2 + theta3)
    
    return [theta1, theta2, theta3, theta4]

# --- TEST EXECUTIONS ---
if __name__ == "__main__":
    # Test cases mapped from Julian's notes (converted to meters, pitch set to 0.0 for horizontal gripper)
    targets = [
        ("Point 1 (+Y Max Safe)", [0.0, 0.25, 0.15, 0.0]),
        ("Point 2 (Close up)", [0.0, 0.20, 0.20, 0.0]),
        ("Point 3 (Diagonal)", [-0.10, 0.10, 0.20, 0.0])
    ]
    
    print("--- IK Solver Test Outputs ---")
    for name, pose in targets:
        joints = solve_ik(*pose)
        if joints:
            print(f"{name}:")
            print(f"  Target [x, y, z, pitch] = {pose}")
            print(f"  Joints [th1, th2, th3, th4] = {[round(j, 4) for j in joints]}\n")
