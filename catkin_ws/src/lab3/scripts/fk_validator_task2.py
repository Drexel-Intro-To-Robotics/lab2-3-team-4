from fk_validator import solve_fk


def main():
    """
    Validate the three IK solutions by running each result through FK.
    """

    test_cases = [
        {
            "name": "Point 1 (+Y Max Safe)",
            "target": [0.0, 0.25, 0.15, 0.0],
            "joints": [1.5708, 1.4318, -2.0838, 0.6520],
        },
        {
            "name": "Point 2 (Close Up)",
            "target": [0.0, 0.20, 0.20, 0.0],
            "joints": [1.5708, 2.0210, -2.1303, 0.1093],
        },
        {
            "name": "Point 3 (Diagonal)",
            "target": [-0.10, 0.10, 0.20, 0.0],
            "joints": [2.3562, 2.5891, -2.2998, -0.2892],
        },
    ]

    print("--- FK Validation for Task 2 ---")

    for case in test_cases:
        target = case["target"]
        joints = case["joints"]

        result = solve_fk(
            joints[0],
            joints[1],
            joints[2],
            joints[3]
        )

        x_fk = result[0]
        y_fk = result[1]
        z_fk = result[2]
        pitch_fk = result[3]

        x_error = x_fk - target[0]
        y_error = y_fk - target[1]
        z_error = z_fk - target[2]

        position_error = (
            x_error ** 2 +
            y_error ** 2 +
            z_error ** 2
        ) ** 0.5

        print("")
        print(case["name"])
        print("Target Pose [x, y, z, pitch]:")
        print(target)

        print("Input Joints [th1, th2, th3, th4]:")
        print(joints)

        print("FK Result [x, y, z, pitch]:")
        print([
            round(x_fk, 4),
            round(y_fk, 4),
            round(z_fk, 4),
            round(pitch_fk, 4)
        ])

        print("Position Error [m]:")
        print(round(position_error, 6))


if __name__ == "__main__":
    main()