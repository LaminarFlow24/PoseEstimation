import pandas as pd
import numpy as np

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle formed by three points (in 3D space).
    p1, p2, p3 are tuples/lists of coordinates (x, y, z).
    """
    v1 = np.array(p1) - np.array(p2)  # Vector from p2 to p1
    v2 = np.array(p3) - np.array(p2)  # Vector from p2 to p3

    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm == 0 or v2_norm == 0:
        return 0  # Return 0 angle if one of the vectors is zero-length

    v1 /= v1_norm
    v2 /= v2_norm

    # Calculate dot product and angle
    dot_product = np.dot(v1, v2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to avoid numerical errors
    angle_deg = np.degrees(angle_rad)  # Convert radians to degrees

    return angle_deg

# Load the original CSV file
input_csv = "pose_data_1733800173.csv"  # Replace with your filename
df = pd.read_csv(input_csv)

# Define all possible joints based on MediaPipe Pose landmarks
landmarks = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer", 7: "left_ear",
    8: "right_ear", 9: "mouth_left", 10: "mouth_right", 11: "left_shoulder",
    12: "right_shoulder", 13: "left_elbow", 14: "right_elbow", 15: "left_wrist",
    16: "right_wrist", 17: "left_pinky", 18: "right_pinky", 19: "left_index",
    20: "right_index", 21: "left_thumb", 22: "right_thumb", 23: "left_hip",
    24: "right_hip", 25: "left_knee", 26: "right_knee", 27: "left_ankle",
    28: "right_ankle", 29: "left_heel", 30: "right_heel", 31: "left_foot_index",
    32: "right_foot_index"
}

# Define connections (triplets) for meaningful angles
connections = [
    (11, 13, 15),  # Left shoulder, elbow, wrist
    (12, 14, 16),  # Right shoulder, elbow, wrist
    (23, 25, 27),  # Left hip, knee, ankle
    (24, 26, 28),  # Right hip, knee, ankle
    (11, 23, 25),  # Left shoulder, hip, knee
    (12, 24, 26),  # Right shoulder, hip, knee
    (13, 11, 12),  # Left elbow, left shoulder, right shoulder
    (14, 12, 11),  # Right elbow, right shoulder, left shoulder
    (23, 11, 12),  # Left hip, left shoulder, right shoulder
    (24, 12, 11),  # Right hip, right shoulder, left shoulder
    (25, 23, 24),  # Left knee, left hip, right hip
    (26, 24, 23),  # Right knee, right hip, left hip
]

# Prepare a new DataFrame for storing angles
angles_data = []

for _, row in df.iterrows():
    angles_row = {}
    for connection in connections:
        j1, j2, j3 = connection

        # Construct column names dynamically based on joint indices
        col_x1, col_y1, col_z1 = f"x_{j1}", f"y_{j1}", f"z_{j1}"
        col_x2, col_y2, col_z2 = f"x_{j2}", f"y_{j2}", f"z_{j2}"
        col_x3, col_y3, col_z3 = f"x_{j3}", f"y_{j3}", f"z_{j3}"

        try:
            # Extract coordinates for the three joints
            p1 = (row[col_x1], row[col_y1], row[col_z1])
            p2 = (row[col_x2], row[col_y2], row[col_z2])
            p3 = (row[col_x3], row[col_y3], row[col_z3])

            # Calculate the angle
            angle = calculate_angle(p1, p2, p3)
            joint_name = f"{landmarks[j1]}_{landmarks[j2]}_{landmarks[j3]}"
            angles_row[joint_name] = angle

        except KeyError as e:
            print(f"Missing data for {e} in row {row.name}, skipping...")
            continue

    # Add class label (last column) to angles
    angles_row["class"] = row["class"]
    angles_data.append(angles_row)

# Create a new DataFrame for the angles
angles_df = pd.DataFrame(angles_data)

# Save the angles DataFrame to a new CSV file
output_csv = "pose_angles_all.csv"
angles_df.to_csv(output_csv, index=False)
print(f"Angles saved to {output_csv}")
