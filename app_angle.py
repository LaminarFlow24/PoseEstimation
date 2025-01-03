import cv2
import mediapipe as mp
import pandas as pd
import time
import math

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize capture mode and CSV data storage
capture_mode = False
recording_class = None
data = []

# Helper function to calculate angle between three points
def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points.
    Args:
    - p1, p2, p3: Mediapipe landmark objects with x, y, z attributes.

    Returns:
    - Angle in degrees between the vectors (p1 -> p2) and (p3 -> p2).
    """
    v1 = [p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]
    v2 = [p3.x - p2.x, p3.y - p2.y, p3.z - p2.z]

    # Calculate dot product and magnitudes
    dot_product = sum(v1[i] * v2[i] for i in range(3))
    magnitude_v1 = math.sqrt(sum(v1[i] ** 2 for i in range(3)))
    magnitude_v2 = math.sqrt(sum(v2[i] ** 2 for i in range(3)))

    # Avoid division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0

    # Calculate angle
    angle = math.acos(dot_product / (magnitude_v1 * magnitude_v2))
    return math.degrees(angle)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Initialize Pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to get pose landmarks
        results = pose.process(rgb_frame)

        # Draw the skeleton and landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        # Check for key presses to control capture mode and recording
        key = cv2.waitKey(10) & 0xFF

        if key == ord('k'):
            capture_mode = True
            print("Capture mode enabled.")
        elif key == ord('l'):
            capture_mode = False
            recording_class = None
            print("Capture mode disabled.")
            # Save recorded data to a CSV file
            if data:
                timestamp = int(time.time())
                df = pd.DataFrame(data)
                df.to_csv(f"pose_data_{timestamp}.csv", index=False)
                print(f"Data saved to pose_data_{timestamp}.csv")
                data = []  # Clear data after saving

        # Check if capture mode is active and a numeric key is held down
        if capture_mode:
            for i in range(1, 10):
                if key == ord(str(i)):
                    recording_class = i  # Set the class for recording
                    if results.pose_landmarks:
                        # Calculate angles for all defined landmark triplets
                        angles = []
                        # Define landmark triplets for angle calculation
                        landmark_indices = [
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
                            (26, 24, 23)   # Right knee, right hip, left hip
                        ]
                        for idx1, idx2, idx3 in landmark_indices:
                            angle = calculate_angle(
                                results.pose_landmarks.landmark[idx1],
                                results.pose_landmarks.landmark[idx2],
                                results.pose_landmarks.landmark[idx3]
                            )
                            angles.append(angle)

                        angles.append(recording_class)  # Append class label
                        data.append(angles)  # Append to data list
                    break  # Only one class per frame

        # Display the output
        cv2.imshow("Body Detection with Skeleton", frame)

        # Break the loop if 'q' key is pressed
        if key == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
