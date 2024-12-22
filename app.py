import cv2
import mediapipe as mp
import pandas as pd
import time

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize capture mode and CSV data storage
capture_mode = False
recording_class = None
data = []

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
                        # Record coordinates for all landmarks
                        joints = []
                        for landmark in results.pose_landmarks.landmark:
                            joints.extend([landmark.x, landmark.y, landmark.z])
                        joints.append(recording_class)  # Append class label
                        data.append(joints)  # Append to data list
                    break  # Only one class per frame

        # Display the output
        cv2.imshow("Body Detection with Skeleton", frame)

        # Break the loop if 'q' key is pressed
        if key == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
