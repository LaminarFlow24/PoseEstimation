import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
model = load_model('pose_classification_model_fixed.h5')

# Load the scaler
with open('scaler.json', 'r') as f:
    scaler_data = json.load(f)
scaler = StandardScaler()
scaler.mean_ = np.array(scaler_data["mean"])
scaler.scale_ = np.array(scaler_data["scale"])

# Load the label encoder
with open('label_encoder.json', 'r') as f:
    label_mapping = json.load(f)
encoder = LabelEncoder()
encoder.classes_ = np.array(label_mapping["classes"])

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(1)

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

            # Extract keypoints
            joints = []
            for landmark in results.pose_landmarks.landmark:
                joints.extend([landmark.x, landmark.y, landmark.z])
            
            # Normalize the input using the scaler
            joints = scaler.transform([joints])

            # Make predictions
            prediction = model.predict(joints)
            class_idx = np.argmax(prediction)
            class_name = encoder.inverse_transform([class_idx])[0]

            # Display prediction on the frame
            cv2.putText(frame, f"Pose: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the output
        cv2.imshow("Posture Detection", frame)

        # Break the loop if 'q' key is pressed
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
