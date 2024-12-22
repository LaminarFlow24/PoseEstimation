import cv2
import numpy as np
import tensorflow as tf
from mediapipe import solutions as mp_solutions
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('pose_classification_model_fixed.h5')

# Set up the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Pose module
mp_pose = mp_solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp_solutions.drawing_utils

# Initialize the StandardScaler used for normalization
scaler = StandardScaler()

# Function to extract pose landmarks and flatten them into a 1D array
def extract_landmarks(frame):
    landmarks = []
    results = pose.process(frame)
    
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])  # 3D coordinates (x, y, z)
    
    return np.array(landmarks).flatten() if landmarks else None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (required by MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Extract pose landmarks from the frame
    landmarks = extract_landmarks(frame_rgb)

    if landmarks is not None:
        # Normalize the landmarks using the same scaler used during training
        landmarks = scaler.fit_transform([landmarks])[0]  # Single instance, reshape to 2D array for fit_transform

        # Predict the pose class using the trained model
        pose_pred = np.argmax(model.predict(np.expand_dims(landmarks, axis=0)))

        # Display the pose number on the frame
        cv2.putText(frame, f'Pose: {pose_pred}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw the pose landmarks on the frame
    if landmarks is not None:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the frame with pose classification and landmarks
    cv2.imshow('Pose Classification', frame)

    # Exit the webcam feed on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
