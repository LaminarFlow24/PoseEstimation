import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the Data
data = pd.read_csv('pose_data_1733800173.csv')  # Replace with your file path

# Step 2: Separate Features and Labels
X = data.iloc[:, :-1].values  # All columns except the last one
y = data.iloc[:, -1].values   # The last column (pose classification)

# Step 3: Normalize Labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)  # Maps labels to 0, 1, 2, ..., num_classes-1

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Step 5: Normalize the Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Step 6: Build the Model
num_classes = len(np.unique(y))
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')  # Ensure correct number of output units
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 7: Define Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Step 8: Train the Model
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping, lr_scheduler])

# Step 9: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Step 10: Detailed Evaluation
y_pred = np.argmax(model.predict(X_test), axis=-1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 11: Save the Model
model.save('pose_classification_model_fixed.h5')
print("Model saved as 'pose_classification_model_fixed.h5'")

import json
from sklearn.preprocessing import StandardScaler

# Save scaler
scaler_data = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
with open('scaler.json', 'w') as f:
    json.dump(scaler_data, f)

# Save label encoder
label_mapping = {"classes": encoder.classes_.tolist()}
with open('label_encoder.json', 'w') as f:
    json.dump(label_mapping, f)


