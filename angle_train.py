import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # For saving the trained model

# Load the dataset
csv_file = "pose_data_1735015919.csv"  # Replace with the correct filename
data = pd.read_csv(csv_file)

# Separate features and labels
X = data.iloc[:, :-1]  # All columns except the last one (angles)
y = data.iloc[:, -1]   # The last column (class labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
print("Training the model...")
clf.fit(X_train, y_train)

# Test the model
print("Testing the model...")
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
model_filename = "pose_classifier.pkl"
joblib.dump(clf, model_filename)
print(f"Model saved as {model_filename}")
