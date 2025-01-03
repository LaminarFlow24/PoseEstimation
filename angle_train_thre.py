import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

# Load the CSV data
csv_file = "kneeraise.csv"  # Replace with your generated CSV file
data = pd.read_csv(csv_file)

# Separate features and labels
X = data.iloc[:, :-1]  # All columns except the last (angles)
y = data.iloc[:, -1]   # The last column (class labels)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Compute mean angles for each class
mean_angles = X.groupby(y).mean()
mean_angles.index = le.inverse_transform(mean_angles.index)  # Map back to string labels

# Save mean angles to a file for inference
mean_angles.to_csv("kneeraise_mean_angles.csv", index=True)
print("Mean angles saved for unknown pose detection.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# Evaluate the model
y_pred = clf.predict(X_test)
target_names = [str(label) for label in le.classes_]
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))



# Save the model and label encoder
joblib.dump(clf, "kneeraise_model.pkl")
joblib.dump(le, "kneeraise_encoder.pkl")
print("Model and Label Encoder saved!")
