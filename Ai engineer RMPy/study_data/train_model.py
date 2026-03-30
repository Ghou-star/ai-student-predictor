import pandas as pd
import joblib

# Load dataset
data = pd.read_csv("study_data.csv")

# Inputs and output
X = data[["hours_studied", "sleep_hours", "phone_usage"]]
y = data["result"]

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(X, y)

# Save model
joblib.dump(model, "student_model.pkl")

print("✅ Model trained and saved!")