import joblib
import pandas as pd

# Load model
model = joblib.load("student_model.pkl")

print("🎓 AI Student Predictor")

# User input
hours = float(input("Hours studied: "))
sleep = float(input("Sleep hours: "))
phone = float(input("Phone usage: "))

# Prepare input
new_data = pd.DataFrame(
    [[hours, sleep, phone]],
    columns=["hours_studied", "sleep_hours", "phone_usage"]
)

# Predict
prediction = model.predict(new_data)

print("\nPrediction:", prediction[0])

if prediction[0] == "Pass":
    print("✅ You are likely to PASS")
else:
    print("❌ You may FAIL. Study harder!")