from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("student_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    hours = float(request.form["hours"])
    sleep = float(request.form["sleep"])
    phone = float(request.form["phone"])

    new_data = pd.DataFrame(
        [[hours, sleep, phone]],
        columns=["hours_studied", "sleep_hours", "phone_usage"]
    )

    prediction = model.predict(new_data)[0]

    if prediction == "Pass":
        result = "✅ Viwe mfoka Phakathi. You are likely to PASS 🎉 But you've just been hacked!"
    else:
        result = "❌ You may FAIL 😬 Reduce phone usage & study more! Im sorry but uhekhiwe, next time dont enter any links.Thanks nge 2 cent engiyithole ku capitec wakho"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
