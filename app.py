from flask import Flask, render_template, request
import joblib
import json
import numpy as np

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Load model, scaler, and features
try:
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")

    with open("features.json", "r") as f:
        features = json.load(f)

    print("✅ Model, Scaler, and Features loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model/scaler/features: {e}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔹 Get all form input values from user
        input_data = [float(x) for x in request.form.values()]
        final_input = np.array(input_data).reshape(1, -1)

        # 🔹 Scale the input data
        final_scaled = scaler.transform(final_input)

        # 🔹 Predict churn probability
        prob = model.predict_proba(final_scaled)[0][1] * 100

        # 🔹 Generate readable result
        if prob > 60:
            result = f"🔴 High Churn Risk ({prob:.2f}%) — Customer is likely to leave soon!"
        elif prob > 30:
            result = f"🟠 Moderate Churn Risk ({prob:.2f}%) — Keep engaging the customer."
        else:
            result = f"🟢 Low Churn Risk ({prob:.2f}%) — Customer is likely to stay."

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"❌ Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)


