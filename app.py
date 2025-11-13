# app.py
from flask import Flask, render_template, request
import pickle, json, os
import numpy as np

app = Flask(__name__)

# Load artifacts
MODEL_PATH = "churn_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.json"

model = None
scaler = None
FEATURES = None

try:
    with open(FEATURES_PATH, "r") as f:
        FEATURES = json.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model, scaler, and feature list loaded.")
except Exception as e:
    print("❌ Error loading model/scaler/features:", e)

def build_feature_vector(form):
    """
    form: flask request.form
    returns: numpy array shape (1, len(FEATURES))
    We expect user-provided friendly fields. We map them to the 30 features list.
    """
    enc = {f: 0 for f in FEATURES}

    # Numeric fields
    enc["SeniorCitizen"] = int(form.get("SeniorCitizen", "0"))
    # fallback zeros if blank
    try:
        enc["tenure"] = float(form.get("tenure", 0) or 0)
    except:
        enc["tenure"] = 0.0
    try:
        enc["MonthlyCharges"] = float(form.get("MonthlyCharges", 0) or 0)
    except:
        enc["MonthlyCharges"] = 0.0
    try:
        enc["TotalCharges"] = float(form.get("TotalCharges", 0) or 0)
    except:
        enc["TotalCharges"] = 0.0

    # Gender
    gender = form.get("gender", "Female")
    enc["gender_Male"] = 1 if gender.lower() == "male" else 0

    # Partner / Dependents
    enc["Partner_Yes"] = 1 if form.get("Partner", "No").lower() == "yes" else 0
    enc["Dependents_Yes"] = 1 if form.get("Dependents", "No").lower() == "yes" else 0

    # PhoneService
    phone = form.get("PhoneService", "No")
    enc["PhoneService_Yes"] = 1 if phone.lower() == "yes" else 0

    # MultipleLines: choices: "No", "No phone service", "Yes"
    ml = form.get("MultipleLines", "No")
    if ml == "No phone service":
        enc["MultipleLines_No phone service"] = 1
    elif ml == "Yes":
        enc["MultipleLines_Yes"] = 1

    # InternetService: "DSL", "Fiber optic", "No"
    isvc = form.get("InternetService", "DSL")
    if isvc == "Fiber optic":
        enc["InternetService_Fiber optic"] = 1
    elif isvc == "No":
        enc["InternetService_No"] = 1

    # OnlineSecurity (Yes/No/No internet service)
    osvc = form.get("OnlineSecurity", "No")
    if osvc == "No internet service":
        enc["OnlineSecurity_No internet service"] = 1
    elif osvc == "Yes":
        enc["OnlineSecurity_Yes"] = 1

    # OnlineBackup
    ob = form.get("OnlineBackup", "No")
    if ob == "No internet service":
        enc["OnlineBackup_No internet service"] = 1
    elif ob == "Yes":
        enc["OnlineBackup_Yes"] = 1

    # DeviceProtection
    dp = form.get("DeviceProtection", "No")
    if dp == "No internet service":
        enc["DeviceProtection_No internet service"] = 1
    elif dp == "Yes":
        enc["DeviceProtection_Yes"] = 1

    # TechSupport
    ts = form.get("TechSupport", "No")
    if ts == "No internet service":
        enc["TechSupport_No internet service"] = 1
    elif ts == "Yes":
        enc["TechSupport_Yes"] = 1

    # StreamingTV
    stv = form.get("StreamingTV", "No")
    if stv == "No internet service":
        enc["StreamingTV_No internet service"] = 1
    elif stv == "Yes":
        enc["StreamingTV_Yes"] = 1

    # StreamingMovies
    sm = form.get("StreamingMovies", "No")
    if sm == "No internet service":
        enc["StreamingMovies_No internet service"] = 1
    elif sm == "Yes":
        enc["StreamingMovies_Yes"] = 1

    # Contract: options "Month-to-month", "One year", "Two year"
    contract = form.get("Contract", "Month-to-month")
    if contract == "One year":
        enc["Contract_One year"] = 1
    if contract == "Two year":
        enc["Contract_Two year"] = 1

    # PaperlessBilling
    enc["PaperlessBilling_Yes"] = 1 if form.get("PaperlessBilling", "No").lower() == "yes" else 0

    # PaymentMethod: bank transfer, credit card (automatic), electronic check, mailed check
    pm = form.get("PaymentMethod", "")
    if pm == "Credit card (automatic)":
        enc["PaymentMethod_Credit card (automatic)"] = 1
    elif pm == "Electronic check":
        enc["PaymentMethod_Electronic check"] = 1
    elif pm == "Mailed check":
        enc["PaymentMethod_Mailed check"] = 1

    # Build ordered vector
    vec = np.array([enc[f] for f in FEATURES], dtype=float).reshape(1, -1)
    return vec

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or FEATURES is None:
        return render_template("index.html", prediction="❌ Server error: model or scaler not loaded. Check logs.")
    try:
        vec = build_feature_vector(request.form)

        # Scale then predict
        scaled = scaler.transform(vec)
        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][1] * 100
        proba = round(float(proba), 2)
        if pred == 1:
            out = f"🔴 High Churn Risk: {proba}%"
        else:
            out = f"🟢 Low Churn Risk: {proba}%"
        return render_template("index.html", prediction=out)
    except Exception as e:
        return render_template("index.html", prediction=f"❌ Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)




