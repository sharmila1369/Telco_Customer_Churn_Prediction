import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Your Exact 30 Features
FEATURES = [
  "SeniorCitizen",
  "tenure",
  "MonthlyCharges",
  "TotalCharges",
  "gender_Male",
  "Partner_Yes",
  "Dependents_Yes",
  "PhoneService_Yes",
  "MultipleLines_No phone service",
  "MultipleLines_Yes",
  "InternetService_Fiber optic",
  "InternetService_No",
  "OnlineSecurity_No internet service",
  "OnlineSecurity_Yes",
  "OnlineBackup_No internet service",
  "OnlineBackup_Yes",
  "DeviceProtection_No internet service",
  "DeviceProtection_Yes",
  "TechSupport_No internet service",
  "TechSupport_Yes",
  "StreamingTV_No internet service",
  "StreamingTV_Yes",
  "StreamingMovies_No internet service",
  "StreamingMovies_Yes",
  "Contract_One year",
  "Contract_Two year",
  "PaperlessBilling_Yes",
  "PaymentMethod_Credit card (automatic)",
  "PaymentMethod_Electronic check",
  "PaymentMethod_Mailed check"
]


def encode_row(row):
    enc = {f: 0 for f in FEATURES}

    # Numeric
    enc["SeniorCitizen"] = int(row["SeniorCitizen"])
    enc["tenure"] = float(row["tenure"])
    try:
        enc["TotalCharges"] = float(row["TotalCharges"])
    except:
        enc["TotalCharges"] = 0
    enc["MonthlyCharges"] = float(row["MonthlyCharges"])

    # Gender
    enc["gender_Male"] = 1 if row["gender"] == "Male" else 0

    # Partner / Dependents
    enc["Partner_Yes"] = 1 if row["Partner"] == "Yes" else 0
    enc["Dependents_Yes"] = 1 if row["Dependents"] == "Yes" else 0

    # Phone Service
    enc["PhoneService_Yes"] = 1 if row["PhoneService"] == "Yes" else 0

    # Multiple Lines
    if row["MultipleLines"] == "No phone service":
        enc["MultipleLines_No phone service"] = 1
    elif row["MultipleLines"] == "Yes":
        enc["MultipleLines_Yes"] = 1

    # Internet Service
    if row["InternetService"] == "Fiber optic":
        enc["InternetService_Fiber optic"] = 1
    elif row["InternetService"] == "No":
        enc["InternetService_No"] = 1

    # Online Security
    if row["OnlineSecurity"] == "No internet service":
        enc["OnlineSecurity_No internet service"] = 1
    elif row["OnlineSecurity"] == "Yes":
        enc["OnlineSecurity_Yes"] = 1

    # Online Backup
    if row["OnlineBackup"] == "No internet service":
        enc["OnlineBackup_No internet service"] = 1
    elif row["OnlineBackup"] == "Yes":
        enc["OnlineBackup_Yes"] = 1

    # Device Protection
    if row["DeviceProtection"] == "No internet service":
        enc["DeviceProtection_No internet service"] = 1
    elif row["DeviceProtection"] == "Yes":
        enc["DeviceProtection_Yes"] = 1

    # Tech Support
    if row["TechSupport"] == "No internet service":
        enc["TechSupport_No internet service"] = 1
    elif row["TechSupport"] == "Yes":
        enc["TechSupport_Yes"] = 1

    # Streaming TV
    if row["StreamingTV"] == "No internet service":
        enc["StreamingTV_No internet service"] = 1
    elif row["StreamingTV"] == "Yes":
        enc["StreamingTV_Yes"] = 1

    # Streaming Movies
    if row["StreamingMovies"] == "No internet service":
        enc["StreamingMovies_No internet service"] = 1
    elif row["StreamingMovies"] == "Yes":
        enc["StreamingMovies_Yes"] = 1

    # Contract
    if row["Contract"] == "One year":
        enc["Contract_One year"] = 1
    if row["Contract"] == "Two year":
        enc["Contract_Two year"] = 1

    # Paperless Billing
    enc["PaperlessBilling_Yes"] = 1 if row["PaperlessBilling"] == "Yes" else 0

    # Payment Method
    pm = row["PaymentMethod"]
    if pm == "Credit card (automatic)":
        enc["PaymentMethod_Credit card (automatic)"] = 1
    elif pm == "Electronic check":
        enc["PaymentMethod_Electronic check"] = 1
    elif pm == "Mailed check":
        enc["PaymentMethod_Mailed check"] = 1

    return [enc[f] for f in FEATURES]


print("Loading CSV...")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Encoding dataset...")
X = []
for i, row in df.iterrows():
    X.append(encode_row(row))
X = np.array(X)

y = (df["Churn"] == "Yes").astype(int)

print("Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training model...")
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_scaled, y)

print("Saving new model files...")
pickle.dump(model, open("churn_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
json.dump(FEATURES, open("features.json", "w"), indent=2)

print("🎉 DONE! New model, scaler, and features.json created (30 features).")
