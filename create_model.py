import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean data
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')
data = data.dropna()

# Encode target column
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical features
cat_cols = data.select_dtypes(include=["object"]).columns
for col in cat_cols:
    if col != "customerID":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# Select features and target
X = data.drop(["customerID", "Churn"], axis=1)
y = data["Churn"]

# Scale numerical columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
pickle.dump(model, open("churn_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ churn_model.pkl and scaler.pkl created successfully!")
