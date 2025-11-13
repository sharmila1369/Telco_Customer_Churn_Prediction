import pickle
import json
import numpy as np

# Load the 30 features list
with open("features.json", "r") as f:
    FEATURES = json.load(f)

# Create a dummy dataset with the correct number of features (30)
dummy_data = np.random.rand(100, len(FEATURES))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dummy_data)

# Save new scaler.pkl
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ New scaler.pkl created with", len(FEATURES), "features!")
