import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from app.features import extract_features

CLAP_DIR = "data/clap"
NOISE_DIR = "data/noise"
MODEL_PATH = "models/clap_detector.pkl"

X = []
y = []

print("Processing CLAP files...")
for file in os.listdir(CLAP_DIR):
    if file.endswith(".wav"):
        path = os.path.join(CLAP_DIR, file)
        X.append(extract_features(path))
        y.append(1)

print("Processing NOISE files...")
for file in os.listdir(NOISE_DIR):
    if file.endswith(".wav"):
        path = os.path.join(NOISE_DIR, file)
        X.append(extract_features(path))
        y.append(0)

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("✅ Model trained successfully")
print("Model saved to:", MODEL_PATH)
