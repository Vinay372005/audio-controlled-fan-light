import joblib
import numpy as np
from features import extract_features

import os


# Get absolute path to models folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "clap_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


def predict_audio(file_path):
    features = extract_features(file_path)
    features = scaler.transform([features])

    prediction = model.predict(features)[0]
    return "CLAP" if prediction == 1 else "NOISE"
