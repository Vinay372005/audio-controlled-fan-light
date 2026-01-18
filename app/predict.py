import joblib
import numpy as np
from features import extract_features

model = joblib.load("models/clap_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_audio(file_path):
    features = extract_features(file_path)
    features = scaler.transform([features])

    prediction = model.predict(features)[0]
    return "CLAP" if prediction == 1 else "NOISE"
