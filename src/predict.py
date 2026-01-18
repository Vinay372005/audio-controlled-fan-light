import os
import joblib
from features import extract_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "clap_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_audio(file_path):
    feat = extract_features(file_path)
    feat_scaled = scaler.transform([feat])
    prediction = model.predict(feat_scaled)[0]
    return prediction
