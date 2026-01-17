import os
import joblib
import numpy as np
from app.features import extract_features

MODEL_PATH = "models/clap_detector.pkl"

def predict(file_path):
    model = joblib.load(MODEL_PATH)

    features = extract_features(file_path)
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]
    return prediction

if __name__ == "__main__":
    test_file = "data/clap/clap_01.wav"
    result = predict(test_file)
    print("Prediction:", result)
