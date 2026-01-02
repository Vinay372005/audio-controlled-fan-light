import os
import joblib
import numpy as np
import librosa
from features import extract_features

MODEL_PATH = "models/clap_detector.pkl"

def predict(file_path):
    # Load trained model
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train.py first.")
        return

    model = joblib.load(MODEL_PATH)

    # Extract features
    features = extract_features(file_path)
    features = np.array(features).reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]
    print(f"File: {file_path} --> Prediction: {prediction}")
    return prediction


if __name__ == "__main__":
    test_file = "data/clap/clap_01.wav"  # Replace with any file
    predict(test_file)
