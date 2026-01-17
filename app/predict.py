import sys
import numpy as np
import joblib
from app.features import extract_features

MODEL_PATH = "models/clap_detector.pkl"

def predict(audio_file):
    model = joblib.load(MODEL_PATH)
    features = extract_features(audio_file)
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]
    return "CLAP" if prediction == 1 else "NOISE"


if __name__ == "__main__":
    audio_path = sys.argv[1]
    result = predict(audio_path)
    print("Prediction:", result)
