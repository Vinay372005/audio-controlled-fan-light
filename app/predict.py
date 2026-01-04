import os
import joblib
import numpy as np

# Import feature extractor from project root
from features import extract_features

class ClapPredictor:
    def __init__(self):
        # Lock project root
        self.base_dir = os.getcwd()

        self.model_path = os.path.join(
            self.base_dir, "models", "clap_detector.pkl"
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "❌ Model not found. Please run train.py first."
            )

        self.model = joblib.load(self.model_path)
        print("✅ Model loaded successfully")

    def predict(self, file_path):
        # Extract features
        features = extract_features(file_path)
        features = np.array(features).reshape(1, -1)

        # Predict
        prediction = self.model.predict(features)[0]
        print(f"File: {os.path.basename(file_path)} → Prediction: {prediction}")

        return prediction


# FSM-compatible function
_predictor = None

def predict(file_path):
    global _predictor
    if _predictor is None:
        _predictor = ClapPredictor()
    return _predictor.predict(file_path)


if __name__ == "__main__":
    test_file = "data/clap/clap_01.wav"
    predict(test_file)
