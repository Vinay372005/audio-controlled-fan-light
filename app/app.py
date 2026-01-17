from flask import Flask, render_template, request
import os
import joblib
import numpy as np
from features import extract_features

app = Flask(__name__)

MODEL_PATH = "models/clap_detector.pkl"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["audio"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    features = extract_features(file_path)

    feature_names = [
        "zcr", "rms", "spectral_centroid",
        "spectral_bandwidth", "spectral_rolloff"
    ] + [f"mfcc_{i+1}" for i in range(10)]

    feature_dict = dict(zip(feature_names, features))

    prediction = model.predict(np.array(features).reshape(1, -1))[0]
    result = "CLAP" if prediction == 1 else "NOISE"

    return render_template(
        "index.html",
        prediction=result,
        features=feature_dict
    )

if __name__ == "__main__":
    app.run(debug=True)
