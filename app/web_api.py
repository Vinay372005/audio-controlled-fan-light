import os
from flask import Flask, request, jsonify
from predict import predict

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "Clap Detection API is running"

@app.route("/predict", methods=["POST"])
def predict_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not file.filename.endswith(".wav"):
        return jsonify({"error": "Only .wav files supported"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    result = predict(file_path)

    label_map = {0: "Noise", 1: "Clap"}

    return jsonify({
        "filename": file.filename,
        "prediction": label_map[result]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
