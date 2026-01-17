import sys
import joblib
import librosa
import numpy as np
from pathlib import Path

model_path = Path('../models/clap_detector.pkl')
if not model_path.exists():
    print("Error: Run train.py first!")
    sys.exit(1)

model = joblib.load(model_path)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=2.0)
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
    rms = librosa.feature.rms(y=y)[0].mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10).mean(axis=1)
    return np.concatenate([[zcr, rms, centroid, bandwidth, rolloff], mfcc]).reshape(1, -1)

if len(sys.argv) > 1:
    test_file = Path(sys.argv[1])
    pred = model.predict(extract_features(test_file))[0]
    print("CLAP" if pred == 1 else "NOISE", f"({test_file.name})")
else:
    print("Usage: python -m app.predict test.wav") [web:6]
