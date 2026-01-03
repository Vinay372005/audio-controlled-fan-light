# app/features.py
import os
import librosa
import numpy as np
import pandas as pd

# ---------- FOLDER PATHS ----------
# Current working directory: /content/audio-controlled-fan-light
DATA_DIR = "data"                # data folder in same directory
OUTPUT_CSV = "data/features.csv" # CSV will be saved here

# ---------- FEATURE EXTRACTION FUNCTION ----------
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=2.0)  # load 2 seconds

    features = []

    # 1. Zero Crossing Rate
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))

    # 2. RMS Energy
    features.append(np.mean(librosa.feature.rms(y=y)))

    # 3. Spectral Centroid
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # 4. Spectral Bandwidth
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    # 5. Spectral Roll-off
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    # 6–15. MFCCs (10 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    features.extend(np.mean(mfcc, axis=1))

    return features

# ---------- MAIN LOOP ----------
features = []
labels = []

# loop over folders
for label, folder in enumerate(["noise", "clap"]):
    folder_path = os.path.join(DATA_DIR, folder)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    else:
        print("Processing folder:", folder_path)

    # loop over files
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            features.append(extract_features(file_path))
            labels.append(label)

# ---------- SAVE FEATURES TO CSV ----------
df = pd.DataFrame(features)
df["label"] = labels
df.to_csv(OUTPUT_CSV, index=False)

print("✅ Features saved to:", OUTPUT_CSV)
print("Feature shape:", df.shape)
