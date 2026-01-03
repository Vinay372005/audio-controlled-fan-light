# features.py
import os
import librosa
import numpy as np
import pandas as pd

# ----------------------- CONFIG -----------------------
DATA_DIR = "data"               # folder containing 'clap' and 'noise'
OUTPUT_CSV = "data/features.csv"  # CSV file to save extracted features

# -------------------- FEATURE EXTRACTION --------------------
def extract_features(file_path):
    """
    Extract 15 audio features from a given wav file:
    1. Zero Crossing Rate
    2. RMS Energy
    3. Spectral Centroid
    4. Spectral Bandwidth
    5. Spectral Roll-off
    6-15. MFCC 1 to 10
    """
    y, sr = librosa.load(file_path, duration=2.0)

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

    # 6-15 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    features.extend(np.mean(mfcc, axis=1))

    return features

# -------------------- MAIN SCRIPT --------------------
features = []
labels = []

# loop over folders: 0 -> noise, 1 -> clap
for label, folder in enumerate(["noise", "clap"]):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            features.append(extract_features(file_path))
            labels.append(label)

# create DataFrame
column_names = [
    "zero_crossing_rate", "rms", "spectral_centroid", 
    "spectral_bandwidth", "spectral_rolloff"
] + [f"mfcc_{i+1}" for i in range(10)]

df = pd.DataFrame(features, columns=column_names)
df["label"] = labels

# save to CSV
df.to_csv(OUTPUT_CSV, index=False)

print("✅ Features saved to", OUTPUT_CSV)
print("Feature shape:", df.shape)
