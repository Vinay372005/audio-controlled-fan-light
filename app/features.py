import os
import librosa
import numpy as np
import pandas as pd

# Correct paths for your current directory structure
DATA_DIR = "data"  # 'data' folder is in the same folder as features.py
OUTPUT_CSV = "data/features.csv"

def extract_features(file_path):
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

    # 6–15 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    features.extend(np.mean(mfcc, axis=1))

    return features

features = []
labels = []

# Loop over your folders: 'noise' and 'clap'
for label, folder in enumerate(["noise", "clap"]):
    folder_path = os.path.join(DATA_DIR, folder)
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            features.append(extract_features(file_path))
            labels.append(label)

# Save features and labels to CSV
df = pd.DataFrame(features)
df["label"] = labels
df.to_csv(OUTPUT_CSV, index=False)

print("Features saved to", OUTPUT_CSV)
print("Feature shape:", df.shape)
