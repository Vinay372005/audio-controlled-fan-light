# app/features.py

"""
Features Extraction Script for Audio-Controlled Fan/Light Project
----------------------------------------------------------------

This script extracts audio features from the 'clap' and 'noise' datasets
and saves them to a CSV file for training/prediction.

Features Extracted:
1. Zero Crossing Rate
2. RMS Energy
3. Spectral Centroid
4. Spectral Bandwidth
5. Spectral Roll-off
6-15. MFCCs (10 coefficients)

Outputs:
- data/features.csv       : CSV file with features + label
- plots/features.png      : Distribution plot of first few features

Usage:
!python app/features.py
"""

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- PATHS -----------------------------
DATA_DIR = "data"                     # folder containing 'clap' and 'noise'
OUTPUT_CSV = "data/features.csv"      # where CSV will be saved
PLOT_FILE = "plots/features.png"      # plot of features

# create plots folder if it doesn't exist
os.makedirs(os.path.dirname(PLOT_FILE), exist_ok=True)

# -------------------------- FEATURE EXTRACTION --------------------------
def extract_features(file_path):
    """
    Extracts 15 audio features + 10 MFCCs from a single audio file.
    Returns a list of features.
    """
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

    # 6-15. MFCCs (10 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    features.extend(np.mean(mfcc, axis=1))

    return features

# -------------------------- PROCESS DATA --------------------------
features = []
labels = []

for label, folder in enumerate(["noise", "clap"]):  # 0: noise, 1: clap
    folder_path = os.path.join(DATA_DIR, folder)
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    print(f"Processing folder: {folder_path}")
    
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            features.append(extract_features(file_path))
            labels.append(label)
            print(f"Processed file: {file}")  # Debug info

# -------------------------- SAVE CSV --------------------------
column_names = [
    "zero_crossing_rate", "rms", "spectral_centroid",
    "spectral_bandwidth", "spectral_rolloff"
]
# MFCC columns
for i in range(1, 11):
    column_names.append(f"mfcc{i}")

df = pd.DataFrame(features, columns=column_names)
df["label"] = labels

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nFeatures saved to {OUTPUT_CSV}")
print("Feature shape:", df.shape)

# -------------------------- PLOT FEATURES --------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(12,6))

# Plot first 5 features distributions
for i, col in enumerate(column_names[:5]):
    plt.subplot(1, 5, i+1)
    sns.histplot(df[col], kde=True, hue=df['label'], palette=['red','green'], alpha=0.5)
    plt.title(col)
    
plt.tight_layout()
plt.savefig(PLOT_FILE)
print(f"Feature distribution plot saved to {PLOT_FILE}")
