import os
import librosa
import numpy as np
import pandas as pd

# ==============================
# PATHS
# ==============================
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
CLAP_DIR = os.path.join(DATA_DIR, "clap")
NOISE_DIR = os.path.join(DATA_DIR, "noise")
OUTPUT_CSV = os.path.join(DATA_DIR, "features.csv")

# ==============================
# FEATURE EXTRACTION FUNCTION
# ==============================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None, duration=2.0)

    features = []

    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    features.append(np.mean(librosa.feature.rms(y=y)))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    features.extend(np.mean(mfcc, axis=1))

    return features

# ==============================
# MAIN
# ==============================
features_list = []
labels_list = []
filenames_list = []

print("Processing NOISE files...")
for file in sorted(os.listdir(NOISE_DIR)):
    if file.endswith(".wav"):
        path = os.path.join(NOISE_DIR, file)
        features_list.append(extract_features(path))
        labels_list.append(0)
        filenames_list.append(file)

print("Processing CLAP files...")
for file in sorted(os.listdir(CLAP_DIR)):
    if file.endswith(".wav"):
        path = os.path.join(CLAP_DIR, file)
        features_list.append(extract_features(path))
        labels_list.append(1)
        filenames_list.append(file)

feature_names = [
    "zcr", "rms", "spectral_centroid",
    "spectral_bandwidth", "spectral_rolloff"
] + [f"mfcc_{i+1}" for i in range(10)]

df = pd.DataFrame(features_list, columns=feature_names)
df["label"] = labels_list
df["filename"] = filenames_list

df.to_csv(OUTPUT_CSV, index=False)

print("\n✅ FEATURES SAVED SUCCESSFULLY")
print("CSV Path:", OUTPUT_CSV)
print("Total samples:", df.shape[0])
print("Noise:", (df["label"] == 0).sum())
print("Clap:", (df["label"] == 1).sum())
print("Feature shape:", df.shape)
