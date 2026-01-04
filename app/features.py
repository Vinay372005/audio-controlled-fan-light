import os
import librosa
import numpy as np
import pandas as pd

# ==============================
# PATHS (DO NOT CHANGE)
# ==============================
BASE_DIR = os.getcwd()                     # /content/audio-controlled-fan-light
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

    # 6–15. MFCCs (10)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    features.extend(np.mean(mfcc, axis=1))

    return features

# ==============================
# MAIN PROCESS
# ==============================
features_list = []
labels_list = []
filenames_list = []

# ---- NOISE FILES (label = 0)
print("Processing NOISE files...")
noise_files = sorted(os.listdir(NOISE_DIR))

for file in noise_files:
    if file.endswith(".wav"):
        file_path = os.path.join(NOISE_DIR, file)
        feats = extract_features(file_path)

        features_list.append(feats)
        labels_list.append(0)
        filenames_list.append(file)

# ---- CLAP FILES (label = 1)
print("Processing CLAP files...")
clap_files = sorted(os.listdir(CLAP_DIR))

for file in clap_files:
    if file.endswith(".wav"):
        file_path = os.path.join(CLAP_DIR, file)
        feats = extract_features(file_path)

        features_list.append(feats)
        labels_list.append(1)
        filenames_list.append(file)

# ==============================
# DATAFRAME CREATION
# ==============================
feature_names = [
    "zcr", "rms", "spectral_centroid",
    "spectral_bandwidth", "spectral_rolloff"
] + [f"mfcc_{i+1}" for i in range(10)]

df = pd.DataFrame(features_list, columns=feature_names)
df["label"] = labels_list
df["filename"] = filenames_list

# ==============================
# SAVE CSV (ONLY ONCE)
# ==============================
df.to_csv(OUTPUT_CSV, index=False)

# ==============================
# FINAL VERIFICATION
# ==============================
print("\n✅ FEATURES SAVED SUCCESSFULLY")
print("CSV Path:", OUTPUT_CSV)
print("Total files processed:", len(df))
print("Noise samples:", df[df["label"] == 0].shape[0])
print("Clap samples:", df[df["label"] == 1].shape[0])
print("Feature shape:", df.shape)
