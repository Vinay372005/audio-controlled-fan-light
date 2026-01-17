import os
import librosa
import pandas as pd
import numpy as np
from pathlib import Path

def extract_features(file_path):
    """Extract 15 Librosa features for clap detection."""
    y, sr = librosa.load(file_path, sr=22050, duration=2.0)
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
    rms = librosa.feature.rms(y=y)[0].mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10).mean(axis=1)
    return np.concatenate([[zcr, rms, centroid, bandwidth, rolloff], mfcc])

clap_dir = Path('../data/clap')
noise_dir = Path('../data/noise')

clap_files = list(clap_dir.glob('*.wav'))
noise_files = list(noise_dir.glob('*.wav'))

print(f"Found {len(clap_files)} claps, {len(noise_files)} noises")

features, labels = [], []
for f in clap_files:
    features.append(extract_features(f))
    labels.append(1)
for f in noise_files:
    features.append(extract_features(f))
    labels.append(0)

df = pd.DataFrame(features, columns=[f'feat_{i}' for i in range(15)])
df['label'] = labels
df.to_csv('../data/features.csv', index=False)
print("✓ Features saved to data/features.csv") [web:6]
