import os
import numpy as np
import librosa

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Feature 1: Short-time energy
    energy = np.sum(y**2) / len(y)

    # Feature 2: Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Feature 3: MFCC (mean of 13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc)

    return [energy, zcr, mfcc_mean]


def load_dataset(data_path="data"):
    X = []
    y = []

    clap_path = os.path.join(data_path, "clap")
    noise_path = os.path.join(data_path, "noise")

    # Clap = 1
    for file in os.listdir(clap_path):
        if file.endswith(".wav"):
            features = extract_features(os.path.join(clap_path, file))
            X.append(features)
            y.append(1)

    # Noise = 0
    for file in os.listdir(noise_path):
        if file.endswith(".wav"):
            features = extract_features(os.path.join(noise_path, file))
            X.append(features)
            y.append(0)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_dataset()
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
