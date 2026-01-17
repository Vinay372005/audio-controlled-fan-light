import librosa
import numpy as np

def extract_features(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    # MFCCs (1–10)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Other audio features
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    )
    spectral_bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sr)
    )

    # Final feature vector (17 features)
    features = np.hstack([
        mfcc_mean,
        zcr,
        rms,
        spectral_centroid,
        spectral_bandwidth
    ])

    return features
