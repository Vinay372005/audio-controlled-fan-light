import librosa
import numpy as np

def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)

    # Remove silence
    y, _ = librosa.effects.trim(y)

    # ----- TIME DOMAIN -----
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))

    # ----- FREQUENCY DOMAIN -----
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # ----- MFCC -----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    features = np.hstack([
        mfcc_mean,
        zcr,
        rms,
        spec_centroid,
        spec_bandwidth,
        rolloff
    ])

    return features
