import pyaudio
import numpy as np
import joblib
import librosa
import time
from fsm import ClapFSM
from pathlib import Path

# Load model
model_path = Path('../models/clap_detector.pkl')
if not model_path.exists():
    print("❌ Run: python -m app.train first!")
    exit(1)
model = joblib.load(model_path)
fsm = ClapFSM()

def extract_features(audio, sr=22050):
    zcr = librosa.feature.zero_crossing_rate(audio)[0].mean()
    rms = librosa.feature.rms(y=audio)[0].mean()
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0].mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0].mean()
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=10).mean(axis=1)
    return np.concatenate([[zcr, rms, centroid, bandwidth, rolloff], mfcc]).reshape(1, -1)

# Audio setup
RATE, CHUNK = 22050, RATE // 2  # 0.5s chunks
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, 
                input=True, frames_per_buffer=CHUNK)

print("👂 Listening for claps... (Ctrl+C to stop)")
print("Odd claps: Toggle Fan | Even claps: Toggle Light")

try:
    last_clap_time = 0
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.float32)
        
        features = extract_features(audio)
        pred = model.predict(features)[0]
        
        if pred == 1 and time.time() - last_clap_time > 0.5:  # Debounce
            fsm.process_clap()
            last_clap_time = time.time()
        else:
            fsm.reset()
            
except KeyboardInterrupt:
    print("\n👋 Stopped.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate() [web:2][web:6]
