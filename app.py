from src.predict import predict_audio
from src.fsm import ClapFSM
import time

fsm = ClapFSM()

audio_files = ["data/clap/clap_01.wav", "data/noise/noise_01.wav"]

for file in audio_files:
    result = predict_audio(file)
    print(f"Audio: {file}, Prediction: {result}")
    if result == "CLAP":
        fsm.update("CLAP")
        time.sleep(0.4)

state = fsm.state
if state == "ON":
    print("💡 Light ON")
    print("🌀 Fan ROTATING")
else:
    print("💡 Light OFF")
    print("🌀 Fan STOPPED")
