import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "data/noise/noise1.wav"  # try clap & noise separately

y, sr = librosa.load(file)

D = np.abs(librosa.stft(y))

plt.figure(figsize=(8, 4))
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram")

plt.savefig("plots/spectrogram.png")
plt.show()
