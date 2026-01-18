import librosa
import librosa.display
import matplotlib.pyplot as plt

file = "data/clap/clap1.wav"   # any clap sample

y, sr = librosa.load(file)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

plt.figure(figsize=(8, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title("MFCC - Clap Sound")

plt.savefig("plots/mfcc_plot.png")
plt.show()
