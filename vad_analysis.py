import numpy as np
import matplotlib.pyplot as plt
import librosa

# ses dosyasını yükle
signal, sr = librosa.load("speech.wav", sr=None)

# pencere boyutu
frame_size = int(0.02 * sr)
hop_size = int(frame_size / 2)

energy = []

# enerji hesaplama
for i in range(0, len(signal) - frame_size, hop_size):
    frame = signal[i:i + frame_size]
    energy.append(np.sum(frame ** 2))

energy = np.array(energy)
threshold = np.mean(energy) * 0.5

# grafik 1
plt.figure(figsize=(10,4))
plt.plot(signal)
plt.title("Speech Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("speech_signal.png")

# grafik 2
plt.figure(figsize=(10,4))
plt.plot(energy)
plt.axhline(threshold, color="r")
plt.title("Energy Plot")
plt.xlabel("Frame")
plt.ylabel("Energy")
plt.tight_layout()
plt.savefig("energy_plot.png")

# zcr
zcr = librosa.feature.zero_crossing_rate(
    signal,
    frame_length=frame_size,
    hop_length=hop_size
)[0]

# grafik 3
plt.figure(figsize=(10,4))
plt.plot(zcr)
plt.title("Voiced / Unvoiced (ZCR)")
plt.xlabel("Frame")
plt.ylabel("ZCR")
plt.tight_layout()
plt.savefig("voiced_unvoiced.png")

plt.show()