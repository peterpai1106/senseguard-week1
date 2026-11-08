import numpy as np
import matplotlib.pyplot as plt

fs = 1000
duration = 1.0
f0 = 20 + 21
noise = 0.8

# 1. time axis
t = np.arange(0, duration, 1 / fs)
# 2. generate signal
rng = np.random.default_rng()
x = np.sin(2 * np.pi * f0 * t) + noise * rng.normal(size=t.size)
# 3. spectrum
spectrum = np.abs(np.fft.rfft(x))
# 4. frequency axis
freq = np.fft.rfftfreq(t.size, 1 / fs)
# 5. find peak
peak_idx = np.argmax(spectrum)
peak_freq = freq[peak_idx]
print(f"Peak frequency = {peak_freq:.2f} Hz")
# 6. plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
mask = t <= 0.2
ax1.plot(t[mask], x[mask])
ax1.set_xlabel("t (s)")
ax1.set_ylabel("Amplitude")
ax1.set_title("Time-domain waveform (first 0.2s)")

ax2.plot(freq, spectrum)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Amplitude")
ax2.set_title("Spectrum")

plt.tight_layout()
plt.savefig("output/fft.png")

