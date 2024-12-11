import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cmath


# Simulate a simple ECG-like signal
def generate_ecg_signal(duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Generating a simple sine wave to simulate heartbeat frequency
    # Heart rate is given in beats per minute, convert it to Hz

    # Simulating heartbeat signal with main component at the heart rate frequency

    # Adding harmonics and some random noise to make it more realistic
    aplha_wave = 0.5 * np.sin(2 * np.pi * 10 * t)
    beta_wave = 0.3 * np.sin(2 * np.pi * 20 * t)
    gamma_wave = 0.2 * np.sin(2 * np.pi * 60 * t)
    noise = 0.1 * np.random.randn(len(t))
    ecg_signal =  aplha_wave + beta_wave + gamma_wave + noise

    return t, ecg_signal


# Parameters for the simulated ECG signal
duration = 10  # seconds
sampling_rate = 409.6  # Hz

# Generate ECG signal
time, ecg_signal = generate_ecg_signal(duration, sampling_rate)

# Plot the original ECG signal
plt.figure(figsize=(10, 4))
plt.plot(time, ecg_signal)
plt.title('Simulated ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.grid(True)
plt.show()

def fft(signal):
    N = len(signal)
    if N <= 1:
        return signal
    even = fft(signal[0::2])
    odd = fft(signal[1::2])
    t = [cmath.exp(-2j * cmath.pi * k / N) * odd[k % len(odd)] for k in range(N)]
    return [even[k] + t[k] for k in range(N // 2)] + [even[k] - t[k] for k in range(N // 2)]


# Apply FFT to the ECG signal
n = len(ecg_signal)
print(n)
fft_result = fft(ecg_signal)
frequencies = np.fft.fftfreq(n, d=1 / sampling_rate)

vector = set([])
# Only take the positive frequencies and magnitude
positive_freqs = frequencies[:n // 2]
magnitude = np.abs(fft_result)[:n // 2]
for i in range(len(magnitude)):
    if magnitude[i] > 200:
        vector.add(i/10)

print(sorted(vector))

# Plot the magnitude of the FFT
plt.figure(figsize=(10, 4))
plt.plot(positive_freqs, magnitude)
plt.title('Frequency Spectrum of ECG Signal (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.axvspan(8, 13, facecolor='yellow', alpha=0.3)
plt.axvspan(13, 30, facecolor='green', alpha=0.3)
plt.axvspan(30, 100, facecolor='red', alpha=0.3)
plt.xlim(0, 100)
legend_elements = [Patch(facecolor='yellow',
                         label='Alpha(8-13 Hz)'),Patch(facecolor='green',
                         label='Beta(13-30 Hz)'),Patch(facecolor='red',
                         label='Gamma(30-100 Hz)')]
plt.legend(handles=legend_elements, loc='upper right')
plt.show()

print(positive_freqs[10:20])
print(magnitude[10:20])