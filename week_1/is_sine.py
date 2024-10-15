import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


# Perform FFT and find dominant frequency
def perform_frequency_analysis(data, dt):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n // 2]  # Only positive frequencies
    power = 2.0 / n * np.abs(yf[:n // 2])

    # Find the dominant frequency
    dominant_frequency = xf[np.argmax(power)]

    # Plot the frequency spectrum
    plt.figure()
    plt.plot(xf, power)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()

    return dominant_frequency, power


# Check if the dominant frequency corresponds to a sine wave
def is_sine_wave(dominant_frequency, power, threshold=0.9):
    # Check if power is highly concentrated at the dominant frequency
    power_ratio = np.max(power) / np.sum(power)  # Ratio of dominant power to total power
    print(power_ratio)
    # If dominant frequency contains most of the signal's energy, it's likely a sine wave
    if power_ratio > threshold:
        return True  # Signal is likely a sine wave
    else:
        return False  # Signal is likely not a pure sine wave


# Generate a sample sine wave and test the function
fs = 100  # Sampling frequency
t = np.linspace(0, 1, fs)
signal = 2 * np.sin(2 * np.pi * 5 * t)  # Sine wave with frequency 5 Hz
plt.figure(figsize=(10, 8))
# Position plot for joint i
plt.plot(signal, label='signal')
# plt.plot(t, label='time', linestyle='--')
plt.title(f'is_sine')
plt.xlabel('Time steps')
plt.ylabel('signal')
plt.legend()
# Perform frequency analysis
dominant_freq, power = perform_frequency_analysis(signal, dt=1 / fs)
print(dominant_freq)
# Check if the signal is a sine wave
if is_sine_wave(dominant_freq, power):
    print(f"The signal is a sine wave with a dominant frequency of {dominant_freq} Hz.")
else:
    print("The signal is not a pure sine wave.")
