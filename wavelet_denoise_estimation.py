import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import pywt
from scipy.signal import welch
import librosa
import time

# Function to capture audio from microphone
def capture_audio(duration=5, sample_rate=44100):
    """
    Captures audio from microphone.

    Parameters:
    - duration (float): Duration to record (in seconds).
    - sample_rate (int): Sampling rate.

    Returns:
    - audio_data (ndarray): Recorded audio signal.
    - sample_rate (int): Sampling rate of the audio.
    """
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait for recording to complete
    return audio_data.flatten(), sample_rate

# Function to perform wavelet denoising
def wavelet_denoising(signal, wavelet='db1', level=6):
    """
    Applies wavelet denoising to the input signal.

    Parameters:
    - signal (ndarray): Input signal.
    - wavelet (str): Wavelet type.
    - level (int): Decomposition level.

    Returns:
    - denoised_signal (ndarray): Denoised signal.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signal

# Function to calculate SNR
def calculate_snr(original_signal, denoised_signal):
    """
    Calculates Signal-to-Noise Ratio (SNR) between original and denoised signals.

    Parameters:
    - original_signal (ndarray): Original signal.
    - denoised_signal (ndarray): Denoised signal.

    Returns:
    - snr (float): Signal-to-Noise Ratio (in dB).
    """
    noise = original_signal - denoised_signal
    signal_power = np.sum(original_signal ** 2)
    noise_power = np.sum(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Function to calculate RMSE
def calculate_rmse(original_signal, denoised_signal):
    """
    Calculates Root Mean Square Error (RMSE) between original and denoised signals.

    Parameters:
    - original_signal (ndarray): Original signal.
    - denoised_signal (ndarray): Denoised signal.

    Returns:
    - rmse (float): Root Mean Square Error.
    """
    rmse = np.sqrt(np.mean((original_signal - denoised_signal) ** 2))
    return rmse

# Function to plot original and denoised signals
def plot_signals(time, original_signal, denoised_signal, title, snr):
    """
    Plots the original and denoised signals.

    Parameters:
    - time (ndarray): Time array.
    - original_signal (ndarray): Original signal.
    - denoised_signal (ndarray): Denoised signal.
    - title (str): Title for the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, original_signal, label='Original Signal', alpha=0.7)
    plt.plot(time, denoised_signal, label='Denoised Signal', alpha=0.9)
    plt.title(f"{title} | SNR: {snr:.2f} dB")
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot PSD
def plot_psd(signal, deoised_signal, fs, rmse: float):
    """
    Plots the Power Spectral Density (PSD) of a signal.

    Parameters:
    - signal (ndarray): Input signal.
    - fs (int): Sampling rate of the signal.
    - title (str): Title for the plot.
    """
    
    f, Pxx = welch(signal, fs=fs, nperseg=1024)
    fd, Pxxd = welch(deoised_signal, fs = fs, nperseg=1024)
    plt.figure(figsize=(8, 6))
    plt.semilogy(f, Pxx, label = 'Original Signal',alpha = 1)
    plt.semilogy(fd, Pxxd, color = 'red', label = 'Denoised Signal', alpha = 0.5)
    plt.title(f'Power Spectral Density | RMSE: {rmse:.4f}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Capture audio from microphone
    duration = 3
    sample_rate = 4000  
    # audio_data, sample_rate = capture_audio(duration, sample_rate)
    file_path = 'Audio_samples\\Breath Audio 2.mp3'
    audio_data, sample_rate = librosa.load(file_path, sr=4000) 
    time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    
    plt.plot(time, audio_data, label = 'Audio Data')
    
    # Perform wavelet denoising
    denoised_signal = wavelet_denoising(audio_data)
    
    # Calculate SNR and RMSE
    snr = calculate_snr(audio_data, denoised_signal)
    rmse = calculate_rmse(audio_data, denoised_signal)
    
    # Plot original and denoised signals
    plot_signals(time, audio_data, denoised_signal, title='Original vs Denoised Signal', snr=snr)
    
    # Plot PSD
    plot_psd(denoised_signal ,audio_data, sample_rate,rmse=rmse)
    # # plot_psd(denoised_signal, sample_rate, title='Denoised Signal PSD')
    
    # # Print SNR and RMSE
    # print(f"SNR: {snr:.2f} dB")
    # print(f"RMSE: {rmse:.4f}")

if __name__ == '__main__':
    main()
        