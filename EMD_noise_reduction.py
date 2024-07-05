# import numpy as np
# import matplotlib.pyplot as plt
# import sounddevice as sd
# import pywt
# from PyEMD import EMD
# from scipy.signal import hilbert, welch

# # Function to capture audio from microphone
# def capture_audio(duration=5, sample_rate=44100):
#     """
#     Captures audio from microphone.

#     Parameters:
#     - duration (float): Duration to record (in seconds).
#     - sample_rate (int): Sampling rate.

#     Returns:
#     - audio_data (ndarray): Recorded audio signal.
#     - sample_rate (int): Sampling rate of the audio.
#     """
#     print(f"Recording for {duration} seconds...")
#     audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
#     sd.wait()  # Wait for recording to complete
#     return audio_data.flatten(), sample_rate

# # Function to perform EMD and extract IMF[0]
# def extract_imf0(signal):
#     emd = EMD()
#     IMFs = emd.emd(signal)
#     imf0 = IMFs[0]
#     return imf0

# # Function to calculate envelope using Hilbert transform
# def calculate_envelope(signal):
#     analytic_signal = hilbert(signal)
#     envelope = np.abs(analytic_signal)
#     return envelope

# # Function to calculate SNR
# def calculate_snr(original_signal, denoised_signal):
#     noise = original_signal - denoised_signal
#     signal_power = np.sum(original_signal ** 2)
#     noise_power = np.sum(noise ** 2)
#     snr = 10 * np.log10(signal_power / noise_power)
#     return snr

# # Function to calculate RMSE
# def calculate_rmse(original_signal, denoised_signal):
#     rmse = np.sqrt(np.mean((original_signal - denoised_signal) ** 2))
#     return rmse

# # Function to plot signals
# def plot_signals(time, original_signal, imf0, envelope_original, envelope_imf0, snr, rmse):
#     plt.figure(figsize=(12, 6))
    
#     plt.subplot(2, 1, 1)
#     plt.plot(time, original_signal, label='Original Signal')
#     plt.plot(time, imf0, label='IMF[0]')
#     plt.title(f"Original Signal vs IMF[0] | SNR: {snr:.2f} dB, RMSE: {rmse:.4f}")
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.grid(True)
    
#     plt.subplot(2, 1, 2)
#     plt.plot(time, envelope_original, label='Envelope Original Signal')
#     plt.plot(time, envelope_imf0, label='Envelope IMF[0]')
#     plt.title("Envelope Comparison")
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()

# # Function to plot PSD
# def plot_psd(original_signal, imf0, fs, rmse):
#     f, Pxx = welch(original_signal, fs=fs, nperseg=1024)
#     fi, Pxxi = welch(imf0, fs=fs, nperseg=1024)
#     plt.figure(figsize=(8, 6))
#     plt.semilogy(f, Pxx, label='Original Signal')
#     plt.semilogy(fi, Pxxi, label='IMF[0]')
#     plt.title(f'Power Spectral Density | RMSE: {rmse:.4f}')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Power/Frequency (dB/Hz)')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # Main function
# def main():
#     # Capture audio from microphone
#     duration = 3  # Duration in seconds
#     sample_rate = 44100  # Sampling rate in Hz
#     audio_data, sample_rate = capture_audio(duration, sample_rate)
#     time_array = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    
#     # Perform EMD and extract IMF[0]
#     imf0 = extract_imf0(audio_data)
#     print(imf0)
    
#     # Calculate envelopes
#     envelope_original = calculate_envelope(audio_data)
#     envelope_imf0 = calculate_envelope(imf0)
    
#     # Calculate SNR and RMSE
#     snr = calculate_snr(audio_data, imf0)
#     rmse = calculate_rmse(audio_data, imf0)
    
#     # Plot signals and envelopes
#     plot_signals(time_array, audio_data, imf0, envelope_original, envelope_imf0, snr, rmse)
    
#     # Plot PSD
#     plot_psd(audio_data, imf0, sample_rate, rmse)

# if __name__ == '__main__':
#     main()



import sounddevice as sd
import numpy as np
from PyEMD import EMD
import librosa
import matplotlib.pyplot as plt

def extract_imf(signal: list):
    emd = EMD()
    IMFs = emd.emd(signal)
    print(*IMFs, sep = ' \n')
    return IMFs

# Function to calculate SNR
def calculate_snr(original_signal, denoised_signal):
    noise = original_signal - denoised_signal
    signal_power = np.sum(original_signal ** 2)
    noise_power = np.sum(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr
    

# load  the file
file_path = 'Audio_samples\\Breath_Audio2.mp3'
audio_data, sample_rate = librosa.load(file_path)

# plot the signal
time = np.linspace(0, len(audio_data), sample_rate)
IMF = extract_imf(audio_data)

for imf in range(len(IMF)):
    plt.plot(time, IMF[imf], label = f'IMF[{imf}]')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.title('EMD Decomposition Signal')
    plt.show()
    
    