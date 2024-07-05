import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import tkinter as tk
import time
from scipy.signal import butter, lfilter, hilbert, kaiserord, filtfilt, firwin, find_peaks, welch
from PyEMD import EMD
import pywt

# Parameters
SAMPLE_RATE = 4000
LOWCUT = 100
HIGHCUT = 1000
ORDER = 5
WINDOW_SIZE = 20
FL_HZ = 10
RIPPLE_DB = 10.0

# Function to record audio 
def record_audio(duration):
    print("Recording...")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    return audio_data.flatten(), SAMPLE_RATE

# Function to display countdown window
def show_countdown():
    countdown_window = tk.Tk()
    countdown_window.title("Countdown")
    countdown_label = tk.Label(countdown_window, text="", font=("Arial", 24))
    countdown_label.pack()

    for i in range(5, 0, -1):
        countdown_label.config(text=f"Get ready!\nStarting in {i}...")
        countdown_window.update()
        time.sleep(1)

    countdown_window.destroy()

# Function to create a Butterworth bandpass filter
def butter_bp(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to filter the audio data
def filter_signal(signal, lowcut, highcut, order, fs):
    b, a = butter_bp(lowcut, highcut, fs, order)
    y = lfilter(b, a, signal)
    return y

# Function to extract the envelope of the signal
def extract_envelope(signal, fs, fL_hz, ripple_db):
    nyq_rate = 0.5 * fs
    width = 1.0 / nyq_rate
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    N, beta = kaiserord(ripple_db, width)
    taps = firwin(N, fL_hz / nyq_rate, window=('kaiser', beta))
    filtered_envelope = filtfilt(taps, 1, envelope)
    return filtered_envelope

# Function to smooth the envelope
def smooth_envelope(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# Function to find peaks in the signal
def find_peaks_signal(signal):
    peaks, _ = find_peaks(signal, distance=30)
    return peaks

# Function to clip the audio signal
def clip_audio(audio_sig, envelope_signal, threshold_ratio, window_size=100):
    prev_avg = 0
    max_avg = 0
    maxima = False
    step_size = 50
    width = len(envelope_signal)

    for i in range(0, len(envelope_signal) - window_size + 1, step_size):
        window_avg = np.mean(envelope_signal[i:i + window_size])
        if window_avg < prev_avg and not maxima:
            maxima = True
            max_avg = prev_avg
        if maxima and window_avg < threshold_ratio * max_avg:
            width = i
            break
        prev_avg = window_avg

    clipped_audio_signal = audio_sig[:width]
    clipped_envelope_signal = envelope_signal[:width]
    return clipped_audio_signal, clipped_envelope_signal, width

# Function to calculate the peak at 1 second
def one_sec_peak(signal, sr):
    return signal[sr]

# Function to apply EMD for noise reduction
def apply_emd(signal):
    emd = EMD()
    IMFs = emd.emd(signal)
    denoised_signal = signal - IMFs[0]
    return denoised_signal, IMFs[0]

# Function to apply wavelet denoising
def wavelet_denoise(signal):
    coeffs = pywt.wavedec(signal, 'db1', level=6)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(denoised_coeffs, 'db1')
    return denoised_signal

# Function to calculate RMSE
def calculate_rmse(original_signal, denoised_signal):
    rmse = np.sqrt(np.mean((original_signal - denoised_signal) ** 2))
    return rmse

# Function to calculate SNR
def calculate_snr(original_signal, denoised_signal):
    signal_power = np.mean(original_signal ** 2)
    noise_power = np.mean((original_signal - denoised_signal) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Function to check if the clipping point is correct
def isCorrect(signal: list , threshold_ratio: float):
    """
    Checks if the sample is correct based on peak analysis.

    Parameters:
    - peaks (ndarray): Indices of peaks in the signal.
    - signal (ndarray): Signal to analyze.
    - threshold (float): Threshold value for peak analysis.

    Returns:
    - bool: True if the sample is correct, False otherwise.
    """
    # maxima = 0
    # for right in range(len(peaks) - 1):
    #     if maxima == 0 and signal[peaks[right + 1]] < signal[peaks[right]]:
    #         maxima = 1
    #         maxx = signal[peaks[right]]
    #         continue
    #     if maxima == 1 and signal[peaks[right]] > 0.4 * maxx:                
    #         return False
    # return True
    
    maxima = False
    pre_avg = 0
    window_size = 100
    step_size = 50
    maxx_window = 0
    for i in range(0, len(signal)-window_size +1, step_size):
        curr_avg = np.mean(signal[i:i+window_size + 1])
        if maxima == False and pre_avg < curr_avg:
            maxima = True
            maxx_window = pre_avg
        if maxima == True and curr_avg > maxx_window * threshold_ratio:
            return False
        pre_avg = curr_avg
    return True

# Function to plot rejection
def plot_rejection(time_array:list, env_sig: list, wavelet_denoised_signal:list):
    plt.figure(figsize=(12, 6))
    plt.plot(time_array, env_sig,color = 'r', label = 'Envelope Signal')
    plt.plot(time_array, wavelet_denoised_signal, color='c', label='Denoised Signal', alpha=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(f' Rejection Audio Signal Before and After ICEEMDAN Filter')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

# Function to plot the results
def plot_results(audio_data, emd_denoised_signal, filtered_signal, wavelet_denoised_signal,peak_1, mx_peak_E ,smooth_env, peaks_E,clipped_E, time_array, rmse_emd, snr_emd, rmse_butter, snr_butter, endpoint):
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_array, audio_data, color = 'b', label = 'Raw Audio Signal')
    plt.plot(time_array, wavelet_denoised_signal, color='c', label='Filtered Audio Signal', alpha = 0.5)
    plt.plot(time_array, smooth_env, color='r', label='Envelope', alpha=0.7)
    plt.plot(time_array[peaks_E], smooth_env[peaks_E], 'go', label='Peaks')
    plt.plot(time_array[endpoint], smooth_env[endpoint], 'mo', label='End Point')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title(f'Filtered Audio Waveform and Envelope | Peak at 1 sec:{abs(peak_1):.2f} | Max Peak: {mx_peak_E:.2f} | SNR: {snr_emd:.2f} | RMSE: {rmse_emd:.4f}')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_array, audio_data, color='b', label='Raw Audio Signal')
    plt.plot(time_array, filtered_signal, color='c', label='Butterworth Filtered Signal', alpha=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(f'Audio Signal Before and After Butterworth Bandpass Filter | RMSE: {rmse_butter:.4f} | SNR: {snr_butter:.2f} dB')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    ctime = np.linspace(0, len(clipped_E) / SAMPLE_RATE, len(clipped_E))
    plt.plot(ctime, clipped_E, color='g', label='Clipped Envelope Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Clipped Envelope Signal')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Main function
def main():
    show_countdown()                                                                            
    duration = 3
    audio_data, sample_rate = record_audio(duration)
    dur = len(audio_data) / sample_rate
    time_array = np.linspace(0, dur, len(audio_data))
    
    # apply filter of raw audio
    butter_worth_filtered = filter_signal(audio_data, LOWCUT, HIGHCUT,ORDER, SAMPLE_RATE)
    
    # Apply EMD for noise reduction
    emd_denoised_signal, imf0 = apply_emd(butter_worth_filtered)
    
    # Apply Butterworth bandpass filter
    # filtered_signal = filter_signal(emd_denoised_signal, LOWCUT, HIGHCUT, ORDER, SAMPLE_RATE)
    
    # Apply wavelet denoising
    wavelet_denoised_signal = wavelet_denoise(emd_denoised_signal)
    
    # Extract envelope and smooth
    envelope = extract_envelope(wavelet_denoised_signal, sample_rate, FL_HZ, RIPPLE_DB)
    smooth_env = smooth_envelope(envelope, WINDOW_SIZE)
    
    # Find peaks in the envelope
    peaks_E = find_peaks_signal(smooth_env)
    
    # Clip the audio signal based on envelope peaks
    clipped_signal, clipped_E, endpoint = clip_audio(wavelet_denoised_signal, smooth_env, 0.2)
    
    # peak at one second
    peak_1 = smooth_env[4000]
    maxx_peaks = np.max(clipped_E)
    
    # Calculate RMSE and SNR
    rmse_emd = calculate_rmse(audio_data, wavelet_denoised_signal)
    snr_emd = calculate_snr(audio_data, wavelet_denoised_signal)
    rmse_butter = calculate_rmse(audio_data, butter_worth_filtered)
    snr_butter = calculate_snr(audio_data, butter_worth_filtered)
    
    # max peaks
    maxx_peaks = np.max(smooth_env)
    plot_results(audio_data, emd_denoised_signal, butter_worth_filtered ,wavelet_denoised_signal,peak_1, maxx_peaks ,smooth_env, peaks_E,clipped_E, time_array, rmse_emd, snr_emd, rmse_butter, snr_butter, endpoint)
    
    # correct = isCorrect(smooth_env, 0.2)
    # correct = True
    # if correct:
    #     plot_results(audio_data, emd_denoised_signal, butter_worth_filtered ,wavelet_denoised_signal,peak_1, maxx_peaks ,smooth_env, peaks_E,clipped_E, time_array, rmse_emd, snr_emd, rmse_butter, snr_butter, endpoint)
    # else:
    #     plot_rejection(time_array,smooth_env, wavelet_denoised_signal )

# Run the main function
if __name__ == '__main__':
    # Allow user to retry if desired
    again = True
    while again:
        main()
        ans = input("Try Again? y/n: ").lower()
        if ans == 'n' or ans =='no':
            again = False
        elif ans == 'yes' or ans == 'y':
            again = True
        else:
            print("Invalid Response!!")
