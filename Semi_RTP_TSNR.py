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
LOWCUT = 1
HIGHCUT = 1999
ORDER = 5
WINDOW_SIZE = 20
FL_HZ = 10
RIPPLE_DB = 10.0

# Function to record audio 
def record_audio(duration):
    """
    Records audio for a specified duration.

    Parameters:
    - duration (float): Duration in seconds to record audio.

    Returns:
    - audio_data (ndarray): Recorded audio signal.
    - SAMPLE_RATE (int): Sampling rate of the audio.
    """
    print("Recording...")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # Wait for recording to complete
    return audio_data.flatten(), SAMPLE_RATE

# Function to display countdown window
def show_countdown():
    """
    Displays a countdown window using Tkinter.
    """
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
    """
    Creates a Butterworth bandpass filter.

    Parameters:
    - lowcut (float): Low cutoff frequency of the filter.
    - highcut (float): High cutoff frequency of the filter.
    - fs (int): Sampling frequency.
    - order (int): Filter order.

    Returns:
    - b (ndarray): Numerator coefficients of the filter.
    - a (ndarray): Denominator coefficients of the filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to filter the audio data
def filter_signal(signal, lowcut, highcut, order, fs):
    """
    Applies a Butterworth bandpass filter to the input signal.

    Parameters:
    - signal (ndarray): Input audio signal to be filtered.
    - lowcut (float): Low cutoff frequency of the filter.
    - highcut (float): High cutoff frequency of the filter.
    - order (int): Filter order.
    - fs (int): Sampling frequency.

    Returns:
    - y (ndarray): Filtered output signal.
    """
    b, a = butter_bp(lowcut, highcut, fs, order)
    y = lfilter(b, a, signal)
    return y

# Function to extract the envelope of the signal
def extract_envelope(signal, fs, fL_hz, ripple_db):
    """
    Extracts the envelope of the input signal.

    Parameters:
    - signal (ndarray): Input signal.
    - fs (int): Sampling frequency.
    - fL_hz (float): Cutoff frequency for the lowpass filter.
    - ripple_db (float): Ripple in the passband of the filter.

    Returns:
    - filtered_envelope (ndarray): Envelope of the input signal.
    """
    nyq_rate = 0.5 * fs
    width = 1.0 / nyq_rate

    # Hilbert transform to get the analytic signal
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    
    # Design a lowpass FIR filter using the Kaiser window method
    N, beta = kaiserord(ripple_db, width)
    taps = firwin(N, fL_hz / nyq_rate, window=('kaiser', beta))
    filtered_envelope = filtfilt(taps, 1, envelope)
    return filtered_envelope

# Function to smooth the envelope
def smooth_envelope(signal, window_size):
    """
    Smoothes the envelope of the signal using a moving average.

    Parameters:
    - signal (ndarray): Input signal (envelope).
    - window_size (int): Size of the moving average window.

    Returns:
    - smoothed_signal (ndarray): Smoothed envelope signal.
    """
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# Function to find peaks in the signal
def find_peaks_signal(signal):
    """
    Finds peaks in the input signal.

    Parameters:
    - signal (ndarray): Input signal.

    Returns:
    - peaks (ndarray): Indices of peaks in the signal.
    """
    peaks, _ = find_peaks(signal, distance=30)
    return peaks

# Function to clip the audio signal
def clip_audio(audio_sig: list, envelope_signal: list, threshold_ratio: int,window_size = 100):
    """
    Clips the audio signal based on envelope peaks.

    Parameters:
    - audio_sig (ndarray): Audio signal to be clipped.
    - envelope_signal (ndarray): Envelope of the audio signal.
    - peaks (ndarray): Indices of peaks in the envelope signal.

    Returns:
    - clipped_audio_signal (ndarray): Clipped audio signal.
    - clipped_envelope_signal (ndarray): Clipped envelope signal.
    """
    prev_avg = 0
    max_avg = 0
    maxima = False
    step_size = 50
    width = len(envelope_signal)

    # Iterate through the signal with the specified step size
    for i in range(0, len(envelope_signal) - window_size + 1, step_size):
        window_avg = np.mean(envelope_signal[i:i + window_size])

        # Detect the point where the signal has reached a local maximum
        if window_avg < prev_avg and not maxima:
            maxima = True
            max_avg = prev_avg

        # Determine the clipping point based on the threshold ratio
        if maxima and window_avg < threshold_ratio * max_avg:
            width = i
            break

        prev_avg = window_avg

    # Clip the signals at the determined width
    clipped_audio_signal = audio_sig[:width]
    clipped_envelope_signal = envelope_signal[:width]

    return clipped_audio_signal, clipped_envelope_signal, width

# Function to calculate the peak at 1 second
def one_sec_peak(signal: list, sr:int):
    """
    Calculates the peak value at 1 second in the signal.

    Parameters:
    - signal (ndarray): Input signal.
    - sr (int): Sampling rate of the signal.

    Returns:
    - peak_value (float): Peak value at 1 second in the signal.
    """
    return signal[sr] 

# Function to apply EMD for noise reduction
def apply_emd(signal):
    """
    Applies Empirical Mode Decomposition (EMD) for noise reduction.

    Parameters:
    - signal (ndarray): Input signal.

    Returns:
    - denoised_signal (ndarray): Signal with noise reduced using EMD.
    """
    emd = EMD()
    IMFs = emd.emd(signal)
    denoised_signal = signal-IMFs[0]
    return denoised_signal, IMFs[0]

# Function to apply wavelet denoising
def wavelet_denoise(signal):
    """
    Applies wavelet denoising to the input signal.

    Parameters:
    - signal (ndarray): Input signal.

    Returns:
    - denoised_signal (ndarray): Signal with noise reduced using wavelet denoising.
    """
    coeffs = pywt.wavedec(signal, 'db1', level=6)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(denoised_coeffs, 'db1')
    return denoised_signal

# Function to check if the sample is correct based on peak analysis
def isCorrect(signal,threshold_ratio, peaks):
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

# plot the power density spetural
def plot_psd(signal, deoised_signal, fs, rmse: float, title:str):
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
    plt.title(f'Power Spectral Density | {title} |RMSE: {rmse:.4f}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
     
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

# Function to calculate SNR
def calculate_snr(original_signal, denoised_signal):
    signal_power = np.mean(original_signal ** 2)
    noise_power = np.mean((original_signal - denoised_signal) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Function to plot results when sample is accepted
def plot_results(audio_data, butterworth, wavelet_denoised_signal, smooth_env, peaks_E, time_array, mx_peak_E, peak_1, clipped_E, width, imf0):
    """
    Plots the results when the sample is accepted.

    Parameters:
    - audio_data (ndarray): Recorded audio signal.
    - emd_denoised_signal (ndarray): EMD denoised audio signal.
    - wavelet_denoised_signal (ndarray): Denoised audio signal.
    - smooth_env (ndarray): Smoothed envelope signal.
    - peaks_E (ndarray): Indices of peaks in the envelope.
    - time_array (ndarray): Time array corresponding to the audio signal.
    - mx_peak_E (float): Maximum peak value in the envelope.
    - peak_1 (float): Peak value at 1 second in the envelope.
    """
    plt.figure(figsize=(14, 6))
    
    # plt.subplot(3, 1, 1)
    # plt.plot(time_array, audio_data, color='b', label='Raw Audio Signal')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Amplitude') 
    # plt.title('Raw Audio Waveform')
    # plt.legend(loc="upper right")
    # plt.grid(True)
    
    # plt.subplot(3, 1, 2)
    # plt.plot(time_array, emd_denoised_signal, color='c', label='EMD Denoised Signal')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Amplitude') 
    # plt.title('EMD Denoised Waveform')
    # plt.legend(loc="upper right")
    # plt.grid(True)
    
    plt.subplot(3, 1, 1)
    plt.plot(time_array, audio_data, color = 'b', label = 'Raw Audio Signal')
    plt.plot(time_array, wavelet_denoised_signal, color='c', label='Filtered Audio Signal', alpha = 0.5)
    plt.plot(time_array, smooth_env, color='r', label='Envelope', alpha=0.7)
    plt.plot(time_array[peaks_E], smooth_env[peaks_E], 'go', label='Peaks')
    plt.plot(time_array[width], smooth_env[width], 'mo', label='End Point')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title(f'Filtered Audio Waveform and Envelope | Peak at 1 sec:{abs(peak_1):.2f} | Max Peak: {mx_peak_E:.2f}')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_array, audio_data, color = 'b', label = 'Raw Audio Signal')
    plt.plot(time_array, butterworth, color='c', label='IMF)', alpha = 0.6)
    plt.plot(time_array, smooth_env, color='r', label='Envelope', alpha=0.8)
    plt.plot(time_array[peaks_E], smooth_env[peaks_E], 'go', label='Peaks')
    plt.plot(time_array[width], smooth_env[width], 'mo', label='End Point')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title(f'IMF0 and Envelope | Peak at 1 sec:{abs(peak_1):.2f} | Max Peak: {mx_peak_E:.2f}')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    # plt.subplot(3, 1, 3)
    # plt.plot(time_array, imf0, color='m', label='IMF0')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Amplitude') 
    # plt.title('EMD Denoised Waveform')
    # plt.legend(loc="upper right")
    # plt.grid(True)

    ctime = np.linspace(0, len(clipped_E) / SAMPLE_RATE, len(clipped_E))
    
    plt.subplot(3, 1, 3)
    plt.plot(ctime, clipped_E, color='g', label='Clipped Envelope Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title('Denoised Audio Waveform')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Function to plot results when sample is rejected
def plot_rejected(audio_data, emd_denoised_signal, wavelet_denoised_signal, smooth_env, peaks_E, time_array, clipped_E):
    """
    Plots the results when the sample is rejected.

    Parameters:
    - audio_data (ndarray): Recorded audio signal.
    - emd_denoised_signal (ndarray): EMD denoised audio signal.
    - wavelet_denoised_signal (ndarray): Denoised audio signal.
    - smooth_env (ndarray): Smoothed envelope signal.
    - peaks_E (ndarray): Indices of peaks in the envelope.
    - time_array (ndarray): Time array corresponding to the audio signal.
    """
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(time_array, audio_data, color='b', label='Raw Audio Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title('Raw Audio Waveform')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(time_array, emd_denoised_signal, color='c', label='EMD Denoised Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title('EMD Denoised Waveform')
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplot(4, 1, 3)
    ctime = np.arange(0, len(clipped_E)) / SAMPLE_RATE
    plt.plot(ctime, clipped_E, color='g', label='Clipped Envelope Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title('Denoised Audio Waveform')
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(time_array, wavelet_denoised_signal, color='b', label='Filtered Audio Signal')
    plt.plot(time_array, smooth_env, color='r', label='Envelope', alpha=0.9)
    plt.plot(time_array[peaks_E], smooth_env[peaks_E], 'go', label='Peaks')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title(f'Filtered Audio Waveform and Envelope | ⚠️⚠️ SAMPLE REJECTED!!')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# Main function
def main():
    """
    Main function to run the audio processing and analysis.

    It records audio, applies noise reduction and filtering,
    extracts features, checks sample validity, and displays results.
    """
    
    show_countdown()                                                                            
    duration = 3
    audio_data, sample_rate = record_audio(duration)
    dur = len(audio_data) / sample_rate
    time_array = np.linspace(0, dur, len(audio_data))
    
    # Apply EMD for noise reduction
    emd_denoised_signal, imf0 = apply_emd(audio_data)
    
    # Apply Butterworth bandpass filter
    filtered_signal = filter_signal(emd_denoised_signal, LOWCUT, HIGHCUT, ORDER, SAMPLE_RATE)
    
    # Apply wavelet denoising
    wavelet_denoised_signal = wavelet_denoise(filtered_signal)
    
    # Extract envelope and smooth
    envelope = extract_envelope(wavelet_denoised_signal, sample_rate, FL_HZ, RIPPLE_DB)
    smooth_env = smooth_envelope(envelope, WINDOW_SIZE)
    
    # Find peaks in the envelope
    peaks_E = find_peaks_signal(smooth_env)
    mx_peak_E = max(smooth_env[peaks_E])     
    
    # Clip the audio signal based on envelope peaks
    clipped_AS, clipped_E, endpoint = clip_audio(wavelet_denoised_signal, smooth_env, threshold_ratio=0.2)
    peaks_clipped = find_peaks_signal(clipped_E)
    
    # Calculate the peak value at 1 second of envelope
    peak_1 = one_sec_peak(smooth_env, SAMPLE_RATE)
    
    # Calculate RMSE 
    rmse_emd = calculate_rmse(audio_data, emd_denoised_signal)
    rmse_imf0 = calculate_rmse(audio_data, imf0)
    
    snr_emd = calculate_snr(audio_data, emd_denoised_signal)
    plot_results(audio_data, emd_denoised_signal, wavelet_denoised_signal, smooth_env, peaks_E, time_array, mx_peak_E, peak_1,clipped_E, endpoint, imf0)
    
    # Check if the sample is correct based on peak analysis
    # if isCorrect(signal=smooth_env, threshold_ratio=0.4,peaks=peaks_E):
    #     plot_results(audio_data, emd_denoised_signal, wavelet_denoised_signal, smooth_env, peaks_E, time_array, mx_peak_E, peak_1,clipped_E, endpoint, imf0)
    #     plot_psd(audio_data, emd_denoised_signal, SAMPLE_RATE, rmse_emd , 'EMD')
    #     plot_psd(audio_data, imf0, SAMPLE_RATE, rmse_imf0 , 'IMF0')
    # else:
    #     # If sample is rejected, plot with rejection indication
    #     plot_rejected(audio_data,emd_denoised_signal, wavelet_denoised_signal, smooth_env, peaks_E, time_array,clipped_E)
    

    
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
