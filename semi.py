import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import tkinter as tk
import time
from scipy.signal import butter, lfilter, hilbert, kaiserord, filtfilt, firwin, find_peaks
from PyEMD import EMD
import pywt

# Parameters
SAMPLE_RATE = 4000
LOWCUT = 100    
HIGHCUT = 1900
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
def clip_audio(audio_sig, envelope_signal, peaks):
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
    width = 0
    for i in range(len(peaks)):
        if envelope_signal[peaks[i]] <= 200:
            diff = peaks[i] - peaks[i-1] // 2
            width = peaks[i] - diff
            break
    
    clipped_audio_signal = audio_sig[:width]
    clipped_envelope_signal = envelope_signal[:width]
    return clipped_audio_signal, clipped_envelope_signal

# Function to calculate the peak at 1 second
def one_sec_peak(signal, sr):
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
    # Assuming that the first IMF contains the most noise and the rest contain the useful signal
    denoised_signal = signal - IMFs[0]
    return denoised_signal

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
def isCorrect(peaks, signal):
    """
    Checks if the sample is correct based on peak analysis.

    Parameters:
    - peaks (ndarray): Indices of peaks in the signal.
    - signal (ndarray): Signal to analyze.
    - threshold (float): Threshold value for peak analysis.

    Returns:
    - bool: True if the sample is correct, False otherwise.
    """
    maxima = 0
    for right in range(len(peaks) - 1):
        if maxima == 0 and signal[peaks[right + 1]] < signal[peaks[right]] and signal[peaks[right]] > 1000:          
            maxima = 1
            maxx = signal[peaks[right]]
            continue
        if maxima == 1 and signal[peaks[right]] > 0.4 * maxx:                
            return False
    return True
     
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
    emd_denoised_signal = apply_emd(audio_data)
    
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
    clipped_AS, clipped_E = clip_audio(wavelet_denoised_signal, smooth_env, peaks_E)
    peaks_clipped = find_peaks_signal(clipped_E)
    
    # Calculate the peak value at 1 second of envelope
    peak_1 = one_sec_peak(smooth_env, SAMPLE_RATE)
    
    # Check if the sample is correct based on peak analysis
    if isCorrect(peaks=peaks_clipped, signal=clipped_AS):
        # If sample is correct, plot the results
        plot_results(audio_data, wavelet_denoised_signal, smooth_env, peaks_E, time_array, mx_peak_E, peak_1,clipped_E)
    else:
        # If sample is rejected, plot with rejection indication
        plot_rejected(audio_data, wavelet_denoised_signal, smooth_env, peaks_E, time_array,clipped_E)

# Function to plot results when sample is accepted
def plot_results(audio_data, wavelet_denoised_signal, smooth_env, peaks_E, time_array, mx_peak_E, peak_1, clipped_E):
    """
    Plots the results when the sample is accepted.

    Parameters:
    - audio_data (ndarray): Recorded audio signal.
    - wavelet_denoised_signal (ndarray): Denoised audio signal.
    - smooth_env (ndarray): Smoothed envelope signal.
    - peaks_E (ndarray): Indices of peaks in the envelope.
    - time_array (ndarray): Time array corresponding to the audio signal.
    - mx_peak_E (float): Maximum peak value in the envelope.
    - peak_1 (float): Peak value at 1 second in the envelope.
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_array, audio_data, color='b', label='Raw Audio Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title('Raw Audio Waveform')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    ctime = np.arange(0, len(clipped_E)) / SAMPLE_RATE
    plt.plot(ctime, clipped_E, color='g', label='Clipped Audio Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title('Denoised Audio Waveform')
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_array, wavelet_denoised_signal, color='b', label='Filtered Audio Signal')
    plt.plot(time_array, smooth_env, color='r', label='Envelope', alpha=0.9)
    plt.plot(time_array[peaks_E], smooth_env[peaks_E], 'go', label='Peaks')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title(f'Filtered Audio Waveform and Envelope | Peak at 1 sec:{abs(peak_1):.2f} | Max Peak: {mx_peak_E:.2f}')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Function to plot results when sample is rejected
def plot_rejected(audio_data, wavelet_denoised_signal, smooth_env, peaks_E, time_array,clipped_E):
    """
    Plots the results when the sample is rejected.

    Parameters:
    - audio_data (ndarray): Recorded audio signal.
    - wavelet_denoised_signal (ndarray): Denoised audio signal.
    - smooth_env (ndarray): Smoothed envelope signal.
    - peaks_E (ndarray): Indices of peaks in the envelope.
    - time_array (ndarray): Time array corresponding to the audio signal.
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_array, audio_data, color='b', label='Raw Audio Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title('Raw Audio Waveform')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    ctime = np.arange(0, len(clipped_E)) / SAMPLE_RATE
    plt.plot(ctime, clipped_E, color='g', label='Clipped Audio Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title('Denoised Audio Waveform')
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplot(3, 1, 3)
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
