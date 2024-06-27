import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import tkinter as tk
import time
from scipy.signal import butter, lfilter, hilbert, kaiserord, filtfilt, firwin, find_peaks

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
    """
    Objective: Record audio from the microphone for a specified duration.

    Input:
        - duration (float): Duration of the audio recording in seconds.

    Output:
        - audio_data (numpy array): The recorded audio data.
        - SAMPLE_RATE (int): The sample rate of the recorded audio.
    """
    print("Recording...")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # Wait for recording to complete
    return audio_data.flatten(), SAMPLE_RATE

# Function to display countdown window
def show_countdown():
    """
    Objective: Display a countdown window to indicate when the recording will start.

    Input: 
        None.

    Output: 
        None.
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
    Objective: Create a Butterworth bandpass filter.

    Input:
        - lowcut (int): The low cutoff frequency.
        - highcut (int): The high cutoff frequency.
        - fs (int): The sample rate.
        - order (int): The order of the filter.

    Output:
        - b, a (numpy arrays): Filter coefficients.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to filter the audio data
def filter_signal(signal, lowcut, highcut, order, fs):
    """
    Objective: Filter the audio data using a Butterworth bandpass filter.

    Input:
        - signal (numpy array): The input audio signal.
        - lowcut (int): The low cutoff frequency.
        - highcut (int): The high cutoff frequency.
        - order (int): The order of the filter.
        - fs (int): The sample rate.

    Output:
        - y (numpy array): The filtered audio signal.
    """
    b, a = butter_bp(lowcut, highcut, fs, order)
    y = lfilter(b, a, signal)
    return y

# Function to extract the envelope of the signal
def extract_envelope(signal, fs, fL_hz, ripple_db):
    """
    Objective: Extract the envelope of the filtered audio signal.

    Input:
        - signal (numpy array): The filtered audio signal.
        - fs (int): The sample rate.
        - fL_hz (int): Lowpass filter cutoff frequency.
        - ripple_db (float): The maximum ripple in the passband.

    Output:
        - filtered_envelope (numpy array): The extracted envelope of the audio signal.
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
    Objective: Smooth the extracted envelope using a moving average filter.

    Input:
        - signal (numpy array): The extracted envelope of the audio signal.
        - window_size (int): The size of the moving average window.

    Output:
        - smoothed_signal (numpy array): The smoothed envelope of the audio signal.
    """
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# Function to find peaks in the signal
def find_peaks_signal(signal):
    """
    Objective: Find peaks in the envelope of the audio signal.

    Input:
        - signal (numpy array): The envelope of the audio signal.

    Output:
        - peaks (numpy array): Indices of the peaks in the signal.
    """
    peaks, _ = find_peaks(signal, distance=30)
    return peaks

# Function to clip the audio signal
def clip_audio(audio_sig, envelope_signal, peaks):
    """
    Objective: Clip the audio signal based on the envelope and peaks.

    Input:
        - audio_sig (numpy array): The input audio signal.
        - envelope_signal (numpy array): The extracted envelope of the audio signal.
        - peaks (numpy array): Indices of the peaks in the envelope signal.

    Output:
        - clipped_audio_signal (numpy array): The clipped audio signal.
        - clipped_envelope_signal (numpy array): The clipped envelope signal.
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
    Objective: Calculate the peak amplitude of the signal at 1 second.

    Input:
        - signal (numpy array): The input audio signal.
        - sr (int): The sample rate of the audio signal.

    Output:
        - peak (float): The peak amplitude at 1 second.
    """
    return signal[sr] 

# Main function
def main():
    """
    Objective: Main function to execute the audio recording, processing, and plotting.

    Input:
        None.

    Output:
        None.
    """
    # Show countdown window
    show_countdown()                                                                              
    
    # Record audio for 3 seconds
    duration = 3
    audio_data, sample_rate = record_audio(duration)

    # Calculate the duration of the audio file
    dur = len(audio_data) / sample_rate

    # Create time array for x-axis
    time_array = np.linspace(0, dur, len(audio_data))
    
    # Filter the audio signal
    filtered_signal = filter_signal(audio_data, LOWCUT, HIGHCUT, ORDER, SAMPLE_RATE)
    envelope = extract_envelope(filtered_signal, sample_rate, FL_HZ, RIPPLE_DB)
    
    # smooth the filter audio and envelope
    # smooth_fs = smooth_envelope(filtered_signal, WINDOW_SIZE)
    smooth_env = smooth_envelope(envelope, WINDOW_SIZE)
    
    # Find the peaks
    peaks_E = find_peaks_signal(smooth_env)
    mx_peak_E = max(smooth_env[peaks_E])     
    
    # Clip the audio signal
    clipped_AS, clipped_E = clip_audio(filtered_signal, smooth_env, peaks_E)
    peaks_clipped = find_peaks_signal(clipped_E)
    
    # Peak at 1 second of envelope 
    peak_1 = one_sec_peak(smooth_env, SAMPLE_RATE)
        
    # Plot the original audio waveform
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_array, filtered_signal, color='b', label='Filtered Audio Signal')
    plt.plot(time_array, smooth_env, color='r', label='Envelope', alpha=0.9)
    plt.plot(time_array[peaks_E], smooth_env[peaks_E], 'go', label='Peaks')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title(f'Original Audio Waveform and Envelope | Peak at 1 sec:{abs(peak_1):.2f} | Max Peak: {mx_peak_E:.2f}')
    plt.legend(loc="upper right")
    plt.grid(True)

    # Plot the clipped audio waveform
    ctime = np.arange(0, len(clipped_E)) / SAMPLE_RATE
    
    plt.subplot(2, 1, 2)
    plt.plot(ctime, clipped_AS, color='b', label='Clipped Audio Signal')
    plt.plot(ctime, clipped_E, color='r', label='Clipped Envelope', alpha=0.9)
    plt.plot(ctime[peaks_clipped], clipped_E[peaks_clipped], 'go', label='Peaks')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(f'Clipped Audio Waveform and Envelope')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
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
        
            
#TODO:
# 1. Clip Audio Signal --> Done
# 2. Find the Peak  --> Done
# 3. Find the Width --> Done
# 4. Find the Peak at 1 sec --> Done
