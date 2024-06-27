# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
from scipy.signal import butter, lfilter, hilbert, kaiserord, filtfilt, firwin, find_peaks
from PyEMD import EMD
import pywt

# Kivy imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg as FCKA
from kivy.clock import Clock

# Parameters for audio processing
SAMPLE_RATE = 4000
LOWCUT = 100    
HIGHCUT = 1900
ORDER = 5
WINDOW_SIZE = 20
FL_HZ = 10
RIPPLE_DB = 10.0

# Function to record audio 
def record_audio(duration):
    print("Recording...")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # Wait for recording to complete
    return audio_data.flatten(), SAMPLE_RATE

# Function to display countdown window
def show_countdown():
    countdown_label.text = "Get ready!\nStarting in 5..."
    Clock.schedule_once(lambda dt: update_countdown(4), 1)

def update_countdown(i):
    if i > 0:
        countdown_label.text = f"Get ready!\nStarting in {i}..."
        Clock.schedule_once(lambda dt: update_countdown(i - 1), 1)
    else:
        countdown_label.text = "Recording..."
        Clock.schedule_once(record_audio_callback, 1)

def record_audio_callback(dt):
    global audio_data, sample_rate
    audio_data, sample_rate = record_audio(duration=3)
    process_audio()

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
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# Function to find peaks in the signal
def find_peaks_signal(signal):
    peaks, _ = find_peaks(signal, distance=30)
    return peaks

# Function to clip the audio signal
def clip_audio(audio_sig, envelope_signal, peaks):
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
    return signal[sr]

# Function to apply EMD for noise reduction
def apply_emd(signal):
    emd = EMD()
    IMFs = emd.emd(signal)
    # Assuming that the first IMF contains the most noise and the rest contain the useful signal
    denoised_signal = signal - IMFs[0]
    return denoised_signal

# Function to apply wavelet denoising
def wavelet_denoise(signal):
    coeffs = pywt.wavedec(signal, 'db1', level=6)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(denoised_coeffs, 'db1')
    return denoised_signal

# Function to process audio
def process_audio():
    global audio_data, sample_rate

    dur = len(audio_data) / sample_rate
    time_array = np.linspace(0, dur, len(audio_data))
    
    # Apply EMD for noise reduction
    emd_denoised_signal = apply_emd(audio_data)
    
    # Apply Butterworth bandpass filter
    filtered_signal = filter_signal(emd_denoised_signal, LOWCUT, HIGHCUT, ORDER, SAMPLE_RATE)
    
    # Apply wavelet denoising
    wavelet_denoised_signal = wavelet_denoise(filtered_signal)
    
    envelope = extract_envelope(wavelet_denoised_signal, sample_rate, FL_HZ, RIPPLE_DB)
    smooth_env = smooth_envelope(envelope, WINDOW_SIZE)
    
    # Find the peaks
    peaks_E = find_peaks_signal(smooth_env)
    mx_peak_E = max(smooth_env[peaks_E])     
    
    # Clip the audio signal
    clipped_AS, clipped_E = clip_audio(wavelet_denoised_signal, smooth_env, peaks_E)
    peaks_clipped = find_peaks_signal(clipped_E)
    
    # Peak at 1 second of envelope 
    peak_1 = one_sec_peak(smooth_env, SAMPLE_RATE)
        
    # Plot the original and denoised audio waveform
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_array, audio_data, color='b', label='Raw Audio Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title('Raw Audio Waveform')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_array, wavelet_denoised_signal, color='g', label='Denoised Audio Signal')
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

# Kivy App Class
class AudioProcessingApp(App):

    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.button = Button(text='Start Recording')
        self.button.bind(on_press=self.start_recording)
        self.layout.add_widget(self.button)
        self.countdown_label = Label(text='')
        self.layout.add_widget(self.countdown_label)
        return self.layout

    def start_recording(self, instance):
        show_countdown()

if __name__ == '__main__':
    AudioProcessingApp().run()
