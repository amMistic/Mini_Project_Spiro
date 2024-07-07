import flet as ft
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter, hilbert, firwin, filtfilt
from PyEMD import EMD
import pywt
from time import sleep
from io import BytesIO
import base64

# Constants
SAMPLE_RATE = 8000
DURATION = 3  # seconds
LOWCUT = 100
HIGHCUT = 2000
ORDER = 5
WINDOW_SIZE = 100
FL_HZ = 10
RIPPLE_DB = 10.0

# Function to record audio
def record_audio():
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    return audio_data.flatten()

# Function to apply Butterworth filter
def butterworth_filter(signal, lowcut, highcut, order, sr):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

# Function to apply EMD filter
def emd_filter(signal):
    emd = EMD()
    imfs = emd.emd(signal)
    denoised_signal = signal - imfs[0]
    return denoised_signal, imfs[0]

# Function to perform wavelet denoising
def wavelet_denoising(signal):
    coeffs = pywt.wavedec(signal, 'db1', level=6)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, 'db1')

# Function to extract envelope
def extract_envelope(signal, sr, fl_hz, ripple_db):
    nyq_rate = 0.5 * sr
    width = 1.0 / nyq_rate
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    n, beta = ss.kaiserord(ripple_db, width)
    taps = firwin(n, fl_hz / nyq_rate, window=('kaiser', beta))
    return filtfilt(taps, 1, envelope)

# Function to smooth envelope
def smooth_envelope(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# Function to find maxima
def maxima(signal):
    index = np.argmax(signal)
    return index, signal[index]

# Function to plot signals and return as image bytes
def plot_signals(time, signal, denoised_signal, envelope, max_peak_time, peak_at_1s):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    axs[0].plot(time, signal, label='Original Signal')
    axs[0].plot(time, denoised_signal, label='Denoised Signal', color='g', alpha=0.5)
    axs[0].set_title('Original vs Denoised Signal')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    axs[1].plot(time, denoised_signal, label='Denoised Signal', color='g', alpha=0.5)
    axs[1].plot(time, envelope, label='Envelope Signal', color='r', alpha=0.7)
    axs[1].plot(time[1 * SAMPLE_RATE], envelope[1 * SAMPLE_RATE], 'go', label='Peak at 1 sec')
    axs[1].plot(time[max_peak_time], envelope[max_peak_time], 'mo', label='Max Peak')
    axs[1].set_title(f'Max Peak: {envelope[max_peak_time]:.2f} | Peak at 1 sec: {envelope[1 * SAMPLE_RATE]:.2f}')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    axs[2].plot(time, envelope, label='Envelope Signal', color='m', alpha=0.7)
    axs[2].set_title('Extracted Envelope')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Amplitude')
    axs[2].legend(loc='upper right')
    axs[2].grid(True)

    plt.tight_layout(pad=3.0)
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    plt.close(fig)
    
    return img_bytes

# Main function to build the app
def main(page: ft.Page):
    page.title = "SpiroMask.ai"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    
    def show_countdown(label):
        for i in range(5, 0, -1):
            label.value = f"GET READY!!\nStarting in {i}..."
            page.update()
            sleep(1)
        label.value = "Recording..."
        page.update()

    def record_and_process(e):
        show_countdown(status_text)
        audio_data = record_audio()
        time = np.linspace(0, len(audio_data) / SAMPLE_RATE, len(audio_data))
        filtered_data = butterworth_filter(audio_data, LOWCUT, HIGHCUT, ORDER, SAMPLE_RATE)
        denoised_data, imf0 = emd_filter(filtered_data)
        denoised_data = wavelet_denoising(denoised_data)
        envelope = extract_envelope(denoised_data, SAMPLE_RATE, FL_HZ, RIPPLE_DB)
        smoothed_envelope = smooth_envelope(envelope, WINDOW_SIZE)
        max_peak_time, max_peak = maxima(smoothed_envelope)
        peak_at_1s = smoothed_envelope[SAMPLE_RATE]
        img_bytes = plot_signals(time, audio_data, denoised_data, smoothed_envelope, max_peak_time, peak_at_1s)
        
        # Convert image bytes to base64 for displaying in Flet
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Update the page with results
        page.controls.clear()
        page.add(
            ft.Text("SpiroMask.ai", size=30, weight="bold"),
            ft.Container(
                ft.Image(src_base64=img_base64, fit=ft.ImageFit.CONTAIN),
                width=page.window_width * 0.8,
                height=page.window_height * 0.6
            ),
            ft.Text(f"Max Peak: {(max_peak*100):.2f}", size=20),
            ft.Text(f"Peak at 1 sec: {(peak_at_1s*100):.2f}", size=20)
        )
        page.update()

    status_text = ft.Text("SpiroMask.ai", size=30, weight="bold", text_align=ft.TextAlign.CENTER)
    start_button = ft.ElevatedButton("Start", on_click=record_and_process)
    
    page.add(
        ft.Container(
            content=status_text,
            alignment=ft.alignment.center,
            expand=True
        )
    )
    page.add(
        ft.Container(
            content=start_button,
            alignment=ft.alignment.center,
            expand=True
        )
    )

ft.app(target=main)
