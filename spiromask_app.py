import flet as ft
import asyncio
from scipy.io.wavfile import write
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="dateutil")


SAMPLE_RATE = 8000

def main(page: ft.Page):
    page.title = 'Spiro Mask'
    page.window_width = 360  # Simulate mobile width
    page.window_height = 640  # Simulate mobile height
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    BG = '#041955'
    FG = '#97b4ff'
    PINK = '#eb06ff'
    
    countdown_label = ft.Text("", color=FG, size=30)
    status_label = ft.Text("", color=FG, size=20)
    peak_label = ft.Text("", color=FG, size=20)
    max_peak_label = ft.Text("", color=FG, size=20)

    start_button = ft.ElevatedButton(
        text="Start",
        bgcolor=PINK,
        color="white",
        on_click=lambda e: asyncio.run(start_process())
    )

    container = ft.Container(
        width=360,
        height=800,
        bgcolor=BG,
        border_radius=20,
        content=ft.Column(
            [
                start_button,
                countdown_label,
                status_label,
                peak_label,
                max_peak_label,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=20,
        ),
    )

    page.add(container)

    async def start_process():
        # Show countdown
        for i in range(5, 0, -1):
            countdown_label.value = f"Starting in {i}..."
            page.update()
            await asyncio.sleep(1)
        
        countdown_label.value = "Recording started!"
        page.update()

        # Record the audio
        audio_data, sr = capture_audio(3)
        status_label.value = "Recording finished. Processing..."
        page.update()

        # Process the audio
        processed_data = process_audio(audio_data, sr)

        # Save the audio file
        current_day = datetime.datetime.now()
        testing_datetime = current_day.strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{testing_datetime}.wav"
        save_file(audio_data, sr, filename)

        # Plot the audio signal
        plot_audio(processed_data)

        # Update labels with peaks
        peak_label.value = f"Peak at 1 sec: {processed_data['peak_at_1sec']:.2f}"
        max_peak_label.value = f"Max Peak: {processed_data['max_peak']:.2f}"
        status_label.value = "Processing finished."
        page.update()

    def capture_audio(duration: int):
        audio_data = sd.rec(int(duration * SAMPLE_RATE), SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        return audio_data.flatten(), SAMPLE_RATE

    def process_audio(audio_data, sr):
        # Apply Butterworth filter
        butter_filt_data = butterworth_filter(audio_data, 100, 2000, 5, SAMPLE_RATE)

        # Extract envelope
        envelope = extract_envelope(butter_filt_data, SAMPLE_RATE, 10, 10.0)

        # Calculate peaks
        peak_at_1sec = envelope[SAMPLE_RATE]
        max_peak_time, max_peak = np.argmax(envelope), np.max(envelope)

        return {
            "original": audio_data,
            "filtered": butter_filt_data,
            "envelope": envelope,
            "peak_at_1sec": peak_at_1sec,
            "max_peak": max_peak,
            "max_peak_time": max_peak_time
        }

    def save_file(audio_data, sr, filename):
        write(filename, sr, audio_data)

    def plot_audio(data):
        time = np.linspace(0, len(data["original"]) / SAMPLE_RATE, len(data["original"]))
        plt.figure(figsize=(10, 6))
        plt.plot(time, data["original"], label='Original Signal')
        plt.plot(time, data["filtered"], label='Filtered Signal', alpha=0.5)
        plt.plot(time, data["envelope"], label='Envelope Signal', alpha=0.5)
        plt.plot(time[data["max_peak_time"]], data["envelope"][data["max_peak_time"]], 'ro', label='Max Peak')
        plt.plot(time[SAMPLE_RATE], data["envelope"][SAMPLE_RATE], 'go', label='Peak at 1 sec')
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title("Audio Signal Processing")
        plt.grid()
        plt.show()

    def butterworth_filter(signal, lowcut, highcut, order, sr):
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = ss.butter(order, [low, high], btype='band')
        y = ss.lfilter(b, a, signal)
        return y

    def extract_envelope(signal, sr, fL_hz, ripple_db):
        nyq_rate = 0.5 * sr
        width = 1.0 / nyq_rate

        analytic_signal = ss.hilbert(signal)
        envelope = np.abs(analytic_signal)

        N, beta = ss.kaiserord(ripple_db, width)
        taps = ss.firwin(N, fL_hz / nyq_rate, window=('kaiser', beta))
        filtered_envelope = ss.filtfilt(taps, 1, envelope)
        return filtered_envelope

ft.app(target=main)
