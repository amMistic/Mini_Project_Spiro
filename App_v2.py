# MAIN ANDROID APP

# TODO:
'''


'''


import sounddevice as sd
import scipy.signal as ss
import numpy as np
from flet import *
from time import sleep
import numpy as np
from PyEMD import EMD
import pywt

# Constants
SAMPLE_RATE = 4000
DURATION = 3  # seconds
LOWCUT = 100
HIGHCUT = 1999
ORDER = 5
WINDOW_SIZE = 100
FL_HZ = 10
RIPPLE_DB = 10.0

# Function to record audio
def record_audio(DURATION: int):
    print("Recording...")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    audio_data = audio_data.flatten()  # Flatten the audio array
    print(f"Audio_data: {audio_data[:10]}")
    return audio_data, SAMPLE_RATE

# Function to apply Butterworth filter
def butterworth_filter(signal, lowcut, highcut, order, sr):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = ss.butter(order, [low, high], btype='band')
    return ss.lfilter(b, a, signal)

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
    analytic_signal = ss.hilbert(signal)
    envelope = np.abs(analytic_signal)
    n, beta = ss.kaiserord(ripple_db, width)
    taps = ss.firwin(n, fl_hz / nyq_rate, window=('kaiser', beta))
    return ss.filtfilt(taps, 1, envelope)

# Function to smooth envelope
def smooth_envelope(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# Function to find maxima
def maxima(signal):
    index = np.argmax(signal)
    return signal[index]

# Main function to build the app
def main(page: Page):
    page.title = "SpiroMask.ai"
    page.horizontal_alignment = CrossAxisAlignment.CENTER
    page.vertical_alignment = MainAxisAlignment.CENTER
    
    status_text = Text("SpiroMask.ai", size=30, weight="bold", text_align=TextAlign.CENTER)
    start_button = ElevatedButton("Start", on_click=lambda e: record_and_process())
    
    def show_countdown():
        start_button.visible = False
        page.update()
        for i in range(5, 0, -1):
            status_text.value = f"GET READY!!\nStarting in {i}..."
            page.update()
            sleep(1)
        status_text.value = "Recording..."
        page.update()

    def record_and_process():
        show_countdown()
        audio_data, sample_rate = record_audio(DURATION)
        time = np.linspace(0, len(audio_data) / SAMPLE_RATE, len(audio_data))
        
        # Apply the Butterworth Filter
        bw_filter = butterworth_filter(audio_data, LOWCUT, HIGHCUT, ORDER, SAMPLE_RATE)
        
        # EMD Filter
        denoised_signal, imf0 = emd_filter(bw_filter)
        filtered_audio = wavelet_denoising(denoised_signal)
        
        # Extract the envelope
        envelope = extract_envelope(filtered_audio, SAMPLE_RATE, FL_HZ, RIPPLE_DB)
        smoothen_envelope = smooth_envelope(envelope, WINDOW_SIZE)
        
        # Calculate Max peaks and Peak at one sec
        mx_peak = maxima(envelope)
        peak_1s = envelope[sample_rate]
        
        # Debug: Print a few samples of the audio data and time
        print(f"Audio Data (first 10 samples): {smoothen_envelope[:10]}")
        print(f"Time Data (first 10 samples): {time[:10]}")
        
        # Convert the audio data into a suitable format for plotting
        smooth_data_points = [LineChartDataPoint(t, v) for t, v in zip(time, smoothen_envelope)]
        envelope_data_points = [LineChartDataPoint(t, v) for t, v in zip(time, envelope)]
        
        # Converting the recorded audio signal into data points for plotting
        signal_data = [
            LineChartData(
                data_points=envelope_data_points,
                stroke_width=2,
                color=colors.BLUE_100,
                curved=True,
                stroke_cap_round=True,
                below_line_gradient=LinearGradient(
                    begin=alignment.top_center,
                    end=alignment.bottom_center,
                    colors=[
                        colors.with_opacity(0.75, colors.BLUE_100),
                        "transparent",
                    ],
                ),
            ),
        ]
        
        # Prepare the chart for plotting
        chart = LineChart(
            data_series=signal_data,
            border=Border(
                bottom=BorderSide(4, colors.with_opacity(0.5, colors.ON_SURFACE))
            ),
            left_axis=ChartAxis(
                labels=[
                    ChartAxisLabel(
                        value=min(smoothen_envelope),
                        label=Text(f"{min(smoothen_envelope):.2f}", size=14, weight=FontWeight.BOLD),
                    ),
                    ChartAxisLabel(
                        value=(min(smoothen_envelope) + max(smoothen_envelope)) / 2,
                        label=Text(f"{(min(smoothen_envelope) + max(smoothen_envelope)) / 2:.2f}", size=14, weight=FontWeight.BOLD),
                    ),
                    ChartAxisLabel(
                        value=max(smoothen_envelope),
                        label=Text(f"{max(smoothen_envelope):.2f}", size=14, weight=FontWeight.BOLD),
                    ),
                ],
                labels_size=40,
            ),
            bottom_axis=ChartAxis(
                labels=[
                    ChartAxisLabel(
                        value=sample_rate * 1,
                        label=Container(
                            Text(
                                "1s",
                                size=16,
                                weight=FontWeight.BOLD,
                                color=colors.with_opacity(0.5, colors.ON_SURFACE),
                            ),
                            margin=margin.only(top=10),
                        ),
                    ),
                    ChartAxisLabel(
                        value=sample_rate * 2,
                        label=Container(
                            Text(
                                "2s",
                                size=16,
                                weight=FontWeight.BOLD,
                                color=colors.with_opacity(0.5, colors.ON_SURFACE),
                            ),
                            margin=margin.only(top=10),
                        ),
                    ),
                    ChartAxisLabel(
                        value=sample_rate * 3,
                        label=Container(
                            Text(
                                "3s",
                                size=16,
                                weight=FontWeight.BOLD,
                                color=colors.with_opacity(0.5, colors.ON_SURFACE),
                            ),
                            margin=margin.only(top=10),
                        ),
                    ),
                ],
                labels_size=32,
            ),
            tooltip_bgcolor=colors.with_opacity(0.8, colors.BLUE_GREY),
            min_y=min(smoothen_envelope),
            max_y=max(smoothen_envelope),
            min_x=0,
            max_x=DURATION * sample_rate,
            expand=False,
        )

        # Clear previous controls and add the chart
        page.controls.clear()
        page.add(
            Text("SpiroMask.ai", size=30, weight="bold"),
            chart,
            Row(
                alignment=MainAxisAlignment.SPACE_BETWEEN,
                controls=[
                    Text(f"Max peaks: {mx_peak:.2f}", size=16, weight=FontWeight.BOLD),
                    Text(f"Peak at 1s: {peak_1s:.2f}", size=16, weight=FontWeight.BOLD),
                ],
            ),
            Text(f"Recording complete. See the plot above.", size=20),
            ElevatedButton("Record Again", on_click=lambda e: reset_page())
        )
        start_button.visible = True
        page.update()

    def reset_page():
        # Reset the page to initial state
        page.controls.clear()
        status_text.value = 'SpiroMask.ai'
        page.add(status_text, start_button)
        page.update()

    # Initial page setup
    page.add(status_text, start_button)

app(target=main)
