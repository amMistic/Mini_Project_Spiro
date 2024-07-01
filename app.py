import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time

from scipy.signal import butter, lfilter, hilbert, kaiserord, filtfilt, firwin, find_peaks 
from kivy.app import App
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import pywt
from PyEMD import EMD

# Global Constants
SAMPLE_RATE = 4000
LOWCUT = 100
HIGHCUT = 1900
ORDER = 5
WINDOW_SIZE = 20
FL_HZ = 10
RIPPLE_DB = 10.0

class CountdownPopup(Popup):
    pass

class ResultPopup(Popup):
    pass

class SpiroMaskMob(App):

    def build(self):
        self.layout = self.root
        self.label = self.root.ids.instruction_label
        self.start_button = self.root.ids.start_button

    def start_recording(self, instance):
        self.show_countdown()
        self.label.text = "Recording..."

    def show_countdown(self):
        self.popup = CountdownPopup()
        self.popup.open()
        
        self.countdown = 5
        Clock.schedule_interval(self.update_countdown, 1)

    def update_countdown(self, dt):
        if self.countdown > 0:
            self.popup.ids.countdown_label.text = f"Get ready! Starting in {self.countdown}..."
            self.countdown -= 1
        else:
            self.popup.dismiss()
            self.record_audio()
            return False

    def record_audio(self):
        duration = 3
        audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        audio_data = audio_data.flatten()
        self.label.text = 'Processing...'
        Clock.schedule_once(lambda dt: self.process_audio(audio_data), 0)

    def process_audio(self, audio_data):
        dur = len(audio_data) / SAMPLE_RATE
        time_array = np.linspace(0, dur, len(audio_data))

        emd = EMD()
        IMFs = emd.emd(audio_data)
        iceeemdan_denoised_signal = audio_data - IMFs[0]

        soft_thresholded_signal = self.soft_thresholding(iceeemdan_denoised_signal)
        filtered_signal = self.filter_signal(soft_thresholded_signal, LOWCUT, HIGHCUT, ORDER, SAMPLE_RATE)
        envelope = self.extract_envelope(filtered_signal, SAMPLE_RATE, FL_HZ, RIPPLE_DB)
        peaks_E = self.find_peaks_signal(envelope)
        clipped_AS, clipped_E = self.clip_audio(filtered_signal, envelope, peaks_E)
        peaks_clipped = self.find_peaks_signal(clipped_E)
        peaks_As = np.max(np.abs(clipped_E))
        peak_1 = self.one_sec_peak(envelope, SAMPLE_RATE)
        is_correct = self.isCorrect(peaks_clipped, clipped_E)
        
        self.show_audio_waveform(time_array, audio_data, filtered_signal, envelope, peaks_E, peaks_As, peak_1, is_correct)
        self.label.text = "Recording and processing completed!"

    def soft_thresholding(self, signal):
        coeffs = pywt.wavedec(signal, 'db1', level=6)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
        denoised_signal = pywt.waverec(denoised_coeffs, 'db1')
        return denoised_signal

    def filter_signal(self, signal, lowcut, highcut, order, fs):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, signal)
        return y

    def extract_envelope(self, signal, fs, fL_hz, ripple_db):
        nyq_rate = 0.5 * fs
        width = 1.0 / nyq_rate

        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)

        N, beta = kaiserord(ripple_db, width)
        taps = firwin(N, fL_hz / nyq_rate, window=('kaiser', beta))
        filtered_envelope = filtfilt(taps, 1, envelope)
        return filtered_envelope

    def find_peaks_signal(self, signal):
        peaks, _ = find_peaks(signal, distance=15)
        return peaks

    def one_sec_peak(self, signal, sr):
        return abs(signal[sr])

    def clip_audio(self, audio_sig, envelope_signal, peaks):
        width = 0
        for i in range(len(peaks)):
            if envelope_signal[peaks[i]] <= 200:
                diff = peaks[i] - peaks[i-1] // 2
                width = peaks[i] - diff
                break
        
        clipped_audio_signal = audio_sig[:width]
        clipped_envelope_signal = envelope_signal[:width]
        return clipped_audio_signal, clipped_envelope_signal

    def isCorrect(self, peaks, signal):
        maxima = 0
        for right in range(len(peaks) - 1):
            # finding first maxima
            if maxima == 0 and signal[peaks[right + 1]] < signal[peaks[right]]:
                maxima = 1
                maxx = signal[peaks[right]]
                continue
            
            # once if achieved the maxima it should be decreasing
            if maxima == 1 and signal[peaks[right]] > 0.4 * maxx:                
                return False
        
        # accepted!
        return True

    def show_audio_waveform(self, time_array, audio_data, filtered_signal, envelope, peaks_E, peaks_As, peak_1, is_correct):
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(time_array, filtered_signal, color='b', label='Filtered Audio Signal')
        ax.plot(time_array, envelope, color='r', label='Envelope', alpha=0.9)
        ax.plot(time_array[peaks_E], envelope[peaks_E], 'go', label='Peaks')

        if is_correct:
            ax.set_title(f'Filtered Audio Waveform and Envelope | Peak at 1 sec: {peak_1:.2f} | Max Peak: {peaks_As:.2f}')
        else:
            ax.set_title('Filtered Audio Waveform and Envelope | ⚠️⚠️ SAMPLE REJECTED!!')

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc="upper right")
        ax.grid(True)

        plot_widget = FigureCanvasKivyAgg(fig)
        popup = ResultPopup(content=plot_widget)
        popup.open()

if __name__ == '__main__':
    SpiroMaskMob().run()
