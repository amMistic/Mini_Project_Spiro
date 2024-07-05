import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, lfilter
import librosa
import sounddevice as sd
import pywt
from PyEMD import CEEMDAN
from PyEMD import EMD 

# PARAMETERS
SAMPLE_RATE = 4000
LOWER = 1
HIGH = 1999


# capture the audio
def capture_audio(duration: int):
    print("Recording...")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    return audio_data.flatten(), SAMPLE_RATE

# load the audio file
def load_file(filepath: str, sr = 44100):
    
    audio_data, sr = librosa.load(filepath)
    return audio_data.flatten(), sr

# Apply the butterworth bandpass
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def ICEEMDAN(signal:list) :
    iceemdan = CEEMDAN()
    IMFs = iceemdan(signal)
    denoised_signal = signal = IMFs[0]
    return denoised_signal, IMFs[0]

# Plot Signal 
def plot_signal(audio_data: list, filtered_audio: list, imf0: list, denoised_sigal: list, sr: int) -> None : 
    
    # time axis
    time = np.linspace(0 , len(audio_data) / sr, len(audio_data))
    
    # Plot the waveform
    #---------------------------------------- Original Audio Signal Waveform --------------------
    plt.figure(figsize=(18, 9))
    plt.subplot(4, 2, 1)
    plt.plot(time, audio_data,label = 'Original Audio')
    plt.title('Original Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Plot the spectrogram
    plt.subplot(4, 2, 2)
    frequencies, times, Sxx = spectrogram(audio_data, sr)
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.title('Spectrogram')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.grid(True)

    # ------------------------------------------------ Butter Worth -------------------------------------
    plt.subplot(4, 2, 3)
    plt.plot(time, filtered_audio,label = 'ButterWorth Audio')
    plt.title('Butter Worth Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 4)
    frequencies, times, Sxx = spectrogram(filtered_audio, sr)
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.title('Spectrogram Butter Worth Audio')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.tight_layout()
    
    # ------------------------------------------------ IMF0 -----------------------------------------------
    plt.subplot(4, 2, 5)
    plt.plot(time, imf0,label = 'IMF0 Audio')
    plt.title('IMF0 Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 6)
    frequencies, times, Sxx = spectrogram(imf0, sr)
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.title('Spectrogram imf0 Audio')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.tight_layout()
    
    # ---------------------------------- Denoised Signal -----------------------------------------------
    plt.subplot(4, 2, 7)
    plt.plot(time, denoised_sigal, label = 'Denoised Signal Audio')
    plt.title('denoised_sigal Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 8)
    frequencies, times, Sxx = spectrogram(denoised_sigal, sr)
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.title('Spectrogram denoised_sigal Audio')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.tight_layout()
    plt.show()

    
def main():
    file_path = 'Audio_sampdles\\Breath_audio_1.mp3'
    audio_data, sr = capture_audio(duration=3)
    
    # filtered the audio signal 
    filtered_signal = bandpass_filter(audio_data, LOWER, HIGH, SAMPLE_RATE)
    
    # ICEEMDAN filter
    ICE_signal, imf0 = ICEEMDAN(audio_data)
        
    plot_signal(audio_data, filtered_signal, imf0, ICE_signal, sr)

if __name__ == '__main__':
    main()