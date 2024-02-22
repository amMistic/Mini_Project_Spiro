import librosa as li
import numpy as np
import matplotlib.pyplot as plt

# construction
class breath_Sample:
    def __init__(self,filename) -> None:
        self.filename = filename
        self.audio = None
        self.sampling_rate = None
        
    def load_audio(self):
        self.audio , self.sampling_rate = li.load(self.filename,sr=None)
    
    def get_sampling_rate(self):
        if self.sampling_rate is None:
            self.load_audio()
        return self.sampling_rate
    
    def visual_FFT(self):
        FFT = np.fft.fft(self.audio)
        frequency = np.fft.fftfreq(len(self.audio),1 / self.sampling_rate)
        
        plt.figure()
        plt.plot(frequency[:len(frequency)//2], np.abs(FFT)[:len(frequency)//2])
        plt.xlabel('Frequency (Hz) ')
        plt.ylabel('Magnitude')
        plt.title('Frequency Content')
        plt.grid(True)
        plt.show()        
    
# Execution
filename = 'Audio_samples\Breath Audio 3.mp3'
sampler = breath_Sample(filename=filename)
sampling_rate = sampler.get_sampling_rate()
print("Sampling Rate:",sampling_rate)


# OUTPUT:
# The sampling rate of 1st Audio sample : 44100 Hz 


