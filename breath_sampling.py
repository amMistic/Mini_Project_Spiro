import librosa as li

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
    
# Execution
filename = 'Audio_samples\Breath Audio 3.mp3'
sampler = breath_Sample(filename=filename)
sampling_rate = sampler.get_sampling_rate()
print("Sampling Rate:",sampling_rate)


# OUTPUT:
# The sampling rate of 1st Audio sample : 44100 Hz 


