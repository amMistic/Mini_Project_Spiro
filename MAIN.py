## ----------- MAIN FILE ----------------------

'''
Data Collection Information:

Name:                                       Gender:
Body Type : 
    - A: Tall and Healthy
    - B: Tall and Skinny  
    - C: Short and Healthy
    - D: Short and Skinny 
Signal:
Filename :  recorded wav file
testing time and date:
Max Peak:
Peak at 1 sec:
Signal Status

'''

# Import the Dependencies
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.io.wavfile import write
from PyEMD import CEEMDAN
import numpy as np
import pandas as pd
import tkinter as tk
import pywt
from time import sleep
import datetime
import os

 
# -----------------------------   GLOBAL PARAMETER   ---------------------------------------
SAMPLE_RATE = 8000
LOWCUT = 100
HIGHCUT = 2000
ORDER = 5
WINDOW_SIZE = 100
FL_HZ = 10
RIPPLE_DB = 10.0

# ------------------------------   RECORD AUDIO SIGNAL   ------------------------------------
def capture_audio(duration: int):
    print('Recording....')
    audio_data = sd.rec(int(duration * SAMPLE_RATE), SAMPLE_RATE, channels=1, dtype = 'int16' )
    sd.wait()
    print('Processing...')
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
        sleep(1)

    countdown_window.destroy()

# --------------------------------------------------   BUTTERWORTH BANDPASS (100 - 1000)    ----------------------------------------------

def butter_bp(lowcut: int, highcut: int, sr: int, order = 5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = ss.butter(order, [low, high], btype='band')
    return b, a

def butterworth_filter(signal: list, lowcut: int, highcut: int, order: int, sr: int):
    b, a = butter_bp(lowcut, highcut, sr, order)
    y = ss.lfilter(b, a, signal)
    return y

# -------------------------------------------------------------  ICEEMDAN FILTER  ---------------------------------------------------------

def iceemdan_filter(signal: list):
    iceemdan = CEEMDAN()
    IMFS = iceemdan(signal)
    denoised_signal = signal - IMFS[0]
    return denoised_signal, IMFS[0]

def wavelet_denoising(signal: list) -> list:
    coeffs = pywt.wavedec(signal, 'db1', level=6)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(denoised_coeffs, 'db1')
    return denoised_signal

# ---------------------------------------------------------------- EVALUATION ------------------------------------------------------------------

def calculate_snr(signal: list, denoise_signal: list) -> float:
    noise = signal - denoise_signal
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log(signal_power / noise_power)
    return snr

def calculate_rmse(signal: list, denoise_signal: list) -> float:
    rmse = np.sqrt(np.mean((signal - denoise_signal)** 2))
    return rmse

# ------------------------------------------------------------ EXTRACT ENEVELOPE AND SMOOTHEN IT   ---------------------------------------------------------------

def extract_envelope(signal: list, sr: int, fL_hz: int, ripple_db: int) -> list:
    nyq_rate = 0.5 * sr
    width = 1.0 / nyq_rate
    
    analytic_signal = ss.hilbert(signal)
    envelope = np.abs(analytic_signal)
    
    N, beta = ss.kaiserord(ripple_db, width)
    taps = ss.firwin(N, fL_hz / nyq_rate, window=('kaiser', beta))
    filtered_envelope = ss.filtfilt(taps, 1, envelope)
    return filtered_envelope

def smooth_envelope(signal: list, window_size: int) -> list:
    return np.convolve(signal, np.ones(window_size) / window_size, mode= 'same')

# --------------------------------------------------------------  CALCULATE PARAMETERS ----------------------------------------------------

def maxima(signal: list) -> tuple:
    signal = np.array(signal)  
    index = np.argmax(signal)  
    maxx_signal = signal[index]
    return index, maxx_signal

#  -------------------------------------------------------------- PLOTS FUNCTION --------------------------------------------------------------------

def plot_signals(time: list, sr: int ,signal: list, denoised_signal: list, envelope: list, maxx_peak_time: int, start: int, end: int) -> None:
    
    plt.figure(figsize=(14, 10))
    plt.subplot(3, 1, 1)
    plt.plot(time, signal, label = 'Oiginal Signal')
    plt.plot(time, denoised_signal, label = 'Denoised Signal', color = 'g', alpha = 0.5)
    plt.title('Original V/s Denoised Signal')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.legend(loc= 'upper right')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time, denoised_signal, label = 'Denoised Signal', color = 'g', alpha = 0.5)
    plt.plot(time, envelope, label = 'Envelope Signal', color = 'r', alpha = 0.7)
    plt.plot(time[sr], envelope[sr], 'go' ,label = 'Peak at 1sec')
    plt.plot(time[maxx_peak_time], envelope[maxx_peak_time], 'mo', label = 'Max Peak')
    plt.plot(time[start], denoised_signal[start], 'ko' ,label = 'Start Point')
    plt.plot(time[end], denoised_signal[end], 'co' ,label = 'End Point')
    plt.title(f'Max Peak: {envelope[maxx_peak_time]:.2f} | Peak at 1sec: {envelope[sr]:.2f}')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.legend(loc= 'upper right')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time, envelope, label = 'Envelope Signal', color = 'm', alpha = 0.7)
    plt.title('Extracted Envelope')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.legend(loc= 'upper right')
    plt.grid(True)
    
    plt.tight_layout(pad= 3.0)
    plt.show()
    
# ------------------------------------------------------------- CLIPPING AUDIO  ------------------------------------------------------------------------

def clipping_audio(signal: list, window_size: int, step_size:int, threshold_ratio: float) -> tuple:
    start = 0
    endpoint = 0
    signal = np.array(signal)
    maxx = np.max(signal) ** 2
    threshold = threshold_ratio * maxx
    for i in range(0, len(signal) - window_size + 1, step_size):
        energy = np.sum((signal[i: i + window_size]) ** 2)
        if start == 0 and energy > threshold:
            start = i
        if start!=0 and energy < threshold :
            endpoint = i 
            break
    
    return start, endpoint

# -------------------------------------------------- RECORDS THE DATA COLLECTION FUNCITON ---------------------------------------------------------------

def record_data(filepath: str, recorded_audio_filename:str,subject_number: int, name: str ,body_type: int, gender:str, testing_datetime:str, diseases: str, max_peak: float, peak_at_1sec:float, signal_status: str ) -> None:
    # Ensure the folder exists
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Open the file in append mode and write the data
    with open(filepath, 'a') as file:
        # If the file is empty, write the header
        if os.path.getsize(filepath) == 0:
            file.write("filename,subject_number,name,signal,body_type,gender,testing_datetime,diseases,max_peak,peak_at_1sec,signal_status\n")
        
        # Write the data
        file.write(f'\n{recorded_audio_filename},{subject_number},{name},{body_type},{gender},{testing_datetime},{diseases},{max_peak},{peak_at_1sec},{signal_status}')
    print(f"Data appended to '{filepath}' successfully.")

def collect_data():
    
    print('\n<--------------------------------------  Subject Details  ------------------------------------------>\n')
    subject_number = int(input('Subject Number: '))
    name = str(input('Subject Full Name: ')).lower().strip().replace(' ','_')
    gender = str(input('Subject Gender: ')).lower().strip()
    body_type = str(input('Subject Body Type[A, B, C, D]: ')).upper().strip()
    diseases = str(input('Any Heart Related Diseases?? Yes/No: ')).lower()
    print('\n<--------------------------------------  Details Collected  ------------------------------------------>\n')
    return subject_number, name,body_type, gender, diseases

def saved_file(audio_data: np.ndarray, sr: int,filename: str, dir: str) ->  None:
    filepath = os.path.join(dir,filename)
    write(filepath, sr, audio_data)
    print(f"Audio data saved to {filepath}")
    
# -------------------------------------------------------------------------- MAIN FUNCTION  ----------------------------------------------------------------------------

def main():
    print("--------------  Initiate System  ------------------")
    sleep(1)

    # Collect subject deta
    subject_number, name, body_type, gender, diseases = collect_data()
    sleep(1)

    # Countdown
    show_countdown()

    # Record the audio
    current_day = datetime.datetime.now()
    testing_datetime = str(current_day.strftime("%Y-%m-%d %H:%M:%S"))
    testing_datetime = testing_datetime.replace('-','').replace(' ','_').replace(':','')

    DATAFILE = 'Data_Collection\\records.csv'
    audio_data, sr = capture_audio(3)
    
    # Time axis
    time = np.linspace(0, len(audio_data) / sr, len(audio_data))
    
    # Apply ICEEMDAN filter on that
    _denoised_signal, _imf0 = iceemdan_filter(audio_data)
    denoised_signal = wavelet_denoising(_denoised_signal)
    
    # Extracting Envelope from the filtered audio
    envelope = extract_envelope(denoised_signal, SAMPLE_RATE, FL_HZ, RIPPLE_DB)
    smoothen_envelope = smooth_envelope(envelope, WINDOW_SIZE)

    # Find the time stemp where the amplitude is maximum
    max_peak_time, maxx_peak = maxima(smoothen_envelope)
    maxx_peak = float(f"{maxx_peak:.2f}")
    
    peak_at_1s = smoothen_envelope[sr]
    peak_at_1s = float(f"{peak_at_1s:.2f}")
    
    # Find the clipps in the audio signal
    start_point, end_point = clipping_audio(envelope, 400, 1, 0.1 )
    filename = f'{name}_{testing_datetime}.wav'
    print(f"Save the plot as: {name}_{testing_datetime}")
    
    # Plot the signal
    plot_signals(time, sr, audio_data, denoised_signal, smoothen_envelope, max_peak_time, start_point, end_point)
    
    # Feedback on this 
    signal_status = str(input("Accepted or not?? Yes or No: ")).lower()
    
    # Classified by human
    if signal_status == 'yes':
        dir = 'E:\\Sprio mini project\\Data_Collection\\collected_audios\\Accepted\\'
    else:
        dir = 'E:\\Sprio mini project\\Data_Collection\\collected_audios\\Rejected\\'
    saved_file(audio_data, sr, filename, dir)
    record_data(DATAFILE, filename, subject_number, name, body_type, gender, testing_datetime, diseases, maxx_peak, peak_at_1s, signal_status)
            

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
           

# save the plot into the Data_collection / PLots directory
# Record the data in the recorder.csv file