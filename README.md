# Mini Spiro Mask
![Project Logo](https://assets.cntraveller.in/photos/60ba1f942267328f9d2457c1/16:9/w_960,c_limit/2C57GB8.jpg)

## Overview
Mini Spiro Mask is a project aimed at analyzing and processing breath audio recordings using Python libraries such as Librosa, NumPy, Matplotlib, and SciPy. The project involves steps to import dependencies, load audio files, calculate FFT (Fast Fourier Transform), visualize frequency content, filter out higher frequencies, and save the filtered audio.

## Dependencies
- Python 3.x
- Librosa
- NumPy
- Matplotlib
- SciPy

## Usage

### Clone the Repository
1. **Clone the repository**: To clone the Mini Spiro Mask repository to your local machine, open a terminal window and run the following command in any folder:
   ```bash
   git clone https://github.com/amMistic/Mini_Project_Spiro.git
   
2. **Navigate to the directory**: Change your current directory to the cloned repository folder:
   ```bash
   cd mini-spiro-mask
   ```
### Install Dependencies
1. **Install Python dependencies**: Before running the project, ensure you have Python 3.x installed on your system. Then, install the required Python dependencies by running:
   
   ```bash
   pip install -r requirements.txt
   ```
   
### Run the Project
1. **Specify input audio file**: Open the `mini_spiro_mask.py` script in a text editor, and specify the input audio file path:

   ```python
   filename = '/path/to/your/audio/file.mp3'
   
2. **Run the script**: Execute the Python script `mini_spiro_mask.py` to analyze and process the audio file:

   ```bash
   python mini_spiro_mask.py
   ```
   
3. **Follow instructions**: Follow the instructions provided in the script for further analysis or processing of the filtered audio.


## Steps
### Step 1: Import Dependencies
Import the necessary Python libraries required for the project.

![image](https://github.com/amMistic/Mini_Project_Spiro/assets/134824444/aa32e21a-7c4b-4158-be87-b5e2276d4710)

### Step 2: Load Audio File and Calculate Sampling Rate
Load the audio file using Librosa and calculate the sampling rate of the audio.

![image](https://github.com/amMistic/Mini_Project_Spiro/assets/134824444/2e41063b-7383-4ef1-baac-cd4438b4214e)

**Question:** Why within mask?
Answer: 👇

**Within Mask**
![image](https://github.com/amMistic/Mini_Project_Spiro/assets/134824444/016668f3-34e9-490b-b004-b67b0172baab)

**Without Mask**
![image](https://github.com/amMistic/Mini_Project_Spiro/assets/134824444/9db69ab9-63ef-4051-9d79-413257fc4b1d)

![image](https://github.com/amMistic/Mini_Project_Spiro/assets/134824444/66bc9a3b-c2f7-43a1-be85-4b45f93a55f7)

### Step 3: Calculate FFT (Fast Fourier Transform)
FFT is used to convert the audio signal from the time domain to the frequency domain. It enables analysis and visualization of frequency components in the audio.

![image](https://github.com/amMistic/Mini_Project_Spiro/assets/134824444/662416d1-81de-46a6-91ed-e2a29e6af1ac)


### Step 4: Plot FFT
Visualize the frequency content of the audio signal using FFT plot. This helps in identifying dominant frequency components and determining the pitch of the sound.

![image](https://github.com/amMistic/Mini_Project_Spiro/assets/134824444/5923fbcf-8973-4dff-8176-f4fb1180144d)

![image](https://github.com/amMistic/Mini_Project_Spiro/assets/134824444/7bc72f86-0b33-4731-a6f0-5ece27e07b5c)



### Step 5: Filter Higher Frequencies
Design and apply a high-pass filter to remove higher frequencies from the audio signal. This step is crucial for isolating breathing information from noise and unwanted components.

![image](https://github.com/amMistic/Mini_Project_Spiro/assets/134824444/156d95b0-eb27-4f8c-a930-074756a9ea4c)

![image](https://github.com/amMistic/Mini_Project_Spiro/assets/134824444/8b8a2c7d-c3c4-4090-a2f8-7f1d398ca788)

### Step 6: Save Filtered Audio
Save the filtered audio after applying the high-pass filter for further analysis or processing.

![image](https://github.com/amMistic/Mini_Project_Spiro/assets/134824444/cda6dac7-d40e-4dc8-97d7-4a29f936ab22)

![image](https://github.com/amMistic/Mini_Project_Spiro/assets/134824444/a69a92d9-0c2b-4e60-b91a-58e8f2eb9b10)


## Results
The project aims to identify breathing information present in the audio signal by analyzing its frequency content and filtering out higher frequencies. The filtered audio can be further used for various applications such as respiratory health monitoring or analysis.


#### TODO:
- Update the Readme file

