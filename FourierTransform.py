import pandas as pd
import numpy as np
import IPython
from IPython.display import Audio
import os
import librosa
import random
import matplolib.pyplot as pyplot
import seaborn as sns
import plotly.express as plt
import data_loading_file as dlf
#Now converting the audio data into mfccs
def mfcc_convertor(audio_data,sampling_rate,n_mfcc,n_fft,hop_length):
    for i in range(len(audio_data)):
        audio_data[i]=librosa.feature.mfcc(audio_data[i],sampling_rate,n_mfcc=n_mfcc,n_fft=n_fft,hop_length=hop_length)
    return audio_data
def mfcc_plot(meta_data,audio_data,sample_rate,hop_length):
    labels=dlf.loader(meta_data)
    for i in range(len(labels)):
        plt.figure(figsize=(10,10))
        sub_meta_data=meta_data[meta_data["primary_label"]==labels[i]].index.tolist()
        librosa.display.specshow(audio_data[i],sr=sample_rate,hop_length=hop_length)
        plt.xlabel("Time")
        plt.ylabel("MFCC Coefficients")
        plt.title("MEL Freq Cepstral Spectrogram")
        plt.colorbar()
        plt.show()
def sound_wave_plot(audio_data,sample_rate):
    labels=dlf.loader(meta_data)
    for i in range(len(labels)):
        plt.figure(figsize=(10,10))
        sub_meta_data=meta_data[meta_data["primary_label"]==labels[i]]
        plt.imshow(sub_meta_data)
def RMS_Energy_plot(audio_data,sample_rate):
    labels=dlf.loader(meta_data)
    for i in range(len(labels)):
        signal,phase=librosa.magphase(librosa.stft(audio_data))
        signal_in_decibles=librosa.amplitude_to_db(signal,ref=np.max)
        root_mean_square=librosa.feature.rms(S=signal)
        final_res=librosa.times_like(root_mean_square)
        plt.figure(figsize=(10,10))
        plt.plot(final_res,root_mean_square[0])
        plt.xlabel("Time")
        plt.ylabel("RMS powers")
        plt.show()
