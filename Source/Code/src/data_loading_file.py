import pandas as pd
import numpy as np
import IPython
from IPython.display import Audio
import os
import librosa
import random
import MFCC as mf
def data_generator(meta_data,path):
    #print("inside data generator function")
    audio_data,number_of_frames=[],[]
    #c=0
    for i in range(len(meta_data)):
        audio_file,sample_rate=librosa.load(path+meta_data.iloc[i][12])
        audio_data.append(audio_file)
        number_of_frames.append(audio_file.shape[0])
        #if c==10: break
        #c=c+1
    return audio_data,number_of_frames,sample_rate
#Kind of code to remove the silence bits or no size bits for the data
def silence_remover(audio_data):
    #print("inside silence remover function")
    for i in range(len(audio_data)):
        audio_data[i],sampler=librosa.effects.trim(audio_data[i])
    return audio_data
#We end to specify the audio duration we need to have so that we can break the
#Based on the audio duration we need we can randomily pick the part of the sample form the audio data_generator
def compress_expander(curr_audio,audio_length):
    #print("inside compress_expander function")
    #print(curr_audio,curr_audio.shape)
    if len(curr_audio)>audio_length:
        max_offset=len(curr_audio)-audio_length
        offset=np.random.randint(max_offset)
        curr_audio=curr_audio[offset:(audio_length+offset)]
    else:
        if len(curr_audio)<audio_length:
            min_offset=audio_length-len(curr_audio)
            offset=np.random.randint(min_offset)
        else:
            offset=0
        curr_audio=np.pad(curr_audio, (offset, audio_length - len(curr_audio) - offset), "constant")
    return curr_audio
def data_manager(audio_data,audio_length):
    #print("inside data manager function")
    for i in range(len(audio_data)):
        print(type(audio_data),type(audio_data[i]))
        audio_data[i]=compress_expander(audio_data[i],audio_length)
    return audio_data
#normalizing is kind feature that need to be used inorder to maintain all teh values in the
#closed intervel range in this case i am compressing the audio data in the range of values [0,1]
def normailze(data):
    #print("normalize function")
    max_value=np.max(data)
    min_value=np.max(data)
    data=(data-min_value)/(max_value-min_value+1000000)
    return data
##creating main function so that we can just call tah function and do every thing
def label_preparation(meta_data):
    #print("inside the label_preparation")
    target=meta_data.primary_label.values
    target=pd.get_dummies(target)
    target=target.values
    return target
#Implementing the above function results in the preparation of output variable for us
#down fucntion is used for returnig the total number of labels in the meat train data
def labler(meta_data):
    #print("inside the labler")
    labels=meta_data.primary_label.value_counts()
    values=labels.index.tolist()
    values.sort()
    return values
def main_func(csv_path,folder_path):
    #Loading the train_metadata file
    meta_data=pd.read_csv(csv_path)
    #Folder path for the audi files
    labels=labler(meta_data)
    Folder_path=folder_path+"/"
    #Now we have file paths and csv file readily loaded we need to call the data loading function
    audio_data,numer_of_frames,sample_rate=data_generator(meta_data,Folder_path)
    audio_data=silence_remover(audio_data)
    #give the numer of minutes you need to consider for selecting the audio
    audio_duration=2
    audio_data=data_manager(audio_data,sample_rate*audio_duration)
    #Now since the audio data got prepared the we need to prepare the target
    target=label_preparation(meta_data)
    n_fft=1000
    n_mfcc=20
    hop_length=500
    audio_data=mf.mfcc_convertor(audio_data,sampling_rate,n_fft,n_mfcc,hop_length)
    return audio_data,target,labels
