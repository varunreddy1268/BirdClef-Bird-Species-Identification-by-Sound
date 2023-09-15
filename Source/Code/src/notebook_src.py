import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import tensorflow.keras as tfk
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Embedding, Normalization, Conv1DTranspose,InputLayer
import tensorflow.keras.layers as tfkl
from tensorflow.keras import Model
import math
import time
import tensorflow as tf
import tensorflow_probability as tfp
import librosa
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import tensorflow_io as tfio
import random
import tensorflow as tf
from pathlib import Path
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

with open('../input/birdclef-2022/scored_birds.json','r') as sb:
  s_b = json.load(sb)
file_path = '../input/birdclef-2022'
train_df = pd.read_csv('../input/birdclef-2022/train_metadata.csv')
train_df = train_df[train_df['primary_label'].isin(s_b)]
bird_label = train_df["primary_label"].unique()

submission_df = pd.read_csv('../input/birdclef-2022/sample_submission.csv')
test_df = pd.read_csv('../input/birdclef-2022/test.csv')
if test_df.shape[0] != submission_df.shape[0]:
    raise ValueError('test submission row number didnt match')

    train_path = '../input/birdclef-2022/train_audio'

    def preprocessing(df, path,bird_label):
      le = 160000
      step = int((le/2))
      sample_rate = 32000
      train = []
      for label in tqdm(bird_label):
        files = librosa.util.find_files(os.path.join(path, label))
        for f in tqdm(files):
          yi = np.where(bird_label == label)
          # load audio\
          #print("1:",type(yi),type(yi.shape),yi[:10])
          y, sr = librosa.load(f,sr=sample_rate)
          #print(y)
          y = ((y-np.amin(y))*2)/(np.amax(y) - np.amin(y)) - 1
          #print("2:",type(y),y.shape,y[:10])

          org_len = len(y)
          intervals = librosa.effects.split(y, top_db= 15, ref= np.max)
          intervals = intervals.tolist()
          #print("3-1:",type(y),y.shape,y[:10])

          y = (y.flatten()).tolist()
          #print("3:",type(y),y[:10])

          nonsilent_y = []

          for p,q in intervals :
           nonsilent_y = nonsilent_y + y[p:q+1]
          #print("4:",type(nonsilent_y),nonsilent_y[:10])
          y = np.array(nonsilent_y).astype('float32')
          if len(y) < le:
            while len(y) < le:
              y = np.concatenate((y, y))
            y = y[:le]
          #print("5:",type(y),y.shape,y[:10])

          # A 1024-point STFT with frames of 5 s and 50% overlap.
          stfts = tf.signal.stft(y, frame_length=le, frame_step=step,
                           fft_length=4096)
          #print("6:stfts",type(stfts),stfts[:10])
          spectrograms = tf.abs(stfts)

          # Warp the linear scale spectrograms into the mel-scale.
          num_spectrogram_bins = stfts.shape[-1]
          #print("7: num_spectrograms",type(num_spectrogram_bins),num_spectrogram_bins)
          lower_edge_hertz, upper_edge_hertz, num_mel_bins = 1000.0, 8000.0, 4096

          linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
            upper_edge_hertz)
          #print("8: linear_to_mel_spectorgrams_matrix",type(linear_to_mel_weight_matrix),linear_to_mel_weight_matrix[:10])


          mel_spectrograms = tf.tensordot(
            spectrograms, linear_to_mel_weight_matrix, 1)
          #print("9: mel_spectrograms",type(mel_spectrograms),mel_spectrograms[:10])

          mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))
          #print("continuous to mel spectrogema the shsape of the mel is ",mel_spectrograms.shape)

          # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
          log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
          #print("9: log_mel_spectrograms:",type(log_mel_spectrograms),log_mel_spectrograms[:10])


          mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
            log_mel_spectrograms)
          print("final",type(mfccs),mfccs.shape,mfccs[:10])

          for mfc in mfccs:
            train.append((mfc, yi))
      return train
train_data = preprocessing(train_df, train_path, bird_label)
@tf.function
def test_step(x_batch_val):
  val_logits = model(x_batch_val, training=False)
  return tf.math.argmax(val_logits,1)
  test_path = '../input/birdclef-2022/test_soundscapes/'
  test_files = os.listdir(test_path)
  def preprocessing_test_dat(test_path, files):
    le = 160000
    step = int((le/2))
    sample_rate = 32000
    test = []
    for file in tqdm(files):
      y, sr = librosa.load(test_path + file, sr=sample_rate)
      # y = y[:le + 1]
      for segment in range(0, len(y), sample_rate*5):
          row_id = file[:-4] + '_' + str(int((segment + (sample_rate * 5)) / (sample_rate)))
          if segment+le > len(y):
              yi = y[segment:]
              while len(yi) < le:
                yi = np.concatenate((yi, yi))
              yi = yi[:le]
          else:
              yi = y[segment:segment+le]

          stfts = tf.signal.stft(yi, frame_length=le, frame_step=le,
                         fft_length=4096)
          spectrograms = tf.abs(stfts)

          # Warp the linear scale spectrograms into the mel-scale.
          num_spectrogram_bins = stfts.shape[-1]
          lower_edge_hertz, upper_edge_hertz, num_mel_bins = 1000.0, 8000.0, 4096

          linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
          upper_edge_hertz)

          mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
          mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))

          # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
          log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

          mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
          test.append((row_id, mfccs))
    return test
class TensorflowDataGenerator_test():
    'Characterizes a dataset for Tensorflow'
    def __init__(self, mel_list, batch_size):
         self.mel_list = mel_list
         self.batch_size = batch_size
         self.index_helper = 0
         self.le = len(mel_list)
    def __len__(self):
        return math.ceil(self.le/ self.batch_size)

    def __getitem__(self, index):
        if self.index_helper >= self.le:
            raise IndexError
        x, y = [], []
        for b in range(self.batch_size):
            if self.index_helper < self.le:
              x.append(self.mel_list[self.index_helper][0])
              y.append(self.mel_list[self.index_helper][1])
              self.index_helper += 1
        return x, np.array(y).astype('float32')

    def reset(self):
        self.index_helper = 0
test_dat = preprocessing_test_dat(test_path, test_files)
batch_size = 32

test_set = TensorflowDataGenerator_test(test_dat,batch_size)
predictions = []
test_set.reset()
for x_batch, y_batch in tqdm(test_set):
    preds = test_step(y_batch)
    for idx, pred in enumerate(preds):
        split_code = x_batch[idx].split('_')
        for bird in bird_label:
            row_id = split_code[0] +'_'+ split_code[1]+'_' + bird+'_'+split_code[2]
            predictions.append([row_id, True if bird == bird_label[pred] else False])
