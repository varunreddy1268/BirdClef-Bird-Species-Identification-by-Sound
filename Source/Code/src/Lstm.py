import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.layers import Dropout
import numpy as np
from sklearn.model_selection import train_test_split
#import data_loading_file as df
#peparing the train and test data
##Basic Model

#

#

####
X_train,X_test,Y_train,Y_test=train_test_split(audio_data,target,test_size=0.1)
X_train,X_validation,Y_train,Y_validation=train_test_split(audio_data,target,test_size=0.1)
X_train=tf.stack(X_train)
Y_train=tf.stack(Y_train)
X_test=tf.stack(X_test)
Y_test=tf.stack(Y_test)
X_validation=tf.stack(X_validation)
Y_validation=tf.stack(Y_validation)
#1-input layer & 3 hidden layers & 1-output layers
classifier=Sequential()
classifier.add(LSTM(units=50,return_sequences=True,input_shape=(128,87)))
classifier.add(Dropout(0.2))
classifier.add(LSTM(units=50,return_sequences=True))
classifier.add(Dropout(0.2))
classifier.add(LSTM(units=50,return_sequences=True))
classifier.add(Dropout(0.2))
classifier.add(LSTM(units=50))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=152,activation="softmax"))
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
classifier.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=["accuracy"])
classifier.fit(X_train,Y_train,validation_data=(X_validation,Y_validation),epochs=25,batch_size=32)

###

# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

##
