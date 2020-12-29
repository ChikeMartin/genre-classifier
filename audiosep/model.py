# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" model library for audiosep Project
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def convnet_model(input_shape):
    """Takes in the tuple shape of the input as only argument
    Returns conv net whose summary can be printed
    """
    # create model
    model = Sequential()
    
    # 1st cond layer
    model.add(Conv2D(32, (3, 3), activation= 'relu', input_shape=input_shape))
    model.add(MaxPool2D((3, 3), strides= (2, 2), padding= 'same'))
    model.add(BatchNormalization())
    
    # 2nd conv layer
    model.add(Conv2D(32, (3, 3), activation= 'relu'))
    model.add(MaxPool2D((3, 3), strides= (2, 2), padding= 'same'))
    model.add(BatchNormalization())
    
    # 3rd conv layer
    model.add(Conv2D(32, (2, 2), activation= 'relu'))
    model.add(MaxPool2D((2, 2), strides= (2, 2), padding= 'same'))
    model.add(BatchNormalization())
    
    # flatten to 1D array and feed to dense
    model.add(Flatten())
    model.add(Dense(64, activation= 'relu'))
    model.add(Dropout(0.3))
    
    # output layer
    model.add(Dense(9, activation= 'softmax'))
    
    # compile model
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0003),
              metrics=['accuracy'])
    
    return model