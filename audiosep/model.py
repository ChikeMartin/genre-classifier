# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" model library for audiosep Project
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

def initialize_model():
    model = Sequential()

    # input layer
    model.add(Flatten(
        input_shape= (inputs.shape[1], inputs.shape[2])))
    
    # 1st hidden layer
    model.add(Dense(512, activation= "relu"))
    
    # 2nd hidden layer    
    model.add(Dense(256, activation= "relu"))
    
    # 3rd hidden layer
    model.add(Dense(64, activation= "relu"))
    
    # output layer
    model.add(Dense(9, activation= 'softmax'))
    
    # compile model
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0005),
              metrics=['accuracy'])
    return model