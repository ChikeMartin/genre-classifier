# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" data library for audiosep Project
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split

JSON_PATH = "../raw_data/genre/data.json"

def load_data(dataset_path = JSON_PATH):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
        
    # convert lists to np.array e.g 'mapping', 'labels' in json file
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    
    return inputs, targets

def split_data(test_size=0.25, validation_size= 0.2):
    """split data into train, vaidation and test sets"""
    # load_data
    X, y = load_data()
    
    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # train / val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size)
    
    # 3D array for each sample eg --> (130, 13, 1)
    X_train = X_train[..., np.newaxis] # 4d array --> (num_samples, 130, 13, 1)
    X_val =X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    inputs, targets = load_data()