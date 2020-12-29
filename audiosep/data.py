# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" Data retrieval and preprocessing library for audiosep Project
"""
import os
import json
import math
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# Constants declared
DATASET_PATH = "../raw_data/genre/genres_original/"
JSON_PATH_MAIN = "../raw_data/genre/data.json"
JSON_PATH_NEW = "../raw_data/genre/data_test.json"
SAMPLE_RATE = 22050
DURATION = 30 # seconds per track
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path= JSON_PATH_MAIN, n_mfcc= 13, n_fft= 2048, hop_length= 512, num_segments= 5, train=True, verbose= True):
    """
    Takes in a path and outputs json with genre, mfccs and labels
    If train is set to False, the path would be for a new track which would be split into 30 sceonds to match the train
    duration. It will then output json file with mfccs for each split.
    """
    SAMPLE_RATE = 22050
    DURATION = 30 # seconds per track
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    
    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_nmfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    if train:
        # loop through all genres
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

            # ensure we are not at root level
            if dirpath is not dataset_path:
                # save the semantic label
                semantic_label = dirpath.split("/")[-1] # genre/blues => ['genre', 'blues']
                data['mapping'].append(semantic_label)
                print(f"\nProcessing {semantic_label}")

                # process files for a specific genre
                for f in filenames[:50]:
                    # load audio file
                    file_path = os.path.join(dirpath, f)
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    # process by segment, extracting mfcc and storing data
                    for s in range(num_segments):
                        start_sample = num_samples_per_segment * s 
                        finish_sample = start_sample + num_samples_per_segment

                        mfcc = librosa.feature.mfcc(signal[start_sample: finish_sample],
                                                    sr=sr,
                                                    n_fft= n_fft,
                                                    n_mfcc= n_mfcc,
                                                    hop_length = hop_length)
                        mfcc = mfcc.T

                        # store mfcc for segment if it has the expected length
                        if len(mfcc) == expected_nmfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            print(f"{file_path}, segment: {s+1}")

            with open(json_path, 'w') as fp:
                json.dump(data, fp, indent=4)

        print(f'Data stored at {json_path}')
                
    if not train:
        json_path = JSON_PATH_NEW
        signal, sr = librosa.load(dataset_path)

        start = 0
        split_data = {}
        duration = int(librosa.get_duration(signal)) # duration

        if duration < 30:
            print("Please upload a song of at least 30 seconds")

        else:
            batch_size = sr * 30  # get num of vectors per 30 secs
            length = len(signal) # total len of signal array
            num_iter = duration // 30 # model trained with 30 second clips
            # num of times to loop over len 
            for i in range(num_iter):
                # slice 30 seconds--> batch size 
                split_data['batch_'+ str(i+1)] = signal[start: start+batch_size]
                start += batch_size  #update start index

        for key, value in split_data.items():
            signal = value
            
            for s in range(num_segments):
                start_sample = num_samples_per_segment * s 
                finish_sample = start_sample + num_samples_per_segment
                    
                mfcc = librosa.feature.mfcc(signal[start_sample: finish_sample],
                                            sr=sr,
                                            n_fft= n_fft,
                                            n_mfcc= n_mfcc,
                                            hop_length = hop_length)
                mfcc = mfcc.T
                    
                # store mfcc for segment if it has the expected length
                if len(mfcc) == expected_nmfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    if verbose:
                        print(f"{key}, segment: {s+1}")

        test_input = np.array(data["mfcc"])[..., np.newaxis]  

        return test_input

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
        
    # convert lists to np.array e.g 'mapping', 'labels' in json file
    inputs = np.array(data["mfcc"])[..., np.newaxis] # 4d array --> (num_samples, 130, 13, 1)
    targets = np.array(data["labels"])
    
    return inputs, targets

def split_data(test_size= 0.25, validation_size= 0.2, dataset_path= JSON_PATH_MAIN, random_state=42):
    """split data into train, vaidation and test sets"""
    # load_data
    X, y = load_data(dataset_path)
    
    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state= random_state)
    
    # train / val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state= random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    inputs, targets = load_data()