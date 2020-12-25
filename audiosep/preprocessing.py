# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" Preprocessing library for audiosep Project
"""

import os
import math
import json
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Constants declared
DATASET_PATH = "../raw_data/genre/genres_original/"
JSON_PATH = "../raw_data/genre/data.json"
SAMPLE_RATE = 22050
DURATION = 30 # seconds per track
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc= 13, n_fft= 2048, hop_length= 512, num_segments= 5):
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
    
    # loop through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # ensure we are not at root level
        if dirpath is not dataset_path:
            # save the semantic label
            semantic_label = dirpath.split("/")[-1] # genre/blues => ['genre', 'blues']
            data['mapping'].append(semantic_label)
            print(f"\nProcessing {semantic_label}")
            
            # process files for a specific genre
            for f in filenames:
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

if __name__ == '__main__':
    # For introspections purpose to quickly get this functions on ipython
    #import audiosep
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
