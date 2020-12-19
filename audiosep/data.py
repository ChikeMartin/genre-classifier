# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" data library for audiosep Project
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "../raw_data/genre/data.json"

def load_data(dataset_path):
    """
    Load the MFCC dataset from saved json file
    """
    with open(dataset_path=et_path=taset_path, "r") as fp:
        data = json.load(fp)
        
    # convert lists to np.array e.g 'mapping', 'labels' in json file
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    
    return inputs, targets

if __name__ == "__main__":
    inputs, targets = load_data(DATASET_PATH)