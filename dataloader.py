# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:48:04 2025

@author: lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch


def normalize_and_split(data, test_size=0.2):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    train_data, test_data = train_test_split(normalized_data, test_size=test_size, shuffle=False)
    return train_data, test_data, scaler

def sliding_window(data, window_size,pred_size,repeat_count):
    windows = []
    for i in range(len(data) - window_size-pred_size):
        x = data[i:i + window_size]
        y = data[i + window_size:i+window_size+pred_size]
        #y = np.repeat(y,repeat_count,axis=1)
        windows.append((x, y))
    return windows

def create_dataset(windows):
    dataset=[]
    
    for window in windows:
        x, y = window
        x_tensor = torch.tensor(x, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.float)  
        
        dataset.append([x_tensor,y_tensor])

    return dataset