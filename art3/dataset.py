#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 11:03:57 2022

@author: jose
"""

from torch.utils.data import Dataset

from scipy import io
import numpy as np

class custom(Dataset):
    def __init__(self, dataset, fold_n, train = True):
        filename = 'data/exportBase_{}_folds_10_exec_{}.mat'.format(dataset, fold_n + 1)
        data_mat = io.loadmat(filename)
        
        if train:
            self.X = np.copy(data_mat['data']['train'][0][0])
            self.y = np.copy(data_mat['data']['classTrain'][0][0].ravel())
            self.y[self.y == -1] = 0
        else:
            self.X = np.copy(data_mat['data']['test'][0][0])
            self.y = np.copy(data_mat['data']['classTest'][0][0].ravel())
            self.y[self.y == -1] = 0
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]
    
    def in_features(self):
        return self.X.shape[1]
