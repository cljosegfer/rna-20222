#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 11:03:57 2022

@author: jose
"""

from torch.utils.data import Dataset

from scipy import io
import numpy as np
import pandas as pd

class custom(Dataset):
    def __init__(self, dataset, fold_n, train = True, downsample = False):
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
        
        if train and downsample:
            gg_path = 'gg/ggBase_{}_folds_10_exec_{}.csv'.format(dataset, fold_n + 1)
            gg = pd.read_csv(gg_path).to_numpy()
            
            scores = []
            for i, row in enumerate(gg):
                vizinhos = np.where(row == 1)[0]
                
                degree = len(vizinhos)
                opposite = 0
                for vizinho in vizinhos:
                    opposite += np.abs(self.y[0] - self.y[vizinho])
                q = 1 - opposite / degree
                scores.append(q)
            border = np.where(np.array(scores) < 1)[0]
            
            self.X = self.X[border, :]
            self.y = self.y[border]
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]
    
    def in_features(self):
        return self.X.shape[1]
