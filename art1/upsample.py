#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 15:54:20 2022

@author: jose
"""

from scipy import io
import numpy as np
import pandas as pd
from sklearn.utils import resample

datasets = ('australian', 
            'banknote', 
            'breastcancer', 
            'breastHess', 
            'bupa', 
            'climate', 
            'diabetes', 
            'fertility', 
            'german', 
            'golub', 
            'haberman', 
            'heart', 
            'ILPD', 
            'parkinsons', 
            'sonar')
K = 10

dataset = 'german'
fold_n = 0

for dataset in datasets:
    for fold_n in range(K):
        filename = 'data/exportBase_{}_folds_10_exec_{}.mat'.format(dataset, fold_n + 1)
        data_mat = io.loadmat(filename)
        
        X_train = np.copy(data_mat['data']['train'][0][0])
        y_train = np.copy(data_mat['data']['classTrain'][0][0].ravel())
        y_train[y_train == -1] = 0
        
        X_test = np.copy(data_mat['data']['test'][0][0])
        y_test = np.copy(data_mat['data']['classTest'][0][0].ravel())
        y_test[y_test == -1] = 0
        
        eta = np.sum(y_train) / len(y_train)
        if eta > 0.5:
            minoria = y_train == 0
        else:
            minoria = y_train == 1
        n_samples = len(minoria) - 2 * np.sum(minoria)
        upsample, classUpsample = resample(X_train[minoria], y_train[minoria],
                                           replace = True, n_samples = n_samples)
        X_train = np.concatenate((X_train, upsample), axis = 0)
        y_train = np.concatenate((y_train, classUpsample), axis = 0)
        eta = np.sum(y_train) / len(y_train)
        assert eta == 0.5
