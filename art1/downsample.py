#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 16:21:30 2022

@author: jose
"""

from scipy import io
import numpy as np
import pandas as pd
from tqdm import tqdm

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

dataset = 'climate'
fold_n = 0

loglog = []
# for dataset in tqdm(datasets):
    # b_size = 0
    # eta = 0
    # for fold_n in range(K):
filename = 'data/exportBase_{}_folds_10_exec_{}.mat'.format(dataset, fold_n + 1)
data_mat = io.loadmat(filename)

X_train = np.copy(data_mat['data']['train'][0][0])
y_train = np.copy(data_mat['data']['classTrain'][0][0].ravel())
y_train[y_train == -1] = 0

X_test = np.copy(data_mat['data']['test'][0][0])
y_test = np.copy(data_mat['data']['classTest'][0][0].ravel())
y_test[y_test == -1] = 0

gg_path = 'gg/ggBase_{}_folds_10_exec_{}.csv'.format(dataset, fold_n + 1)
gg = pd.read_csv(gg_path).to_numpy()

scores = []
for i, row in enumerate(gg):
    vizinhos = np.where(row == 1)[0]
    
    degree = len(vizinhos)
    opposite = 0
    for vizinho in vizinhos:
        opposite += np.abs(y_train[0] - y_train[vizinho])
    q = 1 - opposite / degree
    scores.append(q)
border = np.where(np.array(scores) < 1)[0]
print(border.size)
# b_size += border.size / K

X_train = X_train[border, :]
y_train = y_train[border]
# eta += np.sum(y_train) / len(y_train) / K
print(np.sum(y_train) / len(y_train))
    # loglog.append([dataset, b_size, eta])

# # export
# df = pd.DataFrame(loglog, columns = ['dataset', 'border', 'eta'])
# df.to_csv('output/downsample.csv', index = None)
