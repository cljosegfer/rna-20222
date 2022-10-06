#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 17:19:30 2022

@author: jose
"""

from scipy import io
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from embedding import autoencoder as net

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
dir_path = 'data'
K = 10

dataset = 'german'
fold_n = 1

# data
filename = 'data/exportBase_{}_folds_10_exec_{}.mat'.format(dataset, fold_n + 1)
data_mat = io.loadmat(filename)

X_train = np.copy(data_mat['data']['train'][0][0])
y_train = np.copy(data_mat['data']['classTrain'][0][0].ravel())
y_train[y_train == -1] = 0

X_test = np.copy(data_mat['data']['test'][0][0])
y_test = np.copy(data_mat['data']['classTest'][0][0].ravel())
y_test[y_test == -1] = 0

# p
log = []
ps = np.linspace(1, 100, 100, dtype = int)
loss = 1e6
for p in tqdm(ps):
    temp = net(p = p, l = 0)
    temp.fit(X_train, y_train)
    log.append(temp.loss)
    
    if temp.loss < loss:
        loss = temp.loss
        model = temp
p = model.p
log = np.array(log)

# plot
plt.figure()
plt.plot(ps, log)
plt.xlabel('p')
plt.ylabel('press')
plt.title(dataset)

# lambda
log = []
ls = np.logspace(-5, 1, 100)
loss = 1e6
for l in tqdm(ls):
    temp = net(p = p, l = l)
    temp.fit(X_train, y_train)
    log.append(temp.loss)
    
    if temp.loss < loss:
        loss = temp.loss
        model = temp
l = model.l

# plot
plt.figure()
plt.plot(np.log(ls), log)
plt.xlabel('log lambda')
plt.ylabel('press')
plt.title(dataset)
