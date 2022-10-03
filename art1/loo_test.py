#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:23:50 2022

@author: jose
"""

from scipy import io
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from elm import elm

datasets = ('australian', 
            'banknote', 
            'breastcancer', 
            # 'breastHess', 
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

dataset = 'sonar'
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
ps = np.linspace(1, 100, 1000, dtype = int)
loss = 1e6
for p in tqdm(ps):
    temp = elm(p = p, l = 0)
    temp.fit(X_train, y_train)
    log.append([temp.loss, 
                temp.mse(X_test, y_test), 
                temp.auc(X_test, y_test), 
                temp.acc(X_test, y_test)])
    
    if temp.loss < loss:
        loss = temp.loss
        model = temp
p = model.p

# plot
log = np.array(log)

# ----------------------------------
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(ps, log[:, 0], color = 'b')
ax1.set_xlabel('p')
ax1.set_ylabel('press')
ax1.set_title(dataset)
ax2.plot(ps, log[:, 1], color = 'r')
ax2.set_ylabel('mse')
# ----------------------------------
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(ps, log[:, 0], color = 'b')
ax1.set_xlabel('p')
ax1.set_ylabel('press')
ax1.set_title(dataset)
ax2.plot(ps, log[:, 2], color = 'r')
ax2.set_ylabel('auc')
# ----------------------------------
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(ps, log[:, 0], color = 'b')
ax1.set_xlabel('p')
ax1.set_ylabel('press')
ax1.set_title(dataset)
ax2.plot(ps, log[:, 3], color = 'r')
ax2.set_ylabel('acc')
# ----------------------------------
