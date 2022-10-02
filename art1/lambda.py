#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 22:29:52 2022

@author: jose
"""

from scipy import io
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

dataset = 'ILPD'
fold_n = 1

# param
p = 100

log = []

# data
filename = 'data/exportBase_{}_folds_10_exec_{}.mat'.format(dataset, fold_n + 1)
data_mat = io.loadmat(filename)

X_train = np.copy(data_mat['data']['train'][0][0])
X_train = np.hstack((np.ones(shape = (X_train.shape[0], 1)), X_train))
y_train = np.copy(data_mat['data']['classTrain'][0][0].ravel())
y_train[y_train == -1] = 0

X_test = np.copy(data_mat['data']['test'][0][0])
X_test = np.hstack((np.ones(shape = (X_test.shape[0], 1)), X_test))
y_test = np.copy(data_mat['data']['classTest'][0][0].ravel())
y_test[y_test == -1] = 0

# elm
n = X_train.shape[1]
Z = np.random.normal(size = (n, p))
H = sigmoid(X_train @ Z)
H = np.hstack((np.ones(shape = (H.shape[0], 1)), H))
p += 1

H_test = sigmoid(X_test @ Z)
H_test = np.hstack((np.ones(shape = (H_test.shape[0], 1)), H_test))

# train
log = []
ls = np.linspace(1, 300, 100)
for l in tqdm(ls):
    mathbbH = np.linalg.pinv(H.T @ H + l * np.identity(p)) @ H.T
    W = mathbbH @ y_train
    h = np.diag(H @ mathbbH)
    
    yhat = sigmoid(H @ W)
    r = y_train - yhat
    press = np.sum((r / (1 - h))**2)
    log.append(press)

l = ls[np.argmin(log)]
mathbbH = np.linalg.pinv(H.T @ H + l * np.identity(p)) @ H.T
W = mathbbH @ y_train

# test
yhat = np.tanh(H_test @ W)
acc = mean_squared_error(y_test, yhat)
print('lambda: {} | acc: {}'.format(l, acc))

# plot
plt.figure()
plt.plot(ls, log)
