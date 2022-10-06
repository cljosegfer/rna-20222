#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:51:11 2022

@author: jose
"""

from scipy import io
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

dataset = 'sonar'
fold_n = 1

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
log = []
ps = np.linspace(1, 100, 100)

n = X_train.shape[1]
l = 0
for p in tqdm(ps):
    p = int(p)
    Z = np.random.normal(size = (n, p))
    H = sigmoid(X_train @ Z)
    H = np.hstack((np.ones(shape = (H.shape[0], 1)), H))
    p += 1
    
    mathbbH = np.linalg.pinv(H.T @ H + l * np.identity(p)) @ H.T
    W = mathbbH @ y_train
    h = np.diag(H @ mathbbH)
    
    yhat = sigmoid(H @ W)
    r = y_train - yhat
    press = np.sum((r / (1 - h))**2)
    log.append(press)

p = ps[np.argmin(log)]
p = int(p)
Z = np.random.normal(size = (n, p))
H = sigmoid(X_train @ Z)
H = np.hstack((np.ones(shape = (H.shape[0], 1)), H))
H_test = sigmoid(X_test @ Z)
H_test = np.hstack((np.ones(shape = (H_test.shape[0], 1)), H_test))
p += 1

mathbbH = np.linalg.pinv(H.T @ H + l * np.identity(p)) @ H.T
W = mathbbH @ y_train

# test
yhat = np.tanh(H_test @ W)
acc = mean_squared_error(y_test, yhat)
print('p: {} | acc: {}'.format(p, acc))

# plot
plt.figure()
plt.plot(ps, log)
