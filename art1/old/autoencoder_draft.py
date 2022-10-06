#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 19:06:20 2022

@author: jose
"""

from scipy import io
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
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
l = 0
log = []
ps = np.linspace(1, 100, 100, dtype = np.int)
for p in tqdm(ps):
    temp = MLPRegressor(hidden_layer_sizes = p, 
                        activation = 'logistic', 
                        solver = 'adam')
    temp.fit(X_train, X_train)
    
    Z = temp.coefs_[0]
    b = temp.intercepts_[0]
    H = sigmoid(X_train @ Z + b)
    H = np.hstack((np.ones(shape = (H.shape[0], 1)), H))
    
    mathbbH = np.linalg.pinv(H.T @ H + l * np.identity(p + 1)) @ H.T
    W = mathbbH @ y_train
    h = np.diag(H @ mathbbH)
    
    yhat = sigmoid(H @ W)
    r = y_train - yhat
    press = np.sum((r / (1 - h))**2)
    log.append(press)

p = ps[np.argmin(log)]
model = MLPRegressor(hidden_layer_sizes = p, 
                    activation = 'logistic', 
                    solver = 'adam')
model.fit(X_train, X_train)
Z = model.coefs_[0]
b = model.intercepts_[0]
H = sigmoid(X_train @ Z + b)
H = np.hstack((np.ones(shape = (H.shape[0], 1)), H))

mathbbH = np.linalg.pinv(H.T @ H + l * np.identity(p + 1)) @ H.T
W = mathbbH @ y_train

# test
H_test = sigmoid(X_test @ Z + b)
H_test = np.hstack((np.ones(shape = (H_test.shape[0], 1)), H_test))

yhat = sigmoid(H_test @ W)
acc = mean_squared_error(y_test, yhat)
print('p: {} | acc: {}'.format(p, acc))

# plot
plt.figure()
plt.plot(ps, log)
