#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 23:43:27 2022

@author: jose
"""

from scipy import io
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt

# param
K = 10

base = [1, 3, 5, 7, 9]
ps = []
for multiple in [1, 10, 100]:
    parcial = [item * multiple for item in base]
    ps += parcial
p = 300

ls = []
for multiple in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    parcial = [item * multiple for item in base]
    ls += parcial
l = 0

log = []
# for p in tqdm(ps):
for l in tqdm(ls):
    acc_train = 0
    acc_test = 0
    for fold_n in range(K):
        # data
        filename = 'data/exportBase_german_folds_10_exec_{}.mat'.format(fold_n + 1)
        data_mat = io.loadmat(filename)
        
        X_train = data_mat['data']['train'][0][0]
        y_train = data_mat['data']['classTrain'][0][0].ravel()
        X_test = data_mat['data']['test'][0][0]
        y_test = data_mat['data']['classTest'][0][0].ravel()
        
        # elm
        n = X_train.shape[1] + 1
        Z = np.random.normal(size = (n, p))
        
        # train
        H = np.tanh(np.hstack((np.ones(shape = (X_train.shape[0], 1)), X_train)) @ Z)
        W = np.linalg.pinv(H.T @ H + l * np.identity(p)) @ H.T @ y_train
        yhat = np.sign(H @ W)
        acc_train += mean_squared_error(y_train, yhat) / 10
        
        # test
        H = np.tanh(np.hstack((np.ones(shape = (X_test.shape[0], 1)), X_test)) @ Z)
        yhat = np.tanh(H @ W)
        acc_test += mean_squared_error(y_test, yhat) / 10
        
    log.append([acc_train, acc_test])
log = np.array(log)

plt.figure()
# plt.plot(np.log(ps), log[:, 0], label = 'train')
# plt.plot(np.log(ps), log[:, 1], label = 'test')
# plt.xlabel('log(p)')
plt.plot(np.log(ls), log[:, 0], label = 'train')
plt.plot(np.log(ls), log[:, 1], label = 'test')
plt.xlabel('log(lambda)')
plt.ylabel('mse')
plt.legend()
