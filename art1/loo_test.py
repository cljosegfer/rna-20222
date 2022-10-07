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
from embedding import elm as net

def plot(ps, log, num, dataset, logaritmo):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    if logaritmo:
        lns1 = ax1.plot(np.log(ps)[:num], log[:num, 0], color = 'b', label = 'loo')
        ax1.set_xlabel('log lambda')
        lns2 = ax2.plot(np.log(ps)[:num], log[:num, 1], color = 'r', label = 'mse')
    else:
        lns1 = ax1.plot(ps[:num], log[:num, 0], color = 'b', label = 'loo')
        ax1.set_xlabel('p')
        lns2 = ax2.plot(ps[:num], log[:num, 1], color = 'r', label = 'mse')
    ax1.set_ylabel('loo')
    ax1.set_title(dataset)
    ax2.set_ylabel('mse')
    
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc = 0)
    
    plt.show()    
    fig.savefig('fig/loo_test/{}.png'.format(dataset))
    
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

dataset = 'sonar'
for fold_n in tqdm(range(K)):
    # data
    filename = 'data/exportBase_{}_folds_10_exec_{}.mat'.format(dataset, fold_n + 1)
    data_mat = io.loadmat(filename)
    
    X_train = np.copy(data_mat['data']['train'][0][0])
    y_train = np.copy(data_mat['data']['classTrain'][0][0].ravel())
    y_train[y_train == -1] = 0
    
    X_test = np.copy(data_mat['data']['test'][0][0])
    y_test = np.copy(data_mat['data']['classTest'][0][0].ravel())
    y_test[y_test == -1] = 0

# # reproducao
# H = np.hstack((np.ones(shape = (X_train.shape[0], 1)), X_train))
# y = np.copy(y_train)

# l = 0.1
# p = H.shape[1]
# A = np.linalg.pinv(H.T @ H + l * np.identity(p))

# W = A @ H.T @ y

# n = H.shape[0]
# P = np.identity(n) - H @ A @ H.T

# Py = P @ y
# diagP = np.diag(P)
# loo = Py / diagP
# metrica = loo.T @ loo / n

    # p
    log = []
    loss = 1e6
    l = 0
    p = 30
    # ps = np.linspace(1, 100, 1000, dtype = int)
    ls = np.logspace(-5, 2, 1000)
    # for p in ps:
    for l in ls:
        temp = net(p = p, l = l)
        temp.fit(X_train, y_train)
        # log.append([temp.loss, 
        #             temp.mse(X_test, y_test), 
        #             temp.auc(X_test, y_test), 
        #             temp.acc(X_test, y_test)])
        log.append([temp.loss, temp.mse(X_test, y_test)])
        
    if fold_n == 0:
        loglog = np.array(log) / K
    else:
        loglog += np.array(log) / K

# plot
log = np.copy(loglog)
num = 1000
plot(ls, log, num, dataset, logaritmo = True)

# # ----------------------------------
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()

# num = 1000
# lns1 = ax1.plot(ps[:num], log[:num, 0], color = 'b', label = 'loo')
# # ax1.plot(np.log(ls), log[:, 0], color = 'b')
# ax1.set_xlabel('p')
# ax1.set_ylabel('loo')
# ax1.set_title(dataset)
# lns2 = ax2.plot(ps[:num], log[:num, 1], color = 'r', label = 'mse')
# # ax2.plot(np.log(ls), log[:, 1], color = 'r')
# ax2.set_ylabel('mse')

# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)
# # ----------------------------------
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()

# ax1.plot(ps, log[:, 0], color = 'b')
# # ax1.plot(np.log(ls), log[:, 0], color = 'b')
# ax1.set_xlabel('p')
# ax1.set_ylabel('press')
# ax1.set_title(dataset)
# ax2.plot(ps, log[:, 2], color = 'r')
# # ax2.plot(np.log(ls), log[:, 2], color = 'r')
# ax2.set_ylabel('auc')
# # ----------------------------------
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()

# ax1.plot(ps, log[:, 0], color = 'b')
# # ax1.plot(np.log(ls), log[:, 0], color = 'b')
# ax1.set_xlabel('p')
# ax1.set_ylabel('press')
# ax1.set_title(dataset)
# ax2.plot(ps, log[:, 3], color = 'r')
# # ax2.plot(np.log(ls), log[:, 3], color = 'r')
# ax2.set_ylabel('acc')
# # ----------------------------------
