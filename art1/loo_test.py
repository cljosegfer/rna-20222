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
from embedding import autoencoder as net

def plot(ps, log, num, dataset, logaritmo = False):
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
    fig.savefig('fig/loo_test/autoencoder/p/{}.png'.format(dataset))
    
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

# dataset = 'australian'
for dataset in datasets:
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
    
        # p
        log = []
        l = 0
        p = 30
        ps = np.linspace(1, 100, 1000, dtype = int)
        # ls = np.logspace(-5, 1, 1000)
        for p in tqdm(ps):
        # for l in ls:
            temp = net(p = p, l = l, max_iter = 2000)
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
    plot(ps, log, num, dataset, logaritmo = False)
