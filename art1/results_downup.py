#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:34:22 2022

@author: jose
"""

from scipy import io
import numpy as np
from tqdm import tqdm
from embedding import autoencoder as net
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

# elm
ps = {'australian': 20, 
        'banknote': 100, 
        'breastcancer': 20, 
        'breastHess': 10, 
        'bupa': 15, 
        'climate': 25, 
        'diabetes': 15, 
        'fertility': 5, 
        'german': 25, 
        'golub': 5, 
        'haberman': 5, 
        'heart': 15, 
        'ILPD': 10, 
        'parkinsons': 30, 
      'sonar': 30
      }

ls = {'australian': 0, 
      'banknote': -12, 
      'breastcancer': -2, 
      'breastHess': 0, 
      'bupa': -5, 
      'climate': -2, 
      'diabetes': -2, 
      'fertility': 3, 
      'german': 2, 
      'golub': 1, 
      'haberman': -2, 
      'heart': 0, 
      'ILPD': -1, 
      'parkinsons': -3, 
      'sonar': 2
      }

# autoencoder
ps = {'australian': 20,
        'banknote': 100,
        'breastcancer': 40,
        'breastHess': 5,
        'bupa': 5,
        'climate': 20,
        'diabetes': 10,
        'fertility': 5,
        'german': 20,
        'golub': 5,
        'haberman': 5,
        'heart': 10,
        'ILPD': 10,
        'parkinsons': 10,
        'sonar': 10
        }

ls = {'australian': -12, 
      'banknote': -12, 
      'breastcancer': -8, 
      'breastHess': -1, 
      'bupa': -1, 
      'climate': -1, 
      'diabetes': -1, 
      'fertility': 2, 
      'german': -1, 
      'golub': -2, 
      'haberman': -2, 
      'heart': -1, 
      'ILPD': -1, 
      'parkinsons': -1, 
      'sonar': -1
        }

loglog = []
for dataset in tqdm(datasets):
    # dataset = 'fertility'
    p = ps[dataset]
    l = np.exp(ls[dataset])
    log = []
    # print(dataset, p, l)
    for fold_n in range(K):
        # data
        filename = 'data/exportBase_{}_folds_10_exec_{}.mat'.format(dataset, fold_n + 1)
        data_mat = io.loadmat(filename)
        
        X_train = np.copy(data_mat['data']['train'][0][0])
        y_train = np.copy(data_mat['data']['classTrain'][0][0].ravel())
        y_train[y_train == -1] = 0
        
        X_test = np.copy(data_mat['data']['test'][0][0])
        y_test = np.copy(data_mat['data']['classTest'][0][0].ravel())
        y_test[y_test == -1] = 0
        
        # downsample
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
        
        X_train = X_train[border, :]
        y_train = y_train[border]
        
        # upsample
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
        
        # train
        model = net(p = p, l = l)
        model.fit(X_train, y_train)
        
        # eval
        loss = model.loss
        mse = model.mse(X_test, y_test)
        auc_leve = model.auc(X_test, y_test, discrimina = False)
        
        acc = model.acc(X_test, y_test)
        auc = model.auc(X_test, y_test)
        
        # log
        log.append([loss, mse, auc_leve, acc, auc])
    log = np.array(log)
    loglog.append(np.mean(log, axis = 0))

# export
df = pd.DataFrame(loglog, columns = ['loo', 'mse', 'auc_leve', 
                                     'acc', 'auc'])
df.to_csv('output/{}_downup.csv'.format(type(model)), index = None)
