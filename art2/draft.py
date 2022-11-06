#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:26:55 2022

@author: jose
"""

from scipy import io
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from scipy.spatial import Voronoi

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def projecao(X, model):
    return sigmoid(X @ model.coefs_[0] + model.intercepts_[0])

datasets = ('fertility', 
            'banknote', 
            'breastcancer', 
            'breastHess', 
            'golub', 
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

ps = {'australian': 20, 
        'banknote': 50, 
        'breastcancer': 20, 
        'breastHess': 10, 
        'bupa': 15, 
        'climate': 25, 
        'diabetes': 15, 
        'fertility': 5, 
        'german': 40, 
        'golub': 5, 
        'haberman': 5, 
        'heart': 20, 
        'ILPD': 10, 
        'parkinsons': 30, 
      'sonar': 20
      }

dataset = 'sonar'
# for dataset in datasets:
fold_n = 0

# read
filename = 'data/exportBase_{}_folds_10_exec_{}.mat'.format(dataset, fold_n + 1)
data_mat = io.loadmat(filename)

X_train = np.copy(data_mat['data']['train'][0][0])
y_train = np.copy(data_mat['data']['classTrain'][0][0].ravel())
y_train[y_train == -1] = 0

X_test = np.copy(data_mat['data']['test'][0][0])
y_test = np.copy(data_mat['data']['classTest'][0][0].ravel())
y_test[y_test == -1] = 0

log = []
p = ps[dataset]
ls = np.logspace(-4, 1, 50)
# for l in tqdm(ls):
l = 0.1
# train
model = MLPRegressor(activation = 'logistic', solver = 'lbfgs', 
                     alpha = l, 
                     max_iter = 5000)
model.fit(X_train, y_train)

# projecao
H_train = projecao(X_train, model)
H_test = projecao(X_test, model)

vor = Voronoi(H_train)


# # plot
# plt.figure()
# plt.plot(np.log(ls), log, label = 'train')
# plt.plot(np.log(ls), log2, label = 'test')
# plt.legend()
# plt.title(dataset)
# plt.xlabel('lambda')
# plt.ylabel('silh')
