#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:30:36 2022

@author: jose
"""

from scipy import io
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import silhouette_samples as score
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def projecao(X, model):
    return sigmoid(X @ model.coefs_[0] + model.intercepts_[0])

datasets = ('australian', 
            # 'banknote', 
            'breastcancer', 
            'breastHess', 
            'bupa', 
            'climate', 
            'diabetes', 
            'fertility', 
            # 'german', 
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

dataset = 'golub'
# for dataset in tqdm(datasets):
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
p = ps[dataset] + 100
ls = np.logspace(-5, 2, 100)
print(dataset)
# l = 0.1
for l in tqdm(ls):
    # train
    model = MLPClassifier(activation = 'logistic', solver = 'lbfgs', 
                         alpha = l, hidden_layer_sizes = p, 
                         max_iter = 5000)
    model.fit(X_train, y_train)
    
    # projecao
    H_train = projecao(X_train, model)
    H_test = projecao(X_test, model)
    
    # metrica_train = np.mean(score(H_train, y_train))
    # metrica_test = np.mean(score(H_test, y_test))
    
    metrica_train = np.sum(score(H_train, y_train) < 0) / len(score(H_train, y_train))
    metrica_test = np.sum(score(H_test, y_test) < 0) / len(score(H_test, y_test))
    
    # desempenho
    yhat = model.predict(X_train)
    # acc_train = accuracy_score(yhat, y_train)
    yhat = model.predict(X_test)
    # acc_test = accuracy_score(yhat, y_test)
    
    # log
    log.append([metrica_train, metrica_test])
log = np.array(log)

# plot
plt.figure()
plt.plot(ls, log[:, 0], label = 'q')
# plt.plot(ls, log[:, 2], label = 'acc')
# plt.xlabel('p')
plt.xscale('log')
plt.xlabel('lambda')
plt.ylabel('Q')
# plt.legend()
plt.title('treinamento {}'.format(dataset))

plt.figure()
plt.plot(ls, log[:, 1], label = 'q')
# plt.plot(ls, log[:, 3], label = 'acc')
# plt.legend()
# plt.xlabel('p')
plt.xscale('log')
plt.xlabel('lambda')
plt.ylabel('Q')
plt.title('teste {}'.format(dataset))
