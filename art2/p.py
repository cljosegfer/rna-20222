#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:26:55 2022

@author: jose
"""

from scipy import io
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance_matrix

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def projecao(X, model):
    return sigmoid(X @ model.coefs_[0] + model.intercepts_[0])

def lara_graph(X):
    D = distance_matrix(X, X) ** 2
    D[np.diag_indices(D.shape[0])] = 1e6

    n = X.shape[0]
    adjacencia = np.zeros(shape = (n, n))
    for i in range(n-1):
        for j in range(i+1, n):
            minimo = min(D[i, :] + D[j, :])
            if (D[i, j] <= minimo):
                adjacencia[i, j] = 1
                adjacencia[j, i] = 1
    return adjacencia

def q_index(X, y, gg):
    scores = []
    for i, row in enumerate(gg):
        vizinhos = np.where(row == 1)[0]
        
        degree = len(vizinhos)
        opposite = 0
        for vizinho in vizinhos:
            opposite += np.exp(-np.linalg.norm(X[i] - X[vizinho])) * np.abs(y[i] - y[vizinho]) / 2
        q = 1 - opposite / degree
        scores.append(q)
    scores = np.array(scores)
    return scores

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

dataset = 'sonar'
for dataset in datasets:
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
        
        gg_train = lara_graph(H_train)
        metrica_train = np.mean(q_index(H_train, y_train, gg_train))
        gg_test = lara_graph(H_test)
        metrica_test = np.mean(q_index(H_test, y_test, gg_test))
        
        # desempenho
        yhat = model.predict(X_train)
        acc_train = accuracy_score(yhat, y_train)
        yhat = model.predict(X_test)
        acc_test = accuracy_score(yhat, y_test)
        
        log.append([metrica_train, metrica_test, acc_train, acc_test])
    np.save('output/{}.npy'.format(dataset), log)
