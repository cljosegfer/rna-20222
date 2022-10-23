#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 18:41:58 2022

@author: jose
"""

import numpy as np
from embedding import elm as net
import matplotlib.pyplot as plt

def normal(sd, n):
    mean = (0, 0)
    cov = [[sd, 0], [0, sd]]
    x1 = np.random.multivariate_normal(mean, cov, size = n)
    
    mean = (1, 1)
    cov = [[sd, 0], [0, sd]]
    x2 = np.random.multivariate_normal(mean, cov, size = n)
    
    X = np.concatenate((x1, x2))
    y = np.array([0] * n + [1] * n)
    
    return X, y

def xor(sd, n):
    mean = (-1, -1)
    cov = [[sd, 0], [0, sd]]
    x1 = np.random.multivariate_normal(mean, cov, size = n)
    
    mean = (1, 1)
    cov = [[sd, 0], [0, sd]]
    x2 = np.random.multivariate_normal(mean, cov, size = n)
    
    mean = (-1, 1)
    cov = [[sd, 0], [0, sd]]
    x3 = np.random.multivariate_normal(mean, cov, size = n)
    
    mean = (1, -1)
    cov = [[sd, 0], [0, sd]]
    x4 = np.random.multivariate_normal(mean, cov, size = n)
    
    X = np.concatenate((x1, x2, x3, x4))
    y = np.array([0] * 2 * n + [1] * 2 * n)
    
    return X, y

# param
p = 100
l = 0

# data
n = 250
sd = 0.25
X_train, y_train = xor(sd, n)
X_test, y_test = xor(sd, n)

# train
model = net(p = p, l = l)
model.fit(X_train, y_train)

# eval
loss = model.loss
mse = model.mse(X_test, y_test)
auc_leve = model.auc(X_test, y_test, discrimina = False)

acc = model.acc(X_test, y_test)
auc = model.auc(X_test, y_test)

# resposta
yhat = model.predict(X_test)
idx = np.argsort(yhat)

plt.figure()
plt.scatter(idx, y_test, s = 0.5, label = r'$y$')
plt.plot(yhat[idx], label = r'$\hat{y}$')
plt.legend()
