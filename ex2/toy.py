#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 22:29:52 2022

@author: jose
"""

import numpy as np
import matplotlib.pyplot as plt

def normal(sd, n):
    mean = (0, 0)
    cov = [[sd, 0], [0, sd]]
    x1 = np.random.multivariate_normal(mean, cov, size = n)
    
    mean = (1, 1)
    cov = [[sd, 0], [0, sd]]
    x2 = np.random.multivariate_normal(mean, cov, size = n)
    
    X = np.concatenate((x1, x2))
    y = np.array([-1] * n + [1] * n)
    
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
    y = np.array([-1] * 2 * n + [1] * 2 * n)
    
    return X, y

# param
p = 100
l = 0

# data
n = 250
sd = 0.25
X, y = xor(sd, n)

X_train = np.copy(X)
y_train = np.copy(y)

X_test = np.copy(X)
y_test = np.copy(y)

# elm
n = X_train.shape[1] + 1
Z = np.random.normal(size = (n, p))

# train
H = np.tanh(np.hstack((np.ones(shape = (X_train.shape[0], 1)), X_train)) @ Z)
W = np.linalg.pinv(H.T @ H + l * np.identity(p)) @ H.T @ y_train

# superficie
N = 1000
xv = []
xs = np.linspace(2, -2, N)
ys = np.linspace(-2, 2, N)
for xx in xs:
    for yy in ys:
        xv.append([xx, yy])
xv = np.array(xv)
hv = np.tanh(np.hstack((np.ones(shape = (xv.shape[0], 1)), xv)) @ Z)
yv = hv @ W
yv = yv.reshape(N, N)

# plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.title('p = {}, lambda = {}'.format(p, l))
plt.contour(xs, ys, yv, levels = [0])
