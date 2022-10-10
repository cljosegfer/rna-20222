#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 17:06:22 2022

@author: jose
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

class embedding():
	def __init__(self, p, l):
		self.p = p
		self.l = l
		self.loss = None

	def concatenate(self, X):
		return np.hstack((np.ones(shape = (X.shape[0], 1)), X))

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def projecao(self, X):
		return X

	def train_embedding(self, X):
		self.p = X.shape[1]

	def fit(self, X, y, loo = True):
		# train embedding
		self.train_embedding(X)

		H = self.projecao(X)
		H = self.concatenate(H)

		A = np.linalg.pinv(H.T @ H + self.l * np.identity(self.p + 1))
		self.W = A @ H.T @ y

		if loo:
			N = H.shape[0]
			P = np.identity(N) - H @ A @ H.T
			sigma = P @ y / np.diag(P)
			self.loss = sigma.T @ sigma / N

	def predict(self, X, discrimina = False):
		H = self.projecao(X)
		yhat = self.sigmoid(self.concatenate(H) @ self.W)
		if discrimina:
			yhat[yhat <= 0.5] = 0
			yhat[yhat >= 0.5] = 1
		return yhat

	def acc(self, X, y):
			return accuracy_score(y, self.predict(X, discrimina = True))

	def auc(self, X, y, discrimina = True):
		return roc_auc_score(y, self.predict(X, discrimina = discrimina))

	def mse(self, X, y):
		return mean_squared_error(y, self.predict(X))

class elm(embedding):
	def __init__(self, p = 10, l = 0.1):
		super().__init__(p, l)
		self.Z = None

	def projecao(self, X):
		return self.sigmoid(self.concatenate(X) @ self.Z)

	def train_embedding(self, X):
		n = X.shape[1] + 1
		self.Z = np.random.normal(size = (n, self.p))

class autoencoder(embedding):
	def __init__(self, p = 10, l = 0.1, activation = 'logistic', solver = 'lbfgs', max_iter = 2000):
		super().__init__(p, l)
		self.model = MLPRegressor(hidden_layer_sizes = self.p, activation = activation, solver = solver, max_iter = max_iter)

	def projecao(self, X):
		return self.sigmoid(X @ self.model.coefs_[0] + self.model.intercepts_[0])

	def train_embedding(self, X):
		self.model.fit(X, X)
