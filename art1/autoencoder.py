#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 19:56:12 2022

@author: jose
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

class autoencoder():

	def __init__(self, p = 10, l = 0.1, activation = 'logistic', solver = 'lbfgs', max_iter = 200):
		self.p = p
		self.model = MLPRegressor(hidden_layer_sizes = self.p, activation = activation, solver = solver, max_iter = max_iter)
		self.l = l
		self.loss = None

	def __concatenate(self, X):
		return np.hstack((np.ones(shape = (X.shape[0], 1)), X))

	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def projecao(self, X):
		return self.__sigmoid(X @ self.model.coefs_[0] + self.model.intercepts_[0])

	def fit(self, X, y, loo = True):
		self.model.fit(X, X)
		H = self.projecao(X)

		H = self.__concatenate(H)
		mathbbH = np.linalg.pinv(H.T @ H + self.l * np.identity(self.p + 1)) @ H.T
		self.W = mathbbH @ y

		if loo:
			h = np.diag(H @ mathbbH)
			yhat = self.__sigmoid(H @ self.W)
			r = y - yhat
			press = np.sum((r / (1 - h))**2)

			self.loss = press

	def predict(self, X, discrimina = False):
		H = self.projecao(X)
		yhat = self.__sigmoid(self.__concatenate(H) @ self.W)
		if discrimina:
			yhat = np.sign(yhat)
			yhat[yhat == -1] = 0
		return yhat

	def acc(self, X, y):
		return accuracy_score(y, self.predict(X, discrimina = True))

	def auc(self, X, y):
		return roc_auc_score(y, self.predict(X, discrimina = True))

	def mse(self, X, y):
		return mean_squared_error(y, self.predict(X))
