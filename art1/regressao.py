#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 16:12:42 2022

@author: jose
"""

import numpy as np

class regressao():

	def __init__(self, p = 10, l = 0.1):
		self.p = p
		self.l = l

	def __concatenate(X):
		return np.hstack((np.ones(shape = (X.shape[0], 1)), X))

	def __sigmoid(x):
		return 1 / (1 + np.exp(-x))

	def projecao(self, X):
		return H = self.__sigmoid(X @ self.Z)

	def fit(self, X, y, loo = True):
		X = self.__concatenate(X)
		n = X_train.shape[1]
		self.Z = np.random.normal(size = (n, self.p))
		H = self.projecao(X)

		H = self.__concatenate(H)
		mathbbH = np.linalg.pinv(H.T @ H + self.l * np.identity(self.p + 1)) @ H.T
		self.W = mathbbH @ y_train

		if loo:
			h = np.diag(H @ mathbbH)
			yhat = sigmoid(H @ self.W)
			r = y_train - yhat
			press = np.sum((r / (1 - h))**2)

			return press

	def predict(self, X):
		return sigmoid(self.__concatenate(X) @ self.W)
