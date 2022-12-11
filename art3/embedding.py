#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 10:32:13 2022

@author: jose
"""

import torch
import torch.nn as nn
import numpy as np

class autoencoder(nn.Module):
    def __init__(self, p = 100, l = 1e-3, **kwargs):
        super().__init__()
        self.p = p
        self.l = l
        
        self.encoder = nn.Sequential(
            nn.Linear(kwargs['in_features'], self.p), 
            nn.Sigmoid())
        
        self.decoder = nn.Sequential(
            nn.Linear(self.p, kwargs['in_features']))
        
        self.W = None
    
    def forward(self, x):
        latent = self.encoder(x)
        yhat = self.decoder(latent)
        return yhat
    
    def latent(self, x):
        return self.encoder(x)
    
    def fit(self, x, y):
        H = self.encoder(torch.Tensor(x)).detach().numpy()
        H = self._concatenate(H)
        
        A = np.linalg.pinv(H.T @ H + self.l * np.identity(self.p + 1))
        self.W = A @ H.T @ y
    
    def _concatenate(self, X):
        return np.hstack((np.ones(shape = (X.shape[0], 1)), X))
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, x):
        H = self.encoder(x).detach().numpy()
        yhat = self._sigmoid(self._concatenate(H) @ self.W)
        return yhat

class mlp(nn.Module):
    def __init__(self, p = 100, **kwargs):
        super().__init__()
        self.p = p
        
        self.encoder = nn.Sequential(
            nn.Linear(kwargs['in_features'], self.p), 
            nn.Sigmoid())
        
        self.decoder = nn.Sequential(
            nn.Linear(self.p, 1), 
            nn.Sigmoid())
    
    def forward(self, x):
        latent = self.encoder(x)
        xhat = self.decoder(latent)
        return xhat
    
    def latent(self, x):
        return self.encoder(x)
