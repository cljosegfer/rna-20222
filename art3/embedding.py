#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 10:32:13 2022

@author: jose
"""

import torch.nn as nn

class autoencoder(nn.Module):
    def __init__(self, p = 100, **kwargs):
        super().__init__()
        self.p = p
        
        self.encoder = nn.Sequential(
            nn.Linear(kwargs['in_features'], self.p), 
            nn.Sigmoid())
        
        self.decoder = nn.Sequential(
            nn.Linear(self.p, kwargs['in_features']))
    
    def forward(self, x):
        latent = self.encoder(x)
        yhat = self.decoder(latent)
        return yhat
    
    def latent(self, x):
        return self.encoder(x)

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
