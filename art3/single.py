#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:21:45 2022

@author: jose
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from embedding import autoencoder
from dataset import custom
from hparams import epochs, batch_size

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

# def modificado(xhat, x, H):
#     mse = F.mse_loss(xhat, x)
#     crosstalk = 0
#     for h in H:
#         crosstalk += torch.dot(h, h)
#     crosstalk /= batch.size()[0]
    
#     return mse + crosstalk

def modificado(xhat, x, H):
    mse = F.mse_loss(xhat, x)
    crosstalk = 0
    for h in H:
        for another_h in H:
            if (h == another_h).all():
                continue
            crosstalk += torch.dot(h, another_h)
    crosstalk /= batch.size()[0] ** 2
    
    return mse + crosstalk

dataset = 'sonar'
fold_n = 0

# read
train = custom(dataset, fold_n, downsample = True)
val = custom(dataset, fold_n, train = False)
train_loader = torch.utils.data.DataLoader(train, 
                                           batch_size = batch_size, 
                                           shuffle = True)

# model
model = autoencoder(in_features = train.in_features())
optimizer = optim.Adam(model.parameters(), weight_decay = 1e-4)
criterion = nn.MSELoss()

# train
for epoch in tqdm(range(epochs)):
    for batch, y in train_loader:
        batch = batch.float()
        
        optimizer.zero_grad()
        xhat = model.forward(batch)
        # loss = criterion(xhat, batch)
        loss = modificado(xhat, batch, model.latent(batch))
        loss.backward()
        optimizer.step()

# logistic
model.fit(train.X, train.y)

# eval
yhat = model.predict(torch.Tensor(val.X))
y = val.y
auc = roc_auc_score(y, yhat)

# plot
print(auc)
idx = np.argsort(yhat)
plt.figure()
plt.scatter(np.arange(0, len(idx)), y[idx], label = r'$y$')
plt.plot(yhat[idx], label = r'$\hat{y}$', c = 'orange')
plt.title(dataset)
plt.legend()
