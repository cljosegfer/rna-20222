#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:30:43 2022

@author: jose
"""

import torch
import torch.nn as nn
import torch.optim as optim

from embedding import mlp
from dataset import custom
from hparams import epochs, batch_size

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_auc_score

dataset = 'sonar'
fold_n = 0

# read
train = custom(dataset, fold_n, downsample = True)
val = custom(dataset, fold_n, train = False)
train_loader = torch.utils.data.DataLoader(train, 
                                           batch_size = batch_size, 
                                           shuffle = True)

# model
model = mlp(in_features = train.in_features())
optimizer = optim.Adam(model.parameters(), weight_decay = 1e-4)
criterion = nn.BCELoss()

# train
for epoch in range(epochs):
    for batch, y in train_loader:
        batch = batch.float()
        y = y.reshape(-1, 1).float()
        
        optimizer.zero_grad()
        yhat = model.forward(batch)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()

# eval
with torch.no_grad():
    yhat = model.forward(torch.Tensor(val.X).float())
    loss = criterion(yhat, torch.Tensor(val.y.reshape(-1, 1)))
yhat = yhat.numpy().reshape(-1)
y = val.y.reshape(-1)
auc = roc_auc_score(y, yhat)

# plot
print(auc)
idx = np.argsort(yhat)
plt.figure()
plt.scatter(np.arange(0, len(idx)), y[idx], label = r'$y$')
plt.plot(yhat[idx], label = r'$\hat{y}$', c = 'orange')
plt.title(dataset)
plt.legend()
