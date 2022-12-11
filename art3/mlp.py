#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 10:55:09 2022

@author: jose
"""

import torch
import torch.nn as nn
import torch.optim as optim

from embedding import mlp
from dataset import custom
from hparams import epochs, batch_size, datasets, K

from tqdm import tqdm
# import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

dataset = 'breastcancer'
fold_n = 0

log = []
for dataset in datasets:
    auc = 0
    for fold_n in tqdm(range(K)):
        # read
        train = custom(dataset, fold_n)
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
        # auc = roc_auc_score(y, yhat)
        auc += roc_auc_score(y, yhat) / K
    log.append(auc)
        
    # # plot
    # idx = np.argsort(yhat)
    # plt.figure()
    # plt.scatter(np.arange(0, len(idx)), y[idx], label = r'$y$')
    # plt.plot(yhat[idx], label = r'$\hat{y}$', c = 'orange')
    # plt.title(dataset)
    # plt.legend()
