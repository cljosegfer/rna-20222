#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 18:58:10 2022

@author: jose
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from embedding import autoencoder
from dataset import custom
from hparams import epochs, batch_size, datasets, K

from tqdm import tqdm

from sklearn.metrics import roc_auc_score

# def modificado(xhat, x, H):
#     mse = F.mse_loss(xhat, x)
#     crosstalk = 0
#     for h in H:
#         for another_h in H:
#             if h == another_h:
#                 continue
#             crosstalk += torch.dot(h, another_h)
#     crosstalk /= torch.pow(batch.size()[0], 2)
    
#     return mse + crosstalk

def modificado(xhat, x, H):
    mse = F.mse_loss(xhat, x)
    crosstalk = 0
    for h in H:
        crosstalk += torch.dot(h, h)
    crosstalk /= torch.pow(batch.size()[0], 2)
    
    return mse + crosstalk

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
        model = autoencoder(in_features = train.in_features())
        optimizer = optim.Adam(model.parameters(), weight_decay = 1e-4)
        # criterion = nn.MSELoss()
        
        # train
        for epoch in range(epochs):
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
        auc += roc_auc_score(y, yhat) / K
    log.append(auc)
print(log)