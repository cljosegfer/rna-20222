#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:16:07 2022

@author: jose
"""

import torch
import torch.nn as nn
import torch.optim as optim

from embedding import autoencoder
from dataset import custom
from hparams import epochs, batch_size, datasets, K

from tqdm import tqdm

from sklearn.metrics import roc_auc_score

log = []
for dataset in datasets:
    auc = 0
    for fold_n in tqdm(range(K)):
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
        for epoch in range(epochs):
            for batch, y in train_loader:
                batch = batch.float()
                
                optimizer.zero_grad()
                xhat = model.forward(batch)
                loss = criterion(xhat, batch)
                loss.backward()
                optimizer.step()
        
        # logistic
        model.fit(train.X, train.y)
        
        # eval
        yhat = model.predict(torch.Tensor(val.X))
        y = val.y
        auc += roc_auc_score(y, yhat) / K
    log.append(auc)
