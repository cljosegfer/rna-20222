#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:05:36 2022

@author: jose
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dataset import custom

from hparams import datasets, K

log = []
for dataset in datasets:
    auc = 0
    for fold_n in range(K):
        # read
        train = custom(dataset, fold_n, downsample = False)
        val = custom(dataset, fold_n, train = False)
        
        # model
        model = LogisticRegression()
        
        # train
        model.fit(train.X, train.y)
        
        # eval
        yhat = model.predict(val.X)
        y = val.y
        auc += roc_auc_score(y, yhat) / K
    log.append(auc)
