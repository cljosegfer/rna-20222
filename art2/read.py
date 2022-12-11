#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 23:30:10 2022

@author: jose
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

datasets = ('australian', 
            # 'banknote', 
            'breastcancer', 
            'breastHess', 
            'bupa', 
            'climate', 
            'diabetes', 
            'fertility', 
            # 'german', 
            'golub', 
            'haberman', 
            'heart', 
            'ILPD', 
            'parkinsons', 
            'sonar')

mode = 'p'
# dataset = 'bupa'
for dataset in tqdm(datasets):
    log = np.load('output/{}/{}.npy'.format(mode, dataset))
    
    if mode == 'lambda':
        ls = np.logspace(-5, 2, 100)
    else:
        ls = np.linspace(1, 100, 100, dtype = int)
    
    plt.figure()
    plt.plot(ls, log[:, 0], label = 'q')
    # plt.plot(ls, log[:, 2], label = 'acc')
    plt.xlabel('p')
    # plt.xscale('log')
    # plt.xlabel('lambda')
    plt.ylabel('Q')
    # plt.legend()
    plt.title('treinamento {}'.format(dataset))
    plt.savefig('fig/{}/{}_train.png'.format(mode, dataset), dpi = 300)
    plt.close()
    
    plt.figure()
    plt.plot(ls, log[:, 1], label = 'q')
    # plt.plot(ls, log[:, 3], label = 'acc')
    # plt.legend()
    plt.xlabel('p')
    # plt.xscale('log')
    # plt.xlabel('lambda')
    plt.ylabel('Q')
    plt.title('teste {}'.format(dataset))
    plt.savefig('fig/{}/{}_test.png'.format(mode, dataset), dpi = 300)
    plt.close()
