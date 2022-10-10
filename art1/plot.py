#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 09:22:47 2022

@author: jose
"""

from scipy import io
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from embedding import autoencoder as net

def plot(ps, log, num, dataset, logaritmo = False):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    if logaritmo:
        lns1 = ax1.plot(np.log(ps)[:num], log[:num, 0], color = 'b', label = 'loo')
        ax1.set_xlabel('log lambda')
        lns2 = ax2.plot(np.log(ps)[:num], log[:num, 1], color = 'r', label = 'mse')
    else:
        lns1 = ax1.plot(ps[:num], log[:num, 0], color = 'b', label = 'loo')
        ax1.set_xlabel('p')
        lns2 = ax2.plot(ps[:num], log[:num, 1], color = 'r', label = 'mse')
    ax1.set_ylabel('loo')
    ax1.set_title(dataset)
    ax2.set_ylabel('mse')
    
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc = 0)
    
    plt.show()    
    fig.savefig('fig/{}.png'.format(dataset))

dataset = 'fertility'
log = np.load('old/l/{}.npy'.format(dataset))
ps = np.linspace(1, 100, 1000, dtype = int)
ls = np.logspace(-5, 1, 100)

num = 100
plot(ls, log, num, dataset, logaritmo = True)
