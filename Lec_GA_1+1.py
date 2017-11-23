#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:10:49 2017

@author: pmirbach
"""

import numpy as np

def fitness(x):
    return sum(x)

N = 20
x = [0] * N

f = fitness(x)
sigma = 1. / N

for i in range(50):
    
    x_ = [ (bit+1)%2 if np.random.random() < sigma else bit for bit in x]
    f_ = fitness(x_)
    
    if f_ > f:
        x = x_
        f = f_
    
    print(x, f)