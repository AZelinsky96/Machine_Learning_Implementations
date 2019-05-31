#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:26:25 2019

@author: zeski
"""

import numpy as np


w = 20

for i in range(100): 
    w = w - 0.1*2*w
    print(w)
    



N = 10

D = 3


X = np.zeros((N,D))
X[:,0] = 1


X[:5, 1] = 1
X[5:, 2]  = 1

print(X)



Y = np.array([0] *5 + [1] * 5)
print(Y)




costs = []


w = np.random.randn(D) / np.sqrt(D)

learning_rate = 0.001


for i in range(1000): 
    
    y_hat = np.matmul(X,w)
    delta= y_hat - Y
    
    
X = np.array([[0,4,5],
             [4,5,6]]
             )

print(X.shape[1])
    
