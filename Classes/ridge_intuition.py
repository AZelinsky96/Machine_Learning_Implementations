#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 23:48:34 2019

@author: zeski
"""

import numpy as np
import matplotlib.pyplot as plt

N = 50

X = np.linspace(0,10, N)
Y = 0.5*X + np.random.randn(N)



Y[-1] += 30
Y[-2] += 60 

plt.scatter(X,Y)

X = np.vstack([np.ones(N), X]).T


w_ml = np.linalg.solve(np.matmul(X.T, X), np.matmul(X.T, Y))

Y_hat_ml = np.matmul(X, w_ml)



plt.plot(X[:,1],Y_hat_ml, c = 'y')


lambda_ = 2000

w_map = np.linalg.solve(lambda_*np.identity(3) + np.matmul(X.T, X), np.matmul(X.T,Y))

yhat_l2 = np.matmul(X,w_map)


plt.plot(X[:,1 ], yhat_l2 , c = 'r')
plt.legend(['reg', 'l2'])

plt.show()
