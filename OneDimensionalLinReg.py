#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:19:06 2019

@author: anthonyz
"""
import pandas as pd


from Classes.Regression_Class import Regression
## Loading the data: 
        



def main():
    
    ## Loading the data
    
    df = pd.read_csv("data_1d.csv", header = None)
    df.columns = ['x', 'y']
    
    
    X = df.iloc[:,0].values
    Y = df.iloc[:,1].values
    

    
    ## Defining a Regression object: 
    
    regressor = Regression()
    

    ## performing a linear regression to fit the best values    
    predictions = regressor.one_dim_lr(X,Y)
    
    
    ## Plotting the linear regression 
    regressor.plot_preds()
    
    
    regressor.r_squared()
    
if __name__ == "__main__": 
    main()



        



