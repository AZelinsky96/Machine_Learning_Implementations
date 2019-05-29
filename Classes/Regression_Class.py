#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:54:55 2019

@author: anthonyz
"""



class Regression: 
    
    def __init__(self): 
        return None
        
    def one_dim_lr(self, X, Y):
        """
            This will fit the the one dimensional linear regression to the data being passed in. 
            
            Parameters: 
                
                X : The One dimensional array of X values
                
                Y:  The One dimensional array of Y values
            
            
            Returns:
                
                Predictions of formula
        """
        import numpy as np
        
        X = np.array(X)
        Y = np.array(Y)
        
        ## Calculating the one dimensional solution to LR
    
        ## To find A and B for best linear regression, you will need to take the derivatives with 
        ## Respect to both a and b in terms of the error function: E = sum((y - y^) **2) -> Sum of Squares
    
        ## The two calculas solutions for a and b in y = ax + b: 
        ## a = sum(yi*xi) - mean(y)*sum(xi) / sum(xi**2) - mean(x)*sum(xi)
        ## b = mean(y)sum(xi**2) - mean(x) * sum(yi * xi) / sum(xi**2) - mean(x)*sum(xi)
    
        ## The summation of xi**2 in denom is the same as taking x as a vector and dot producting w/self
        # PROOF

        #print("Proof X dot X == sum(Xi **2) in for loop")
        #print(round(np.sum([i ** 2 for i in X]), 2) ==  round(X.dot(X), 2))

        denominator = X.dot(X) - (np.mean(X)* np.sum(X))
        
        a = (X.dot(Y) - np.mean(Y) * np.sum(X) )/ denominator
    
        b = (np.mean(Y) * X.dot(X) - np.mean(X)* X.dot(Y)) / denominator

        self.X = X
        self.Y = Y
        self.predictions = a * X +b 
        
        
        return a * X + b
    
    
    def multi_reg(self, X,Y): 
        import numpy as np
        X = np.array(X)
        Y = np.array(Y)
        
        w = np.linalg.solve(np.matmul(X.T, X), np.matmul(X.T, Y ))
        
        predictions = np.matmul(X,w)
        self.X = X
        self.Y = Y
        self.predictions = predictions
        
        return predictions
    
    def univariate_polynomial(self, X, Y):
        import numpy as np
        X = np.array(X)
        Y = np.array(Y)
        
        
        if X.shape != (len(X), 1): 
            print(X.shape)
            print("Errors, X must be univariate and of shape ({},1)".format(len(X)))
        else: pass
        
    
        x = [[1, i, i **2] for i in X]
        
        
        X_poly = np.array(x)
        
        
        w = np.linalg.solve(np.matmul(X_poly.T, X_poly ), np.matmul(X_poly.T, Y)) 
        
        
        predictions = np.matmul(X_poly, w)
        
        
        self.X = X_poly
        self.Y = Y
        self.predictions = predictions
        
        return predictions
        
    
    def plot_preds(self): 
        
        import matplotlib.pyplot as plt
            
        plt.scatter(self.X, self.Y)
        
        plt.plot(self.X, self.predictions, c = 'red')
            
        plt.show()
        
    def r_squared(self): 
        """
        This will compute the R-squared for the fit model. 
        Return R_squared
        """
        import numpy as np
        

        #print(np.sum([(i - p) **2 for i,p in zip(self.Y, self.predictions)]))
        
        SSres = np.sum([(i-p) ** 2 for i,p in zip(self.Y, self.predictions)])
        
        SStot = np.sum([(i - np.mean(self.Y)) ** 2 for i in self.Y])
        
        r_squared = 1 - (SSres/ SStot)
        print("R_squared:")
        if r_squared >=0.75: 
            print("Great Job!: {}".format(r_squared))
        elif (r_squared >50) and (r_squared < 75): 
            print("Not bad: {}".format(r_squared))
        else: 
            print("Model Needs improvement: {}".format(r_squared))
            
            
        return r_squared
