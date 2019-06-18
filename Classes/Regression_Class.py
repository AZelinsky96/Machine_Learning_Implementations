#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:54:55 2019

@author: anthonyz
"""



class LinearRegression:

    def __init__(self):
        return None

    def fit(self, X, Y):
        """
            This will fit the the one dimensional linear regression to the data being passed in.

            Parameters:

                X : The One dimensional array of X values

                Y:  The One dimensional array of Y values


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

        self.a = a
        self.b = b

        return print("Successfully Fit.")


    def predict(self, X):
        """
        Purpose: To utilize the parameters derived to find


        """

        self.X = X
        return self.a * X + self.b


    def r_squared(self, X, Y):
        """
        This will compute the R-squared for the fit model.
        Return R_squared
        """
        import numpy as np

        predictions = self.a * X + self.b

        SSres = np.sum([(i-p) ** 2 for i,p in zip(Y, predictions)])

        SStot = np.sum([(i - np.mean(Y)) ** 2 for i in Y])

        r_squared = 1 - (SSres/ SStot)
        print("R_squared:")
        if r_squared >=0.75:
            print("Great Job!: {}".format(r_squared))
        elif (r_squared >50) and (r_squared < 75):
            print("Not bad: {}".format(r_squared))
        else:
            print("Model Needs improvement: {}".format(r_squared))


        return r_squared


class multiple_regression:



    def __init__(self):

        return None

    def fit(self,X,Y):
        import numpy as np
        X = np.array(X)
        Y = np.array(Y)

        w = np.linalg.solve(np.matmul(X.T, X), np.matmul(X.T, Y ))

        self.w = w

        return w

    def predict(self, X):
        import numpy as np
        self.X = X


        return np.matmul(X,self.w)


    def r_squared(self, X, Y):
        """
        This will compute the R-squared for the fit model.
        Return R_squared
        """
        import numpy as np



        predictions = np.matmul(X, self.w)


        SSres = np.sum([(i-p) ** 2 for i,p in zip(Y, predictions)])

        SStot = np.sum([(i - np.mean(Y)) ** 2 for i in Y])

        r_squared = 1 - (SSres/ SStot)
        print("R_squared:")
        if r_squared >=0.75:
            print("Great Job!: {}".format(r_squared))
        elif (r_squared >50) and (r_squared < 75):
            print("Not bad: {}".format(r_squared))
        else:
            print("Model Needs improvement: {}".format(r_squared))


        return r_squared







class Univariate_Polynomial:

    def __init__(self):

        return None

    def make_polynomial(self, X, deg):
        """
        Purpose: This function will output a polynomial with one interaction term. Hence Univariate polynomial.

        Parameters:

            X    = The input Matrix of shape (n, 1)

            deg  = The degree you wish to create the polynomial

        Returns:

            Polynomial Data

        """

        import numpy as np

        ## Creating a
        n = len(X)
        data = [np.ones(n).reshape(-1,1)]

        for d in range(deg):
            data.append(X**(d+1))

        return np.hstack(data)

    def fit(self, X, Y):
        """
        Purpose: To solve for w in the regression formula


        """


        import numpy as np

        w = np.linalg.solve(np.matmul(X.T, X), np.matmul(X.T, Y))

        self.w = w

        return w

    def predict(self, X):
        """
        Purpose: To predict Yhat from the input X matrix

        Parameters:

            X = The input (n,d) matrix

        Returns:

            Yhat = A (n,1) vector


        """

        import numpy as np

        return np.matmul(X, self.w)


    def r_squared(self, X, Y):
        """
        Purpose: This will compute the R-squared for the fit model.

        Parameters:

            X = A

        Return R_squared
        """
        import numpy as np



        predictions = np.matmul(X, self.w)


        SSres = np.sum([(i-p) ** 2 for i,p in zip(Y, predictions)])

        SStot = np.sum([(i - np.mean(Y)) ** 2 for i in Y])

        r_squared = 1 - (SSres/ SStot)
        print("R_squared:")
        if r_squared >=0.75:
            print("Great Job!: {}".format(r_squared))
        elif (r_squared >50) and (r_squared < 75):
            print("Not bad: {}".format(r_squared))
        else:
            print("Model Needs improvement: {}".format(r_squared))


        return r_squared


class RidgeRegression:

    def __init__(self):
        return None

    def make_univariate_polynomial(self, X, deg):
        """
        Purpose: This function will output a polynomial with one interaction term. Hence Univariate polynomial.

        Parameters:

            X    = The input Matrix of shape (n, 1)

            deg  = The degree you wish to create the polynomial

        Returns:

            Polynomial Data

        """

        import numpy as np

        ## Creating a
        n = len(X)
        data = [np.ones(n).reshape(-1,1)]

        for d in range(deg):
            data.append(X**(d+1))

        self.poly = True

        return np.hstack(data)



    def fit(self, X, Y, lambda_):
        """
        Purpose: To solve for with an l2 regularization


        Parameters:

            X = A (n,d) matrix of input features values

            Y = A (n,1) vector of targets

            lambda_ = This will be the regularization tuner, with increase in lambda, there is increased regularization.

        Returns:

            weights
        """
        import numpy as np

        X = np.array(X)
        Y = np.array(Y)

        N = len(X)

        if self.poly != True:
            X = np.vstack([np.ones(N)], X)


        w_map = np.linalg.solve(lambda_ * np.identity(X.shape[1]) + np.matmul(X.T, X), np.matmul(X.T, Y))
        self.w = w_map


        print(w_map.shape)
        print(X.T.shape)
        return self.w


    def predict(self, X):
       """
       Purpose: This will predict the outputs of the regression

       Parameters:

           X = A (n,n) matrix of input features


       Returns:

           Predictions

       """


       import numpy as np
       X = np.array(X)

       N = len(X)

       if self.poly != True:
           X = np.vstack([np.ones(N)], X)

       predictions = np.matmul(X, self.w)

       return predictions


    def r_squared(self, X, Y):
        """
        Purpose: This will compute the R-squared for the fit model.

        Parameters:

            X = A (n,n) matrix of input features

            Y = A (n,1) vector of target values

        Return R_squared
        """
        import numpy as np


        X = np.array(X)

        N = len(X)

        if self.poly != True:
            X = np.vstack([np.ones(N)], X)
        predictions = np.matmul(X, self.w)


        SSres = np.sum([(i-p) ** 2 for i,p in zip(Y, predictions)])

        SStot = np.sum([(i - np.mean(Y)) ** 2 for i in Y])

        r_squared = 1 - (SSres/ SStot)
        print("R_squared:")
        if r_squared >=0.75:
            print("Great Job!: {}".format(r_squared))
        elif (r_squared >50) and (r_squared < 75):
            print("Not bad: {}".format(r_squared))
        else:
            print("Model Needs improvement: {}".format(r_squared))


        return r_squared

class lasso_regression:

    def __init__(self):
        return None



    def fit(self,X,Y, l1_term, learning_rate, steps, plot_rmse = False):


        import numpy as np
        import matplotlib.pyplot as plt

        X = np.array(X)
        Y = np.array(Y)

        w = np.random.randn(X.shape[1])#/ np.sqrt(X.shape[1])
        w = w.reshape(-1,1)

        cost = []

        for i in range(steps):
            y_hat = X.dot(w)

            ## Derivative of lr cost J = transpose(X) * delta(Yhat - Y)
            delta = y_hat - Y

            ## performing gradient descent
            w = w -  learning_rate * (X.T.dot(delta) + l1_term*np.sign(w))



        if plot_rmse == True:
            plt.plot(cost)
            plt.xlabel("Step")
            plt.ylabel("RMSE")
            plt.title("Cost function through gradient descent steps")
            plt.show()
        self.w = w

    def predict(self, X):

        import numpy as np
        X = np.array(X)

        return np.matmul(X, self.w)



class Logistic_Regression:

    def __init__(self):
        return None



    def fit(self, X,Y,learning_rate, epoch, plot_optim = False, l2 = False):

        import numpy as np
        ##  Setting the shapes of the dataframe
        
        D = X.shape[1]
        N = len(X)

        ## Creating N X 1 array and concatenating as a bias term for the N X D Matrix of features
        ones = np.array([[1] * N]).T
        Xb = np.concatenate((ones, X), axis = 1)

        ## Randomly initializing the weights
        w = np.random.randn(D + 1)

        #P_Y_with_X    = sigmoid(np.matmul(Xb, W))


        errors = {}
        preds = self.sigmoid(np.matmul(Xb, w))


        for i in range(epoch):
            if i % 10 == 0:
                errors[i] = (self.cross_entropy(Y, preds))
            if l2 == True:
                lambda_ = input("Enter Float value for lambda: ")
                try:
                    lambda_ = int(lambda_)
                except:
                    lambda_ = float(lambda_)
                except:
                    raise TypeError ("Invalid Entry: {}, you must enter a  numerical value".format(lambda_))


                w += learning_rate * (np.matmul((Y - preds).T, Xb) - lambda_ * w)

            preds = self.sigmoid(np.matmul(Xb,w))

        if plot_optim == True:
            self.plot_grad(errors)

        self.w = w

    def predict(self, X):
        import numpy as np
        ##  Setting the shapes of the dataframe
        D = X.shape[1]
        N = len(X)

        ## Creating N X 1 array and concatenating as a bias term for the N X D Matrix of features
        ones = np.array([[1] * N]).T
        Xb = np.concatenate((ones, X), axis = 1)

        preds = self.sigmoid(np.matmul(Xb, self.w))

        self.preds = preds

        return preds

    @staticmethod
    def sigmoid(a):
        """
        Sigmoid function applied to Linear to output values between 0 and 1
        """
        import numpy as np
        return 1 / (1 + np.exp(-a))


    @staticmethod
    def cross_entropy( Y, preds):
        """
        Function for cross entropy
        """
        import numpy as np
        E = 0
        for i in range(len(Y)):

            if preds[i] == 0:
                round_ = preds[i] + 0.0001
            elif preds[i] == 1:
                round_ = preds[i] - 0.0001
            else:
                round_ = preds[i]

            if Y[i] == 1:
                E -= np.log(round_)
            else:
                E -= np.log(1- round_)

        return E

    @staticmethod
    def plot_grad( errors):
        import matplotlib.pyplot as plt
        index_ = errors.keys()
        plt.plot(index_, errors.values())
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.title("Graph of error in iterations of gradient descent")
        plt.show()
