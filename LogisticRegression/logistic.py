import pandas as  pd
import numpy as np
import os

import sys


import matplotlib.pyplot as plt


sys.path.append("/home/zeski/Documents/PythonLessons/MLImplementations/Classes")

import Regression_Class as RegClass


np.random.seed(10)
def min_max(x):
    """
    Implementation of min max scaler to apply to true numerical data. This will scale data between 0 and 1.
    X must be a pandas series.
    """
    min_ = min(x)
    max_ = max(x)

    def scale(z):
        return (z - min_) / (max_ - min_)

    x = x.apply(scale)
    return x


## Implementing get_dummies from scratch
removed_dummy = {}

def dummy(x, remove, column_name):
    """
    This is a basic implementation of one hot encoding. It takes in three parameters to be called inside of a for
    loop in the get data function. This will one hot encode the variables to be encoded.

    """
    ## Setting the dimension of the encoded values
    D = len(x.unique())
    ## Setting the length of the dataset
    N = len(x)

    ## Creating an empty matrix N,D of 0's to be filled in later
    X = np.zeros((N,D))

    unique_ = x.unique()
    ## Looping through the unique values, setting all values of arrray to the column sourced from dataframe
    def replacer(val,array_):
        ## Defining my one hot encoder.
        ## This will be applied in unique loop below to replace values
        def dummy_map(a):
             if a == val:
                 return 1
             else:
                 return 0
        ## The dummy map function is used to replace all values in an iteration of unique
        ## numbers to one if they match and 0 if they dont.
        return np.array(list(map(dummy_map, array_)))

    ## This is where function above is implemented
    for i,k in enumerate(unique_):
        X[:, i]  = x
        X[:, i] = replacer(i, X[:,i])
    ## This is set in place in order to deal with the dummy variable trap.
    if remove == True:
        removed_dummy[i] = X[:, -1]

    ## Creating the column names for the one hot encode
    columns_name = []
    return_df = pd.DataFrame(X)
    ## Setting the variable names and replacing columns
    columns = [str(column_name) + str(k) for k in return_df.columns ]
    return_df.columns = columns
    return return_df


def get_data():
    """
    The get data function. This will read in the csv

    """
    df = pd.read_csv("ecommerce_data.csv")

    print(df.columns)

    #Scaling the numerical data
    for i in ["n_products_viewed", "visit_duration"]:
        df[i] = min_max(df[i])


    ## Implementation of one hot encoding. Passing the column in for loop. If len of unique values is
    ## Greater than 2, then it is passed to dummy. This in turn returns
    ## append_ which is appended to end of dataframe and the original column removed
    """
    Potentially encapsulate inside of one function
    """
    for col in ['is_mobile', "is_returning_visitor", "time_of_day", ]:

        if len(df[col].unique()) <= 2:
            pass
        else:
            append_ = dummy(df[col], False, col)

            df = pd.concat([df, append_], axis = 1, ignore_index = False)
            df.drop([col], axis = 1, inplace = True)

    ## Since this is not softmax, I will be filtering for binary data.
    df = df[df['user_action'] <= 1]
    X = df.drop(["user_action"], axis = 1).values
    Y = df.loc[:, "user_action"].values

    return X,Y



def main():
    ## Loading in the data for the course project
    X, Y = get_data()

    log_reg = RegClass.Logistic_Regression()

    log_reg.fit(X,Y, 0.01, 100, plot_optim = True)

    print(Y, log_reg.predict(X))





main()
