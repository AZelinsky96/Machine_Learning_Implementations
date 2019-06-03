import pandas as  pd
import numpy as np
import os

import sys

sys.path.append("/home/zeski/Documents/PythonLessons/MLImplementations/Classes")

import Regression_Class as RegClass


def main():
    ## Loading in the data for the course project
    df = pd.read_csv("ecommerce_data.csv")
    ## Printing the dataframe head
    print(df.head())



main()
