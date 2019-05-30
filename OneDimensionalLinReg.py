#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:19:06 2019

@author: anthonyz
"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as bs

import matplotlib.pyplot as plt


## This is my class for my regression implementations
import Classes.Regression_Class as RegClass
from mpl_toolkits.mplot3d import Axes3D

        
def plotter(df, Y, X):
    """
    This function will be used in main to plot dataframe features against year. 


    """  
    
    if X == Y: 
        return None
    ## Creating the figure with figure size
    plt.figure(figsize = (8, 5))
    ## Scattering the figure
    plt.scatter(df[X], df[Y])
    ## Setting a title 
    plt.title("Scatter of {} versus {}".format(Y,X))
    ## Assigning x label 
    plt.xlabel("{}".format(X))
    ## Assigning Y label
    plt.ylabel("{}".format(Y))
    ## Setting X tick peculiarites
    plt.xticks(rotation = 75, fontsize = 10)
    ## Showing individual plots in loop
    plt.show()
       


def scraper():
    """
    Purpose: This function will scrape two separate websites to pull data related to GDP and population. It will created and return a Dataframe for manipulation.
   
   
    """
    
    def unicode_decode(_list): 
        """
        This function will decode the unicode characters for line break.
        
        """
        _list = [i.replace(u"\xa0", "") for i in _list]
        return _list


    def na_replace(x): 
        """
        The scaped data has NA as an object value. This will encode those values properly as None. 
        
        """
        
        if "NA" in x:
            return None
        else: 
            return x
  
    
    
    ## Requesting the first page
    page = requests.get("https://www.thebalance.com/us-gdp-by-year-3305543")
    ## Requesting the second page
    page_2 = requests.get("https://www.multpl.com/united-states-population/table/by-year")
    
    ## Utilizing beautiful soup to manipulate html data
    soup = bs(page.content, "html.parser")
    
    ## Retrieving the header tags
    table_head = soup.find_all('th')
    ## Retrieving the row tags
    table_body = soup.find_all("tr")
    
    column_head = []
    year_data   = []
    
    ## The year data and the column heads are both placed as th tags instead of year being td tags. The first five th tags are the column heads wheras the remaining are actually the column year data
    counter = 1
    for i in table_head:
        if counter <= 5:
            column_head.append(i.get_text())
            counter += 1
        
        else: 
            year_data.append(i.get_text())
            
            
            
    ## Creating the dataframe: 
    
    df = pd.DataFrame(columns = column_head)
    df = df.iloc[:, 1:]    
    df['Year'] = year_data
    df = df[['Year'] + list(df.columns)[:-1]]
    
    
    nom_gdp   = []
    real_gdp  = []
    growth    = []
    events    = []
    
    ## Retrieving the remaining column data. 
    for tr in table_body: 
        td_tags = tr.find_all("td")
        
        td_ingest = []
        for td in td_tags: 
            td_ingest.append(td.get_text())
        ## Unfortunately, there is missing data where nothing is input in the td tag. Therefore I know there should be a sequence of 4. If the sequence is broken where there is less than. The last row where
        ## The missing data is will be imputed as NA
        if len(td_ingest) != 4: 
            td_ingest.append("N/A")
        ## Looping through the td ingest. and assigning the data properly     
        counter = 1
        for i in td_ingest: 
            if counter == 1: 
                nom_gdp.append(i)
                counter += 1 
            elif counter == 2: 
                real_gdp.append(i)
                counter +=1 
            elif counter == 3: 
                growth.append(i)
                counter += 1
            else: 
                events.append(i)
                counter = 1
    
        
    ## Nominal GDP has a random NA  in there. Multiple td 
    nom_gdp = nom_gdp[1:]    
    
    ## Making sure the unicode characters are stripped
    df['Nominal GDP (trillions)'] = unicode_decode(nom_gdp)
    df['Real GDP (trillions)']    = unicode_decode(real_gdp)
    df['GDP Growth Rate']         = unicode_decode(growth)
    df['Events Affecting GDP']    = unicode_decode(events)
    
    
    
    

    
    
    ## Performing some data manipulatio. Cleaning up the numerical data. 
    
    for i in ["Year", 'Nominal GDP (trillions)' , 'Real GDP (trillions)', 'GDP Growth Rate']:
        
        if i == 'GDP Growth Rate' : 
            df[i] = df[i].apply(na_replace)
            df[i] = df[i].str.replace("%", "").astype("float")
            

            
        else: 
            df[i] = df[i].replace("NA", None)    
            df[i] = df[i].str.replace("$", "").astype("float64")        
    
    
    ## Now loading the second information: 
    soup2 = bs(page_2.content,"html.parser")
    
    ## Finding all of the table tow data
    table_body_2 = soup2.find_all("tr")
    
    header = []
    date   = []
    pop    = [] 
    for i in table_body_2: 
        if len(i.find_all("th")) > 0: 
            th_tags = i.find_all("th")
            [header.append(th.get_text().strip()[:5]) for th in th_tags]
        else: 
            td_tags = i.find_all("td")
            counter = 1
            
            for i in td_tags: 
               if counter == 1: 
                   date_text = i.get_text()[-4:]
                   date.append(float(date_text))
                   counter += 1
               else: 
                   value_text  = i.get_text().replace("million", "")
                   pop.append(float(value_text))
                   counter = 1
    
    pop_df = pd.DataFrame(columns = header)
    #print(pop_df.columns)
    
    pop_df[header[0]] = date
    pop_df[header[1]] = pop
    
    df = pd.merge(df, pop_df, left_on = "Year", right_on = "Date").drop("Date", axis = 1).rename({"Value": "Population"}, axis = 1)
  
    
    return df 











def main():
    
    # Loading the data by webscraping
    df = scraper()    
    
    ## Done loading the data: 
## ------------------------------------------------------------------- 
    ## Examining the relationships between the variables and year: 
    
    print(df.columns)
    for i in   ['Nominal GDP (trillions)' , 'Real GDP (trillions)', 'GDP Growth Rate','Population']: 
         plotter(df, i, "Real GDP (trillions)")
         
         
    ## Given This data I will explore the trends with Linear regression for the second plot Real GDP
    
    print("-"*65, "\nFitting a Regression Line to Population by Year.")
    X = df.iloc[:,0].values
    Y = df.iloc[:,-1].values
    

    
    ## Defining a linear Regression object: 
    
    regressor = RegClass.LinearRegression()
    

    ## performing a linear regression to fit the best values    
    regressor.fit(X,Y)
    
    
    predictions = regressor.predict(X)
    
    regressor.r_squared(X,Y)
    
    ## Plotting the Linear Regression to the values
    
    
    plt.figure(figsize = (10,8))
    plt.scatter(X,Y)
    plt.plot(X,predictions, c = "r")
    
    plt.title("Linear Regression for Population over Time")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend(['Regression', "Y"])
    plt.show()
    
    
    print("You can see a clear positive trend towards a growth in population as year increases.")
    
    
    
    ### Applying a multi linear regression 
    print("\n"+ "-"*65, "\nPerforming a multi linear regression.")
    print("-"*15, "\nExamining the increase of GDP as a result of population and Time")
    
    
    X = df.iloc[:,[0, -1]].values
    Y = df.iloc[:, 2].values
    
    
    multi_reg = RegClass.multiple_regression()
    
    
    multi_reg.fit(X,Y)
    
    predictions = multi_reg.predict(X)
    
    
    multi_reg.r_squared(X,Y)



        
    ## Performing a 3 dimensional plot 
    
    fig = plt.figure(figsize = (8,5))
    
    
    
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_title("3d scatterplot of GDP as a result of year and population")
    ax.scatter(X[:,0], X[:, 1], Y)
    ax.scatter(X[:,0], X[:, 1], predictions)
    
    plt.show()
    print("\nExamining the above plot, we can see that there is a clear trend for increase in GDP as a result of increase in time and population. However, The multi linear regression line we show above does not fit the curve very well. Therefore, a polynomial approach will need to be taken. ")
    
    
    
    
    
    ## Experimenting with polynomial regression: 
    
    print("\n" + "-"*65, "\nExperimenting with Polynomial Regression: GDP as a result of population")
    
    
    
    
    X = df.iloc[:, -1].values.reshape(-1,1)
    Y = df.iloc[:, 2].values.reshape(-1,1)

    
    
    
    
    univariate_poly = RegClass.Univariate_Polynomial()
    
    
    
    X_poly = univariate_poly.make_polynomial(X,2)
    
    print("Created the polynomial below: ")
    print(X_poly[:5])
    
    univariate_poly.fit(X_poly, Y)
    
    predictions = univariate_poly.predict(X_poly)
    
    
    
    
    

    
    ## Plotting 
    plt.scatter(X,Y) 
    plt.plot(X, predictions, c = 'r')
    plt.title("GDP resultant of Population")
    plt.xlabel("Population")
    plt.ylabel("GDP")
    plt.show()
    
    univariate_poly.r_squared(X_poly,Y)
    

    ## Playing around with l2 regularization 
    
    
    Y[-2] += 15
    Y[-3] += 18
    Y[-5] += 21

    plt.scatter(X,Y)
    
    
    poly_overfit = RegClass.Univariate_Polynomial()
    
    poly_overfit.fit(X_poly, Y)
    
    
    
    predictions = poly_overfit.predict(X_poly)

    
    plt.plot(X, predictions, c = 'r', label = "Outlier")

    poly_l2 = RegClass.RidgeRegression()
    
    X_poly = poly_l2.make_univariate_polynomial(X, 2)

    poly_l2.fit(X_poly, Y, 100)
    
    
    predictions = poly_l2.predict(X_poly)
    
    
    plt.plot(X, predictions, c = 'g', label = "Regularized")
    plt.title("Regularized versus Non-Regularized Polynomial on Univariate Data")
    plt.xlabel("Population")
    plt.ylabel("GDP")
    plt.legend()
    plt.show()
    
    
    poly_overfit.r_squared(X_poly, Y)
    poly_l2.r_squared(X_poly, Y)
if __name__ == "__main__": 
    main()



        



