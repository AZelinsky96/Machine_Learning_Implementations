#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:19:06 2019

@author: anthonyz
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs

import matplotlib.pyplot as plt


from Classes.Regression_Class import Regression

def unicode_decode(_list): 
    _list = [i.replace(u"\xa0", "") for i in _list]
    return _list


        



def main():
    
    # Loading the data by webscraping
    page = requests.get("https://www.thebalance.com/us-gdp-by-year-3305543")
    
    
    soup = bs(page.content, "html.parser")
    table_head = soup.find_all('th')
    table_body = soup.find_all("tr")
    
    column_head = []
    year_data   = []
    
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
    

    for tr in table_body: 
        td_tags = tr.find_all("td")
        
        td_ingest = []
        for td in td_tags: 
            td_ingest.append(td.get_text())
            
        if len(td_ingest) != 4: 
            td_ingest.append("NA")
            
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
    
        
    
    nom_gdp = nom_gdp[1:]    
    
    
    df['Nominal GDP (trillions)'] = unicode_decode(nom_gdp)
    df['Real GDP (trillions)']    = unicode_decode(real_gdp)
    df['GDP Growth Rate']         = unicode_decode(growth)
    df['Events Affecting GDP']    = unicode_decode(events)
    
    
   

    
    
    ## Performing some data manipulation
    
    def na_replace(x): 
        if "NA" in x:
            return None
        else: 
            return x
    
    
    for i in ["Year", 'Nominal GDP (trillions)' , 'Real GDP (trillions)', 'GDP Growth Rate']:
        
        if i == 'GDP Growth Rate' : 
            df[i] = df[i].apply(na_replace)
            df[i] = df[i].str.replace("%", "").astype("float")
            

            
        else: 
            df[i] = df[i].replace("NA", None)    
            df[i] = df[i].str.replace("$", "").astype("float64")    

    print(df[['Nominal GDP (trillions)', 'Real GDP (trillions)', 'GDP Growth Rate']].head())
    ## Done loading the data: 
## ------------------------------------------------------------------- 
    ## Examining the relationships between the variables and year: 
    
    def plotter(df, column):
        
        plt.figure(figsize = (10, 5))
        plt.scatter(df['Year'], df[column])
        plt.title("Scatter of {} versus Year".format(column))
        plt.xlabel("Year")
        plt.ylabel("{}".format(column))
        plt.xticks(rotation = 75, fontsize = 10)
        plt.show()
        
    for i in   ['Nominal GDP (trillions)' , 'Real GDP (trillions)', 'GDP Growth Rate']: 
         plotter(df, i)
         
         
    ## Given This data I will explore the trends with Linear regression for the second plot Real GDP
    
    X = df.iloc[:,0].values
    Y = df.iloc[:,2].values
    

    
    ## Defining a Regression object: 
    
    regressor = Regression()
    

    ## performing a linear regression to fit the best values    
    predictions = regressor.one_dim_lr(X,Y)
    
    
    ## Plotting the linear regression 
    regressor.plot_preds()
    
    
    regressor.r_squared()
    
if __name__ == "__main__": 
    main()



        



