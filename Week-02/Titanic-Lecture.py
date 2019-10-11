# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:46:00 2019

@author: alexl
"""

import pandas as pd

df = pd.read_csv('Data/train.csv', index_col = 0)


#---- Data Exploration ----#


#show missing values
print(df.isna().sum())


#---- Feature Engineering ----#


#Dropping NA



#MUST USE DOUBLE SQUARES TO KEEP MODEL AS DF NOT A SERIES
#Matrix not vector
X = df[['Pclass','Sex','Age','Embarked']]

#Single squared bracks to make it a vector
y = df['Survived']

#create a training set and a test set of data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 0)


#---- Building the model ----#
from sklearn.linear_model import LogisticRegression
m = LogisticRegression() #place hyperparameters here

m.fit(X,y)

a = m.coef_ #coefficient(s)
print(a)
b = m.intercept_ #constant
print(b)

print(m.predict_proba([[90]]))





