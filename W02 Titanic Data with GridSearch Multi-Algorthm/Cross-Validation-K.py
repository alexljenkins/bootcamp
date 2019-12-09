# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:27:19 2019
Cross Validation K
@author: alexl
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Data/train.csv', index_col = 0)
df = df[['Age','Pclass','SibSp','Survived']]
df = df.dropna()

X = df[['Pclass','Age', 'SibSp']]
y = df['Survived']

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)

print(Xtrain.shape, ytrain.shape)

from sklearn.linear_model import LogisticRegression
m = LogisticRegression(solver = 'lbfgs')

m.fit(Xtrain,ytrain)
print(m.score(Xtrain,ytrain))

from sklearn.model_selection import cross_val_score
#High variance means overfitting or sampling bias
result = cross_val_score(m,Xtrain,ytrain, cv=3)
print(result)
print(m.score(Xtrain,ytrain))

