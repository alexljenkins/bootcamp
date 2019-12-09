# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:33:13 2019

@author: alexl
"""

import pandas as pd
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
m = DecisionTreeClassifier()


df = pd.read_csv('../Data/exported_data.csv', index_col=0)

X = df.drop('Survived')
y = df['Survived']

m.fit(X, y)
