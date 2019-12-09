# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:12:36 2019
@author: alexl
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Data/train.csv")

data = pd.read_csv('Data/train.csv', index_col=0, parse_dates = True)
#df.index.weekday_name
#print(data.groupby('season').count())

print(data.columns)

print(data['registered'].sum() /
      data['count'].count())

print(data['count'].count())

#sns.heatmap(data.corr())
#plt.show()

data['year'] = pd.DatetimeIndex(data.index).year

#print(data.month)
data['month'] = pd.DatetimeIndex(data.index).month
data['day'] = pd.DatetimeIndex(data.index).day
data['hour'] = pd.DatetimeIndex(data.index).hour
data['dayofweek'] = pd.DatetimeIndex(data.index).dayofweek

sns.barplot(data['season'], data['count'], hue=data['year'])
plt.show()
sns.boxplot(data['season'], data['count'], hue=data['year'])
plt.show()


sns.barplot(data['month'], data['count'], hue=data['year'])
plt.show()
sns.boxplot(data['month'], data['count'], hue=data['year'])
plt.show()

sns.barplot(data['dayofweek'], data['count'], hue=data['year'])
plt.show()
sns.boxplot(data['dayofweek'], data['count'], hue=data['year'])
plt.show()

sns.barplot(data['year'], data['count'], hue=data['weather'])
plt.show()
sns.barplot(data['year'], data['count'], hue=data['weather'])
plt.show()

X = data.drop(['count','registered','casual'], axis = 1)
y = data['count']

m = LinearRegression()
m.fit(X,y)

ypred = m.predict(X)

plt.plot (X,y,'ro')
plt.plot(X,ypred, 'b-')
plt.show()

print(m.coef_)
print(m.intercept_)
print(m.score(X,y))

#
##residuals aproaches 0?
#residuals = y - ypred
##print(residuals.mean().round(5))
#
##normally distributed?
#plt.hist(residuals, bins = 20)


