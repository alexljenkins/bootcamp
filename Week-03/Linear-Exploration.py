# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:48:32 2019

@author: alexl
"""
import FeatureEngineerer as fe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Data/train.csv', index_col=0, parse_dates = True)
X_pred = pd.read_csv('Data/test.csv', index_col=0, parse_dates = True)
#randomise the order of data
data = fe.FeatureAdder(data)
print(data.columns)

print(data['registered'].sum() /
      data['count'].count())

print(data['count'].count())

sns.heatmap(data.corr())
plt.show()

#Descriptors

plt.hist(data.windspeed,bins = 20)
plt.show()


#data['dayofweek'] = pd.DatetimeIndex(data.index).dayofweek

sns.barplot(data['season'], data['count'], hue=data['year'])
plt.show()
sns.boxplot(data['season'], data['count'], hue=data['year'])
plt.show()

sns.barplot(data['month'], data['count'], hue=data['year'])
plt.show()
sns.boxplot(data['month'], data['count'], hue=data['year'])
plt.show()

sns.barplot(data['weekday_hour'], data['count'], hue=data['year'])
plt.show()
sns.boxplot(data['weekday_hour'], data['count'], hue=data['year'])
plt.show()

sns.barplot(data['weekend_hour'], data['count'], hue=data['year'])
plt.show()
sns.boxplot(data['weekend_hour'], data['count'], hue=data['year'])
plt.show()


sns.barplot(data['year'], data['count'], hue=data['weather'])
plt.show()
sns.barplot(data['year'], data['count'], hue=data['weather'])
plt.show()



#plt.plot (X,y,'ro')
#plt.plot(X_test,ypred, 'b-')
#plt.show()
#
#print("" + m.coef_)
#print("intercept:" + m.intercept_)
#print("model score is:" + m.score(X_test,ypred))
#
#
##residuals aproaches 0?
#residuals = y - ypred
#print(residuals.mean())


#data[data['workingday'] == 1]['count'].describe()
#data[data['workingday'] == 0]['count'].describe()
#
#data['demandzone'] = data[data['count'] < 42] = 'low'
#
#plt.scatter(data.index, data['count'])
#data['demandzone'] = data[data['count'] < 42] = 'low'
#print(data.demandzone)