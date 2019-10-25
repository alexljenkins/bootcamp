# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:16:54 2019
@author: alexl
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = sns.load_dataset("anscombe")

print(df)

print(df.info())
print(df.describe())

print()

df.dataset.unique()

print(df.groupby(df.dataset).mean())
print(df.groupby(df.dataset).count())
plt.scatter(df.x,df.y)

sns.pairplot(df, hue='dataset')
sns.boxplot(df.dataset,df.x)
sns.boxplot(df.dataset,df.y)


# TO DATETIME

date1= 'jan 15th 2017'
date2= '2017/01/15'
date3= '2017 05 05'

print(pd.to_datetime(date1))
print(pd.to_datetime(date2))
print(pd.to_datetime(date3))

data = pd.read_csv('Data/stock_px.csv'), index_col = 0, parse_dates = True)

data.info()
data.columns

df['year'] = df.index.year #creates a year 

#adding timedelta to timestamps
import numpy as np

#pd.to_timedelta(numbers, frequency)
pd.to_timedelta(np.ones(2214), 'D') #creates a list of 1 days of length of dataframe
data.index = data.index + pd.to_timedelta(np.ones(2214), 'D')

#can't add to datetimes together, but can add invidual
data.index[0].year + df.index[1].year


data.index[0].strftime('%U-')














