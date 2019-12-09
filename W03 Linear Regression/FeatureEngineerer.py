# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:08:36 2019

@author: alexl
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os

def Difference(df):
#    creating a difference column
    diflist = list()
    for i in range(1, len(df)):
    	diflist.append(df['count'][i] - df['count'][i - 1])
    diflist.insert(0, 0)
    df['difference'] = diflist
    
    return df

def FeatureAdder(df):
#    
#    #adds year, month, day, hour to df
    df['year'] = pd.DatetimeIndex(df.index).year
    df['month'] = pd.DatetimeIndex(df.index).month
    df['day'] = pd.DatetimeIndex(df.index).day
    df['hour'] = pd.DatetimeIndex(df.index).hour
    
    #adding weekend hours and weekday hours
    df['weekday_hour'] = (df['hour']+1) * df['workingday']
    df['weekend_hour'] = (df['hour']+1) * (1 - df['workingday'])
    
    #creating month-year identifier
    df['month_year'] = df['month'] + df['year'] * 100 - 200000
    
    return df

def FillWindspeed(df):
    """
    By Vivekanandan Srinivasan
    https://medium.com/analytics-vidhya/how-to-finish-top-10-percentile-in-bike-sharing-demand-competition-in-kaggle-part-2-29e854aaab7d
    """
    
    from sklearn.ensemble import RandomForestClassifier
    wCol= ["season","weather","humidity","month","temp","year","atemp"]
    
    dataWind0 = df[df["windspeed"]==0]
    dataWindNot0 = df[df["windspeed"]!=0]
    dataWindNot0["windspeed"] = dataWindNot0["windspeed"].astype(str)
    
    rfModel_wind = RandomForestClassifier()
    rfModel_wind.fit(dataWindNot0[wCol], dataWindNot0["windspeed"])
    wind0Values = rfModel_wind.predict(X= dataWind0[wCol])
    dataWind0["windspeed"] = wind0Values
    df = dataWindNot0.append(dataWind0)
    df["windspeed"] = df["windspeed"].astype(float)
#    df.reset_index(inplace=True)
#    df.drop('index',inplace=True,axis=1)
    
    return df


def OneHotEncoder(df, column):
    """
    Encodes a column into multiple binary columns.
    Requires DF and column name
    """
    ohencoder = pd.get_dummies(df[column])
    #merge and drop useless columns
    df = pd.concat((df, ohencoder), axis=1)
    df.drop([column], axis=1, inplace=True)

    return df


def FeatureScaler(df):
    """
    Scales features to values between 0-1
    Requires DF
    """
    scaler = MinMaxScaler()
    for column in df.columns:
        try:
            df[column] = scaler.fit_transform(df[[column]])
        except:
            continue
            
    return df
    

def Validator(X, y, ypred):
    
    resids = ypred - y
    #Durbin Watson
    from statsmodels.stats.stattools import durbin_watson
    print(f"Durbin Watson Score (around 2 is good): {round(durbin_watson(resids, axis=0), 4)}")
    
    #RMSLE
    from sklearn.metrics import mean_squared_log_error
    print(f'Root Mean Squared Log Error is: {np.sqrt(mean_squared_log_error(y, ypred))}')
    
    #QQ-plot for our residuals --> no homoscedasticity
    import statsmodels.api as sm
    sm.qqplot(resids, line='r')
    plt.title(f"How much the quartiles are represented by the dataset")
    plt.show()

    #VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
    vifs = [VIF(X.values, i) for i, colname in enumerate(X)]
    s = pd.Series(vifs, index=X.columns)
    s.plot.bar()
    plt.title(f"VIF Analysis. Testing linear dependencies (<5 is good)\nthe residual is (0 is good): {round(resids.sum(), 2)}")
    plt.show()
    
    #correlation heatmap
#    sns.heatmap(X.corr())
#    plt.show()

    #Y vs Ypred
#    plt.scatter(x = y, y = y)
#    plt.plot(y,ypred, color = 'red')
#    plt.title(f"Y vs Ypred. (following the line is good)")
#    plt.show()
    
    #plot residuals
    plt.hist(resids, bins = 20)
    plt.title(f"Residuals in 20 bins (should be normally distributed)")
    plt.show()


def BinTime(df):
    #group
    demand = [-100, 41, 277, 10000]
    labels = [0,1,2]
    df['time'] = pd.cut(df["count"], demand, labels = labels)



















def graph(avp, month, year):
    """
    Creates a scatter plot of actual vs predicted values for a given timeperiod,
    life expectancy and population for
    each country for a single year
    """
    start = pd.to_datetime(f'{year}-{month}-01')
    if month == 12:
        end = pd.to_datetime(f'{year+1}-1-01')
    else:
        end = pd.to_datetime(f'{year}-{month+1}-01')
    
    timedf = avp.loc[(avp.index > start) & (avp.index < end)]
    
    #dont know how to extract a single year from this
    plt.scatter(timedf.index, timedf['actual'], c = 'b')
    plt.scatter(timedf.index, timedf['prediction'], c = 'r')
    plt.xlim(timedf.index.min(), timedf.index.max())
    plt.ylim(0, 1000) #timedf.values.max()+2)
    plt.title(f'Actual vs Predicted bike demand {month}-{year}')
#    plt.scatter(X.index[year,month], y[year,month], c='b')
#    plt.scatter(X.index[0:100],ypred[0:100], c='r')
    plt.savefig('temp_image.png')
    plt.close()
     #kept this outside function, so function is still useful for single use

def PredictionCompare(df, ypred):
    
    images = []
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    years = [2011,2012]
    avp = pd.DataFrame()
    avp['actual'] = df['count']
    avp['prediction'] = 10**ypred
    avp['prediction'] = round(avp['prediction'],0)
    for year in years:
        for month in months:
            graph(avp,month,year)
            images.append(imageio.imread('temp_image.png'))
            os.remove('temp_image.png')
    
    imageio.mimsave('final.gif', images, fps=1)
     
#plt.scatter(X.index[0:100], y[0:100], c='b')
#plt.scatter(X.index[0:100],ypred[0:100], c='r')
#plt.show()

#
#plt.scatter(X.loc[2011-01-01:2011-01-05], y[2011-01-01:2011-01-05], c='b')
#plt.scatter(X.index[2011-01-01:2011-01-05], y[2011-01-01:2011-01-05], c='b')
#plt.scatter(X.index[2011][01], y[2011][01], c='b')
#plt.scatter(X.index[2011, 01], y[2011, 01], c='b')
#
#plt.scatter(y.index['2011-01-01':'2011-01-05'], y['2011-01-01':'2011-01-05'], c='b')
#
#plt.scatter(y.index, y)
#year = pd.DataFrame(y['2011'])
#
#day = y['2011-07-04':'2011-07-04']
#print(day)
#plt.scatter(day.index, day.values)
#plt.xlim(day.index[0] - timedelta(hours=1), day.index[-1] + timedelta(hours=1))
#plt.ylim(0, day.values.max()+2)
#plt.show()
#
#day = y['2011-07-04':'2011-07-04']
#day_pred = ypred['2011-07-04':'2011-07-04']
#y['prediction'] = ypred

#avp = pd.DataFrame()
#avp['actual'] = data['count']
#avp['prediction'] = 10**ypred
#avp['prediction'] = round(avp['prediction'],0)
#print(avp)
#print(avp.loc[['2011-01-01'],['actual']])
#
#
#start = '2011-01-01'
#end = '2011-01-02'
#print(avp.loc[(avp.index > start) & (avp.index < end), 'actual'])
##dont know how to extract a single year from this
#plt.scatter(avp['2011-01-01'].index, avp.loc[['2011-01-01'],['actual']], c = 'b')
#plt.scatter(avp['2011-01-01'].index, avp['prediction'], c = 'r')
#plt.xlim(avp.index.min(), avp.index.max())
#plt.ylim(0, avp.values.max()+2)
#plt.show()

















