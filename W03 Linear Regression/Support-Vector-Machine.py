# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:14:30 2019

@author: alexl
"""
import LinearRegression as lr
import FeatureEngineerer as fe
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, PowerTransformer
from sklearn.svm import SVR

#Supress ALL Warnings
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('Data/train.csv', index_col=0, parse_dates = True)
X_pred = pd.read_csv('Data/test.csv', index_col=0, parse_dates = True)
#randomise the order of data
data = data.sample(len(data))

#y = data['count']
y = np.log10(data['count']) #use linear regression y values


def DropFeatures(df):
    if 'count' in df.columns:
        X = df.drop(['count','registered','casual'], axis = 1)
    else:
        X = df
    X = X.drop(['atemp','humidity'], axis = 1)
    return X

def AddFeatures(df):
#    #adds year, month, day, hour to df
    df['year'] = pd.DatetimeIndex(df.index).year
    df['month'] = pd.DatetimeIndex(df.index).month
    df['day'] = pd.DatetimeIndex(df.index).day
    df['hour'] = pd.DatetimeIndex(df.index).hour
    
    return df


def Features(df):
#    df = DropFeatures(df)
#    df = AddFeatures(df)
    df = lr.XFeatures(df) #use Linear Regression Feature Engineering
    return df


X = Features(data)
X_pred = Features(X_pred)

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42) #comment out

pipeline = make_pipeline(
        PolynomialFeatures(1),
        MinMaxScaler(),
        PowerTransformer(copy=True, method = 'yeo-johnson',standardize=True),
        SVR()
        )

g = GridSearchCV(pipeline, cv = 2, n_jobs = -1, param_grid = {
        'polynomialfeatures__degree':[1],
        'svr__kernel':['rbf'],
        'svr__C':[1],
        'svr__gamma':[0.1],
        'svr__epsilon':[10]
        })

g.fit(X,y)

#predict test data
y_pred = g.predict(X_pred) #kaggle predictions
y_pred = 10**y_pred

export = pd.DataFrame(y_pred, columns = ['count'], index = X_pred.index) ##################
export.to_csv(f'Data/SVR_kaggle.csv')

#dir(g)
g.best_params_
print(g.best_score_)