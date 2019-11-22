# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:12:36 2019
@author: alexl
"""

import FeatureEngineerer as fe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('Data/train.csv', index_col=0, parse_dates = True)
X_pred = pd.read_csv('Data/test.csv', index_col=0, parse_dates = True)
#randomise the order of data
data = data.sample(len(data))


def share_results(a,b):
    iterater = 0
    for i in a:
        print(f'coefficient of {X.columns[iterater]}: {round(i,4)}')
        iterater +=1
    print(f'constant is: {round(b, 4)}')
    
def build_model(model, X, y):
    #Fit model
    model.fit(X, y)
    print(f'Model score was: {model.score(X,y)}')
    share_results(model.coef_, model.intercept_)
    
    return model.predict(X)

def XFeatures(data):
    #----------- Feature Engineering and Selection -----------#
    if 'count' in data.columns:
        X = data.drop(['count','registered','casual'], axis = 1)
    else:
        X = data
    
    #----------- Add year, month, day, hour to df -----------#
    X = fe.FeatureAdder(X)
    
    #----------- windspeed = 0 to RFclassifier -----------#
    #X = fe.FillWindspeed(X)
    
    #----------- buckets -----------#
    X['temp'] = pd.qcut(X['temp'], 3, labels = [0, 1, 2])
    #X = fe.OneHotEncoder(X,'temp').rename(columns = {0:'Cold',1:'Nice',2:'Hot'})
    
    #----------- one hot encode -----------#
    X = fe.OneHotEncoder(X,'season').rename(columns = {1:'Spring',2:'Summer',3:'Autumn',4:'Winter'})
    X = fe.OneHotEncoder(X,'weekday_hour').rename(columns = {0:'wh0',1:'wh1',2:'wh2',3:'wh3',4:'wh4',5:'wh5',
                        6:'wh6',7:'wh7',8:'wh8',9:'wh9',10:'wh10',11:'wh11',12:'wh12',13:'wh13',14:'wh14',15:'wh15',
                        16:'wh16',17:'wh17',18:'wh18',19:'wh19',20:'wh20',21:'wh21',22:'wh22',23:'wh23',24:'wh24'})
    
    X = fe.OneHotEncoder(X,'weekend_hour').rename(columns = {0:'we0',1:'we1',2:'we2',3:'we3',4:'we4',5:'we5',
                        6:'we6',7:'we7',8:'we8',9:'we9',10:'we10',11:'we11',12:'we12',13:'we13',14:'we14',15:'we15',
                        16:'we16',17:'we17',18:'we18',19:'we19',20:'we20',21:'we21',22:'we22',23:'we23',24:'we24'})
    X = fe.OneHotEncoder(X,'weather')
    
    #----------- Scale everything to 0-1 -----------#
    fe.FeatureScaler(X)
    
    #----------- Feature Selection -----------#
    print(X.columns)
    X = X.drop([1,'we0','wh0', 'Autumn','atemp','humidity',
            'hour','month','year','day',
            'workingday','holiday'], axis = 1)
    
    
    return X

#y = data['count']
y = np.log10(data['count'])

X = XFeatures(data)
X_test = XFeatures(X_pred)

#currently not being used!!!!!!!!!!!!!!!!!!!!
#poly = PolynomialFeatures(2)
#X = poly.fit_transform(X)


#initialise the model
#m = LinearRegression()
m = Ridge(0.1)
#L2NORM = loss = MSE+(regularisation strength)*SUM((coefficients of x)^2)

#fit the model
#m.fit(X_train,y_train)
predition_model = build_model(m, X, y)

#predict test data
ypred = m.predict(X) #data from training set
y_pred = m.predict(X_test) #kaggle predictions


from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(Polynomial, minmax,PowerTransformer(copy=True, method = 'yeo-johnson'), standardize=True),Ridge(0.001))
    
    
        )
g = GridSearchCV(pipeline, param_grid = {
        'featurename__attribute':[]
        
        })
g.fit(X,y)

#data[data['workingday'] == 1]['count'].describe()
#data[data['workingday'] == 0]['count'].describe()
#
#data['demandzone'] = data[data['count'] < 42] = 'low'
#
#plt.scatter(data.index, data['count'])
#data['demandzone'] = data[data['count'] < 42] = 'low'
#print(data.demandzone)
#re-evaluate negatives to 0
print(f'Sum of ypredictions < 0: {ypred[ypred < 0].sum()}')
print(f'Sum of kaggle ys < 0: {y_pred[y_pred < 0].sum()}')
ypred[ypred < 0] = 0
y_pred[y_pred < 0] = 0

#validate results
fe.Validator(X, y, ypred)    

#prediction compare graph
#fe.PredictionCompare(data,ypred)


#unlog and export predictions for Kaggle
y_pred = 10**y_pred
export = pd.DataFrame(y_pred, columns = ['count'], index = X_test.index) ##################
export.to_csv(f'Data/count_predictions2.csv')

