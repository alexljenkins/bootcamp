import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

# Metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

df = pd.read_fwf("Data/climate_data.txt", comment="%", skiprows =96)

df.rename(mapper = {'nt':'month',
                    'mperatu':'raw_temp',
                    'Anomal':'raw_anomaly',
                    'peratu':'adjusted_temp',
                    'Anomal.1':'adjusted_anomaly',
                    'atu':'regional_temp',
                    'mal':'regional_anomaly'},
                    axis = 1, copy=False, inplace=True)

df['day'] = 1
df['date'] = pd.to_datetime(df[['Year','month', 'day']], format="%Y %m %d")
df.set_index(keys = 'date', drop = True, inplace = True)
df.drop(['Year','month','e','k','day'], axis = 1, inplace = True)

#%%
#Chop off the top where there is blank datax
df = df.loc['1756-01-01':'2013-10-01']

# df.fillna(method='ffill', inplace = True, )
# df['raw_temp'] = df.groupby(df.index.month)['raw_temp'].fillna(method='ffill', limit = 1) # only for 1 row

# forward fill data from previous year instead of previous datapoint.
df = df.groupby(df.index.month).fillna(method='ffill', limit = 1)
# df.loc['1866'] # the year with NAs
y = df['adjusted_temp']
df.drop('adjusted_temp', axis = 1, inplace = True)

#%%

for i, column in zip(range(len(df.columns)), df.columns):
    for years in range(12, 132, 12):
        df[f'{column}_{years}'] = df.iloc[:,i].shift(years)

df = df.iloc[120:]
y = y.iloc[120:]


df.head()
# %%

# split dataset
X_train, X_test = df[1:len(X)-12], df[len(X)-12:]

# lasso_regressor = GridSearchCV(lasso, parameters, scoring = 'neg_mean_squared_error', cv = 5)


data = pd.read_csv('Data/export_all.csv')
X = data.drop(labels='Survived',axis=1)
y = data['Survived']

names = ['LinearRegression',
         'Ridge',
         'Lasso',
         'ElasticNet'
         ]

classifiers = [LinearRegression(),
               Ridge(),
               Lasso(),
               ElasticNet()
               ]
#print(LogisticRegression().get_params().keys())

parameters = [{'C':[0.1,0.5,0.8,1],'solver':['newton-cg', 'lbfgs'],'class_weight':['auto','balanced']},
               {},
               {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-2,1,5,10,20]},
               {}
               ]

answer = pd.DataFrame(columns=['Name','Score'])#,'TP','TN','FP','FN','AUC'])

for name, classifier, parameter in zip(names, classifiers, parameters):
    clf = GridSearchCV(classifier,parameter)

    m = clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    answer = answer.append({'Name':name, 'Score':score}, ignore_index=True)

print(answer.sort_values(by = 'Score', ascending = False))
