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

def shift_10_years(df, answer):
    #shift data back 10 times from 1 to 10 years
    data = df.copy(deep = True)
    for i, column in zip(range(len(data.columns)), data.columns):
        for years in range(12, 132, 12):
            data[f'{column}_{years}'] = data.iloc[:,i].shift(years)
    data = data.iloc[120:]
    answer = answer.iloc[120:]
    return data, answer


def shift_a_year(df, answer):
    #shift data back 1 year
    data = df.copy(deep = True)
    for column in data.columns:
        data[f'{column}_12'] = data[f'{column}'].shift(12)
        del data[f'{column}']
    data = data.iloc[12:]
    answer = answer.iloc[12:]
    return data, answer

X, y = shift_a_year(df, y)
# X, y = shift_10_years(df, y)
# %% build model data
X

# split dataset into train and test
split = 12
X_train, X_test = X[1:len(X) - split], X[len(X) - split:]
y_train, y_test = y[1:len(y) - split], y[len(y) - split:]

m = LinearRegression()

m.fit(X_train, y_train)
ypred = m.predict(X_test)

m.score(X_test, y_test)

future_pred = m.predict(df[-13:])

future_X = pd.date_range(start=df.iloc[-1].name, end=None, periods=13, freq='1MS')


plt.plot(X_test.index, ypred, 'r-')
plt.plot(X_test.index, y_test, 'g-')
plt.plot(future_X, future_pred, 'b-')
plt.show()

# %% Grid Search for optimal model and params

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

parameters = [{'normalize':[True,False]},
              {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-2,1,5,10,20],'normalize':[True,False]}, #'alpha':[1e-15,1e-10,1e-8,1e-4,1e-2,1,5,10,20],'normalize':[True,False],'solver':['lsqr','sag','saga']
              {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-2,1,5,10,20],'normalize':[True,False]}, #'alpha':[1e-15,1e-10,1e-8,1e-4,1e-2,1,5,10,20],'normalize':[True,False]
              {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-2,1,5,10,20],'normalize':[True,False]} #'alpha':[1e-15,1e-10,1e-8,1e-4,1e-2,1,5,10,20],'l1_ratio':['0','0.1','0.2','0.5','0.8','0.9','1'],'normalize':[True,False]
              ]

answer = pd.DataFrame(columns=['Name','Score'])#,'TP','TN','FP','FN','AUC'])

for name, classifier, parameter in zip(names, classifiers, parameters):
    clf = GridSearchCV(classifier,parameter)
    m = clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    answer = answer.append({'Name':name, 'Score':score}, ignore_index=True)

print(answer.sort_values(by = 'Score', ascending = False))
