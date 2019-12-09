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
y = df['raw_temp']
# df.drop('raw_temp', axis = 1, inplace = True)
#%%

def shift_10_years_x(data, answer):
    #shift data back 10 times from 1 to 10 years
    for i, column in zip(range(len(data.columns)), data.columns):
        for years in range(12, 132, 12):
            data[f'{column}_{years}'] = data.iloc[:,i].shift(years)
    data = data.iloc[120:,1:]
    answer = answer.iloc[120:]
    return data, answer


def shift_10_years(df, answer):
    #shift data back 1 year
    data = df.copy(deep = True)
    for column in data.columns:
        data[f'{column}_120'] = data[f'{column}'].shift(120)
        del data[f'{column}']
    data = data.iloc[120:]
    answer = answer.iloc[120:]

    return data, answer

def remove_cols(df):
    #shift data back 1 year
    data = df.copy(deep = True)
    data.drop(['raw_anomaly','adjusted_anomaly','regional_anomaly'], axis = 1, inplace = True)
    data.drop(['regional_temp','adjusted_temp'], axis = 1, inplace = True)

    return data


df = remove_cols(df)
# X, y = shift_10_years(df, y)
X, y = shift_10_years_x(df, y)
# %% build model data

# split dataset into train and test
split = 12
X_train, X_test = X[:-split], X[-split:]
y_train, y_test = y[:-split], y[-split:]

m = LinearRegression()

m.fit(X_train, y_train)
ypred = m.predict(X_test)
m.score(X_test, y_test)
X.head()
future_pred = m.predict(df.iloc[-12:,:-1])

future_X = pd.date_range(start=df.iloc[-1].name, end=None, periods=13, freq='1MS')[1:]

plt.plot(X.iloc[-48:].index, y.iloc[-48:], 'r-')
plt.plot(X_test.index, ypred, 'b-')
# plt.plot(X_test.index, y_test, 'y-')
plt.plot(future_X, future_pred, 'g-')
plt.show()


# %% Creating DF to House all the predictions

future = pd.date_range(start=df.iloc[-1].name, end=None, periods=121, freq='1MS')[1:]
future_df = pd.DataFrame(index=future)
future_df.index.name = 'date'
X.head()


for i in range(12):
    m = LinearRegression(normalize = True)
    # m.fit(X, y) #no refit 
    # ypred = m.predict(X_test)
    future_pred = m.predict(df[-12:])
    future_df[f'i'] = future_pred









# %% Metircs
print(f'Rsquared: {m.score(X_test,y_test)}')

# Mean-Squared-Error (MSE) SUM(ytrue - ypred)^2 / n
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, ypred)
print(f'Mean Squared Error: {mse}')

# mean_absolute_percentage_error
mape = np.sum(np.abs(ypred-y_test) / y_test * 100) / len(ypred)
print(f'Mean Absolute Percentage Error: {mape}')

#mean absolute error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, ypred)
print(f'Mean Absolute Error: {mae}')
