import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

df = sns.load_dataset('flights')

df['date'] = df['year'].astype(str) + '-' + df['month'].astype(str)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)


#shit X variables 12 months into the past to predict present as if in the future
df['passengers_minus_12months'] = df['passengers'].shift(12)
df['passengers_rolling_mean'] = df['passengers_minus_12months'].rolling(12).mean()
features = ['passengers_minus_12months', 'passengers_rolling_mean']


# %% plotting data
df['passengers'].plot()
df['passengers_minus_12months'].plot()
df['passengers_rolling_mean'].plot()
df.dropna(inplace=True)

# %% train test split
X_train = df.iloc[:-20][features]
y_train = df.iloc[:-20]['passengers']
X_test = df.iloc[-20:][features]
y_test = df.iloc[-20:]['passengers']


from sklearn.linear_model import LinearRegression
m = LinearRegression()
m.fit(X_train, y_train)

ypred = m.predict(X_test)

plt.plot(df['passengers'])
plt.plot(X_test.index,ypred)
plt.legend(['original data','prediction'])

# %% # METRICS on how good the prediction model was

# Mean-Squared-Error (MSE) SUM(ytrue - ypred)^2 / n

from sklearn.metrics import mean_squared_error
np.sum((ypred - y_test)**2) / len(ypred)
mse = mean_squared_error(y_test, ypred)
mse

# mean_absolute_percentage_error
mape = np.sum(np.abs(ypred-y_test) / y_test * 100) / len(ypred)
mape

# r^2 value
m.score(X_test, y_test)
m.score(X_train, y_train)

#then feed it real current data to predict the future

#stats model that already does shifting data for you:
from statsmodels.tsa.ar_model import AR

#fbprophet is a facebook package and model you can use out of the box. Uses exponencial smoothing
