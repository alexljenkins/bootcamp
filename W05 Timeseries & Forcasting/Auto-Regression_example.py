import pandas as pd

from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error


df = pd.read_fwf("Data/climate_data.txt", comment="%", skiprows =96)
df = df.drop([0])
df.rename(mapper = {'nt':'month','mperatu':'temperature'},axis = 1, copy=False, inplace=True)
df['day'] = 1
df['date'] = pd.to_datetime(df[['Year','month', 'day']], format="%Y %m %d")
df.set_index(keys = 'date', drop = False, inplace = True)
df.head()
temperature = df['temperature']
temperature.dropna(inplace = True)

# split dataset
X = temperature.values
X
train, test = X[1:len(X)-12], X[len(X)-12:]

# print('Lag: %s' % model_fit.k_ar)
# print('Coefficients: %s' % model_fit.params)

# train autoregression
model = AR(train)
model_fit = model.fit()


# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

for i in range(len(predictions)):
	print(f'prediction={round(predictions[i],2)}, actual={round(test[i],2)}')


error = mean_squared_error(test, predictions)
print(f'Test MSE: {error}')
# plot results
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
