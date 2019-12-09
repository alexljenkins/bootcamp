# %% # reads in packages and data
import numpy as np
import pandas as pd
import re
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_fwf("Data/climate_data.txt", comment="%", skiprows =96)

# %% # Clean the dataframe into useful datetime format
df = df.drop([0])
df.rename(mapper = {'nt':'month','mperatu':'temperature'},axis = 1, copy=False, inplace=True)
df['day'] = 1
df['date'] = pd.to_datetime(df[['Year','month', 'day']], format="%Y %m %d")
df.set_index(keys = 'date', drop = False, inplace = True)

temp_data = pd.DataFrame()
temp_data['temp'] = df['temperature']

# %% # Expand data to include NAN values for missing months and years

full_index = pd.DataFrame(pd.date_range(start=(df['date'].min()), end=(df['date'].max()), freq='1MS'))

data = pd.DataFrame()
data=pd.merge(full_index,temp_data, how='outer', left_on=0, right_index=True)

data.set_index(data[0], inplace=True)
data.index.name = 'datetime'
data.drop([0], axis=1, inplace=True)

# %% # Exploring the data

plt.plot(data[::12], '-')
plt.show()
plt.scatter(data.index[::12],data[::12],s=2)
plt.show()
#yearly mean temperature
data['temp'].groupby(data.index.year).mean()
plt.plot(data['temp'].groupby(data.index.year).mean())
plt.show()
#monthly mean temerature
data['temp'].groupby(data.index.month).mean()
plt.plot(data['temp'].groupby(data.index.month).mean())
plt.show()

data['temp'].groupby(data.index.year).count()

# %%
data['moving_average'] = data['temp'].rolling(12).mean()

# %% # auto-regressor from stats model
from statsmodels.tsa.ar_model import AR
data['temp']

ar = AR(data['temp'], dates=None, missing='drop')

ar.fit()

ar.predict(data['temp'], start='2013-01-01', end='2013-05-01')
