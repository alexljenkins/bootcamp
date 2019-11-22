#extracting each component from a timeseries dataset

import seaborn as sns
import numpy as np

df = sns.load_dataset('flights')

df['passengers'].plot()

df['diff'] = df['passengers'].diff() # removes trend
df['diff'].mean() # every month the number of passengers goes up by 2.24
# df['diff'].plot()

#relative change. removes hetroscedasticity and trend
df['pct'] = df['passengers'].pct_change()
# df['pct'].plot()

#leave a little trend but removes hetroscedasticity
df['log'] = np.log(df['passengers'])

#the pct change increase by an average of 1.5%
df['pct'].mean()

#Don't forget to reverse your outcome at the other end
# cumsum reversed is .cumprod()

#removing seasonality either Fourier Transform (FFT) if seasonality is unknown
#or calculate the averages for every seasonal component

monthmeans = df.groupby('month')['pct'].mean().to_list()*12 #concat list to itself 12 times to make it the same length as the df

df['unseasoned'] = df['pct'] / monthmeans
df['unseasoned'].plot()
# df['unseasoned'].hist() #looking for a normal dist for the data to be perfectly described by the current transformations with just a little noise

#breaks down the coefficients of each varibale.
from statsmodels.tsa.seasonal import seasonal_decompose

sdr = seasonal_decompose(df['passengers'].values, freq=12, model='additive')
sdr.plot()
