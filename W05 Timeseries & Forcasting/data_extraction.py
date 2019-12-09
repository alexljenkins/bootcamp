# %% # reads in packages and data
import numpy as np
import pandas as pd
import re
import seaborn as sns
from matplotlib import pyplot as plt
#mean monthly temp for the next 10 years

data = pd.read_fwf("Data/climate_data.txt", comment="%", skiprows =96)

# %% #Creates an empty df of the complete month and years that the data has
dates_list = []

for year in range(int(data['Year'].min()), int(data['Year'].max()+1)):
    for month in range(1, 13):
        dates_list.append([year, month])

df = pd.DataFrame(dates_list, columns = ['Year','Month'])
df['Day'] = 1

timeline = pd.DataFrame()
timeline['datetime'] = pd.to_datetime(df[['Year','Month','Day']], unit="d")

timeline.set_index(keys = 'datetime', drop = True, inplace = True)
# print(timeline.head())
# timeline.index.max()
# %% # Turns data of year and month into datetime

data = data.drop([0])
data['Year'] = data['Year'].astype(int)
data['month'] = data['nt'].astype(int)
data['day'] = 1

data['date'] = pd.to_datetime(data[['Year','month', 'day']], format="%Y %m %d")
data.set_index(keys = 'date', drop = False, inplace = True)

# print(data.head())
# %% # merge the two dataframes to show the NAN values in your dataset

x=pd.merge(data,timeline, how='outer', left_index=True, right_index=True)
# print(x.head(10))
a = pd.DataFrame()
a['temp'] = x['mperatu']
a[a['temp'].isnull()]

x['1866'] #last year with NAN data in it
pd.isnull(a['temp'])
a[(pd.isnull(a['temp'])).idxmin():]

# %% #

sns.scatterplot(data = a)
plt.show()
sns.lineplot(data = a)
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))

plot = sns.pointplot(
    ax=ax,
    data=a, x=a.index, y="temp"
)
ax.set_xticklabels(a.index)
plt.show()

# %%


# full_idx = pd.date_range(start=(data['date'].min()), end=(data['date'].max()), freq='1MS')
# alex = pd.DataFrame()
# alex['datetime'] = full_idx
full_idx = pd.DataFrame(pd.date_range(start=(data['date'].min()), end=(data['date'].max()), freq='1MS'))
temp_data = pd.DataFrame()

temp_data['temp'] = data['mperatu']

ab=pd.merge(full_idx,temp_data, how='outer', left_on=0, right_index=True)

ab.set_index(ab[0], inplace=True)
ab.index.name = 'datetime'
ab.drop([0], axis=1, inplace=True)
print(ab.head())


# print(2343)
#
# data_temp = pd.DataFrame()
# data_temp = data.drop('date', axis=1)
# data_temp.reset_index(inplace=True)
#
# data2 = (data_temp.groupby(data_temp['date'], as_index=False).apply(lambda group: group.reindex(full_idx)))
# .reset_index(level=0, drop=True).sort_index())
# data2
