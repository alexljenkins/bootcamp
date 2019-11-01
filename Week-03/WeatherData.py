# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:40:54 2019
@author: alexl
How to connect to and handle APIs
"""

import pandas as pd
import json
from datetime import datetime
import requests

lat = '38.9072'
long = '-77.0369'
SECRET_KEY = "32e3fea63b56e30a83a74e87ee076da6"

data = pd.read_csv('Data/train.csv')

data['datetime'] = pd.to_datetime(data['datetime']).dt.tz_localize('EST')
data.set_index(keys = 'datetime', drop = False, inplace = True)
data['timestamp'] = data['datetime'].apply(pd.Timestamp.timestamp) #to seconds timestamp

for time in range(data.timestamp.min(), data.timestamp.max(),86400):

    request = requests.get(path)
    jr = request.json()
    df = pd.DataFrame(jr['hourly']['data'])
    df.columns
    for result in df['time']:

        data.loc[data['datetime'] == result].index #grabs the row with same timestamp

    """
    make a request for the days weather

    for each hour:
        etract temp, atemp, humidity, windspeed
        add them to the dataframe
    """



path = f'https://api.darksky.net/forecast/{SECRET_KEY}/{lat},{long},{time}'
    
result = requests.get(path)

##set timezone
#data['datetime'] = data['datetime'].apply(lambda x : pd.to_datetime(x, unit='s').tz_localize('EST'))
#
##there (to timestamp)
#data['timestamp'] = data['datetime'].apply(pd.Timestamp.timestamp)
#
##and back again
#data['date2'] = data['timestamp'].apply(lambda x : pd.to_datetime(x, unit='s').tz_localize('EST'))


#Dictionary version
result_json = result.json()
print(result_json)

result_json.keys()
result_json.values()
result_json.items()

#hourly is a dict
#data is a list
result_json['hourly']['data']

temps = pd.DataFrame(result_json['hourly']['data'])
temps.columns
temps.time
temps['datetime'] = pd.to_datetime(temps.time, unit='s') #set units to the accuracy of the timestamp (seconds)

temps = temps.set_index(temps['datetime'])
temps.drop(['windGust',
           'cloudCover',
           'icon',
           'datetime',
           'apparentTemperature',
           'dewPoint',
           'uvIndex',
           'summary',
           'pressure',
           'ozone',
           'precipProbability'
           ])

temps.datetime

print(temps)

pd.to_datetime(int(datetime.today().timestamp()),unit='s')

data

data['datetime'] == 1293858000
