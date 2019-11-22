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
#time = datetime.fromtimestamp(1571270626)
time = 1571270626
print(time)
path = f'https://api.darksky.net/forecast/{SECRET_KEY}/{lat},{long},{time}'

result = requests.get(path)
#actual text returned from the request
print(result.text)

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

temps.columns

print(temps)

pd.to_datetime(int(datetime.today().timestamp()),unit='s')





