# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:28:13 2019
@author: alexl
"""

import pandas as pd
import os
import json
from datetime import datetime
import requests

# setup weather API
lat = '38.9072'
long = '-77.0369'
SECRET_KEY = "32e3fea63b56e30a83a74e87ee076da6"


def GatherBikeData():
    """    read in all csvs in folder path    """
    
    PATH = 'Data/Bikes'
    df_2017 = pd.DataFrame()
    for csv in os.listdir(path=PATH):
      df = pd.read_csv(f'{PATH}/{csv}')
      df_2017 = pd.concat([df_2017, df], sort=True)
      
    return df_2017


def SetDatetimeTimestamp(df, column):
    """    convert date strings to datetime and timestamps    """
    
    df[column] = pd.to_datetime(df[column]).dt.tz_localize('EST')
    df.set_index(keys = column, drop = False, inplace = True)
    df['timestamp'] = df[column].apply(pd.Timestamp.timestamp) #to seconds timestamp
    
    return df


def ConvertTimeStamp(df, column):
    """    convert timestamp column to ETS datetimes    """
    
    df[column] = pd.to_datetime(df[column], unit = 's').dt.tz_localize('UTC').dt.tz_convert('EST')

    return df

def WeatherRequest(time):
    """    Gets weather data for the day of which the timestamp is in    """
    
    path = f'https://api.darksky.net/forecast/{SECRET_KEY}/{lat},{long},{time}'
    result = requests.get(path)
    result_json = result.json()
    # explore the json results
#    result_json.keys()
#    result_json.values()
#    result_json.items()
    weather_data = pd.DataFrame(result_json['hourly']['data'])
    
    return weather_data


def Workflow():
    """"    Runs the program    """
    # Gather Bike Data and Covert times
    data = GatherBikeData()
    data = SetDatetimeTimestamp(data, 'Start date')
    
    # Make a request for weather data and convert times
#    for time in range(data.timestamp.min(), data.timestamp.max(),86400):
    request_return = WeatherRequest(1293858000)
    results = ConvertTimeStamp(request_return, 'time')


    return data, results



data, results = Workflow()



data.columns
results.columns

data['timestamp']
results.time

#df.merge( how = 'inner') Merge dataframes using index

