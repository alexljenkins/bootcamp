import pandas as pd
import json
from datetime import datetime
import requests

SECRET_KEY = "e5be932c860abc13a3ba18121bc14a8a"
https://api.darksky.net/forecast/[key]/[latitude],[longitude],[time]
LAT = '38.9072'
LONG = '-77.0369'
#time = datetime.fromtimestamp(1571270626)
TIME = 1571270626
PATH = f'https://api.darksky.net/forecast/{SECRET_KEY}/{LAT},{LONG},{TIME}'

result = requests.get(path)
#Dictionary version
result_json = result.json()

result_json.keys()
result_json.values()
result_json.items()

#hourly is a dict
#data is a list
result_json['hourly']['data']

temps = pd.DataFrame(result_json['hourly']['data'])
temps.columns

temps['datetime'] = pd.to_datetime(temps.time, unit='s')
