"""
to run this python script from another python script AS IF IT WAS ANYTHING
we use bash.

import os
import time

while True:
    print('running the etl job again:')
    os.system('etl_process.py')
    print('-' * 60)
    time.sleep(60)

"""
# import os
#pw = os.getenv('environment_variable_password')

# ETL Job
from sqlalchemy import create_engine
import pandas as pd
import time
import mongo
import logging #logging works over prints because this could be running on an ec2 machine
#extract data from database

db = 'babynames'
table = 'names'
user = 'kristian'
host = '0.0.0.0'
password = '1234''

def extract_data():

    pg_conn = create_engine(f'postgres://{user}:{password}@{host}/{db}')
    df = pd.read_sql(f'SELECT * FROM {table} LIMIT 10', pg_conn)
    logging.info(str(df.shape)) #logging
    return df


# transform the database
def transform_data(df):

    df['length'] = df['name'].apply(len)
    df.set_index('name', inplace= True)
    logging.debug(str(df)) #logging
    return df


def load_df(df):
    # load into db2
    json = df.to_dict()
    json['timestamp'] = time.asctime()
    logging.warning(str(json)) #logging
    mongo_connection = pymongo.MongoClient()
    mongo_connection.db.collections.names.insert(json)


# docker run -it -d -p 27017:27017 mongo

""" alternative way, selecting a different port
mongo_connection = pymongo.MongoClient("0.0.0.0")
# docker run -it -d -p 27016:27017 mongo
"""





"""
If you had 3 functions to run extract, transform, load:

1.
#object oriented format
df = extract_data()
df = transform_data(df)
load_df()

2.
#functional format
extract_data(transform_data(load_df()))

3.
# Airflow Style:
extract_data >> transform_data >> load_df
"""



"""
#logging messages:

logging.basicConfig(filename='debug.log', level=logging.CRITICAL)

logging.info
logging.debug
logging.warn
logging.warning
logging.error
logging.exception
logging.critical
