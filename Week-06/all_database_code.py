import os
import pandas as pd
from sqlalchemy import create_engine

USER = 'postgres'
PW = 'cartamon'
HOST = 'babynames.c1ipxvq2s9x8.eu-central-1.rds.amazonaws.com'
PORT = '5432'
DBNAME = 'king_alex'
connection_string = f'postgres://{USER}:{PW}@{HOST}:{PORT}/{DBNAME}'

db = create_engine(connection_string)


# %% load all files in one table:
import sys
sys.path.append('C:\\Users\\alexl\\Documents\\GitPython')

for filename in os.listdir('../Week-06/Data/States/'):
     df = pd.read_csv('C:\\Users\\alexl\\Documents\\GitPython\\cartamon-code\\Week-06\\Data\\States\\' + filename,
                        sep = ',', header = None,
                        names = ['state','gender','year','name','count'])
     df.to_sql('names', db, if_exists = 'append', index = False)


# %% load all files into seperate tables:

import sys
sys.path.append('C:\\Users\\alexl\\Documents\\GitPython')

for filename in os.listdir('../Week-06/Data/States/'): #for file in files in folder
     df = pd.read_csv('C:\\Users\\alexl\\Documents\\GitPython\\cartamon-code\\Week-06\\Data\\States\\' + filename)
     df.columns = [c.lower() for c in df.columns]
     df.to_sql(filename[:-4], db, if_exists = 'replace', index = False)


#%% load a single file into a SQL database:

df = pd.read_csv('C:\\Users\\alexl\\Documents\\GitPython\\cartamon-code\\Week-06\\Data\\large_countries_2015.csv', index_col=0)
# lowercase ALL column headers so sql reads them nicely
df.columns = [c.lower() for c in df.columns]

df.to_sql('data', db, index = False, if_exists='replace') # or append.
# much faster in SQL
"""
\COPY <destination table name> FROM <filename.x> DELIMITER ','
"""

#%% Create a database from python

import os
os.system(createdb <dbname>)
