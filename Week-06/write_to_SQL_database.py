import os
import sys
import pandas as pd
sys.path.append('C:\\Users\\alexl\\Documents\\GitPython')
from password import password
from sqlalchemy import create_engine

conns = f'postgres://alex:{password()}@localhost/babynames'

db = create_engine(conns)

for filename in os.listdir('../Week-06/Data/States/'):
     df = pd.read_csv('C:\\Users\\alexl\\Documents\\GitPython\\cartamon-code\\Week-06\\Data\\States\\' + filename,
                        sep = ',', header = None,
                        names = ['state','gender','year','name','count'])
     df.to_sql('names', db, if_exists = 'append', index = False)


# much faster in SQL
"""
\COPY <destination table name> FROM <filename.x> DELIMITER ','
"""
