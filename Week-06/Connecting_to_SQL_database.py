import sys
sys.path.append('C:\\Users\\alexl\\Documents\\GitPython')
from password import password
from sqlalchemy.sql import select, column
from sqlalchemy import create_engine
import pandas as pd

USER = 'postgres'
PW = 'postgres'
HOST = 'localhost' #127.0.0.1
PORT = '5432'
DBNAME = 'babynames'
connection_string = f'postgres://{USER}:{PW}@{HOST}:{PORT}/{DBNAME}'

db = create_engine(connection_string)

df = pd.read_csv('C:\\Users\\alexl\\Documents\\GitPython\\cartamon-code\\Week-06\\Data\\large_countries_2015.csv', index_col=0)

df.to_sql('data', db)

# %%

QUERY = """
SELECT * FROM data;
"""
#creates an object that's only avaliable once
result = db.execute(QUERY)
#see how big the data is
result.rowcount
pd.DataFrame(result, columns=result.keys())


results = pd.read_sql_table('data', db, index_col=1)


# %% queries


q4 = "WITH foo as(SELECT name, SUM(count) FROM names WHERE year = 2000 GROUP BY name HAVING SUM(count) >1000) SELECT count(*) from foo;"
q5 = "SELECT name, SUM(count)::REAL / (SELECT SUM(count) FROM names WHERE year = 2000)::REAL *100 FROM names where year = 2000 GROUP BY name ORDER BY name;"
