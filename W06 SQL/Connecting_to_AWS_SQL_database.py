# %% connecting to AWS RDS Server
import sys
sys.path.append('C:\\Users\\alexl\\Documents\\GitPython')
from password import password
from sqlalchemy.sql import select, column
from sqlalchemy import create_engine
import pandas as pd
USER = 'postgres'
PW = 'cartamon'
HOST = 'babynames.c1ipxvq2s9x8.eu-central-1.rds.amazonaws.com'
PORT = '5432'
DBNAME = 'king_alex'
connection_string = f'postgres://{USER}:{PW}@{HOST}:{PORT}/{DBNAME}'

db = create_engine(connection_string)

# %%

QUERY = """
SELECT COUNT(*) FROM names;
"""

#creates an object that's only avaliable once
result = db.execute(QUERY)
print(pd.DataFrame(result, columns=result.keys()))
# pd.DataFrame(result, columns=result.keys())



# %% queries
temp = "SELECT year, SUM(count) FROM names INNER JOIN life ON life.time=names.year GROUP BY year;"
top_5_names_over_time = "WITH foo AS(SELECT name AS topnames FROM names GROUP BY name ORDER BY SUM(count) DESC LIMIT 5) SELECT name, year, SUM(Count) FROM names INNER JOIN foo ON foo.topnames=names.name GROUP BY name, year ORDER BY year;"
names_popularity_over_time = "SELECT name, year, SUM(count) FROM names WHERE name IN('Alex','Morris','Helen') GROUP BY name, year;"
average_name_length_by_year = "WITH foo AS (SELECT year, LENGTH(name), SUM(count) FROM names GROUP BY LENGTH(name), year) SELECT year, AVG(length) AS avg_name_length  FROM foo GROUP BY year;"
q3 = "SELECT year, LENGTH(name), SUM(count) FROM names GROUP BY LENGTH(name), year;"
q4 = "WITH foo as(SELECT name, SUM(count) FROM names WHERE year = 2000 GROUP BY name HAVING SUM(count) >1000) SELECT count(*) from foo;"
q5 = "SELECT name, SUM(count)::REAL / (SELECT SUM(count) FROM names WHERE year = 2000)::REAL *100 FROM names where year = 2000 GROUP BY name ORDER BY name;"

create_date_and_select_year = "SELECT to_char(date(orderdate),'YYYY') as year FROM orders;"

# %% connecting to AWS EC2 Computer

ec2_comp = 'ec2-18-195-160-34.eu-central-1.compute.amazonaws.com'
