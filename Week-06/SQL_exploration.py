import sys
import pandas as pd
sys.path.append('C:\\Users\\alexl\\Documents\\GitPython')
from password import password
from sqlalchemy.sql import select, column
from sqlalchemy import create_engine
import psycopg2

conns = f'postgres://alex:{password()}@localhost/babynames'


db = create_engine(conns)

# s = select('Alex').where(names(column('count')))
#
#
# for i in conn.execute(s):
#     print(i)

#
# def get_vendors():
#     """ query data from the vendors table """
#     conn = None
#     try:
#         params = config()
#         conn = psycopg2.connect(**params)
#         cur = conn.cursor()
#         cur.execute("SELECT name, count FROM names ORDER BY count")
#         print("The number of parts: ", cur.rowcount)
#         row = cur.fetchone()
#
#         while row is not None:
#             print(row)
#             row = cur.fetchone()
#
#         cur.close()
#     except (Exception, psycopg2.DatabaseError) as error:
#         print(error)
#     finally:
#         if conn is not None:
#             conn.close()
#
# get_vendors()
