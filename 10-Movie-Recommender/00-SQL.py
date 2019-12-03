from sqlalchemy import create_engine
import pandas as pd
#%% adding data to the nubflex SQL database
conns = 'postgres://postgres:postgres@localhost/nubflex'
db = create_engine(conns)

# df.to_sql('links', db, if_exists = 'append', index = False)

links = pd.read_csv("data/links.csv", index_col = "movieId")
movies = pd.read_csv("data/movies.csv", index_col = "movieId")
ratings = pd.read_csv("data/ratings.csv")
tags = pd.read_csv("data/tags.csv")


movies.columns
