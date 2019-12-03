"""
Building a movie recommender system (an unsupervised learning problem).
Let's first explor the data a bit by grabbing the smaller data set:
https://grouplens.org/datasets/movielens/latest/
And playing around a bit in pandas before deciding the structure
for a SQL database.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dump all the data in (from the smaller dataset)
links = pd.read_csv("data/links.csv")
movies = pd.read_csv("data/movies.csv") #, index_col = "movieId"
ratings = pd.read_csv("data/ratings.csv")
tags = pd.read_csv("data/tags.csv")


# Checkout the shapes, columns and a bit of the data
# also lowercase my column names just to get used to them as SQL table columns
data = [links, movies, ratings, tags]
for df in data:
    df.columns = map(str.lower, df.columns)
    print(df.shape, df.columns, "\n", df.tail(), "\n\n")


# What are the most common ratings given to any given movie?
most_common_ratings = ratings.groupby('rating')['rating'].count()
sns.barplot(x = most_common_ratings.index, y = most_common_ratings)


# Overall, what are the ratings of movies on average?
avg_rating = ratings.groupby('movieid')['rating'].mean().sort_values(ascending=False)
sns.distplot(avg_rating,bins=11)


# How many ratings has each movie got?
num_of_ratings = ratings.groupby('movieid')['rating'].count().sort_values(ascending=False)
sns.distplot(num_of_ratings)
print(num_of_ratings.max(), round(num_of_ratings.mean(),1), num_of_ratings.median(), num_of_ratings.min())

# Is there a pattern in # of ratings vs rating score?
sns.jointplot(x=avg_rating, y=num_of_ratings)


# What are most movies' average rating?
sns.jointplot(x=avg_rating, y=num_of_ratings, kind="hex")


# Does peoples' rating vary greatly on a single movie?
max_rating = ratings.groupby('movieid')['rating'].max()
min_rating = ratings.groupby('movieid')['rating'].min()
varience_in_rating = max_rating - min_rating
sns.distplot(varience_in_rating)


# Obviously this might have to be standardised against the number of ratings
ax = sns.jointplot(x=varience_in_rating, y=num_of_ratings)
ax.set_axis_labels("rating varience", "# of ratings")


# What are the 5 movies with the most 5 star ratings?
ratings_count = ratings.groupby(by=['movieid','rating'])['rating'].count()
most_five_stars = ratings_count.loc[:,5].sort_values(ascending=False).head(5)
print(movies[movies['movieid'].isin(list(most_five_stars.index))][['movieid','title']])

# How does this list compare to the "top 5" rated movies?
# Requirement: has at least 100 user ratings.
above_min_ratings = num_of_ratings[num_of_ratings > 100].index
avg_rating = ratings.groupby(by=['movieid'])['rating'].mean()

avg_rating = avg_rating[avg_rating.index.isin(above_min_ratings)]

top_five_movies = avg_rating.sort_values(ascending=False).head(5)
print(movies[movies['movieid'].isin(list(top_five_movies.index))][['movieid','title']])
