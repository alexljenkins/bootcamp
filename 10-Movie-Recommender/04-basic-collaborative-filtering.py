"""
Cosine similarity.
Calulating the angle of the two vectors
(Between -1 and 1)
cosim = cos(a) = (X . y)/||x||*||y||
Should normalize/de-mean so the angles of "opposite" vectors are captured
as 180' rather than 0' apart. Called: Centred cosine similarity.
This makes it the same as "pearson correlation".
"""
import math
import pandas as pd

df = pd.read_csv('data\\cart_movies.csv')

# Expand df into a matrix
matrix = df.set_index(['User','Movie'])['Rating'].unstack(1)
matrix.fillna(0.0, inplace=True)


def cosim(x, y, matrix):
    """
    Given a matrix, calulates the cosine similarity of x and y.
    Assuming they are both arrays/vectors
    """
    num = 0
    for movie in matrix.columns:
        num += x[movie] * y[movie]
        xsum = math.sqrt(sum(x ** 2))
        ysum = math.sqrt(sum(y ** 2))
        denom = xsum * ysum

    return num/denom

# Cosine between two users:
cosim(matrix.loc['alex'],matrix.loc['moritzs'], matrix)

# To check it's working by comparing a user with themself
cosim(matrix.loc['alex'],matrix.loc['alex'], matrix)

# cosine similarity of all users with each other
cosine_similarities = []
for name1 in matrix.index:
    for name2 in matrix.index:
        c = cosim(matrix.loc[name1],matrix.loc[name2], matrix)

        cosine_similarities.append((name1, name2, c))

sim_matrix = pd.DataFrame(cosine_similarities).set_index([0,1])[2].unstack(1)

# graph the results to see who is most similar to each other
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))
sns.heatmap(sim_matrix, annot=True)
plt.show()



##%% Shortcut Method:
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarities(matrix)


"""
Advantages over NMF:
- Keep a sparse matrix
- A bit more intuitive
- Add additional features
- Faster than euclidian distance O(nm)
- parellelizeable (dask)

Disadvantages:
- Might need more data incase similar users haven't seen different movies
- run-time is longer... new user vs whole table

Note:
- Could use PCA to reduce "users" dimension to speed things up AND
    stop users being similar to others where no new infomation is present.
"""
