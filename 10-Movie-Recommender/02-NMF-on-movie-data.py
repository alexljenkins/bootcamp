import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.decomposition import NMF

ratings = pd.read_csv("data/ratings.csv") # only file we really need for NMF
ratings.columns = map(str.lower, ratings.columns)

def zero_nmf_model(ratings):
    # Create a sparse matrix of all user ratings for all movies
    reviews = pd.pivot_table(ratings, 'rating', 'userid', 'movieid')
    reviews = reviews.fillna(0)

    # instantiate the NMF model
    model = NMF(n_components=42, init='random', random_state=42)

    model.fit(reviews)

    # Note: R matrix is reviews
    Q = model.components_  # movie-feature matrix
    P = model.transform(reviews)  # user-feature matrix

    # dot product of P and Q is our Rhat (Rpredictions)
    Rhat = np.dot(P,Q)

    # 610 movies, 9724 users
    # Rhat.shape

    # only useful to compare against other models
    # model.reconstruction_err_

    return Rhat, model

zero_Rhat, zero_model = zero_nmf_model(ratings)
print(zero_Rhat.shape, zero_model.reconstruction_err_)


#%%
"""
Would filling the matrix with a more meaningful set of values (other than 0)
reduce the reconstruction error?
"""

def median_nmf_model(ratings):
    """
    Same as zero_nmf_model function, but fills empty reviews
    with a median of the column instead of zero values.
    """
    reviews = pd.pivot_table(ratings, 'rating', 'userid', 'movieid')
    # Fillna with the median of each column
    reviews.fillna(reviews.median(), inplace=True)

    model = NMF(n_components=42, init='random', random_state=42)

    model.fit(reviews)
    Q = model.components_  # movie-feature matrix
    P = model.transform(reviews)  # user-feature matrix
    Rhat = np.dot(P,Q)

    return Rhat, model

median_Rhat, median_model = median_nmf_model(ratings)

# Wooh! way better.
print(median_Rhat.shape, median_model.reconstruction_err_)
