import pandas as pd
import random

def deep_recommender(num=5):
    movies = pd.read_csv('movies.csv')['title'].tolist()
    random.shuffle(movies)

    # create a new user vector for their movies and ratings

    # load the trained nmf model

    # feed in the user vector into the model (transform)

    # results np.dot(profile, nmf_model.components_)

    # results final = convert to names from nmf output

    return movies[:num]
