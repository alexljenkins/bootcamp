from flask import Flask, render_template, request
from recommender import deep_recommender

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/movies')
def recommender():
    return render_template('movies.html')

@app.route('/results')

def recommenders():
    user_input = request.args
    user_input = dict(user_input).values
    user_movies = list(user_input)[0::2]
    user_ratings = list(user_input)[1::2]

    movie_list = deep_recommender(3)
    return render_template('results.html', movies=movie_list)
