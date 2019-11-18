# https://www.youtube.com/watch?v=iwhuPxJ0dig

from flask import Flask
from flask import render_template, json


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/test')
def test():
    my_dict = {"name":"Alex","title":"Data Scientist"}
    return json.dumps(my_dict) #beautified and json recognised
    # return my_dict #just returns the dict format

if __name__ == "__main__":
    app.run(debug=True) #host='0.0.0.0', port=80
