from flask import Flask, render_template
from flask_bootstrap3 import Bootstrap
import requests
import pandas as pd
import json

app = Flask(__name__)
bootstrap = Bootstrap(app)

# flask-bootstrap
@app.route('/')
def index():
    name = 'Home'
    return render_template('index.html', name=name)

@app.route('/dataset', methods=("POST", "GET"))
def table():
    
    data = pd.read_csv("winequality-red.csv")
    json_list = json.loads(json.dumps(list(data.T.to_dict().values())))
    return render_template('table.html', tables=json_list)

# dohvacam api
@app.route('/tvshows/<num>')
def show(num):
    app.logger.info("Pozvana stranica show.html")
    parameters = {'page': num}
    url = 'https://api.tvmaze.com/shows?'
    response = requests.get(url, params=parameters)
    jsonShow = response.json()    
    return render_template('tvshows.html', jsonShow=jsonShow, num=num)

@app.route('/links/')
def links():
    links = ['https://github.com/PMF-Data-Science/PMF-DataScience',
             'https://github.com/PMF-Data-Science/computer-science-educators-stack-exchange-data']
    return render_template('links.html', links=links)