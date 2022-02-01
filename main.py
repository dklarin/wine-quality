from flask import Flask, render_template
from flask_bootstrap3 import Bootstrap
import requests, json
import pandas as pd
import matplotlib.pyplot as plt
import os.path



app = Flask(__name__)
bootstrap = Bootstrap(app)
data = pd.read_csv("winequality-red.csv")

# flask-bootstrap
@app.route('/')
def index():
    name = 'Home'
    return render_template('index.html', name=name)

@app.route('/dataset', methods=("POST", "GET"))
def dataset():
    
    #data = pd.read_csv("winequality-red.csv")
    json_list = json.loads(json.dumps(list(data.T.to_dict().values())))
    return render_template(
        'table.html', tables=json_list,
        link1='dataset',link2='describe',link3='shape',
        switch=1)

@app.route('/describe')
def describe():

    describe=data.describe()
    json_list = json.loads(json.dumps(list(describe.T.to_dict().values())))
    return render_template(
        'table.html', tables=json_list,
        link1='dataset',link2='describe',link3='shape')

@app.route('/shape')
def shape():

    shape=data.shape
    
    return render_template(
        'info.html',shape=shape,rows=shape[0],columns=shape[1],
        link1='dataset',link2='describe',link3='shape',
        )

@app.route('/reg_lin')
def reg_lin():

    
    
    return render_template(
        'info.html',
        link1='gradijentni_spust',link2='unakrsna_validacija',link3='shape',
        button1='Gradijentni spust',
        reg_lin='reg_lin')

@app.route('/gradijentni_spust')
def gradijentni_spust():

    
    
    return render_template(
        'info.html',
        link1='distribution_alcohol',link2='distribution_quality',
        button1='Distribution alcohol',
        reg_lin='reg_lin')

@app.route('/unakrsna_validacija')
def unakrsna_validacija():

    
    
    return render_template(
        'info.html',
        link1='gradijentni_spust',link2='unakrsna_validacija',
        
        reg_lin='reg_lin')

@app.route('/distribution_alcohol')
def distribution_alcohol():

    x = data['alcohol']
    y = data['quality']
    fig, ax = plt.subplots()
  
    plt.hist(x, density=True, bins=30)
    plt.ylabel('Count')
    plt.xlabel('alcohol')

    file_exists = os.path.exists('static/pics/distr_alcohol.png')

    if file_exists:        
        img = os.path.join('static/pics/distr_alcohol.png')
    else:
        fig.savefig('static/pics/distr_alcohol.png')    
    
    return render_template(
        'info.html',
        link1='distribution_alcohol',link2='distribution_quality',
        button1='Distribution alcohol',
        image=img)

@app.route('/distribution_quality')
def distribution_quality():

    x = data['alcohol']
    y = data['quality']
    fig,ax=plt.subplots()  
    plt.hist(y, density=True, bins=30)
    plt.ylabel('Count')
    plt.xlabel('quality')

    file_exists = os.path.exists('static/pics/distr_quality.png')

    if file_exists:
        img = os.path.join('static/pics/distr_quality.png')
    else:
        fig.savefig('static/pics/distr_quality.png')   
    
    return render_template(
        'info.html',
        link1='distribution_alcohol',link2='distribution_quality',
        button1='Distribution alcohol',
        image=img)











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