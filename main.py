from flask import Flask, render_template
from flask_bootstrap3 import Bootstrap
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

from methods import *

app = Flask(__name__)
bootstrap = Bootstrap(app)
data = pd.read_csv("winequality-red.csv")

# metoda koja razdvaja string


@app.template_filter('nesto')
def reverse_filter(s):
    st = s.split("_")
    if len(st) < 2:
        return st[0]
    else:
        return st[0]+' '+st[1]

# flask-bootstrap


@app.route('/')
def index():
    name = 'Home'
    return render_template('index.html', name=name)

# prikaz podataka


@app.route('/dataset', methods=("POST", "GET"))
def dataset():
    json_list = json.loads(json.dumps(
        list(data.loc[0:10].T.to_dict().values())))
    return render_template(
        'table.html', tables=json_list,
        link1='dataset', link2='describe', link3='shape',
        switch=1)

# deskriptivna statistika


@app.route('/describe')
def describe():
    describe = data.describe()
    json_list = json.loads(json.dumps(list(describe.T.to_dict().values())))
    return render_template(
        'table.html', tables=json_list,
        link1='dataset', link2='describe', link3='shape')

# retci i stupci


@app.route('/shape')
def shape():
    shape = data.shape
    return render_template(
        'info.html', shape=shape, rows=shape[0], columns=shape[1],
        link1='dataset', link2='describe', link3='shape')

# linearna regresija


@app.route('/linear_regression')
def reg_lin():

    return render_template(
        'info.html',
        link1='gradijentni_spust', link2='unakrsna_validacija')


@app.route('/gradijentni_spust')
def gradijentni_spust():

    return render_template(
        'info.html',
        link1='distribution_alcohol', link2='distribution_quality')


@app.route('/unakrsna_validacija')
def unakrsna_validacija():

    return render_template(
        'info.html',
        link1='gradijentni_spust', link2='unakrsna_validacija',
        reg_lin='reg_lin')


@app.route('/distribution_alcohol')
def distribution_alcohol():

    x = data['alcohol']
    y = data['quality']
    #fig, ax = plt.subplots()

    fig = figax()

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
        link1='distribution_alcohol', link2='distribution_quality',
        image=img)


@app.route('/distribution_quality')
def distribution_quality():

    x = data['alcohol']
    y = data['quality']
    #fig, ax = plt.subplots()

    fig = figax()
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
        link1='distribution_alcohol', link2='distribution_quality',
        button1='Alcohol Distribution',
        image=img)


@app.route('/gradient_descent')
def scaling_data():

    # Skaliranje podataka
    x = data['alcohol']
    y = data['quality']
    x_scaled = (x - x.mean()) / x.std()
    x_scaled = np.c_[np.ones(x_scaled.shape[0]), x_scaled]

    # Hiperparametri za gradijentni spust
    alpha = 0.01
    iterations = 2000
    m = y.size
    np.random.seed(123)
    theta = np.random.rand(2)

    past_thetas, past_costs = gradient_descent(
        x_scaled, y, theta, iterations, alpha)
    theta = past_thetas[-1]

    return render_template(
        'info.html',
        link1='distribution_alcohol', link2='distribution_quality',
        button1='Alcohol Distribution', rows=theta[0], columns=theta[1])


@app.route('/funkcija_troska')
def funkcija_troska():

    # Skaliranje podataka
    x = data['alcohol']
    y = data['quality']
    x_scaled = (x - x.mean()) / x.std()
    x_scaled = np.c_[np.ones(x_scaled.shape[0]), x_scaled]

    # Hiperparametri za gradijentni spust
    alpha = 0.01
    iterations = 2000
    m = y.size
    np.random.seed(123)
    theta = np.random.rand(2)

    past_thetas, past_costs = gradient_descent(
        x_scaled, y, m, theta, iterations, alpha)
    theta = past_thetas[-1]

    fig = figax()
    plt.title('Cost Function J')
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.plot(past_costs)
    # plt.show()

    file_exists = os.path.exists('static/pics/funkcija_troska.png')

    if file_exists:
        img = os.path.join('static/pics/funkcija_troska.png')
    else:
        fig.savefig('static/pics/funkcija_troska.png')

    return render_template(
        'info.html',
        link1='distribution_alcohol', link2='distribution_quality',
        button1='Alcohol Distribution',
        image=img)


@app.route('/links/')
def links():
    links = ['https://github.com/PMF-Data-Science/PMF-DataScience',
             'https://github.com/PMF-Data-Science/computer-science-educators-stack-exchange-data']
    return render_template('links.html', links=links)
