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

links = ['distribution_alcohol', 'distribution_quality',
         'gradient_descen', 'funkcija_troska', 'izgled_regresije']


data = pd.read_csv("winequality-red.csv")
x = data['alcohol']
y = data['quality']

# Skaliranje podataka
x_scaled = (x - x.mean()) / x.std()
x_scaled = np.c_[np.ones(x_scaled.shape[0]), x_scaled]

# Hiperparametri za gradijentni spust
alpha = 0.01
iterations = 2000
#m = y.size
np.random.seed(123)
theta = np.random.rand(2)

# Izračun gradijentnog spusta
past_thetas, past_costs = gradient_descent(
    x_scaled, y, theta, iterations, alpha)
theta = past_thetas[-1]

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
        'table.html', tables=json_list, title='dataset',
        link1='dataset', link2='describe', link3='shape',
        switch=0, stat=0)

# deskriptivna statistika


@app.route('/describe')
def describe():
    describe = data.describe()
    describe = describe.round(2)

    stats = pd.Series(['count', 'mean', 'std', 'min', '25%', '50%',
                       '75%', 'max'], index=[0, 1, 2, 3, 4, 5, 6, 7])
    describe['statistics'] = stats.values

    json_list = json.loads(json.dumps(
        list(describe.T.to_dict().values())))
    return render_template(
        'table.html', tables=json_list, title='describe',
        link1='dataset', link2='describe', link3='shape',
        stat=1)

# retci i stupci


@app.route('/shape')
def shape():
    shape = data.shape
    return render_template(
        'info.html', shape=shape, rows=shape[0], columns=shape[1], title='shape',
        link1='dataset', link2='describe', link3='shape')

# linearna regresija


@app.route('/linear_regression')
def reg_lin():

    return render_template(
        'info.html',
        link1='gradijentni_spust', link2='unakrsna_validacija', title='linear regression')


# gradijentni spust
@app.route('/gradijentni_spust')
def gradijentni_spust():

    return render_template(
        'info.html',
        link1=links[0],
        link2=links[1],
        link3=links[2],
        link4=links[3],
        link5=links[4],
        title='gradijentni spust'
    )

# unakrsna validacija


@app.route('/unakrsna_validacija')
def unakrsna_validacija():

    return render_template(
        'info.html',
        link1='gradijentni_spust', link2='unakrsna_validacija',
        reg_lin='reg_lin')


# distribucija alkohola
@app.route('/distribution_alcohol')
def distribution_alcohol():

    fig = figax()
    plt.hist(x, density=True, bins=30)
    plt.ylabel('Count')
    plt.xlabel('alcohol')

    file_exists = os.path.exists('static/images/distr_alcohol.png')

    if file_exists:
        image = os.path.join('static/images/distr_alcohol.png')
    else:
        fig.savefig('static/images/distr_alcohol.png')

    return render_template(
        'info.html',
        link1=links[0],
        link2=links[1],
        link3=links[2],
        link4=links[3],
        link5=links[4],
        title='distribution alcohol',
        image=image)

# distribucija kvalitete


@app.route('/distribution_quality')
def distribution_quality():

    fig = figax()
    plt.hist(y, density=True, bins=30)
    plt.ylabel('Count')
    plt.xlabel('quality')

    file_exists = os.path.exists('static/images/distr_quality.png')

    if file_exists:
        image = os.path.join('static/images/distr_quality.png')
    else:
        fig.savefig('static/images/distr_quality.png')

    return render_template(
        'info.html',
        link1=links[0],
        link2=links[1],
        link3=links[2],
        link4=links[3],
        link5=links[4],
        button1='Alcohol Distribution',
        title='distribution quality',
        image=image)

# skaliranje, hiperparametri, izračun gradijentnog spusta


@app.route('/gradient_descen')
def gradient_descen():

    # Hiperparametri za gradijentni spust
    #alpha = 0.01
    #iterations = 2000
    #m = y.size
    # np.random.seed(123)
    #theta = np.random.rand(2)

    # Izračun gradijentnog spusta
    # past_thetas, past_costs = gradient_descent(
    #    x_scaled, y, theta, iterations, alpha)
    #theta = past_thetas[-1]

    return render_template(
        'info.html',
        link1=links[0],
        link2=links[1],
        link3=links[2],
        link4=links[3],
        link5=links[4],
        button1='Alcohol Distribution',
        title='gradient descent',
        rows=theta[0], columns=theta[1])

# funkcija troška


@app.route('/funkcija_troska')
def funkcija_troska():

    # Hiperparametri za gradijentni spust
    #alpha = 0.01
    #iterations = 2000
    #m = y.size
    # np.random.seed(123)
    #theta = np.random.rand(2)

    # past_thetas, past_costs = gradient_descent(
    #    x_scaled, y, theta, iterations, alpha)
    #theta = past_thetas[-1]

    fig = figax()
    plt.title('Cost Function J')
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.plot(past_costs)

    file_exists = os.path.exists('static/images/funkcija_troska.png')

    if file_exists:
        image = os.path.join('static/images/funkcija_troska.png')
    else:
        fig.savefig('static/images/funkcija_troska.png')

    return render_template(
        'info.html',
        link1=links[0],
        link2=links[1],
        link3=links[2],
        link4=links[3],
        link5=links[4],
        button1='Alcohol Distribution',
        title='funkcija troška',
        image=image)


@app.route('/izgled_regresije')
def izgled_regresije():

    plt.figure(figsize=(10, 6))
    plt.scatter(x_scaled[:, 1], y, color='black')
    x = np.linspace(-5, 20, 1000)
    s = theta[1] * x + theta[0]

    plt.title("our prediction visualization")
    plt.xlabel('Alcohol percent')
    plt.ylabel('Quality of wine')

    plt.plot(x, s)

    file_exists = os.path.exists('static/images/izgled_regresije.png')

    if file_exists:
        image = os.path.join('static/images/izgled_regresije.png')
    else:
        plt.savefig('static/images/izgled_regresije.png')

    return render_template(
        'info.html',
        link1=links[0],
        link2=links[1],
        link3=links[2],
        link4=links[3],
        link5=links[4],
        button1='Alcohol Distribution',
        title='izgled regresije',
        image=image)


@app.route('/linkovi/')
def linkovi():
    linkovi = ['https://github.com/PMF-Data-Science/PMF-DataScience',
               'https://github.com/PMF-Data-Science/computer-science-educators-stack-exchange-data']
    return render_template('links.html', linkovi=linkovi)
