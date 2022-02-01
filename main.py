from flask import Flask, render_template
from flask_bootstrap3 import Bootstrap
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import os.path


app = Flask(__name__)
bootstrap = Bootstrap(app)
data = pd.read_csv("winequality-red.csv")


def figax():
    fig, ax = plt.subplots()
    return fig


@app.template_filter('nesto')
def reverse_filter(s):
    st = s.split("_")
    #print("Koja je vrijednost: ", st[1])
    #print("Koji je tip: ", type(st[1]))

    if len(st) < 2:
        return st[0]
    else:
        return st[0]+' '+st[1]

    # return s[::-1]


# flask-bootstrap
@app.route('/')
def index():
    name = 'Home'
    return render_template('index.html', name=name)


'''@app.route('/dataset', methods=("POST", "GET"))
def dataset():
    m = 0
    n = 9
    #data = pd.read_csv("winequality-red.csv")
    json_list = json.loads(json.dumps(
        list(data.loc[m:n].T.to_dict().values())))
    return render_template(
        'table.html', tables=json_list,
        link1='dataset', link2='describe', link3='shape',
        switch=1)'''


@app.route('/dataset/<num>', methods=("POST", "GET"))
def dataset(num):
    s = 0
    k = 9
    m = int(num) * 10 - 10
    n = int(num) * 10 - 1
    print("Broj1:", m)
    print("Broj2:", n)
    print("Num: ", num)
    print("Tip: ", type(num))
    num1 = 0
    num2 = 1
    num3 = 2
    num4 = 3
    #num = int(num) + 1
    num1 = num1 + int(num)
    num2 = num2 + int(num)
    num3 = num3 + int(num)
    num4 = num4 + int(num)
    #data = pd.read_csv("winequality-red.csv")
    json_list = json.loads(json.dumps(
        list(data.loc[m:n].T.to_dict().values())))

    return render_template(
        'table.html', tables=json_list,
        link1='describe', link2='describe', link3='shape',
        switch=1,
        num1=num1, num2=num2, num3=num3, num4=num4, num=str(num))


@app.route('/describe')
def describe():

    describe = data.describe()
    json_list = json.loads(json.dumps(list(describe.T.to_dict().values())))
    return render_template(
        'table.html', tables=json_list,
        link1='dataset', link2='describe', link3='shape')


@app.route('/shape')
def shape():

    shape = data.shape

    return render_template(
        'info.html', shape=shape, rows=shape[0], columns=shape[1],
        link1='dataset', link2='describe', link3='shape',
    )


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
