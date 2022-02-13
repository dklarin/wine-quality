from ast import keyword
from unittest import result
from flask import Flask, render_template
from flask_bootstrap3 import Bootstrap
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

from methods import *

from reg_lin_gra_spu import reg_lin_page
from reg_lin_una_val import reg_lin_una_val_page
from reg_pol import reg_pol_page

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.register_blueprint(reg_lin_page)
app.register_blueprint(reg_pol_page)
app.register_blueprint(reg_lin_una_val_page)


data = pd.read_csv("winequality-red.csv")


# metoda koja razdvaja string
@app.template_filter('nesto')
def reverse_filter(s):
    st = s.split("_")
    if len(st) < 2:
        return st[0]
    else:
        return st[0]+' '+st[1]


@app.template_filter('tocka')
def reverse_filter(s):
    if (s.find('.') != -1):
        st = s.split(".")
        print(st)
        st = st[1].split("_")
        print(st)
    else:
        st = s.split("_")
    if len(st) < 2:
        return st[0]
    elif len(st) == 3:
        return st[2]
    elif len(st) == 4:
        return st[2]+' '+st[3]
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
        'table.html',
        tables=json_list,
        title='dataset',
        link1='dataset',
        link2='describe',
        link3='shape',
        switch=0
    )


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
        'table.html',
        tables=json_list,
        title='describe',
        link1='dataset',
        link2='describe',
        link3='shape',
        switch=1
    )


# retci i stupci
@app.route('/shape')
def shape():
    shape = data.shape
    return render_template(
        'info.html',
        keyword1='retci',
        keyword2='stupci',
        value1=shape[0],
        value2=shape[1],
        title='shape',
        link1='dataset',
        link2='describe',
        link3='shape'
    )


@app.route('/linkovi/')
def linkovi():
    linkovi = ['https://www.pmfst.unist.hr/',
               'https://www.pmfst.unist.hr/kalendar/']
    return render_template('links.html', linkovi=linkovi)
