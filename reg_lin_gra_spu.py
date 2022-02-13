from flask import Blueprint, render_template
from methods import *
import numpy as np
import os.path
import pandas as pd
from reg_lin_methods import *

reg_lin_gra_spu_page = Blueprint('reg_lin_gra_spu_page', __name__,
                                 template_folder='templates')

sp = 'reg_lin_gra_spu_page.'

gra_spu_links = ['distribution_alcohol', 'distribution_quality',
                 'gradient_descen', 'funkcija_troska', 'reg_lin_izgled_regresije']

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


# 1.1.1
@reg_lin_gra_spu_page.route('/distribution_alcohol')
def distribution_alcohol():

    x_ax = 'alcohol'
    pic = 'static/images/distr_alcohol.png'

    image = handle_image(x, x_ax, pic)

    return render_template(
        'info.html',
        link1=sp+gra_spu_links[0],
        link2=sp+gra_spu_links[1],
        link3=sp+gra_spu_links[2],
        link4=sp+gra_spu_links[3],
        link5=sp+gra_spu_links[4],
        title='distribution alcohol',
        image=image
    )


# 1.1.2
@reg_lin_gra_spu_page.route('/distribution_quality')
def distribution_quality():

    x_ax = 'quality'
    pic = 'static/images/distr_quality.png'

    image = handle_image(y, x_ax, pic)

    return render_template(
        'info.html',
        link1=sp+gra_spu_links[0],
        link2=sp+gra_spu_links[1],
        link3=sp+gra_spu_links[2],
        link4=sp+gra_spu_links[3],
        link5=sp+gra_spu_links[4],
        button1='Alcohol Distribution',
        title='distribution quality',
        image=image)


# 1.1.3
@reg_lin_gra_spu_page.route('/gradient_descen')
def gradient_descen():

    return render_template(
        'info.html',
        keyword1='theta 1',
        keyword2='theta 2',
        link1=sp+gra_spu_links[0],
        link2=sp+gra_spu_links[1],
        link3=sp+gra_spu_links[2],
        link4=sp+gra_spu_links[3],
        link5=sp+gra_spu_links[4],
        button1='Alcohol Distribution',
        title='gradient descent',
        value1=theta[0].round(2),
        value2=theta[1].round(2))


# 1.1.4
@reg_lin_gra_spu_page.route('/funkcija_troska')
def funkcija_troska():

    pic = 'static/images/funkcija_troska.png'

    fig = plot(past_costs)

    file_exists = os.path.exists(pic)

    if file_exists:
        image = os.path.join(pic)
    else:
        fig.savefig(pic)

    return render_template(
        'info.html',
        link1=sp+gra_spu_links[0],
        link2=sp+gra_spu_links[1],
        link3=sp+gra_spu_links[2],
        link4=sp+gra_spu_links[3],
        link5=sp+gra_spu_links[4],
        button1='Alcohol Distribution',
        title='funkcija troška',
        image=image)


# 1.1.5
@reg_lin_gra_spu_page.route('/reg_lin_izgled_regresije')
def reg_lin_izgled_regresije():

    pic = 'static/images/izgled_regresije.png'

    plt = plot_regression(x_scaled, y, theta[0], theta[1])

    file_exists = os.path.exists(pic)

    if file_exists:
        image = os.path.join(pic)
    else:
        plt.savefig(pic)

    return render_template(
        'info.html',
        link1=sp+gra_spu_links[0],
        link2=sp+gra_spu_links[1],
        link3=sp+gra_spu_links[2],
        link4=sp+gra_spu_links[3],
        link5=sp+gra_spu_links[4],
        button1='Alcohol Distribution',
        title='izgled regresije',
        image=image)
