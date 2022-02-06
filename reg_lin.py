from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
from methods import *
import os.path
import pandas as pd

reg_lin_page = Blueprint('reg_lin_page', __name__,
                         template_folder='templates')

lin_reg_links = ['gradijentni_spust', 'unakrsna_validacija']

sp = 'reg_lin_page.'

grad_spust_links = ['distribution_alcohol', 'distribution_quality',
                    'gradient_descen', 'funkcija_troska', 'izgled_regresije_lin']

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


# 1
@reg_lin_page.route('/regression_linear')
def regression_linear():

    return render_template(
        'info.html',
        link1=sp+lin_reg_links[0],
        link2=sp+lin_reg_links[1],
        title='linear regression')


# 1.1
@reg_lin_page.route('/gradijentni_spust')
def gradijentni_spust():

    return render_template(
        'info.html',
        link1=sp+grad_spust_links[0],
        link2=sp+grad_spust_links[1],
        link3=sp+grad_spust_links[2],
        link4=sp+grad_spust_links[3],
        link5=sp+grad_spust_links[4],
        title='gradijentni spust'
    )


# 1.2
@reg_lin_page.route('/unakrsna_validacija')
def unakrsna_validacija():

    return render_template(
        'info.html',
        link1='simple_page.gradijentni_spust',
        link2='simple_page.unakrsna_validacija',
        title='unakrsna validacija'
    )


# 1.1.1
@reg_lin_page.route('/distribution_alcohol')
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
        link1=sp+grad_spust_links[0],
        link2=sp+grad_spust_links[1],
        link3=sp+grad_spust_links[2],
        link4=sp+grad_spust_links[3],
        link5=sp+grad_spust_links[4],
        title='distribution alcohol',
        image=image
    )


# 1.1.2
@reg_lin_page.route('/distribution_quality')
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
        link1=sp+grad_spust_links[0],
        link2=sp+grad_spust_links[1],
        link3=sp+grad_spust_links[2],
        link4=sp+grad_spust_links[3],
        link5=sp+grad_spust_links[4],
        button1='Alcohol Distribution',
        title='distribution quality',
        image=image)


# 1.1.3
@reg_lin_page.route('/gradient_descen')
def gradient_descen():

    return render_template(
        'info.html',
        keyword1='theta 1',
        keyword2='theta 2',
        link1=sp+grad_spust_links[0],
        link2=sp+grad_spust_links[1],
        link3=sp+grad_spust_links[2],
        link4=sp+grad_spust_links[3],
        link5=sp+grad_spust_links[4],
        button1='Alcohol Distribution',
        title='gradient descent',
        value1=theta[0].round(2),
        value2=theta[1].round(2))


# 1.1.4
@reg_lin_page.route('/funkcija_troska')
def funkcija_troska():

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
        link1=sp+grad_spust_links[0],
        link2=sp+grad_spust_links[1],
        link3=sp+grad_spust_links[2],
        link4=sp+grad_spust_links[3],
        link5=sp+grad_spust_links[4],
        button1='Alcohol Distribution',
        title='funkcija troška',
        image=image)


# 1.1.5
@reg_lin_page.route('/izgled_regresije_lin')
def izgled_regresije_lin():

    plt.figure(figsize=(10, 6))
    plt.scatter(x_scaled[:, 1], y, color='black')
    x = np.linspace(-5, 7.5, 1000)
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
        link1=sp+grad_spust_links[0],
        link2=sp+grad_spust_links[1],
        link3=sp+grad_spust_links[2],
        link4=sp+grad_spust_links[3],
        link5=sp+grad_spust_links[4],
        button1='Alcohol Distribution',
        title='izgled regresije',
        image=image)
