from flask import Blueprint, render_template
import numpy as np
from reg_lin_methods import *
from data_read import *

reg_lin_gra_spu_page = Blueprint('reg_lin_gra_spu_page', __name__,
                                 template_folder='templates')

sp = 'reg_lin_gra_spu_page.'

gra_spu_links = ['rlgd_distribution_alcohol', 'rlgd_distribution_ph',
                 'rlgd_gradient_descent', 'rlgd_cost_function', 'rlgd_regression_appearance']

x = x_alcohol()
y = x_ph()

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
@reg_lin_gra_spu_page.route('/rlgd_distribution_alcohol')
def rlgd_distribution_alcohol():

    x_ax = 'alcohol'
    pic = 'static/images/rlgd_distribution_alcohol.png'

    image = handle_image(x, x_ax, pic, 'hist', 0)

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
@reg_lin_gra_spu_page.route('/rlgd_distribution_ph')
def rlgd_distribution_ph():

    x_ax = 'ph'
    pic = 'static/images/rlgd_distribution_ph.png'

    image = handle_image(y, x_ax, pic, 'hist', 0)

    return render_template(
        'info.html',
        link1=sp+gra_spu_links[0],
        link2=sp+gra_spu_links[1],
        link3=sp+gra_spu_links[2],
        link4=sp+gra_spu_links[3],
        link5=sp+gra_spu_links[4],
        button1='Alcohol Distribution',
        title='Distribution pH',
        image=image)


# 1.1.3
@reg_lin_gra_spu_page.route('/rlgd_gradient_descent')
def rlgd_gradient_descent():

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
@reg_lin_gra_spu_page.route('/rlgd_cost_function')
def rlgd_cost_function():

    pic = 'static/images/rlgd_cost_function.png'

    image = handle_image(0, 0, pic, 'plot', past_costs)

    return render_template(
        'info.html',
        link1=sp+gra_spu_links[0],
        link2=sp+gra_spu_links[1],
        link3=sp+gra_spu_links[2],
        link4=sp+gra_spu_links[3],
        link5=sp+gra_spu_links[4],
        button1='Alcohol Distribution',
        title='Cost Function',
        image=image)


# 1.1.5
@reg_lin_gra_spu_page.route('/rlgd_regression_appearance')
def rlgd_regression_appearance():

    pic = 'static/images/rlgd_regression_appearance.png'

    image = handle_image_reg(x_scaled, y, theta[0], theta[1], pic)

    return render_template(
        'info.html',
        link1=sp+gra_spu_links[0],
        link2=sp+gra_spu_links[1],
        link3=sp+gra_spu_links[2],
        link4=sp+gra_spu_links[3],
        link5=sp+gra_spu_links[4],
        button1='Alcohol Distribution',
        title='regression appearance',
        image=image)
