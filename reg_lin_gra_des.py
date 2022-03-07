from flask import Blueprint, render_template
import numpy as np
from data_read import *
from reg_lin_gra_des_methods import *

reg_lin_gra_des_page = Blueprint('reg_lin_gra_des_page', __name__,
                                 template_folder='templates')

sp = 'reg_lin_gra_des_page.'
path = 'static/images/rlgd_'

# Links name
gra_des_links = ['rlgd_distribution_alcohol', 'rlgd_distribution_ph',
                 'rlgd_gradient_descent', 'rlgd_cost_function', 'rlgd_regression_appearance']


# Data
x = x_alcohol()
y = x_ph()


# Data scaling
x_scaled = x_scale(x)


# Hyperparameters
alpha = 0.01
iterations = 2000
np.random.seed(123)
theta = np.random.rand(2)


# Cost parameters
past_thetas, past_costs = gradient_descent(
    x_scaled, y, theta, iterations, alpha)
theta = past_thetas[-1]


# 1 Distribution Alcohol
@reg_lin_gra_des_page.route('/rlgd_distribution_alcohol')
def rlgd_distribution_alcohol():

    variable = 'alcohol'
    image_filename = path+'1_distribution_alcohol.png'
    image = handle_distribution(image_filename, x, variable)

    return render_template(
        'info.html',
        link1=sp+gra_des_links[0],
        link2=sp+gra_des_links[1],
        link3=sp+gra_des_links[2],
        link4=sp+gra_des_links[3],
        link5=sp+gra_des_links[4],
        title='distribution alcohol',
        image=image
    )


# 2 Distribution pH
@reg_lin_gra_des_page.route('/rlgd_distribution_ph')
def rlgd_distribution_ph():

    variable = 'ph'
    image_filename = path+'2_distribution_ph.png'
    image = handle_distribution(image_filename, y, variable)

    return render_template(
        'info.html',
        link1=sp+gra_des_links[0],
        link2=sp+gra_des_links[1],
        link3=sp+gra_des_links[2],
        link4=sp+gra_des_links[3],
        link5=sp+gra_des_links[4],
        button1='Alcohol Distribution',
        title='Distribution pH',
        image=image)


# 3 Gradient Descent
@reg_lin_gra_des_page.route('/rlgd_gradient_descent')
def rlgd_gradient_descent():

    return render_template(
        'info.html',
        keyword1='theta 1',
        keyword2='theta 2',
        link1=sp+gra_des_links[0],
        link2=sp+gra_des_links[1],
        link3=sp+gra_des_links[2],
        link4=sp+gra_des_links[3],
        link5=sp+gra_des_links[4],
        button1='Alcohol Distribution',
        title='gradient descent',
        value1=theta[0].round(2),
        value2=theta[1].round(2))


# 4 Cost Function
@reg_lin_gra_des_page.route('/rlgd_cost_function')
def rlgd_cost_function():

    image_filename = path+'3_cost_function.png'
    image = handle_cost(image_filename, past_costs)

    return render_template(
        'info.html',
        link1=sp+gra_des_links[0],
        link2=sp+gra_des_links[1],
        link3=sp+gra_des_links[2],
        link4=sp+gra_des_links[3],
        link5=sp+gra_des_links[4],
        button1='Alcohol Distribution',
        title='Cost Function',
        image=image)


# 5 Regression Appearance
@reg_lin_gra_des_page.route('/rlgd_regression_appearance')
def rlgd_regression_appearance():

    image_filename = path+'4_regression_appearance.png'
    image = handle_regression(image_filename, x_scaled, y, theta[0], theta[1])

    return render_template(
        'info.html',
        link1=sp+gra_des_links[0],
        link2=sp+gra_des_links[1],
        link3=sp+gra_des_links[2],
        link4=sp+gra_des_links[3],
        link5=sp+gra_des_links[4],
        button1='Alcohol Distribution',
        title='regression appearance',
        image=image)
