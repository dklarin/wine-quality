from flask import Blueprint, render_template
#from methods import *
import numpy as np
import os.path
import pandas as pd
from reg_lin_methods import *
from data_read import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model, metrics
from reg_pol_methods import *
from reg_lin_methods import *

vina_page = Blueprint('vina_page', __name__,
                      template_folder='templates')

x = x_alcohol()
y = x_ph()

d = {'col1': [13, 12], 'col2': [3.24, 2.9]}
df = pd.DataFrame(data=d)
i = df['col1']
j = df['col2']

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


min_deg = 1
max_deg = 10
rmses, min_rmse, min_deg = rmses_deg_from_range(x, y, min_deg, max_deg)

poly_reg = PolynomialFeatures(degree=min_deg)
x_poly = poly_reg.fit_transform(x.to_numpy().reshape(-1, 1))
lr = linear_model.LinearRegression(fit_intercept=True, copy_X=True)

lr.fit(x_poly, y)
y_predict = lr.predict(x_poly)


@vina_page.route('/vino_1')
def vino_1():

    pic_lin = 'static/images/lin_vino_1.png'
    pic_pol = 'static/images/pol_vino_1.png'

    image_lin = handle_image_reg_lin_not_scaled(
        x_scaled, x, y, theta[0], theta[1], pic_lin, i[0], j[0])
    image_pol = handle_image_reg_pol(
        x, y, y_predict, lr, min_deg, pic_pol, i[0], j[0])

    return render_template(
        'vina.html',
        link1='vina_page.'+'vino_1',
        link2='vina_page.'+'vino_2',
        image=image_lin,
        image2=image_pol,
        text='Vino 1, Alkohol: 13%, pH: 3,24'
    )


@vina_page.route('/vino_2')
def vino_2():

    pic_lin = 'static/images/lin_vino_2.png'
    pic_pol = 'static/images/pol_vino_2.png'

    image_lin = handle_image_reg_lin_not_scaled(
        x_scaled, x, y, theta[0], theta[1], pic_lin, i[1], j[1])
    image_pol = handle_image_reg_pol(
        x, y, y_predict, lr, min_deg, pic_pol, i[1], j[1])

    return render_template(
        'vina.html',
        link1='vina_page.'+'vino_1',
        link2='vina_page.'+'vino_2',
        image=image_lin,
        image2=image_pol,
        text='Vino 1, Alkohol: 12%, pH: 2,9'

    )
