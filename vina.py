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

vina_page = Blueprint('vina_page', __name__,
                      template_folder='templates')

x = x_ph()
y = y_quality()


@vina_page.route('/vina')
def vina():

    name = 'pošip'

    min_deg = 1
    max_deg = 10
    rmses, min_rmse, min_deg = rmses_deg_from_range(x, y, min_deg, max_deg)

    poly_reg = PolynomialFeatures(degree=min_deg)
    x_poly = poly_reg.fit_transform(x.to_numpy().reshape(-1, 1))
    lr = linear_model.LinearRegression(
        fit_intercept=True, copy_X=True)

    lr.fit(x_poly, y)
    y_predict = lr.predict(x_poly)

    file_exists = os.path.exists('static/images/posip.png')

    X_grid = np.arange(min(x), max(x), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(x, y, color='red')
    plt.scatter(x, y_predict, color='green')

    plt.scatter(x[0], y[0], color='blue')
    #plt.scatter(x, y_predict, color='yellow')

    plt.plot(X_grid, lr.predict(
        poly_reg.fit_transform(X_grid)), color='black')
    plt.title('Polynomial Regressioan')
    plt.xlabel('pH level')
    plt.ylabel('Quality')

    if file_exists:
        image = os.path.join('static/images/posip.png')
    else:
        plt.savefig('static/images/posip.png')
        image = os.path.join('static/images/posip.png')
        plt.clf()
        plt.cla()
        plt.close()

    return render_template(
        'vina.html',
        name=name,
        image=image

    )
