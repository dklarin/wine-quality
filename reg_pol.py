import numpy as np
from reg_pol_methods import *
from data_read import *
import json
import os.path
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from flask import Blueprint, render_template


import matplotlib
matplotlib.use('Agg')


reg_pol_page = Blueprint('reg_pol_page', __name__,
                         template_folder='templates')

sp = 'reg_pol_page.'

reg_pol_links = ['rp_regression_look', 'rp_metrics',
                 'rp_forced_access', 'model_training', 'model_appearance']


x = x_alcohol()
y = x_ph()


poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x.to_numpy().reshape(-1, 1))
lr = linear_model.LinearRegression(fit_intercept=True, copy_X=True)
lr.fit(x_poly, y)
y_predict = lr.predict(x_poly)


# 2
@reg_pol_page.route('/regression_polynomial')
def regression_polynomial():

    return render_template(
        'info.html',
        link1=sp+reg_pol_links[0],
        link2=sp+reg_pol_links[1],
        link3=sp+reg_pol_links[2],
        link4=sp+reg_pol_links[3],
        link5=sp+reg_pol_links[4],
        title='polynomial regression')


# 2.1
@reg_pol_page.route('/rp_regression_look')
def rp_regression_look():

    poly_reg = PolynomialFeatures(degree=4)
    x_poly = poly_reg.fit_transform(x.to_numpy().reshape(-1, 1))
    lr = linear_model.LinearRegression(
        fit_intercept=True, copy_X=True)
    lr.fit(x_poly, y)
    y_predict = lr.predict(x_poly)

    pic_pol = 'static/images/pol_regresija.png'
    image_pol = handle_image_reg_pol(x, y, y_predict, lr, 4, pic_pol, 0, 0)

    return render_template(
        'info.html',
        link1=sp+reg_pol_links[0],
        link2=sp+reg_pol_links[1],
        link3=sp+reg_pol_links[2],
        link4=sp+reg_pol_links[3],
        link5=sp+reg_pol_links[4],
        title='polynomial regression',
        image=image_pol
    )


# 2.2
@reg_pol_page.route('/rp_metrics')
def rp_metrics():

    poly_reg = PolynomialFeatures(degree=4)
    x_poly = poly_reg.fit_transform(x.to_numpy().reshape(-1, 1))
    lr = linear_model.LinearRegression(
        fit_intercept=True, copy_X=True)
    lr.fit(x_poly, y)
    y_predict = lr.predict(x_poly)

    mae = metrics.mean_absolute_error(y, y_predict)
    mse = metrics.mean_squared_error(y, y_predict)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_predict))
    r2_square = metrics.r2_score(y, y_predict)

    return render_template(
        'info.html',
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        value1=mae.round(4),
        value2=mse.round(4),
        value3=rmse.round(4),
        value4=r2_square.round(4),
        link1=sp+reg_pol_links[0],
        link2=sp+reg_pol_links[1],
        link3=sp+reg_pol_links[2],
        link4=sp+reg_pol_links[3],
        link5=sp+reg_pol_links[4],
        title='metrike',
    )


# 2.3
@reg_pol_page.route('/rp_forced_access')
def rp_forced_access():

    min_deg = 1
    max_deg = 10
    rmses, min_rmse, min_deg = rmses_deg_from_range(x, y, min_deg, max_deg)
    degrees = range(1, 10)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.plot(degrees, rmses)
    ax.set_yscale('log')
    ax.set_xlabel('Degree')
    ax.set_ylabel('RMSE')

    file_exists = os.path.exists('static/images/rmse_degree.png')

    if file_exists:
        image = os.path.join('static/images/rmse_degree.png')
    else:
        fig.savefig('static/images/rmse_degree.png')

    imageText = 'Best degree {} with RMSE {}'.format(
        min_deg, min_rmse.round(4))

    return render_template(
        'info.html',
        link1=sp+reg_pol_links[0],
        link2=sp+reg_pol_links[1],
        link3=sp+reg_pol_links[2],
        link4=sp+reg_pol_links[3],
        link5=sp+reg_pol_links[4],
        title='prisilni pristup',
        image=image,
        imageText=imageText
    )


# 2.4
@reg_pol_page.route('/model_training')
def model_training():

    min_deg = 1
    max_deg = 10
    rmses, min_rmse, min_deg = rmses_deg_from_range(x, y, min_deg, max_deg)

    poly_reg = PolynomialFeatures(degree=min_deg)
    x_poly = poly_reg.fit_transform(x.to_numpy().reshape(-1, 1))
    lr = linear_model.LinearRegression(
        fit_intercept=True, copy_X=True)
    lr.fit(x_poly, y)
    y_predict = lr.predict(x_poly)

    df = pd.DataFrame({'Real Values': y, 'Predictions': y_predict})

    json_list = json.loads(json.dumps(
        list(df.head(14).T.to_dict().values())))

    return render_template(
        'table.html',
        link1=sp+reg_pol_links[0],
        link2=sp+reg_pol_links[1],
        link3=sp+reg_pol_links[2],
        link4=sp+reg_pol_links[3],
        link5=sp+reg_pol_links[4],
        tables=json_list,
        title='Trained Model Table',
        switch=2,
    )


# 2.5
@reg_pol_page.route('/model_appearance')
def model_appearance():

    min_deg = 1
    max_deg = 10
    rmses, min_rmse, min_deg = rmses_deg_from_range(x, y, min_deg, max_deg)

    poly_reg = PolynomialFeatures(degree=min_deg)
    x_poly = poly_reg.fit_transform(x.to_numpy().reshape(-1, 1))
    lr = linear_model.LinearRegression(
        fit_intercept=True, copy_X=True)
    lr.fit(x_poly, y)
    y_predict = lr.predict(x_poly)

    pic_pol = 'static/images/pol_model.png'
    image_pol = handle_image_reg_pol(
        x, y, y_predict, lr, min_deg, pic_pol, 0, 0)

    return render_template(
        'info.html',
        link1=sp+reg_pol_links[0],
        link2=sp+reg_pol_links[1],
        link3=sp+reg_pol_links[2],
        link4=sp+reg_pol_links[3],
        link5=sp+reg_pol_links[4],
        title='Trained Model Appearance',
        image=image_pol,
        switch=2,
    )
