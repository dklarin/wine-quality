from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
from methods import *
import os.path
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from data_read import *

import seaborn as sns


reg_lin_una_val_page = Blueprint('reg_lin_una_val_page', __name__,
                                 template_folder='templates')

sp = 'reg_lin_una_val_page.'

x_scaled = StandardScaler().fit_transform(x.to_numpy().reshape(-1, 1))

x_scaled = pd.DataFrame(x_scaled, columns=["alcohol"])

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, shuffle=True, test_size=0.2)

lr = linear_model.LinearRegression(
    fit_intercept=True, normalize=False, copy_X=True)

lr.fit(x_train, y_train)

y_predict = lr.predict(x_test)


# 2.2
@reg_lin_una_val_page.route('/reg_lin_metrike')
def reg_lin_metrike():

    mae = metrics.mean_absolute_error(y_test, y_predict)
    mse = metrics.mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_predict))
    r2_square = metrics.r2_score(y_test, y_predict)

    return render_template(
        'info.html',
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        value1=mae.round(4),
        value2=mse.round(4),
        value3=rmse.round(4),
        value4=r2_square.round(4)*100,
        link1=sp+'reg_lin_metrike',
        link2=sp+'k_fold_validacija',
        link3=sp+'matrica_konfuzije',
        link4=sp+'reg_lin_treniranje_modela',
        title='metrike'
    )


# K fold validacija i traženje najbolje kombinacije parametara
@reg_lin_una_val_page.route('/k_fold_validacija')
def k_fold_validacija():

    grid_linear = GridSearchCV(linear_model.LinearRegression(), param_grid={'fit_intercept': [
                               True, False], 'normalize': [True, False], 'copy_X': [True, False]}, cv=5)
    grid_linear.fit(x_train, y_train)
    grid_linear.best_params_
    lr_model = grid_linear.best_estimator_
    lr_model.fit(x_train, y_train)
    predictions_linear = lr_model.predict(x_test)

    mae = metrics.mean_absolute_error(y_test, predictions_linear)
    mse = metrics.mean_squared_error(y_test, predictions_linear)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions_linear))
    r2_square = metrics.r2_score(y_test, predictions_linear)

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
        link1=sp+'reg_lin_metrike',
        link2=sp+'k_fold_validacija',
        link3=sp+'matrica_konfuzije',
        link4=sp+'reg_lin_treniranje_modela',
        title='metrike'
    )


@reg_lin_una_val_page.route('/matrica_konfuzije')
def matrica_konfuzije():

    data_copy = copy_data()
    ax = sns.heatmap(data_copy.corr())
    fig = ax.get_figure()
    fig.savefig('static/images/matrica_konfuzije.png')
    plt.close(fig)

    image = os.path.join('static/images/matrica_konfuzije.png')

    return render_template(
        'info.html',
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        # value1=mae.round(4),
        # value2=mse.round(4),
        # value3=rmse.round(4),
        # value4=r2_square.round(4),
        link1=sp+'reg_lin_metrike',
        link2=sp+'k_fold_validacija',
        link3=sp+'matrica_konfuzije',
        link4=sp+'reg_lin_treniranje_modela',
        title='metrike',
        image=image
    )


@reg_lin_una_val_page.route('/reg_lin_treniranje_modela')
def reg_lin_treniranje_modela():

    x = copy_data2()
    y = y_quality()

    x_scaled = StandardScaler().fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled, index=x.index, columns=x.columns)
    x_scaled.isna().sum()
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, shuffle=True, test_size=0.2)

    lr = linear_model.LinearRegression(
        fit_intercept=True, normalize=False, copy_X=True)
    lr.fit(x_train, y_train)

    predictions_linear = lr.predict(x_test)
    mae = metrics.mean_absolute_error(y_test, predictions_linear)
    mse = metrics.mean_squared_error(y_test, predictions_linear)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions_linear))
    r2_square = metrics.r2_score(y_test, predictions_linear)

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
        link1=sp+'reg_lin_metrike',
        link2=sp+'k_fold_validacija',
        link3=sp+'matrica_konfuzije',
        link4=sp+'reg_lin_treniranje_modela',
        title='treniranje modela',
        switch=2,
    )
