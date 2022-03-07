from flask import Blueprint, render_template
from methods import *
import os.path
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


from data_read import *

import seaborn as sns

from reg_lin_cro_val_methods import *


reg_lin_cro_val_page = Blueprint('reg_lin_cro_val_page', __name__,
                                 template_folder='templates')

sp = 'reg_lin_cro_val_page.'


data = pd.read_csv("winequality-red.csv")
data_copy = data[["residual sugar", "sulphates", "alcohol", "pH"]]


x_1 = data['alcohol']
y_1 = data['pH']


x_3 = data_copy[["residual sugar", "sulphates", "alcohol"]]
y_3 = data_copy["pH"]


# Link names
cro_val_links = ['rlcv_metrics', 'rlcv_k_fold_validation',
                 'rlcv_matrics', 'rlcv_model_training']


# Data scaling
x_scaled_1 = x_scale(x_1)
x_scaled_3 = x_scale_3(x_3)


x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(
    x_scaled_1, y_1, shuffle=True, test_size=0.2)
x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(
    x_scaled_3, y_3, shuffle=True, test_size=0.2)


lr = linear_model.LinearRegression(fit_intercept=True, copy_X=True)


# 1 Linear Regression Metrics
@reg_lin_cro_val_page.route('/rlcv_metrics')
def rlcv_metrics():

    lr.fit(x_train_1, y_train_1)
    y_predict = lr.predict(x_test_1)
    mae, mse, rmse, r2_square = linear_metrics(y_test_1, y_predict)

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
        link1=sp+cro_val_links[0],
        link2=sp+cro_val_links[1],
        link3=sp+cro_val_links[2],
        link4=sp+cro_val_links[3],
        title='Metrics'
    )


# K-fold Validation
@reg_lin_cro_val_page.route('/rlcv_k_fold_validation')
def rlcv_k_fold_validation():

    grid_linear = GridSearchCV(linear_model.LinearRegression(), param_grid={'fit_intercept': [
                               True, False],  'copy_X': [True, False]}, cv=5)

    grid_linear.fit(x_train_1, y_train_1)
    grid_linear.best_params_
    lr_model = grid_linear.best_estimator_
    lr_model.fit(x_train_1, y_train_1)
    predictions_linear = lr_model.predict(x_test_1)

    mae, mse, rmse, r2_square = linear_metrics(y_test_1, predictions_linear)

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
        link1=sp+cro_val_links[0],
        link2=sp+cro_val_links[1],
        link3=sp+cro_val_links[2],
        link4=sp+cro_val_links[3],
        title='K-Fold Validation Metrics'
    )


# Miltiple Linear Regression Matrices
@reg_lin_cro_val_page.route('/rlcv_matrics')
def rlcv_matrics():

    ax = sns.heatmap(data_copy.corr())
    fig = ax.get_figure()
    fig.savefig('static/images/rlcv_matrics.png')
    plt.clf()
    plt.cla()
    plt.close(fig)

    image = os.path.join('static/images/rlcv_matrics.png')

    return render_template(
        'info.html',
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        link1=sp+cro_val_links[0],
        link2=sp+cro_val_links[1],
        link3=sp+cro_val_links[2],
        link4=sp+cro_val_links[3],
        title='Confusion Matrics',
        image=image
    )


# Model Training
@reg_lin_cro_val_page.route('/rlcv_model_training')
def rlcv_model_training():

    lr.fit(x_train_3, y_train_3)
    predictions_linear = lr.predict(x_test_3)
    mae, mse, rmse, r2_square = linear_metrics(y_test_3, predictions_linear)

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
        link1=sp+cro_val_links[0],
        link2=sp+cro_val_links[1],
        link3=sp+cro_val_links[2],
        link4=sp+cro_val_links[3],
        title='Model Training Metrics',
        switch=2,
    )
