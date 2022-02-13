from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
from methods import *
import os.path
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

reg_lin_una_val_page = Blueprint('reg_lin_una_val_page', __name__,
                                 template_folder='templates')

sp = 'reg_lin_una_val_page.'

data = pd.read_csv("winequality-red.csv")

x = data['alcohol']
y = data['quality']

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
        value4=r2_square.round(4),
        link1=sp+'reg_lin_metrike',
        link2=sp+'reg_lin_metrike',
        link3=sp+'reg_lin_metrike',
        title='metrike'
    )
