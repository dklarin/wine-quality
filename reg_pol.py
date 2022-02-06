from ast import keyword
from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model, metrics
import pandas as pd
from methods import *
import os.path

reg_pol_page = Blueprint('reg_pol_page', __name__,
                         template_folder='templates')

data = pd.read_csv("winequality-red.csv")
x = data['pH']
y = data['quality']

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x.to_numpy().reshape(-1, 1))

lr = linear_model.LinearRegression(
    fit_intercept=True, normalize=False, copy_X=True)

lr.fit(x_poly, y)
y_predict = lr.predict(x_poly)


@reg_pol_page.route('/', defaults={'page': 'index'})
@reg_pol_page.route('/<page>')
def show(page):
    try:
        return render_template(f'pages/{page}.html')
    except TemplateNotFound:
        abort(404)


# 2
@reg_pol_page.route('/regression_polynomial')
def regression_polynomial():

    return render_template(
        'info.html',
        link1='reg_pol_page.'+'izgled_regresije_pol',
        link2='index',
        title='polynomial regression')


# 2.1
@reg_pol_page.route('/izgled_regresije_pol')
def izgled_regresije_pol():

    X_grid = np.arange(min(x), max(x), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(x, y, color='red')
    plt.scatter(x, y_predict, color='green')
    plt.plot(X_grid, lr.predict(poly_reg.fit_transform(X_grid)), color='black')
    plt.title('Polynomial Regression')
    plt.xlabel('pH level')
    plt.ylabel('Quality')
    # plt.show()

    file_exists = os.path.exists('static/images/polinomijalna_regresija.png')

    if file_exists:
        image = os.path.join('static/images/polinomijalna_regresija.png')
    else:
        plt.savefig('static/images/polinomijalna_regresija.png')

    return render_template(
        'info.html',
        link1='reg_pol_page.'+'izgled_regresije_pol',
        link2='reg_pol_page.'+'metrike',
        title='polynomial regression',
        image=image
    )


# 2.2
@reg_pol_page.route('/metrike')
def metrike():

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
        link1='reg_pol_page.'+'izgled_regresije_pol',
        link2='reg_pol_page.'+'metrike',
        title='metrike',

    )
