from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import pandas as pd
from methods import *
import os.path

reg_pol_page = Blueprint('reg_pol_page', __name__,
                         template_folder='templates')

data = pd.read_csv("winequality-red.csv")
x = data['pH']
y = data['quality']


@reg_pol_page.route('/', defaults={'page': 'index'})
@reg_pol_page.route('/<page>')
def show(page):
    try:
        return render_template(f'pages/{page}.html')
    except TemplateNotFound:
        abort(404)


# 2
@reg_pol_page.route('/polynomial_regression')
def polynomial_regression():

    return render_template(
        'info.html',
        link1='reg_pol_page.'+'vengaboys',
        link2='index',
        title='polynomial regression')


# 2.1
@reg_pol_page.route('/vengaboys')
def vengaboys():

    poly_reg = PolynomialFeatures(degree=4)
    x_poly = poly_reg.fit_transform(x.to_numpy().reshape(-1, 1))

    lr = linear_model.LinearRegression(
        fit_intercept=True, normalize=False, copy_X=True)

    lr.fit(x_poly, y)
    y_predict = lr.predict(x_poly)

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
        link1='reg_pol_page.'+'vengaboys',
        link2='index',
        title='polynomial regression',
        image=image
    )
