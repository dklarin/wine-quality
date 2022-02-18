import pandas as pd
from sklearn import linear_model, metrics
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from math import degrees
import numpy as np
import matplotlib
import os.path
matplotlib.use('Agg')


# Ne koristi se
def plot(x, y, y_pred, lin_reg, degree, pic):
    poly_reg = PolynomialFeatures(degree=degree)
    X_grid = np.arange(min(x), max(x), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(x, y, color='red')
    plt.scatter(x, y_pred, color='green')

    plt.title('Polynomial Regression')
    plt.xlabel('pH level')
    plt.ylabel('Quality')

    plt.plot(X_grid, lin_reg.predict(
        poly_reg.fit_transform(X_grid)), color='black')

    file_exists = os.path.exists(pic)

    if file_exists:
        image = os.path.join(pic)
        return image
    else:
        plt.savefig(pic)
        image = os.path.join('static/images/broken_page.png')
        plt.clf()
        plt.cla()
        plt.close()

        return image


# reg_pol_prisilni_pristup
# reg_pol_treniranje_modela
# reg_pol_izgled_modela
def rmses_deg_from_range(x, y, min_deg, max_deg):
    degrees = np.arange(min_deg, max_deg)
    min_rmse, min_deg = 1e10, 0
    rmses = []

    # print(degrees)

    for deg in degrees:

        poly_features = PolynomialFeatures(degree=deg)
        x_poly = poly_features.fit_transform(x.to_numpy().reshape(-1, 1))

        poly_reg = linear_model.LinearRegression()
        poly_reg.fit(x_poly, y)

        y_predict = poly_reg.predict(x_poly)
        poly_mse = metrics.mean_squared_error(y, y_predict)
        poly_rmse = np.sqrt(poly_mse)
        rmses.append(poly_rmse)

        if min_rmse > poly_rmse:
            min_rmse = poly_rmse
            min_deg = deg

    # print(rmses)
    return (rmses, min_rmse, min_deg)


# reg_pol_treniranje_modela
# reg_pol_izgled_modela
def model_training(x, y, degree):
    poly_reg = PolynomialFeatures(degree=degree)
    x_poly = poly_reg.fit_transform(x.to_numpy().reshape(-1, 1))
    lr = linear_model.LinearRegression(
        fit_intercept=True, normalize=False, copy_X=True)
    lr.fit(x_poly, y)
    y_predict = lr.predict(x_poly)
    return (y_predict, lr)


def handle_image_reg_pol(x, y, y_pred, lin_reg, degree, pic, i, j):

    file_exists = os.path.exists(pic)

    if file_exists:
        image = os.path.join(pic)
        return image
    else:
        poly_reg = PolynomialFeatures(degree=degree)
        X_grid = np.arange(min(x), max(x), 0.1)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(x, y, color='red')
        plt.scatter(x, y_pred, color='green')
        if type(i) == np.int64:
            plt.scatter(i, j, color='blue')

        plt.title('Polynomial Regression')
        plt.xlabel('pH level')
        plt.ylabel('Quality')

        plt.plot(X_grid, lin_reg.predict(
            poly_reg.fit_transform(X_grid)), color='black')
        plt.savefig(pic)
        plt.clf()
        plt.cla()
        plt.close()
        image = os.path.join('static/images/broken_page.png')
        return image
