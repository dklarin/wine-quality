import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model, metrics

poly_reg = PolynomialFeatures(degree=4)

# reg_pol_izgled_regresije


def plot(X, y, y_pred, lin_reg):
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X, y, color='red')
    plt.scatter(X, y_pred, color='green')
    plt.plot(X_grid, lin_reg.predict(
        poly_reg.fit_transform(X_grid)), color='black')
    plt.title('Polynomial Regression')
    plt.xlabel('pH level')
    plt.ylabel('Quality')
    # plt.show()


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
def model_training(x, y, min_deg):
    poly_reg = PolynomialFeatures(degree=min_deg)
    x_poly = poly_reg.fit_transform(x.to_numpy().reshape(-1, 1))
    lr = linear_model.LinearRegression(
        fit_intercept=True, normalize=False, copy_X=True)
    lr.fit(x_poly, y)
    y_predict = lr.predict(x_poly)
    return y_predict
