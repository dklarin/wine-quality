import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model, metrics


data = pd.read_csv("winequality-red.csv")
x = data['pH']
y = data['quality']

poly_reg = PolynomialFeatures(degree=4)


'''def figax():
    fig, ax = plt.subplots()
    return fig'''

# Izračun gradijentnog spusta


def gradient_descent(x, y, theta, iterations, alpha):
    past_costs = []
    past_thetas = [theta]
    for i in range(iterations):
        prediction = np.dot(x, theta)
        error = prediction - y
        cost = 1/(2*y.size) * np.dot(error.T, error)
        past_costs.append(cost)
        theta = theta - (alpha * (1/y.size) * np.dot(x.T, error))
        past_thetas.append(theta)

    return past_thetas, past_costs


'''@app.route('/dataset/<num>', methods=("POST", "GET"))
def dataset(num):
    s = 0
    k = 9
    m = int(num) * 10 - 10
    n = int(num) * 10 - 1
    print("Broj1:", m)
    print("Broj2:", n)
    print("Num: ", num)
    print("Tip: ", type(num))
    num1 = 0
    num2 = 1
    num3 = 2
    num4 = 3
    #num = int(num) + 1
    num1 = num1 + int(num)
    num2 = num2 + int(num)
    num3 = num3 + int(num)
    num4 = num4 + int(num)
    #data = pd.read_csv("winequality-red.csv")
    json_list = json.loads(json.dumps(
        list(data.loc[m:n].T.to_dict().values())))

    return render_template(
        'table.html', tables=json_list,
        link1='describe', link2='describe', link3='shape',
        switch=1,
        num1=num1, num2=num2, num3=num3, num4=num4, num=str(num))'''


# Polynomial regression


def rmses_deg_from_range(min_deg, max_deg):
    degrees = np.arange(min_deg, max_deg)
    min_rmse, min_deg = 1e10, 0
    rmses = []

    print(degrees)

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

    print(rmses)
    return (rmses, min_rmse, min_deg)
