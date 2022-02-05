import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)


def figax():
    fig, ax = plt.subplots()
    return fig

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
    plt.show()
