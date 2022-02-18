import os.path
import numpy as np
import matplotlib.pyplot as plt


# izracun_gradijentnog spusta
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


# distribution_alcohol
# distribution_quality
def histogram(x, x_ax):
    fig, ax = plt.subplots()
    plt.hist(x, density=True, bins=30)
    plt.ylabel('Count')
    plt.xlabel(x_ax)
    return fig


# funkcija_troska
def plot(past_costs):
    fig, ax = plt.subplots()
    plt.title('Cost Function J')
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.plot(past_costs)
    return fig


# reg_lin_izgled_regresije
def plot_regression(x_scaled, y, theta_0, theta_1):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_scaled[:, 1], y, color='black')
    x = np.linspace(-5, 20, 1000)
    y = theta_1 * x + theta_0
    plt.plot(x, y)
    plt.title('Prediction visualization')
    plt.xlabel('Alcohol percent')
    plt.ylabel('Quality of wine')
    return plt


# distribution_alcohol
# distribution_quality
# funkcija_troska
def handle_image(x, x_ax, pic, visual, past_costs):
    if visual == 'hist':
        fig = histogram(x, x_ax)
    else:
        fig = plot(past_costs)
    file_exists = os.path.exists(pic)
    if file_exists:
        image = os.path.join(pic)
        return image
    else:
        fig.savefig(pic)


# reg_lin_izgled_regresije
def handle_image_reg(x_scaled, y, theta_0, theta_1, pic):
    plt = plot_regression(x_scaled, y, theta_0, theta_1)
    file_exists = os.path.exists(pic)
    if file_exists:
        image = os.path.join(pic)
        return image
    else:
        plt.savefig(pic)


# reg_lin_izgled_regresije
def handle_image_reg_lin(x_scaled, x, y, theta_0, theta_1, pic, i, j):

    file_exists = os.path.exists(pic)

    if file_exists:
        image = os.path.join(pic)
        return image
    else:
        plt.figure(figsize=(6, 5))
        plt.scatter(x_scaled[:, 1], y, color='black')
        #plt.scatter(x, y, color='red')
        plt.scatter(i, j, color='blue')
        x = np.linspace(-5, 20, 1000)
        y = theta_1 * x + theta_0
        plt.plot(x, y)
        plt.title('Prediction visualization')
        plt.xlabel('Alcohol percent')
        plt.ylabel('Quality of wine')
        plt.savefig(pic)
        plt.clf()
        plt.cla()
        plt.close()
        return image

# reg_lin_izgled_regresije


def handle_image_reg_lin_not_scaled(x_scaled, x, y, theta_0, theta_1, pic, i, j):

    file_exists = os.path.exists(pic)

    if file_exists:
        image = os.path.join(pic)
        return image
    else:
        #plt.figure(figsize=(6, 5))
        plt.scatter(x, y, color='black')
        #plt.scatter(x, y, color='red')
        plt.scatter(i, j, color='blue')
        x = np.linspace(9, 15, 1000)
        y = theta_1 * x + theta_0

        plt.title('Prediction visualization')
        plt.xlabel('Alcohol percent')
        plt.ylabel('Quality of wine')

        plt.plot(x, y)
        plt.savefig(pic)
        plt.clf()
        plt.cla()
        plt.close()
        return image
