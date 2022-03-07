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
