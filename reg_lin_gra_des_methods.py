import numpy as np
import matplotlib.pyplot as plt
import os.path


# 1 from which file
# 2 from which method(s)


# Method plots histogram for distributions
# 1 self
# 2 handle_distribution
def histogram(x, variable):
    fig, ax = plt.subplots()
    plt.hist(x, density=False, bins=30)
    plt.ylabel('Count')
    plt.xlabel(variable)
    return fig


# Method plots cost function
# 1 self
# 2 handle_cost
def plot(past_costs):
    fig, ax = plt.subplots()
    plt.title('Cost Function')
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.plot(past_costs)
    return fig


# Method plots regression
# 1 self
# 2 handle_regression
def plot_regression(x_scaled, y, theta_0, theta_1):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_scaled[:, 1], y, color='black')
    x = np.linspace(-5, 20, 1000)
    y = theta_1 * x + theta_0
    plt.plot(x, y)
    plt.title('Prediction visualization')
    plt.xlabel('Alcohol Percent')
    plt.ylabel('pH Values')
    return plt


# Method checks if image exists
# 1 self
# 2 handle distribution
# 2 handle_cost
# 2 handle_regression
def handle_image(image_filename, fig):
    file_exists = os.path.exists(image_filename)
    if file_exists:
        image = os.path.join(image_filename)
        return image
    else:
        fig.savefig(image_filename)
        # plt.clf()
        # plt.cla()
        # plt.close()


# Method scales x variable
# Regression Linear Gradient Descent
def x_scale(x):
    x_scaled = (x - x.mean()) / x.std()
    x_scaled = np.c_[np.ones(x_scaled.shape[0]), x_scaled]
    return x_scaled


# Method calculates gradient descent
# Regression Linear Gradient Descent
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


# 1 Regression Linear Gradient Descent
# 2 Distribution Alcohol
# 2 Distribution pH
def handle_distribution(image_filename, x, variable):
    fig = histogram(x, variable)
    image = handle_image(image_filename, fig)
    return image


# 1 Regression Linear Gradient Descent
# 2 Cost Function
def handle_cost(image_filename, past_costs):
    fig = plot(past_costs)
    image = handle_image(image_filename, fig)
    return image


# 1 Regression Linear Gradient Descent
# 2 Regression Appearance
def handle_regression(image_filename, x_scaled, y, theta_0, theta_1):
    plt = plot_regression(x_scaled, y, theta_0, theta_1)
    image = handle_image(image_filename, plt)
    return image
