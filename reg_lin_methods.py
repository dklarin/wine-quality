import os.path
import numpy as np
import matplotlib.pyplot as plt


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
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))
    plt.scatter(x_scaled[:, 1], y, color='black')
    x = np.linspace(-5, 20, 1000)
    y = theta_1 * x + theta_0
    plt.title('Prediction visualization')
    plt.xlabel('Alcohol percent')
    plt.ylabel('Quality of wine')
    plt.plot(x, y)
    return fig

# distribution_alcohol
# distribution_quality


def handle_image(x, x_ax, pic):
    fig = histogram(x, x_ax)
    file_exists = os.path.exists(pic)
    if file_exists:
        image = os.path.join(pic)
        return image
    else:
        fig.savefig(pic)
