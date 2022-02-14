import pandas as pd

data = pd.read_csv("winequality-red.csv")


def x_alcohol():
    x = data['alcohol']
    return x


def x_ph():
    x = data['pH']
    return x


def y_quality():
    y = data['quality']
    return y
