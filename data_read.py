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


def copy_data():
    data_copy = data[["residual sugar", "sulphates", "alcohol", "quality"]]
    return data_copy


def copy_data2():
    data_copy = data[["residual sugar", "sulphates", "alcohol"]]
    return data_copy
