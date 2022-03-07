import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import metrics


# Method scales x variable
# 1 Regression Linear Cross Validation
def x_scale(x):
    x_scaled = StandardScaler().fit_transform(x.to_numpy().reshape(-1, 1))
    x_scaled = pd.DataFrame(x_scaled, columns=["alcohol"])
    return x_scaled


# Method to scale multiple linear regression
# 1 Regression Linear Cross Validation
def x_scale_3(x):
    x_scaled = StandardScaler().fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled, index=x.index, columns=x.columns)
    return x_scaled


# Method for metrics
# 1 Regression Linear Cross Validation
# 2 Linear Regression Metrics
# 2 K-fold Validation
# 2 Model Training
def linear_metrics(y_test, y_predict):
    mae = metrics.mean_absolute_error(y_test, y_predict)
    mse = metrics.mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_predict))
    r2_square = metrics.r2_score(y_test, y_predict)
    return mae, mse, rmse, r2_square
