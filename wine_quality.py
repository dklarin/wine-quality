import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import plotly.express as px
from flask import Blueprint, render_template
import matplotlib.pyplot as plt
import os.path
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import json

from sklearn import metrics

from data_read import data_read, y_quality

wine_quality_page = Blueprint('wine_quality_page', __name__,
                              template_folder='templates')

sp = 'wine_quality_page.'


@wine_quality_page.route('/wine_quality')
def wine_quality():

    name = 'wine_quality'

    df = data_read()

    corr = df.corr()
    plt.subplots(figsize=(15, 10))
    ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                     annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

    #ax = sns.heatmap(data_copy.corr())
    fig = ax.get_figure()
    fig.savefig('static/images/mat_konfuzije.png')

    image = os.path.join('static/images/mat_konfuzije.png')

    return render_template(
        'info.html',
        link1=sp+'wine_quality',
        link2=sp+'model1',
        link3=sp+'model2',
        link4=sp+'model3',
        link5=sp+'model4',
        link6=sp+'model5',
        name='wine_quality_page.'+name,
        image=image
    )


@wine_quality_page.route('/model1')
def model1():

    name = 'wine_quality'

    df = data_read()

    # Create Classification version of target variable
    df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
    # Separate feature variables and target variable
    X = df.drop(['quality', 'goodquality'], axis=1)
    y = df['goodquality']

    print(df['goodquality'].value_counts())

    # Splitting the data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=0)

    model1 = DecisionTreeClassifier(random_state=1)
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    print(type(classification_report(y_test, y_pred1)))

    mae = metrics.mean_absolute_error(y_test, y_pred1)
    mse = metrics.mean_squared_error(y_test, y_pred1)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred1))
    r2_square = metrics.r2_score(y_test, y_pred1)

    return render_template(
        'info.html',
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        value1=mae.round(4),
        value2=mse.round(4),
        value3=rmse.round(4),
        value4=r2_square.round(4),
        link1=sp+'wine_quality',
        link2=sp+'model1',
        link3=sp+'model2',
        link4=sp+'model3',
        link5=sp+'model4',
        link6=sp+'model5',
        name='wine_quality_page.'+name,
        # image=image
    )


@wine_quality_page.route('/model2')
def model2():

    name = 'wine_quality'

    df = data_read()

    # Create Classification version of target variable
    df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
    # Separate feature variables and target variable
    X = df.drop(['quality', 'goodquality'], axis=1)
    y = df['goodquality']

    print(df['goodquality'].value_counts())

    # Splitting the data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=0)

    model2 = RandomForestClassifier(random_state=1)
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    print(classification_report(y_test, y_pred2))

    mae = metrics.mean_absolute_error(y_test, y_pred2)
    mse = metrics.mean_squared_error(y_test, y_pred2)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred2))
    r2_square = metrics.r2_score(y_test, y_pred2)

    feat_importances = pd.Series(model2.feature_importances_, index=X.columns)
    ax = feat_importances.nlargest(25).plot(kind='barh', figsize=(10, 10))
    fig = ax.get_figure()
    fig.savefig('static/images/randomforest.png')

    return render_template(
        'info.html',
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        value1=mae.round(4),
        value2=mse.round(4),
        value3=rmse.round(4),
        value4=r2_square.round(4),
        link1=sp+'wine_quality',
        link2=sp+'model1',
        link3=sp+'model2',
        link4=sp+'model3',
        link5=sp+'model4',
        link6=sp+'model5',
        name='wine_quality_page.'+name,
    )


@wine_quality_page.route('/model3')
def model3():

    name = 'wine_quality'

    df = data_read()

    # Create Classification version of target variable
    df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
    # Separate feature variables and target variable
    X = df.drop(['quality', 'goodquality'], axis=1)
    y = df['goodquality']

    print(df['goodquality'].value_counts())

    # Splitting the data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=0)

    model3 = AdaBoostClassifier(random_state=1)
    model3.fit(X_train, y_train)
    y_pred3 = model3.predict(X_test)
    print(classification_report(y_test, y_pred3))

    mae = metrics.mean_absolute_error(y_test, y_pred3)
    mse = metrics.mean_squared_error(y_test, y_pred3)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred3))
    r2_square = metrics.r2_score(y_test, y_pred3)

    return render_template(
        'info.html',
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        value1=mae.round(4),
        value2=mse.round(4),
        value3=rmse.round(4),
        value4=r2_square.round(4),
        link1=sp+'wine_quality',
        link2=sp+'model1',
        link3=sp+'model2',
        link4=sp+'model3',
        link5=sp+'model4',
        link6=sp+'model5',
        name='wine_quality_page.'+name,
    )


@wine_quality_page.route('/model4')
def model4():

    name = 'wine_quality'

    df = data_read()

    # Create Classification version of target variable
    df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
    # Separate feature variables and target variable
    X = df.drop(['quality', 'goodquality'], axis=1)
    y = df['goodquality']

    print(df['goodquality'].value_counts())

    # Splitting the data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=0)

    model4 = GradientBoostingClassifier(random_state=1)
    model4.fit(X_train, y_train)
    y_pred4 = model4.predict(X_test)
    print(classification_report(y_test, y_pred4))

    mae = metrics.mean_absolute_error(y_test, y_pred4)
    mse = metrics.mean_squared_error(y_test, y_pred4)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred4))
    r2_square = metrics.r2_score(y_test, y_pred4)

    return render_template(
        'info.html',
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        value1=mae.round(4),
        value2=mse.round(4),
        value3=rmse.round(4),
        value4=r2_square.round(4),
        link1=sp+'wine_quality',
        link2=sp+'model1',
        link3=sp+'model2',
        link4=sp+'model3',
        link5=sp+'model4',
        link6=sp+'model5',
        name='wine_quality_page.'+name,
    )


@wine_quality_page.route('/model5')
def model5():

    name = 'wine_quality'

    df = data_read()

    # Create Classification version of target variable
    df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
    # Separate feature variables and target variable
    X = df.drop(['quality', 'goodquality'], axis=1)
    y = df['goodquality']

    print(df['goodquality'].value_counts())

    # Splitting the data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=0)

    model5 = xgb.XGBClassifier(random_state=1)
    model5.fit(X_train, y_train)
    y_pred5 = model5.predict(X_test)
    print(classification_report(y_test, y_pred5))

    mae = metrics.mean_absolute_error(y_test, y_pred5)
    mse = metrics.mean_squared_error(y_test, y_pred5)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred5))
    r2_square = metrics.r2_score(y_test, y_pred5)

    #feat_importances = pd.Series(model2.feature_importances_, index=X.columns)
    #ax = feat_importances.nlargest(25).plot(kind='barh', figsize=(10, 10))
    #fig = ax.get_figure()
    # fig.savefig('static/images/randomforest.png')

    feat_importances = pd.Series(model5.feature_importances_, index=X.columns)
    ax = feat_importances.nlargest(25).plot(kind='barh', figsize=(10, 10))
    fig = ax.get_figure()
    fig.savefig('static/images/gbc.png')

    return render_template(
        'info.html',
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        value1=mae.round(4),
        value2=mse.round(4),
        value3=rmse.round(4),
        value4=r2_square.round(4),
        link1=sp+'wine_quality',
        link2=sp+'model1',
        link3=sp+'model2',
        link4=sp+'model3',
        link5=sp+'model4',
        link6=sp+'model5',
        name='wine_quality_page.'+name,
    )


@wine_quality_page.route('/good')
def good():
    df = data_read()

    # Create Classification version of target variable
    df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
    # Separate feature variables and target variable
    X = df.drop(['quality', 'goodquality'], axis=1)
    y = df['goodquality']

    # Filtering df for only good quality
    df_temp = df[df['goodquality'] == 1]
    df_temp = df_temp.describe()
    describe = df_temp.round(2)

    stats = pd.Series(['count', 'mean', 'std', 'min', '25%', '50%',
                       '75%', 'max'], index=[0, 1, 2, 3, 4, 5, 6, 7])
    describe['statistics'] = stats.values

    json_list = json.loads(json.dumps(
        list(describe.T.to_dict().values())))

    return render_template(
        'table.html',
        tables=json_list,
        title='describe',
        link1='dataset',
        link2='describe',
        link3='shape',
        switch=1
    )


@wine_quality_page.route('/bad')
def bad():
    df = data_read()

    # Create Classification version of target variable
    df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
    # Separate feature variables and target variable
    X = df.drop(['quality', 'goodquality'], axis=1)
    y = df['goodquality']

    # Filtering df for only good quality
    df_temp = df[df['goodquality'] == 0]
    df_temp = df_temp.describe()
    describe = df_temp.round(2)

    stats = pd.Series(['count', 'mean', 'std', 'min', '25%', '50%',
                       '75%', 'max'], index=[0, 1, 2, 3, 4, 5, 6, 7])
    describe['statistics'] = stats.values

    json_list = json.loads(json.dumps(
        list(describe.T.to_dict().values())))

    return render_template(
        'table.html',
        tables=json_list,
        title='describe',
        link1='dataset',
        link2='describe',
        link3='shape',
        switch=1
    )


@wine_quality_page.route('/quality8')
def quality8():
    df = data_read()

    # Create Classification version of target variable
    #df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
    # Separate feature variables and target variable
    #X = df.drop(['quality', 'goodquality'], axis=1)
    #y = df['goodquality']

    # Filtering df for only good quality
    df_temp = df[df['quality'] == 8]
    df_temp = df_temp.describe()
    describe = df_temp.round(2)

    stats = pd.Series(['count', 'mean', 'std', 'min', '25%', '50%',
                       '75%', 'max'], index=[0, 1, 2, 3, 4, 5, 6, 7])
    describe['statistics'] = stats.values

    json_list = json.loads(json.dumps(
        list(describe.T.to_dict().values())))

    return render_template(
        'table.html',
        tables=json_list,
        title='describe',
        link1='dataset',
        link2='describe',
        link3='shape',
        switch=1
    )


@wine_quality_page.route('/quality7')
def quality7():
    df = data_read()

    # Create Classification version of target variable
    #df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
    # Separate feature variables and target variable
    #X = df.drop(['quality', 'goodquality'], axis=1)
    #y = df['goodquality']

    # Filtering df for only good quality
    df_temp = df[df['quality'] == 7]
    df_temp = df_temp.describe()
    describe = df_temp.round(2)

    stats = pd.Series(['count', 'mean', 'std', 'min', '25%', '50%',
                       '75%', 'max'], index=[0, 1, 2, 3, 4, 5, 6, 7])
    describe['statistics'] = stats.values

    json_list = json.loads(json.dumps(
        list(describe.T.to_dict().values())))

    return render_template(
        'table.html',
        tables=json_list,
        title='describe',
        link1='dataset',
        link2='describe',
        link3='shape',
        switch=1
    )


@wine_quality_page.route('/quality6')
def quality6():
    df = data_read()

    # Create Classification version of target variable
    #df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
    # Separate feature variables and target variable
    #X = df.drop(['quality', 'goodquality'], axis=1)
    #y = df['goodquality']

    # Filtering df for only good quality
    df_temp = df[df['quality'] == 6]
    df_temp = df_temp.describe()
    describe = df_temp.round(2)

    stats = pd.Series(['count', 'mean', 'std', 'min', '25%', '50%',
                       '75%', 'max'], index=[0, 1, 2, 3, 4, 5, 6, 7])
    describe['statistics'] = stats.values

    json_list = json.loads(json.dumps(
        list(describe.T.to_dict().values())))

    return render_template(
        'table.html',
        tables=json_list,
        title='describe',
        link1='dataset',
        link2='describe',
        link3='shape',
        switch=1
    )
