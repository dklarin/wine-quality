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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import json

from sklearn import metrics

from data_read import data_read

wine_quality_page = Blueprint('wine_quality_page', __name__,
                              template_folder='templates')

sp = 'wine_quality_page.'

wine_quality_links = ['wine_quality', 'decision_tree', 'random_forest',
                      'ada_boost', 'gradient_boost', 'xg_boost', 'good_wines', 'bad_wines']


@wine_quality_page.route('/wine_quality')
def wine_quality():

    name = 'wine_quality'

    df = data_read()

    corr = df.corr()
    plt.subplots(figsize=(15, 14))
    ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                     annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
    fig = ax.get_figure()
    fig.savefig('static/images/wq_matrix.png')

    image = os.path.join('static/images/wq_matrix.png')

    return render_template(
        'wine_quality.html',
        link1=sp+wine_quality_links[0],
        link2=sp+wine_quality_links[1],
        link3=sp+wine_quality_links[2],
        link4=sp+wine_quality_links[3],
        link5=sp+wine_quality_links[4],
        link6=sp+wine_quality_links[5],
        link7=sp+wine_quality_links[6],
        link8=sp+wine_quality_links[7],
        name='wine_quality_page.'+name,
        image=image
    )


@wine_quality_page.route('/naive_bayes')
def naive_bayes():

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

    model1 = GaussianNB()
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    #print(type(classification_report(y_test, y_pred1)))

    report = classification_report(y_test, y_pred1, output_dict=True)
    df1 = pd.DataFrame(report).transpose()
    df1 = df1.round(2)
    stats = pd.Series(['Good Wines', 'Bad Wines', 'Accuracy',
                       'Macro AVG', 'Weighted AVG'], index=[0, 1, 2, 3, 4])
    df1['class'] = stats.values
    json_list = json.loads(json.dumps(
        list(df1.T.to_dict().values())))

    mae = metrics.mean_absolute_error(y_test, y_pred1)
    mse = metrics.mean_squared_error(y_test, y_pred1)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred1))
    r2_square = metrics.r2_score(y_test, y_pred1)

    print(accuracy_score(y_test, y_pred1))

    return render_template(
        'wine_quality.html',
        tables=json_list,
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        value1=mae.round(4),
        value2=mse.round(4),
        value3=rmse.round(4),
        value4=r2_square.round(4),
        link1=sp+wine_quality_links[0],
        link2=sp+wine_quality_links[1],
        link3=sp+wine_quality_links[2],
        link4=sp+wine_quality_links[3],
        link5=sp+wine_quality_links[4],
        link6=sp+wine_quality_links[5],
        link7=sp+wine_quality_links[6],
        link8=sp+wine_quality_links[7],
        name='wine_quality_page.'+name,
        title='Decision Tree Classifier'
    )


@wine_quality_page.route('/decision_tree')
def decision_tree():

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
    #print(type(classification_report(y_test, y_pred1)))

    report = classification_report(y_test, y_pred1, output_dict=True)
    df1 = pd.DataFrame(report).transpose()
    df1 = df1.round(2)
    stats = pd.Series(['Good Wines', 'Bad Wines', 'Accuracy',
                       'Macro AVG', 'Weighted AVG'], index=[0, 1, 2, 3, 4])
    df1['class'] = stats.values
    json_list = json.loads(json.dumps(
        list(df1.T.to_dict().values())))

    mae = metrics.mean_absolute_error(y_test, y_pred1)
    mse = metrics.mean_squared_error(y_test, y_pred1)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred1))
    r2_square = metrics.r2_score(y_test, y_pred1)

    print(accuracy_score(y_test, y_pred1))

    #print('Točnost: {:.2f}%'.format(metrics.accuracy_score(y, y_pred)*100))
    #print('Preciznost: {:.2f}%'.format(metrics.precision_score(y, y_pred, average='macro')*100))
    #print('Odziv: {:.2f}%'.format(metrics.recall_score(y, y_pred, average='macro')*100))
    #print('F1: {:.2f}%'.format(metrics.f1_score(y, y_pred, average='macro')*100))

    return render_template(
        'wine_quality.html',
        tables=json_list,
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        value1=mae.round(4),
        value2=mse.round(4),
        value3=rmse.round(4),
        value4=r2_square.round(4),
        link1=sp+wine_quality_links[0],
        link2=sp+wine_quality_links[1],
        link3=sp+wine_quality_links[2],
        link4=sp+wine_quality_links[3],
        link5=sp+wine_quality_links[4],
        link6=sp+wine_quality_links[5],
        link7=sp+wine_quality_links[6],
        link8=sp+wine_quality_links[7],
        name='wine_quality_page.'+name,
        title='Decision Tree Classifier'
    )


@wine_quality_page.route('/random_forest')
def random_forest():

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
    #print(classification_report(y_test, y_pred2))

    report2 = classification_report(y_test, y_pred2, output_dict=True)

    df2 = pd.DataFrame(report2).transpose()
    df2 = df2.round(2)
    stats = pd.Series(['Good Wines', 'Bad Wines', 'Accuracy',
                       'Macro AVG', 'Weighted AVG'], index=[0, 1, 2, 3, 4])
    df2['class'] = stats.values
    json_list = json.loads(json.dumps(
        list(df2.T.to_dict().values())))

    mae = metrics.mean_absolute_error(y_test, y_pred2)
    mse = metrics.mean_squared_error(y_test, y_pred2)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred2))
    r2_square = metrics.r2_score(y_test, y_pred2)

    feat_importances = pd.Series(model2.feature_importances_, index=X.columns)
    ax = feat_importances.nlargest(25).plot(kind='barh', figsize=(10, 10))
    fig = ax.get_figure()
    fig.savefig('static/images/randomforest.png')

    return render_template(
        'wine_quality.html',
        tables=json_list,
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        value1=mae.round(4),
        value2=mse.round(4),
        value3=rmse.round(4),
        value4=r2_square.round(4),
        link1=sp+wine_quality_links[0],
        link2=sp+wine_quality_links[1],
        link3=sp+wine_quality_links[2],
        link4=sp+wine_quality_links[3],
        link5=sp+wine_quality_links[4],
        link6=sp+wine_quality_links[5],
        link7=sp+wine_quality_links[6],
        link8=sp+wine_quality_links[7],
        name='wine_quality_page.'+name,
        title='Random Forest Classifier'
    )


@wine_quality_page.route('/ada_boost')
def ada_boost():

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
    #print(classification_report(y_test, y_pred3))

    report3 = classification_report(y_test, y_pred3, output_dict=True)

    df3 = pd.DataFrame(report3).transpose()
    df3 = df3.round(2)
    stats = pd.Series(['Good Wines', 'Bad Wines', 'Accuracy',
                       'Macro AVG', 'Weighted AVG'], index=[0, 1, 2, 3, 4])
    df3['class'] = stats.values
    json_list = json.loads(json.dumps(
        list(df3.T.to_dict().values())))

    mae = metrics.mean_absolute_error(y_test, y_pred3)
    mse = metrics.mean_squared_error(y_test, y_pred3)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred3))
    r2_square = metrics.r2_score(y_test, y_pred3)

    return render_template(
        'wine_quality.html',
        tables=json_list,
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        value1=mae.round(4),
        value2=mse.round(4),
        value3=rmse.round(4),
        value4=r2_square.round(4),
        link1=sp+wine_quality_links[0],
        link2=sp+wine_quality_links[1],
        link3=sp+wine_quality_links[2],
        link4=sp+wine_quality_links[3],
        link5=sp+wine_quality_links[4],
        link6=sp+wine_quality_links[5],
        link7=sp+wine_quality_links[6],
        link8=sp+wine_quality_links[7],
        name='wine_quality_page.'+name,
        title='Ada Boost Classifier'
    )


@wine_quality_page.route('/gradient_boost')
def gradient_boost():

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
    #print(classification_report(y_test, y_pred4))

    report4 = classification_report(y_test, y_pred4, output_dict=True)

    df4 = pd.DataFrame(report4).transpose()
    df4 = df4.round(2)
    stats = pd.Series(['Good Wines', 'Bad Wines', 'Accuracy',
                       'Macro AVG', 'Weighted AVG'], index=[0, 1, 2, 3, 4])
    df4['class'] = stats.values
    json_list = json.loads(json.dumps(
        list(df4.T.to_dict().values())))

    mae = metrics.mean_absolute_error(y_test, y_pred4)
    mse = metrics.mean_squared_error(y_test, y_pred4)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred4))
    r2_square = metrics.r2_score(y_test, y_pred4)

    return render_template(
        'wine_quality.html',
        tables=json_list,
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        value1=mae.round(4),
        value2=mse.round(4),
        value3=rmse.round(4),
        value4=r2_square.round(4),
        link1=sp+wine_quality_links[0],
        link2=sp+wine_quality_links[1],
        link3=sp+wine_quality_links[2],
        link4=sp+wine_quality_links[3],
        link5=sp+wine_quality_links[4],
        link6=sp+wine_quality_links[5],
        link7=sp+wine_quality_links[6],
        link8=sp+wine_quality_links[7],
        title='Gradient Boosting Classifier'
    )


@wine_quality_page.route('/xg_boost')
def xg_boost():

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

    report5 = classification_report(y_test, y_pred5, output_dict=True)

    df5 = pd.DataFrame(report5).transpose()
    df5 = df5.round(2)
    stats = pd.Series(['Good Wines', 'Bad Wines', 'Accuracy',
                       'Macro AVG', 'Weighted AVG'], index=[0, 1, 2, 3, 4])
    df5['class'] = stats.values
    json_list = json.loads(json.dumps(
        list(df5.T.to_dict().values())))

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
        'wine_quality.html',
        tables=json_list,
        keyword1='MAE',
        keyword2='MSE',
        keyword3='RMSE',
        keyword4='R2 SQUARE',
        value1=mae.round(4),
        value2=mse.round(4),
        value3=rmse.round(4),
        value4=r2_square.round(4),
        link1=sp+wine_quality_links[0],
        link2=sp+wine_quality_links[1],
        link3=sp+wine_quality_links[2],
        link4=sp+wine_quality_links[3],
        link5=sp+wine_quality_links[4],
        link6=sp+wine_quality_links[5],
        link7=sp+wine_quality_links[6],
        link8=sp+wine_quality_links[7],
        name='wine_quality_page.'+name,
        title='XGBClassifier'
    )


@wine_quality_page.route('/good_wines')
def good_wines():
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
        title='Good Wines',
        link1=sp+wine_quality_links[0],
        link2=sp+wine_quality_links[1],
        link3=sp+wine_quality_links[2],
        link4=sp+wine_quality_links[3],
        link5=sp+wine_quality_links[4],
        link6=sp+wine_quality_links[5],
        link7=sp+wine_quality_links[6],
        link8=sp+wine_quality_links[7],
        switch=1,
        wine='wine'
    )


@wine_quality_page.route('/bad_wines')
def bad_wines():
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
        title='Bad Wines',
        link1=sp+wine_quality_links[0],
        link2=sp+wine_quality_links[1],
        link3=sp+wine_quality_links[2],
        link4=sp+wine_quality_links[3],
        link5=sp+wine_quality_links[4],
        link6=sp+wine_quality_links[5],
        link7=sp+wine_quality_links[6],
        link8=sp+wine_quality_links[7],
        switch=1,
        wine='wine'
    )


@wine_quality_page.route('/quality8')
def quality8():
    df = data_read()

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
