"""
Best ranking 891/8935

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from xgboost import XGBRegressor

# Evaluate mean_absolute_error value produced by LogisticalRegressor model
def logistical_MAE_score(X_train, X_valid, y_train, y_valid):
    global model
    model = LogisticRegression(random_state=1, solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, y_train)
    predictors = model.predict(X_valid)
    return mean_absolute_error(y_valid, predictors)

# Evaluate mean_absolute_error value produced by elasticnet model
def elasticnet_MAE_score(X_train, X_valid, y_train, y_valid):
    global model
    model = ElasticNet(random_state=1, fit_intercept=True)
    model.fit(X_train, y_train)
    predictors = model.predict(X_valid)
    return mean_absolute_error(y_valid, predictors)

# Evaluate mean_absolute_error value produced by RandomForestRegressor model
def randomforest_MAE_score(X_train, X_valid, y_train, y_valid):
    global model
    model = RandomForestRegressor(n_estimators=1000, random_state=1)
    model.fit(X_train, y_train)
    predictors = model.predict(X_valid)
    return mean_absolute_error(y_valid, predictors)

# Evaluate mean_absolute_error value produced by XGBRegressor model
def xgb_MAE_score(X_train, X_valid, y_train, y_valid):
    global model
    model = XGBRegressor(n_estimators=800, random_state=2)
    # model = XGBRegressor(learning_rate=0.12, min_child_weight=2, max_depth=3, reg_lambda=0.98, subsample=1, n_estimators=800, random_state=1)
    model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)
    predictors = model.predict(X_valid)
    return mean_absolute_error(y_valid, predictors)

def gradientBoosting_MAE_score(X_train, X_valid, y_train, y_valid):
    global model
    model = GradientBoostingRegressor(loss='huber', random_state=1, n_estimators=600)
    model.fit(X_train, y_train)
    predictors = model.predict(X_valid)
    return mean_absolute_error(y_valid, predictors)

def adaBoosting_MAE_score(X_train, X_valid, y_train, y_valid):
    global model
    model = AdaBoostRegressor(n_estimators=800, random_state=1)
    model.fit(X_train, y_train)
    predictors = model.predict(X_valid)
    return mean_absolute_error(y_valid, predictors)

# Create table for missing data analysis
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False) #同一个dataset中，为空总数越多的列所占的比例就越高
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    return plt

# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='best')
    plt.xlabel('Parameter')
    plt.ylabel('Score')
    plt.ylim(ylim)

# Detect whether missing data existing in the object dataset
def missing_values_exist(dataset):
    col_with_missing = []
    for col in dataset.columns:
        if dataset[col].isnull().any():
            col_with_missing.append(col)
    return col_with_missing != []

# Impute the missing data with SimpleImpute function
def impute_missing_data(data):
    imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    imputed_data = pd.DataFrame(imputer.fit_transform(data))
    imputed_data.columns = data.columns
    return imputed_data

# Preprocess the data
def normalize_data(data):
    # Delete missing data with high missing rate(over 35%)
    missing_data = draw_missing_data_table(data)
    high_missing_rate_cols = []
    for index, row in missing_data.iterrows():
        if row['Percent'] > 0.35:
            high_missing_rate_cols.append(index)
    high_missing_rate_cols.append('Id')
    data.drop(high_missing_rate_cols, axis=1, inplace=True)

    """
    # Transform categorical variable into Dummy variables
    # creates new columns indicating the presence (or absence) of each possible value in the origin
    # Firstly, transform all the variables with object types into category type
    for col in data.columns:
        if data[col].dtypes == 'object':
            data[col] = pd.Categorical(data[col])
    data = pd.get_dummies(data, drop_first=True)
    """

    # labelEncoder = LabelEncoder()
    # data = labelEncoder.fit_transform(data)

    # Delete all the 'object' variables
    data = data.select_dtypes(exclude='object')
    # Impute missing data
    imputed_data = impute_missing_data(data).copy()
    #
    # p = re.compile('^[A-Za-z]+$')
    #
    # object_cols = [cols for cols in imputed_data.columns if p.match(imputed_data[cols])]
    # for col in object_cols:
    #     imputed_data[col] = LabelEncoder().fit_transform(imputed_data[col])


    return imputed_data

if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)  #解决pycharm中pandas输出过长显示位省略号的问题
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    train_data = pd.read_csv('/Users/csun/Code/PycharmProjects/microCourse/data/train.csv', engine='python')
    test_data = pd.read_csv('/Users/csun/Code/PycharmProjects/microCourse/data/test.csv', engine='python')
    id = test_data.Id

    impute_train_data = normalize_data(train_data)
    X = impute_train_data[impute_train_data.loc[:, impute_train_data.columns != 'SalePrice'].columns]
    y = impute_train_data['SalePrice']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)


    print("MAE score by LogisticalRegressor model")
    print(logistical_MAE_score(X_train, X_valid, y_train, y_valid))
    logistical_model = model

    print("MAE score by RandomForestRegressor model ")
    print(randomforest_MAE_score(X_train, X_valid, y_train, y_valid))
    randomforest_model = model

    print("MAE score by XGBRegressor model ")
    print(xgb_MAE_score(X_train, X_valid, y_train, y_valid))
    xgb_model = model


    print("MAE score by ElasticNet mode")
    print(elasticnet_MAE_score(X_train, X_valid, y_train, y_valid))
    elasticnet_model = model

    print("MAE score by Gradient Boosting mode")
    print(gradientBoosting_MAE_score(X_train, X_valid, y_train, y_valid))
    gboosting_model = model

    print("MAE score by AdaBoosting mode")
    print(adaBoosting_MAE_score(X_train, X_valid, y_train, y_valid))
    adaboosting_model = model


    """
        Using the provided models to predict the test_data
    """
    test_data = normalize_data(test_data)
    # LR_price = logistical_model.predict(test_data)
    RandomForest_price = randomforest_model.predict(test_data)
    XGB_price = xgb_model.predict(test_data)
    elasticnet_price = elasticnet_model.predict(test_data)
    gboosting_price = gboosting_model.predict(test_data)

    submission = gboosting_price
    # submission = RandomForest_price * 0.1 + XGB_price * 0.4 + gboosting_price * 0.5
    submission = RandomForest_price * 0.25 + XGB_price * 0.35 + elasticnet_price * 0.1 + gboosting_price * 0.3
    submission_df = pd.DataFrame(data={'Id': id, 'SalePrice': submission})
    submission_df.to_csv('./output/submission11.csv', index=False)


    """
    # Plot learning curve

    # scores = cross_val_score(logistical_model, X_train, y_train, cv=10)
    # print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    # title1 = 'Learning Curve (Logistical Regressor)'
    # plt = plot_learning_curve(estimator=logistical_model, title=title1, X=X_train, y=y_train, ylim=(0, 1.1), cv=10, n_jobs=2)
    # plt.show()

    # scores = cross_val_score(randomforest_model, X_train, y_train, cv=10)
    # print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    # title2 = 'Learning Curve (RandomForest Regressor)'
    # plt = plot_learning_curve(estimator=randomforest_model, title=title2, X=X_train, y=y_train, ylim=(0, 1.1), cv=10, n_jobs=2)
    # plt.show()
    
    scores = cross_val_score(xgb_model, X_train, y_train, cv=10)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    title3 = 'Learning Curve (XGboost Regressor)'
    plt = plot_learning_curve(estimator=xgb_model, title=title3, X=X_train, y=y_train, ylim=(0, 1.1), cv=10, n_jobs=2)
    plt.show()
    
    """


    """
        Transform all the object variables into categorical variables and then impute the missing data
        Results:
        logistical_MAE_score: 39930.17123287671
        randomforest_MAE_score: 16471.331993150685
        xgb_MAE_score: 14917.441299229453
        
        Only impute the missing data without creating new columns
        Results:          
        logistical_MAE_score: 40904.08561643836
        randomforest_MAE_score: 16897.479965753428
        xgb_MAE_score: 14764.69789437072
    """

