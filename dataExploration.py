import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats

import warnings

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    df_train = pd.read_csv('./data/train.csv')

    # Correlation mastrix (heatmap)
    corrmat = df_train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True, annot_kws={'size': 10})
    plt.savefig('./img/heatmap.png')

    # saleprice correlation matrix
    k = 10  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.savefig('./img/heatmap_saleprice_correlated.png')
    """
    Analysis:
        1. "OverallQual" "GrLivArea" "TotalBsmtSF" are strongly correlated with "SalePrice"
        2. "GarageCars" and "GarageArea" are like twin brothers, the number of cars that fit into the garage is a consequence of the garage.
         Therefore, we just need one of these variables in our analysis(we keep 'GarageCars' since its correlation with 'SalePrice' is higher)
        3. "TotalBsmtSF" and "1stFloor" also seems to be twin brothers, we keep "TotalBsmtSF" here
        4. "TotRmsAbvGrd" and "GrLivArea" twin brothers again
        5. "FullBath" seems not so related by subjective
        6. "YearBuilt" seems slightly correlated with 'SalePrice', maybe a time-series analysis should be done on the factor
    """

    # Scatter plot between 'SalePrice' and correlated variables
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(df_train[cols], size=2.5)
    plt.savefig('./img/scatter_saleprice_correlated.png')

    # missing data
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    """
    Analysis:
        1. Data with more than 15% missing rate will be deleted in the future analysis. Since none of these variables seem to be very important,
        most of them are not aspect in which we think about when buying a house. Variables like "PoolQC" "MiscFeature" and "FireplaceQu" are strong
        candidates for outliers.(delete 'PoolQC' 'MiscFeature' 'Alley' 'Fence' 'FireplaceQu' 'LotFrontage' )
        2. 'GarageX' variables have the same number of missing data. Since the most important information regarding garages is expressed by 'GarageCars',
        varaibales 'GarageX' will be deleted for the furture analysis
        3. for the same reason, 'BasmtX' variables are deleted(variable 'TotalBsmtSF' keeps the most useful information for the basement)
        4. 'MasVnrArea' and 'MasVnrType' have a strong correlation with 'YearBuilt' and 'OverallQual', so these two variables are not essential
        5. there is only one missing observation in 'Electrical', so delete this observation and keep the variable
    """

    # dealing with missing data
    df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
    df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
    df_train.isnull().sum().max()  # just checking that there's no missing data missing...

    # Outliars
    # outliars can markedly affect our models and can be a valuable source of information
    # establish a threshold that defines an observation as an outlier
    # standardizing data
    saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis]);
    low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
    high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
    print('outer range (low) of the distribution:')
    print(low_range)
    print('\nouter range (high) of the distribution:')
    print(high_range)
    """
    Analysis:
        1. Low range values are similar and not too far from 0.
        2. High range values are far from 0 and the 7.something values are really out of range.
    """

    # bivariate analysis saleprice/grlivarea
    var = 'GrLivArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
    plt.savefig('./img/scatter_bivariate_saleprice_grlivarea.png')
    """
    Analysis:
        1. The two values with bigger 'GrLivArea' seems strange and they are not following the crowd.
        These two points are not representative of the typical case. we define them as outilers and delte them
        2. The two observation in the top of the plot look like two special cases, however they seem to be following the trend, keep them
    """
    # deleting points
    df_train.sort_values(by='GrLivArea', ascending=False)[:2]
    df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
    df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

    # bivariate analysis saleprice/totalbsmtsf
    var = 'TotalBsmtSF'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
    plt.savefig('./img/scatter_bivariate_saleprice_totalbsmtsf.png')


    # histogram and normal probability plot
    sns.distplot(df_train['SalePrice'], fit=norm)
    fig = plt.figure()
    plt.savefig('./img/histogram_saleprice.png')
    res = stats.probplot(df_train['SalePrice'], plot=plt)
    plt.savefig('./img/probability_plot_saleprice.png')
    # applying log transformation
    # in case of positive skewness, log transformations usually works well
    df_train['SalePrice'] = np.log(df_train['SalePrice'])

    # histogram and normal probability plot
    sns.distplot(df_train['GrLivArea'], fit=norm);
    fig = plt.figure()
    plt.savefig('./img/histogram_grlivarea.png')
    res = stats.probplot(df_train['GrLivArea'], plot=plt)
    plt.savefig('./img/probability_plot_grlivarea.png')
    # data transformation
    df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

    # histogram and normal probability plot
    sns.distplot(df_train['TotalBsmtSF'], fit=norm);
    fig = plt.figure()
    plt.savefig('./img/histogram_totalbsmtsf.png')
    res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
    plt.savefig('./img/probability_plot_totalbsmtsf.png')

    """
    Skewness exist, but numbers of observations with value zero(houses without basement), value zero cannot apply log transformations
    Create a variable that can get the effect of having or not having basement
    """
    # create column for new variable (one is enough because it's a binary categorical feature)
    # if area>0 it gets 1, for area==0 it gets 0
    df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
    df_train['HasBsmt'] = 0
    df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
    # transform data
    df_train.loc[df_train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

    # convert categorical variable into dummy
    df_train = pd.get_dummies(df_train)
