

"""
    - impute missing ata
    - transform some numerical variables to categorical
    - Label Encoding some categorical variable
    - get dummy variables for categorical variable
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

if __name__ == '__main__':
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    train_ID = train_data['Id']
    test_ID = test_data['Id']
    train_data.drop("Id", axis=1, inplace=True)
    test_data.drop("Id", axis=1, inplace=True)
    """
        train data shape: before->(1460, 81)  after->(1460, 80)
        test data shape: before->(1459,80) after->(1459, 79)
    """

    # Delete outliers
    index = train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 300000)].index
    train_data = train_data.drop(index)
    """
        train data shape: (1458, 80)
    """

    # clean skewness of the target variable -> 'SalePrice'
    train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
    train_rows = train_data.shape[0]
    test_rows = test_data.shape[0]
    saleprice = train_data.SalePrice.values
    all_data = pd.concat([train_data, test_data], sort=False)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    """
        all data shape: (2917, 79)
    """

    # impute missing data
    missing_data_count = all_data.isnull().sum().sort_values(ascending=False)
    missing_data_ratio = (all_data.isnull().sum() / len(all_data)).sort_values(ascending=False)
    missing_data = pd.concat([missing_data_count, missing_data_ratio], axis=1, keys=['count', 'ratio'])
    missing_data = missing_data.drop(missing_data[missing_data['count'] == 0].index)

    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=missing_data.index, y=missing_data['count'])
    plt.show()

    # all_data[missing_data.index] = all_data[missing_data.index].fillna(None)
    all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
    all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
    all_data['Alley'] = all_data['Alley'].fillna('None')
    all_data['Fence'] = all_data['Fence'].fillna('None')
    all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
    # object -> none
    for col in ('GarageCond', 'GarageQual', 'GarageFinish', 'GarageType'):
        all_data[col] = all_data[col].fillna('None')
    all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(all_data['GarageYrBlt'].mean())
    for col in ('BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1'):
        all_data[col] = all_data[col].fillna('None')
    all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
    all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode())
    # 100% 'AllPub' in Utilities
    all_data.drop(['Utilities'], axis=1, inplace=True)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)

    # Transfoming some numeric variables that are really categorical

    # Label Encoding some categorical variables that may contain information in their ordering set
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir')
    for col in cols:
        lbl = LabelEncoder()
        all_data[col] = lbl.fit_transform(list(all_data[col].values))

    # Adding one more important feature
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    # Skewed features: check the skew of all numeric features
    numeric_cols = all_data.select_dtypes(exclude='object')
    skew_feats = numeric_cols.skew(axis=0).sort_values(ascending=False)

    # Box Cox Transformation of (highly) skewed features
    skewness = skew_feats[abs(skew_feats)>0.75]
    skewed_features = skewness.index
    for feat in skewed_features:
        all_data[feat] = boxcox1p(all_data[feat], 0.15)

    # Getting dummy categorical features
    all_data = pd.get_dummies(all_data)

    # Getting the new train and test sets
    train = all_data[:train_rows]
    test = all_data[train_rows:]

    # Modeling
    X_train, X_valid, y_train, y_valid = train_test_split(train, saleprice, train_size=0.8, test_size=0.2, random_state=2)
    model_xgb = xgb.XGBRegressor(colsample_bylevel=0.25,colsample_bytree=0.6,max_depth=4,min_child_weight=4.8,n_estimators=3000,reg_lambda=0.1,subsample=0.8, learning_rate=0.01, n_jobs=4,seed=42)
    model_xgb.fit(X_train, y_train)
    predictors = model_xgb.predict(X_valid)
    print(mean_absolute_error(y_valid, predictors))

    submission = np.expm1(model_xgb.predict(test))
    submission_df = pd.DataFrame(data={'Id': test_ID, 'SalePrice': submission})
    submission_df.to_csv('./output/submission13.csv', index=False)


