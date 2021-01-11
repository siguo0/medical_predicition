import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

def calc_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def data_process(df):
    default = list(set(df[df.isnull().values == True].index))
    df['default'] = 0
    df.loc[default, 'default'] = 1

    df = df.drop(df.loc[df['Molecular weight'] == 0, 'Molecular weight'].index)

    #处理缺失值
    df.RO5_violations.fillna(df.RO5_violations.mode()[0], inplace=True)
    df.AlogP.fillna(df.AlogP.median(), inplace=True)

    df.loc[df['Molecular weight'] >= 0.6, 'Molecular weight'] = df.loc[df['Molecular weight'] < 0.6, 'Molecular weight'].median()
    df.loc[df['AlogP'] < 0.2, 'AlogP'] = df.loc[df['AlogP'] >= 0.2, 'AlogP'].median()

    df['x1'] = df['Molecular weight'] * df.Molecule_max_phase
    df['x2'] = df.Molecule_max_phase / (df.AlogP + 1)

    return df

def standformat(y_pred):
    df=pd.read_csv('submit.csv')
    df.Label=y_pred
    df.to_csv('submit.csv',index=False)


if __name__=='__main__':
    df=pd.read_pickle('trains0.pkl')
    df=data_process(df)

    test=pd.read_pickle('test.pkl')
    test=data_process(test)

    data=df.drop(['Label'],axis=1)
    target=df.Label

    xgb=XGBRegressor(n_estimators=100,
                              learning_rate=0.1,
                              max_depth=10,
                              min_child_weight=13,
                              subsample=0.6,
                              colsample_bytree=0.6,
                              reg_alpha=3.25,
                              reg_lambda=3.5,
                              objective='reg:squarederror',
                              eval_metric='rmse')

    # xgb=XGBRegressor(num_leaves=64,max_depth=10,learning_rate=0.1,n_estimators=1000,subsample=0.8,feature_fraction=0.8,reg_alpha=0.5,reg_lambda=0.5,metric=None,min_child_weight=7)

    kf = KFold(n_splits=10, shuffle=True)
    pre = [0]
    for k, (train, test0) in enumerate(kf.split(data, target)):
        kf = xgb.fit(data.iloc[train], target.iloc[train].values.ravel(), verbose=True)
        pred_kf = kf.predict(test)
        pre += pred_kf / 10
        print(pre)
    standformat(pre)
