import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

def Features_process(df):#对训练集 测试集都可以进行
    # 处理Features字符串列表
    df.Features = df.Features.apply(lambda x: x[1:-1])
    spl = pd.DataFrame(df.Features.str.split(',', expand=True).values.astype(np.float64))
    # 在结合前将spl做一些统计
    df['feature_min'] = spl.min(axis=1)
    df['feature_max']=spl.max(axis=1)
    df['feature_mean']=spl.mean(axis=1)
    df['feature_sum'] = spl.sum(axis=1)
    df = df.drop(['ID'], axis=1)
    # 结合原df以及列表元素升维后的特征df
    df = pd.concat([df, spl], axis=1)

    # 处理缺失值
    df.RO5_violations.fillna(df.RO5_violations.median(), inplace=True)
    df.AlogP.fillna(df.AlogP.mean(), inplace=True)

    df['x1'] = df['Molecular weight'] * df.Molecule_max_phase
    df['x2'] = df.Molecule_max_phase / (df.AlogP + 1)

    df.fillna(df.median(),inplace=True)

    return df


def train_process(df):
    df.loc[df['Molecular weight'] >= 0.6, 'Molecular weight'] = df.loc[
        df['Molecular weight'] < 0.6, 'Molecular weight'].median()
    df.loc[df['AlogP'] < 0.2, 'AlogP'] = df.loc[df['AlogP'] >= 0.2, 'AlogP'].median()

    return df


def test_process(df):
    df=df.drop(['Features'],axis=1)
    return df


def standformat(y_pred):#将预测值标准化输入文件
    df=pd.read_csv('submit.csv')
    df.Label=y_pred
    df.to_csv('submit.csv',index=False)


if __name__=='__main__':
    train = pd.read_csv('train_0312.csv')
    train = Features_process(train)
    train=train_process(train)

    test=pd.read_csv('test.csv')
    test=Features_process(test)
    test=test_process(test)

    data=train.drop(['Label'],axis=1)
    target=train.Label

    xgb = XGBRegressor(n_estimators=500,
                              learning_rate=0.1,
                              max_depth=10,
                              min_child_weight=13,
                              subsample=0.6,
                              colsample_bytree=0.6,
                              reg_alpha=3.25,
                              reg_lambda=3.5,
                              objective='reg:squarederror',
                              eval_metric='rmse')

    kf = KFold(n_splits=20, shuffle=True)
    pre=[0]
    for k, (train, test0) in enumerate(kf.split(data, target)):

        kf = xgb.fit(data.iloc[train], target.iloc[train].values.ravel(), verbose=True)
        pred_kf = kf.predict(test)
        pre+=pred_kf/20
        print(pre)

    standformat(pre)