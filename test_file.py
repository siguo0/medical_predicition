import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import time
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import featuretools as ft
import scipy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from xgboost import XGBRegressor
import pickle
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge


def calc_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def data_process(df):
    df.loc[df['Molecular weight'] >= 0.6, 'Molecular weight'] = df.loc[
        df['Molecular weight'] < 0.6, 'Molecular weight'].median()
    df.loc[df['AlogP'] < 0.2, 'AlogP'] = df.loc[df['AlogP'] >= 0.2, 'AlogP'].median()
    df=df.drop(df.loc[df['Molecular weight']==0,'Molecular weight'].index)
    # df.drop_duplicates(subset=['Molecule_max_phase','Molecular weight','RO5_violations','AlogP','Label'],keep='first',inplace=True)


    #处理缺失值
    df.RO5_violations.fillna(df.RO5_violations.mode()[0], inplace=True)
    df.AlogP.fillna(df.AlogP.median(), inplace=True)
    # df=df.drop(['RO5_violations'],axis=1)
    # df['x1'] =df['Molecular weight'] * df.Molecule_max_phase
    # df['x2']=df.Molecule_max_phase/(df.AlogP+1)

    return df

if __name__=='__main__':

    df=pd.read_pickle('trains0.pkl')
    df=data_process(df)



    # df.plot(kind='scatter',x='Molecule_max_phase',y='Label')
    # plt.show()

    data=df.drop(['Label'],axis=1)
    target=df.Label

    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3, random_state=0)

    # train_x=pd.concat([train_x,train_y],axis=1)
    # train_x.drop_duplicates(subset=['Molecule_max_phase','Molecular weight','RO5_violations','AlogP','Features','Label'],keep=False,inplace=True)
    # train_x=train_x.drop(['Features'],axis=1)
    # train_y=train_x.Label
    # train_x=train_x.drop(['Label'],axis=1)
    # test_x=test_x.drop(['Features'],axis=1)
    # print(train_x)
    # print(train_y)
    # print(test_x)
    # print(test_y)

    # print(np.any(np.isinf(train_x)))
    # print(np.any(np.isinf(train_y)))

    # gbdt=GradientBoostingRegressor()
    # gbdt.fit(train_x,train_y)
    # print(calc_rmse(gbdt.predict(test_x),test_y))


    xgb = XGBRegressor(n_estimators=500,learning_rate=0.1,colsample_bytree=1,max_depth=10,subsample=0.8,min_child_weight=7)
    # xgb=XGBRegressor(num_leaves=64,max_depth=10,learning_rate=0.1,n_estimators=100,subsample=0.8,feature_fraction=0.8,reg_alpha=0.5,reg_lambda=0.5,metric=None,min_child_weight=7)


    # kf = KFold(n_splits=5,shuffle=True)
    # recall_score_kf_list = []
    # for k, (train, test) in enumerate(kf.split(data, target)):
    #     start=time.time()
    #     kf = xgb.fit(data.iloc[train], target.iloc[train].values.ravel(),verbose=True)
    #     pred_kf = kf.predict(data.iloc[test])
    #     print(pred_kf)
    #     recall_score_kf = calc_rmse(kf.predict(data.iloc[test]),target.iloc[test])
    #     print(recall_score_kf)
    #     recall_score_kf_list.append(recall_score_kf)
    #     print('iteration', k, 'recall score', recall_score_kf)
    #     print(time.time()-start)
    # print(np.mean(recall_score_kf_list))

    xgb.fit(train_x,train_y)
    print(calc_rmse(xgb.predict(test_x),test_y))






    # high_dimension=df.iloc[:,5:].T.values
    # model_pca=PCA(n_components=100)
    # X_pca = model_pca.fit(high_dimension).transform(high_dimension)
    # low_dimension=pd.DataFrame(model_pca.components_,columns=df.index)
    # # print("降维后各主成分方向：\n", model_pca.components_)
    # # print("降维后各主成分的方差值：", model_pca.explained_variance_)
    # # print("降维后各主成分的方差值与总方差之比：", model_pca.explained_variance_ratio_)
    # # print("奇异值分解后得到的特征值：", model_pca.singular_values_)
    # # print("降维后主成分数：", model_pca.n_components_)


    # df=df.drop(df.iloc[:,5:].columns,axis=1)
    # df=pd.concat([df,low_dimension.T],axis=1)