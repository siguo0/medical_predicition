import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn import preprocessing
from xgboost import XGBRegressor
import shap


def calc_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def train_process(df):

    d1=df.drop_duplicates(subset=['Molecule_max_phase','Molecular weight','RO5_violations','AlogP','Features'],keep='first')
    d2=df.drop_duplicates(subset=['Molecule_max_phase','Molecular weight','RO5_violations','AlogP','Features'],keep=False)
    df3=d2.append(d1).drop_duplicates(subset=['Molecule_max_phase','Molecular weight','RO5_violations','AlogP','Features'],keep=False)
    df=df.append(df3)

    df = df.drop(['Features'], axis=1)
    df = df.reset_index(drop=True)
    return df

def test_process(df):
    df = df.drop(['Features'], axis=1)
    return df

def Features_process(df):#对训练集 测试集都可以进行

    # 处理Features字符串列表
    df.Features = df.Features.apply(lambda x: x[1:-1])
    spl = pd.DataFrame(df.Features.str.split(',', expand=True).values.astype(np.float64))
    # 在结合前将spl做一些统计
    df['feature_min'] = spl.min(axis=1)
    df['feature_sum'] = spl.sum(axis=1)
    # df = df.drop(['ID','Features'], axis=1)
    df = df.drop(['ID'], axis=1)
    # 结合原df以及列表元素升维后的特征df
    df = pd.concat([df, spl], axis=1)

    # 处理缺失值
    df.RO5_violations.fillna(df.RO5_violations.median(), inplace=True)
    df.AlogP.fillna(df.AlogP.mean(), inplace=True)

    df['x1'] = df['Molecular weight'] * df.Molecule_max_phase
    df['x2'] = df.Molecule_max_phase / (df.AlogP + 1)

    df.fillna(df.median(),inplace=True)

    df.loc[df['Molecular weight'] >= 0.6, 'Molecular weight'] = df.loc[df['Molecular weight'] < 0.6, 'Molecular weight'].median()
    df.loc[df['AlogP'] < 0.2, 'AlogP'] = df.loc[df['AlogP'] >= 0.2, 'AlogP'].median()

    df['Molecular weight1']=preprocessing.scale(df['Molecular weight'])
    df['AlogP1']=preprocessing.scale(df.AlogP)

    return df

def stacking_first_lgb(train_x,train_y,test_x):
    gbm = lgb.LGBMRegressor(n_estimators=5000,
                            learning_rate=0.01,
                            max_depth=10,
                            min_child_weight=7,
                            subsample=0.5,
                            colsample_bytree=0.6,
                            reg_alpha=3.25,
                            reg_lambda=3.5,
                            boosting_type='gbdt')

    gbm.fit(train_x, train_y)

    #此时不重要特征的删除是对于未划分的训练集来划分 是否考虑在多折验证中添加？
    important = pd.DataFrame(gbm.feature_importances_, columns=['importance'])
    important = important.loc[important['importance'] == 0, 'importance'].index
    train_x = train_x.drop(important, axis=1)
    test_x=test_x.drop(important,axis=1)


    score=[]
    new_test=pd.DataFrame()
    new_test_=[0]

    real_target=pd.DataFrame()#新训练集的label
    pre_target=pd.DataFrame()#新训练集的特征
    kf = KFold(n_splits=10, shuffle=False,random_state=0)#如果要使划分后的测试集相同 因为要将多个模型预测结果作为特征 划分的测试集的真实值作为标记值，要么使用相同的划分 要么这个划分要在主函数中进行 然后各个子函数只用于获得这些预测值

    #将划分后的训练集 再次划分出训练集和测试集 使用训练集多折划分 对于
    for k, (train, test) in enumerate(kf.split(train_x, train_y)):
        gbm.fit(train_x.iloc[train], train_y.iloc[train].values.ravel())
        pre_target=pre_target.append(list(gbm.predict(train_x.iloc[test])),ignore_index=True)
        real_target=real_target.append(list(train_y[test]),ignore_index=True)
        new_test_+=gbm.predict(test_x)/10
        score.append(calc_rmse(gbm.predict(train_x.iloc[test]),train_y[test]))
    print('lgb_new_test:',new_test_)
    print('lgb:',np.mean(score))
    return list(pre_target.values),list(real_target.values),new_test_#每次预测得到的关于训练集的预测值叠加

def stacking_first_xgb(train_x,train_y,test_x):
    xgb = XGBRegressor(n_estimators=500,
                     learning_rate=0.1,
                     colsample_bytree=1,
                     max_depth=10,
                     subsample=0.5,
                     min_child_weight=7)

    xgb.fit(train_x, train_y)
    important = pd.DataFrame(xgb.feature_importances_, columns=['importance'])
    important = important.loc[important['importance'] == 0, 'importance'].index

    train_x.columns=range(train_x.shape[1])
    train_x=train_x.drop(important,axis=1)
    test_x.columns=range(test_x.shape[1])
    test_x=test_x.drop(important,axis=1)

    score=[]
    new_test = pd.DataFrame()
    new_test_ = [0]
    real_target=pd.DataFrame()
    pre_target=pd.DataFrame()
    kf = KFold(n_splits=10, shuffle=False,random_state=0)#如果要使划分后的测试集相同 因为要将多个模型预测结果作为特征 划分的测试集的真实值作为标记值，要么使用相同的划分 要么这个划分要在主函数中进行 然后各个子函数只用于获得这些预测值

    #将划分后的训练集 再次划分出训练集和测试集 使用训练集多折划分 对于
    for k, (train, test) in enumerate(kf.split(train_x, train_y)):
        xgb.fit(train_x.iloc[train], train_y.iloc[train].values.ravel())
        pre_target=pre_target.append(list(xgb.predict(train_x.iloc[test])),ignore_index=True)
        real_target=real_target.append(list(train_y[test]),ignore_index=True)
        new_test_+=xgb.predict(test_x)/10
        score.append(calc_rmse(xgb.predict(train_x.iloc[test]),train_y[test]))
    print(np.mean(score))

    return list(pre_target.values),list(real_target.values),new_test_

if __name__=='__main__':
    train = pd.read_csv('train_0312.csv')
    train = Features_process(train)

    data=train.drop(['Label'],axis=1)
    target=train.Label

    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3, random_state=0)
    #划分完训练测试集 合并x、y方便进行训练集的处理
    train_x = pd.concat([train_x, train_y], axis=1)
    train_x=train_process(train_x)
    #训练集处理结束 重新划分开x、y
    train_y=train_x.Label
    train_x=train_x.drop(['Label'],axis=1)
    #对测试集的处理
    test_x=test_process(test_x)

    new_df=pd.DataFrame()
    new_test=pd.DataFrame()
    new_df['lgb'],new_df['lgb_target'],new_test['lgb']=stacking_first_lgb(train_x,train_y,test_x)
    new_df['xgb'],new_df['xgb_target'],new_test['xgb']=stacking_first_xgb(train_x,train_y,test_x)

    new_df=new_df.astype(np.float64)

    xgb=XGBRegressor(max_depth=3,n_estimators=400,learning_rate=0.01)
    xgb.fit(new_df.drop(['lgb_target','xgb_target'],axis=1),new_df.lgb_target,
            eval_set=[(new_test,test_y)],early_stopping_rounds=100,verbose=True)

    print(calc_rmse(xgb.predict(new_test),test_y))