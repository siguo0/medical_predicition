#与训练集train_0312放于同一根目录下 需配置以下库 最后输出结果读取当前目录下的Molecule_prediction_20200312文件夹，读取其中submit_examp_0312提交示例文件 生成提交文件
#总体流程为 读取训练集 进行特征处理 进入stacking第一层模型训练 最后结果通过第二层stacking模型得到
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import catboost as ctb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import shap
import time


def calc_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def standformat(y_pred):
    df=pd.read_csv('./Molecule_prediction_20200312/submit_examp_0312.csv')
    df.Label=y_pred
    df.to_csv('submit.csv',index=False)

def train_process(df):
    #d1为保留首项的删除
    # d1=df.drop_duplicates(subset=['Molecule_max_phase','Molecular weight','RO5_violations','AlogP','Features'],keep='first')
    #d2为不保留的删除
    # d2=df.drop_duplicates(subset=['Molecule_max_phase','Molecular weight','RO5_violations','AlogP','Features'],keep=False)
    #d3为d1、d2两者叠加 然后根据相同原则删除 则 只保留非交集 即为d1中所保留的项
    # d3=d2.append(d1).drop_duplicates(subset=['Molecule_max_phase','Molecular weight','RO5_violations','AlogP','Features'],keep=False)
    # df=df.append(d3)
    df = df.drop(['Features'], axis=1)
    df = df.reset_index(drop=True)
    return df

def test_process(df):
    df = df.drop(['Features'], axis=1)
    return df

def missing_value_process(df):
    miss_index=df[df.isnull().values == True].index#得到有缺失值的行索引

    print(miss_index)
    train_x=df.drop(miss_index,axis=0).drop(['RO5_violations','AlogP','Features'],axis=1)
    train_y1=df.drop(miss_index,axis=0).RO5_violations
    train_y2=df.drop(miss_index,axis=0).AlogP

    test_x=df[df.isnull().values == True].drop(['RO5_violations','AlogP','Features'],axis=1)

    gbm=lgb.LGBMRegressor()
    gbm.fit(train_x,train_y2)
    df.loc[miss_index,'AlogP']=gbm.predict(test_x)

    gbm=lgb.LGBMClassifier()
    gbm.fit(train_x,train_y1)
    df.loc[miss_index,'RO5_violations']=gbm.predict(test_x)

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
    # df=missing_value_process(df)

    df['x1'] = df['Molecular weight'] * df.Molecule_max_phase
    df['x2'] = df.Molecule_max_phase / (df.AlogP + 1)

    df.fillna(df.median(),inplace=True)

    df.loc[df['Molecular weight'] >= 0.6, 'Molecular weight'] = df.loc[df['Molecular weight'] < 0.6, 'Molecular weight'].median()
    df.loc[df['AlogP'] < 0.2, 'AlogP'] = df.loc[df['AlogP'] >= 0.2, 'AlogP'].median()

    df['Molecular weight1']=preprocessing.scale(df['Molecular weight'])
    df['AlogP1']=preprocessing.scale(df.AlogP)

    return df


def stacking_first_lgb(train,test):
    data=train.drop(['Label'],axis=1)
    target=train.Label

    gbm = lgb.LGBMRegressor(n_estimators=870,
                            learning_rate=0.135,
                            max_depth=6,
                            subsample=0.79,
                            colsample_bytree=0.55,
                            num_leaves=195,
                            boosting_type='gbdt')

    gbm.fit(data,train.Label)
    important=pd.DataFrame(gbm.feature_importances_,columns=['importance'])
    important=important.loc[important['importance']==0,'importance'].index
    data=data.drop(important,axis=1)
    test=test.drop(important,axis=1)

    kf = KFold(n_splits=10, shuffle=False,random_state=0)
    new_test_ = [0]
    real_target=pd.DataFrame()
    pre_target=pd.DataFrame()
    pre_train=[0]
    for k, (train0, test0) in enumerate(kf.split(data, target)):
        kf = gbm.fit(data.iloc[train0], target.iloc[train0].values.ravel(),
                     eval_set=[(data.iloc[test0],target.iloc[test0])],verbose=False,eval_metric='rmse',early_stopping_rounds=100)
        pre_target = pre_target.append(list(gbm.predict(data.iloc[test0])), ignore_index=True)#作为新训练集特征的预测值
        real_target = real_target.append(list(target[test0]), ignore_index=True)#作为新训练集标记值的值
        new_test_ += gbm.predict(test)/10
    return list(pre_target.values), list(real_target.values), new_test_

def stacking_first_ctb(train,test):
    data=train.drop(['Label'],axis=1)
    target=train.Label
    cat = ctb.CatBoostRegressor(iterations=570,depth=8,bagging_temperature=2,l2_leaf_reg=1.0,loss_function='RMSE',learning_rate=0.1)

    cat.fit(data, target)
    important = pd.DataFrame(cat.feature_importances_, columns=['importance'])
    important = important.loc[important['importance'] == 0, 'importance'].index

    data.columns=range(data.shape[1])
    train_x=data.drop(important,axis=1)
    test.columns=range(test.shape[1])
    test_x=test.drop(important,axis=1)

    score=[]
    new_test_ = [0]
    real_target=pd.DataFrame()
    pre_target=pd.DataFrame()
    kf = KFold(n_splits=10, shuffle=False,random_state=0)#如果要使划分后的测试集相同 因为要将多个模型预测结果作为特征 划分的测试集的真实值作为标记值，要么使用相同的划分 要么这个划分要在主函数中进行 然后各个子函数只用于获得这些预测值

    #将划分后的训练集 再次划分出训练集和测试集 使用训练集多折划分 对于
    for k, (train0, test0) in enumerate(kf.split(data, target)):
        kf = cat.fit(data.iloc[train0], target.iloc[train0].values.ravel(),
                     eval_set=[(data.iloc[test0],target.iloc[test0])],verbose=True,early_stopping_rounds=50)
        pre_target = pre_target.append(list(cat.predict(data.iloc[test0])), ignore_index=True)#作为新训练集特征的预测值
        real_target = real_target.append(list(target[test0]), ignore_index=True)#作为新训练集标记值的值
        new_test_ += cat.predict(test)/10
    return list(pre_target.values), list(real_target.values), new_test_


def stacking_first_rf(train,test):
    data=train.drop(['Label'],axis=1)
    target=train.Label
    rf = RandomForestRegressor(n_estimators=35,max_depth=38)

    rf.fit(data, target)
    important = pd.DataFrame(rf.feature_importances_, columns=['importance'])
    important = important.loc[important['importance'] == 0, 'importance'].index

    data.columns=range(data.shape[1])
    train_x=data.drop(important,axis=1)
    test.columns=range(test.shape[1])
    test_x=test.drop(important,axis=1)

    score=[]
    new_test_ = [0]
    real_target=pd.DataFrame()
    pre_target=pd.DataFrame()
    kf = KFold(n_splits=5, shuffle=False,random_state=0)#如果要使划分后的测试集相同 因为要将多个模型预测结果作为特征 划分的测试集的真实值作为标记值，要么使用相同的划分 要么这个划分要在主函数中进行 然后各个子函数只用于获得这些预测值

    #将划分后的训练集 再次划分出训练集和测试集 使用训练集多折划分 对于
    for k, (train0, test0) in enumerate(kf.split(data, target)):
        kf = rf.fit(data.iloc[train0], target.iloc[train0].values.ravel())
        pre_target = pre_target.append(list(rf.predict(data.iloc[test0])), ignore_index=True)#作为新训练集特征的预测值
        real_target = real_target.append(list(target[test0]), ignore_index=True)#作为新训练集标记值的值
        new_test_ += rf.predict(test)/5
    return list(pre_target.values), list(real_target.values), new_test_

def stacking_first_adb(train,test):
    data=train.drop(['Label'],axis=1)
    target=train.Label
    adb = AdaBoostRegressor(DecisionTreeRegressor(max_depth=35),learning_rate=0.1)

    adb.fit(data, target)
    important = pd.DataFrame(adb.feature_importances_, columns=['importance'])
    important = important.loc[important['importance'] == 0, 'importance'].index

    data.columns=range(data.shape[1])
    train_x=data.drop(important,axis=1)
    test.columns=range(test.shape[1])
    test_x=test.drop(important,axis=1)

    score=[]
    new_test_ = [0]
    real_target=pd.DataFrame()
    pre_target=pd.DataFrame()
    kf = KFold(n_splits=5, shuffle=False,random_state=0)#如果要使划分后的测试集相同 因为要将多个模型预测结果作为特征 划分的测试集的真实值作为标记值，要么使用相同的划分 要么这个划分要在主函数中进行 然后各个子函数只用于获得这些预测值

    #将划分后的训练集 再次划分出训练集和测试集 使用训练集多折划分 对于
    for k, (train0, test0) in enumerate(kf.split(data, target)):
        kf = adb.fit(data.iloc[train0], target.iloc[train0].values.ravel())
        pre_target = pre_target.append(list(adb.predict(data.iloc[test0])), ignore_index=True)#作为新训练集特征的预测值
        real_target = real_target.append(list(target[test0]), ignore_index=True)#作为新训练集标记值的值
        new_test_ += adb.predict(test)/5
        print(calc_rmse(adb.predict(data.iloc[test0]),target[test0]))
    return list(pre_target.values), list(real_target.values), new_test_


def stacking_first_xgb(train,test):
    data=train.drop(['Label'],axis=1)
    target=train.Label
    xgb = XGBRegressor(n_estimators=400,
                     learning_rate=0.1,
                     colsample_bytree=1,
                     max_depth=10,
                     subsample=0.5,
                     min_child_weight=7)

    xgb.fit(data, target)
    important = pd.DataFrame(xgb.feature_importances_, columns=['importance'])
    important = important.loc[important['importance'] == 0, 'importance'].index

    data.columns=range(data.shape[1])
    train_x=data.drop(important,axis=1)
    test.columns=range(test.shape[1])
    test_x=test.drop(important,axis=1)

    score=[]
    new_test_ = [0]
    real_target=pd.DataFrame()
    pre_target=pd.DataFrame()
    kf = KFold(n_splits=5, shuffle=False,random_state=0)#如果要使划分后的测试集相同 因为要将多个模型预测结果作为特征 划分的测试集的真实值作为标记值，要么使用相同的划分 要么这个划分要在主函数中进行 然后各个子函数只用于获得这些预测值

    #将划分后的训练集 再次划分出训练集和测试集 使用训练集多折划分 对于
    for k, (train0, test0) in enumerate(kf.split(data, target)):
        kf = xgb.fit(data.iloc[train0], target.iloc[train0].values.ravel(),
                     eval_set=[(data.iloc[test0],target.iloc[test0])],verbose=True,eval_metric='rmse',early_stopping_rounds=50)
        pre_target = pre_target.append(list(xgb.predict(data.iloc[test0])), ignore_index=True)#作为新训练集特征的预测值
        real_target = real_target.append(list(target[test0]), ignore_index=True)#作为新训练集标记值的值
        new_test_ += xgb.predict(test)/5
    return list(pre_target.values), list(real_target.values), new_test_


if __name__=='__main__':
    train=pd.read_csv('train_0312.csv')
    train=Features_process(train)
    train=train_process(train)

    test=pd.read_csv('test.csv')
    test=Features_process(test)
    test=test_process(test)

    data = train.drop(['Label'], axis=1)
    target = train.Label

    new_df = pd.DataFrame()
    new_test = pd.DataFrame()
    # new_df['rf'], new_df['rf_target'], new_test['rf'] = stacking_first_rf(train,test)
    # new_df['lgb'], new_df['lgb_target'], new_test['lgb'] = stacking_first_lgb(train,test)
    new_df['lgb'], new_df['lgb_target'], new_test['lgb'] = stacking_first_lgb(train,test)


    new_df['xgb'], new_df['xgb_target'], new_test['xgb'] = stacking_first_xgb(train,test)
    new_df['ctb'], new_df['ctb_target'], new_test['ctb'] = stacking_first_ctb(train,test)

    new_df = new_df.astype(np.float64)
    new_df.to_csv('my_stacking_first.csv',index=False)
    new_test.to_csv('my_new_test.csv',index=False)

    print(new_df)

    train_x = new_df.drop(['lgb_target', 'ctb_target','xgb_target'], axis=1)
    train_y = new_df.lgb_target#因为所有模型的真实值都是相同的 所以没有关系
    #
    # print(train_x)
    # print(train_y)

    cat = ctb.CatBoostRegressor(depth=8, learning_rate=0.1, loss_function='RMSE',
                                iterations=50)
    cat.fit(train_x, train_y)
    pre=cat.predict(new_test)
    # xgb = XGBRegressor(n_estimators=500,
    #                    learning_rate=0.01,
    #                    max_depth=2,
    #                    objective='reg:squarederror',
    #                    subsample=0.4)
    # xgb.fit(train_x,train_y)
    # pre=xgb.predict(new_test)
    # standformat(pre)





