import pandas as pd
import random

t1=pd.read_csv('2.16484.csv')
t2=pd.read_csv('2.16838.csv')
t3=pd.read_csv('best.csv')
data=pd.DataFrame()
data['t1']=t1.Label
data['t2']=t2.Label
data['t3']=t3.Label
data['t4']=data.median(axis=1)
data['t5']=data.mean(axis=1)
data['t6']=data.drop(['t1','t2','t3'],axis=1).mean(axis=1)
t1.Label=data.t6
print(t1)
t1.to_csv('submit.csv',index=False)