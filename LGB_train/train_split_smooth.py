import lightgbm as lgb
import numpy as np
import pandas as pd

import argparse
import os
from time import time

def invalid(raw_data):
    cond = (raw_data['Patv'] <= 0) & (raw_data['Wspd'] > 2.5) 

    return cond

df=pd.read_csv('wtbdata_245days_filter.csv')

sim=np.load('sim.npy')
parser = argparse.ArgumentParser(description='LGB for KDD')
parser.add_argument('--start', type=int, default=0,
                    help='start time')
parser.add_argument('--end', type=int, default=134,
                    help='end time')

args = parser.parse_args()
try:
    name=str(args.start)+'-'+str(args.end)+'_feature_filter_smooth_le8'
    os.makedirs('/output/'+name)
except:
    pass

np.random.seed(66)
####
for c in df.columns[3:]:
    m=df[c].mean()
    df[c]=df[c].fillna(m)
df=df[df.Day>=0].reset_index(drop='true')
para={'objective': 'regression',
#  'metric': 'auc',
 'boosting_type': 'gbdt',
 'learning_rate': 0.05,
 'num_leaves':8,
 'bagging_fraction':1,
 'bagging_freq': 1,
 'bagging_seed': 66,
 'feature_fraction': 1,
 'feature_fraction_seed': 66,
 'max_bin': 30,
 'max_depth': 4,
'num_threads':30,
    'lambda_l1':0,
 'verbose': -1}
print(para)
col = ['Wspd', 'Etmp','Itmp', 'Patv']
#col = ['Etmp','Patv']
for k in range(args.start,args.end):
    print('start turb ',k)
    df1=df[df.TurbID==(k+1)]
    df1=df1[col]
    #val=df1.values
    #for i in range(1,len(val)-1):
        #val[i]=val[i-1]/6+2*val[i]/3+val[i+1]/6
    n=2
    sind=np.where(sim[k]!=0)[0]
    val=sim[k][k]*df[df.TurbID==(k+1)][col].values
    for si in sind:
        if si!=k:
            val+=sim[k][si]*df[df.TurbID==(si+1)][col].values
    if n>1:
        val=np.vstack([val[n*i:n*(i+1)].mean(0) for i in range(len(val)//n)])
    df2=pd.DataFrame(val,columns=col)
    model_list=[]
    dpre_list=[]
    inp=int(144*1.5//n)
    le=8
    l=96//n+192//(le*2)
    t=96//n
#####Actually, we only use the first 48 models to provide predictions in the final strategy.
    col = ['Wspd', 'Etmp','Itmp', 'Patv']
    df3=pd.DataFrame()
    for i in range(inp):
        for c in col:
            df3[c+str(i)]=df2[c].shift(i,axis=0)

    for d in range(1,l+1):
        t1=time()
        if d>t:
            dk=[t+(d-t)*le-i for i in range(le)]
            df3['label']=sum([df2['Patv'].shift(-dk[j],axis=0) for j in range(le)])/le
            data_all = df3.values[inp:-dk[0], :]
        else:
            df3['label']=df2['Patv'].shift(-d,axis=0)
            data_all=df3.values[inp:-d,:]
        x=data_all[:,:-1]
        y=data_all[:,-1]
        ind=int(len(x)*0.8)
        alpha=0
        train_x,train_y=x[int(alpha*ind):ind],y[int(alpha*ind):ind]
        eval_x,eval_y=x[ind:],y[ind:]


        t2=time()
        print('feature time:',round(t2-t1,2))
        trn_data = lgb.Dataset(train_x,label=train_y)
        val_data = lgb.Dataset(eval_x,label=eval_y)
        print(m)
        model = lgb.train(para, trn_data, num_boost_round=100000, valid_sets=[trn_data, val_data],
                                    verbose_eval=-1,
                                    early_stopping_rounds=200)

        t3=time()

        print('train time:',round(t3-t2,2))
        model.save_model('/output/'+name+'/turb'+str(k)+'_drift'+str(d)+'.txt')
        dpre_list.append(np.mean((eval_y-model.predict(eval_x))**2))
    print('eval result:',np.mean(dpre_list))