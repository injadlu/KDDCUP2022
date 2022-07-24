import numpy as np
import pandas as pd
##### generate the matrix S
loc=pd.read_csv('kdd_loc.csv')
def soft(arr):
    return np.exp(arr)/np.exp(arr).sum(1).reshape(-1,1)
loc=pd.read_csv('kdd_loc.csv')
l=loc[['x','y']].values
m=np.sqrt(((l.reshape(134,1,2)-l)**2).sum(-1))
sim=np.zeros((134,134))
top=np.argsort(m,axis=1)
sim[np.arange(134).reshape(-1,1),top[:,:5]]=soft(-m[np.arange(134).reshape(-1,1),top[:,:5]]/250)
np.save('sim.npy',sim)

##### generate the clean training dataset

data=pd.read_csv('wtbdata_245days.csv')
df_median=data.groupby(['Day','Tmstamp'])['Wspd','Etmp','Itmp','Patv'].median().reset_index()
new_col=[]
for s in df_median.columns:
    if s!='Day' and s!='Tmstamp':
        new_col.append(s+'_m')
    else:
        new_col.append(s)
df_median.columns=new_col
data=pd.merge(data,df_median,how='left',on=['Day','Tmstamp'])
print(data.Etmp.mean())
col=['Etmp','Itmp']
for s in col:
    v1=data[s].values
    v2=data[s+'_m'].values
    ind=np.abs(v1-v2)>10
    v1[ind]=v2[ind]
data=data[data.columns[:13]]
print(data.Etmp.mean())
data.to_csv('wtbdata_245days_filter.csv',index=None)