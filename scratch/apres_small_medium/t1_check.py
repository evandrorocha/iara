import sys, os, numpy as np, pandas as pd, sklearn
sys.path.insert(0,"scratch"); from eval_ship_split_5x2cv import metadata
from sklearn.linear_model import LogisticRegression; from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
m=metadata(); A=pd.read_pickle("data/iara_processed/cache_amb_min_psd24k.pkl"); sm=m.loc[A.index.intersection(m.index)]; sm=sm[sm.Target<2]
X,y,g=A.loc[sm.index].values,sm.Target.values,sm["Ship ID"].values
s=pd.Series(g); w=1.0/s.map(s.value_counts()).values
for c in [0,1]: w[y==c]/=w[y==c].sum()
w=w*len(y)/w.sum(); a=[]
for r in range(10):
    p=np.zeros(len(y))
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=r).split(X,y,g):
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=2000)); clf.fit(X[tr],y[tr],logisticregression__sample_weight=w[tr]); p[te]=clf.predict_proba(X[te])[:,1]
    a.append(roc_auc_score(y,p,sample_weight=w))
print("sklearn",sklearn.__version__,"AUC média %.3f min %.3f max %.3f"%(np.mean(a),min(a),max(a)))
