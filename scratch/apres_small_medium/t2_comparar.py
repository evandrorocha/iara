"""T2 do teste de impressão de ambiente para vários modelos (assinatura 'nivel'): acerto por tercil de vizinhos da outra classe."""
import sys, os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE)); sys.path.insert(0,"notebooks/src")
from eval_ship_split_5x2cv import metadata, FOLDS_CSV
import iara_results as R
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
m=metadata(); A=pd.read_pickle("data/iara_processed/cache_amb_min_psd24k.pkl"); X=A.loc[A.index.isin(m.index)]
folds=pd.read_csv(FOLDS_CSV,dtype={"Ship ID":str}); folds=folds[folds.condicao=="split_navio"]
def t2(exp):
    P=R.file_predictions(exp); rows=[]
    for k in range(10):
        d=folds[folds.fold==k]; tr=d[(d.Subset=="trn")&(d.Target<3)].File.values; tr=tr[np.isin(tr,X.index)]
        sc=StandardScaler().fit(X.loc[tr]); nn=NearestNeighbors(n_neighbors=10).fit(sc.transform(X.loc[tr]))
        pk=P[k].copy(); pk["File"]=pk.File.astype(int); pk=pk[(pk.Target<2)&pk.File.isin(X.index)]
        _,ind=nn.kneighbors(sc.transform(X.loc[pk.File])); ytr=m.loc[tr,"Target"].values
        for (_,r),nb in zip(pk.iterrows(),ind):
            v=np.bincount(ytr[nb],minlength=3)/10; rows.append((r.Target,r.Prediction,v[1] if r.Target==0 else v[0]))
    T=pd.DataFrame(rows,columns=["y","pred","viz_outra"]); out={}
    for y,cn in [(0,"SMALL"),(1,"MEDIUM")]:
        t=T[T.y==y].copy(); t["terc"]=pd.qcut(t.viz_outra.rank(method="first"),3,labels=["baixo","médio","alto"])
        a=t.groupby("terc",observed=True).pred.apply(lambda s:(s==y).mean()*100)
        out[cn]=a.round(1).to_dict()
        out[cn]["queda"]=round(a["baixo"]-a["alto"],1)
    return out
if __name__=="__main__":
    for e in sys.argv[1:]: print(e, t2(e))
