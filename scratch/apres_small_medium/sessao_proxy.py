import sys, numpy as np, pandas as pd
sys.path.insert(0,"scratch"); from eval_ship_split_5x2cv import metadata
m=metadata(); m["Classe"]=m.Target.map({0:"SMALL",1:"MEDIUM",2:"LARGE",3:"BG"})
m["sess"]=m.Dataset.str.replace("E","E")+"|"+m.Wind.astype(str)+"|"+m["Sea state"].astype(str)+"|"+m["Rain state"].astype(str)
m["sessW"]=m.Wind.astype(str)+"|"+m["Sea state"].astype(str)+"|"+m["Rain state"].astype(str)  # sem subconjunto
for col in ["sess","sessW"]:
    g=m.groupby(col)
    print(col,": nº sessões",g.ngroups," gravações/sessão mediana",g.size().median()," máx",g.size().max())
# validação: IDs consecutivos compartilham sessão?
ids=m.sort_index(); same=(ids.sess.values[1:]==ids.sess.values[:-1]).mean()
rng=np.random.default_rng(0); rnd=np.mean([ (ids.sess.values==rng.permutation(ids.sess.values)).mean() for _ in range(20)])
print(f"IDs consecutivos na mesma sessão: {same*100:.1f}%  (acaso {rnd*100:.1f}%)")
# BG compartilha clima com navios?
bgW=set(m[m.Classe=="BG"].sessW); print("BG: gravações com clima igual a alguma gravação de navio:", m[m.Classe=="BG"].sessW.isin(set(m[m.Classe!="BG"].sessW)).mean().round(2))
# sessões de SMALL/MEDIUM: quantos navios por sessão e mistura de classes
s=m[m.Target<3].groupby("sess").agg(n=("Name","size"),navios=("Ship ID","nunique"),
   small=("Target",lambda x:(x==0).sum()),medium=("Target",lambda x:(x==1).sum()),large=("Target",lambda x:(x==2).sum()))
print("sessões com >=2 navios:",(s.navios>=2).sum(),"de",len(s))
print(s.sort_values("n",ascending=False).head(12).to_string())
# fração de gravações SMALL/MEDIUM cuja sessão tem outro navio da mesma classe
for c in [0,1,2]:
    d=m[m.Target==c]; ok=[]
    for i,r in d.iterrows():
        o=d[(d.sess==r.sess)&(d["Ship ID"]!=r["Ship ID"])]; ok.append(len(o)>0)
    print(m.Classe[d.index[0]],"com outro navio da mesma classe na mesma sessão:",round(np.mean(ok)*100,1),"%")
m[["sess","sessW"]].to_csv("scratch/apres_small_medium/sessao_proxy.csv")
