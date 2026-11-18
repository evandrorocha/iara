import sys, numpy as np, pandas as pd
sys.path.insert(0,"notebooks/src"); sys.path.insert(0,"scratch")
import iara_results as R
from eval_ship_split_5x2cv import metadata
m = metadata()
print("gravações por subconjunto:"); print(pd.crosstab(m.Target.map(dict(enumerate(R.CLASSES))), m.Dataset).to_string())
# navios com gravações em >1 subconjunto
for c in [0,1]:
    d=m[m.Target==c]; s=d.groupby("Ship ID").Dataset.nunique()
    print(R.CLASSES[c],"navios em 1 subconj:",(s==1).sum(),"em 2+:",(s>1).sum())
for exp in ["cnn_lofar","split_id_mlp_lofar_4c","split_navio_mlp_lofar_4c","split_navio_svm2000_mel_4c"]:
    d=pd.concat(R.file_predictions(exp).values()); d["File"]=d.File.astype(int); d=d.join(m[["Dataset"]],on="File")
    d["ok"]=d.Target==d.Prediction
    t=d[d.Target<2].groupby(["Target","Dataset"]).ok.mean().mul(100).round(0).unstack()
    t.index=[R.CLASSES[i] for i in t.index]; print("\n",exp); print(t.to_string())
