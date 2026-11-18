import sys, numpy as np, pandas as pd
sys.path.insert(0,"notebooks/src"); sys.path.insert(0,"scratch")
import iara_results as R
from eval_ship_split_5x2cv import metadata
m = metadata(); m["Length"]=pd.to_numeric(m.Length,errors="coerce")
nship = m.groupby("Ship ID").size()
def tipo(r):
    t=r["DETAILED TYPE"]
    if r.Target==0: return {"Pilot Vessel":"Pilot","Tug":"Tug","Fishing Vessel":"Fishing"}.get(t,"outros")
    if r.Target==1: return {"Motor Hopper":"Motor Hopper","Offshore Supply Ship":"OSS"}.get(t,"outros supply/AHTS")
    return "-"
m["tipo"]=m.apply(tipo,axis=1)
def rec(exp):
    d=pd.concat(R.file_predictions(exp).values()); d["File"]=d.File.astype(int)
    d=d.join(m[["tipo","Ship ID"]],on="File"); d["ok"]=d.Target==d.Prediction
    d["pred"]=d.Prediction.map(dict(enumerate(R.CLASSES)))
    return d
for c,cn in [(0,"SMALL"),(1,"MEDIUM")]:
    print(f"\n##### {cn}")
    rows={}
    for lab,exp in [("CNN LOFAR (id)","cnn_lofar"),("MLP LOFAR id","split_id_mlp_lofar_4c"),("MLP LOFAR navio","split_navio_mlp_lofar_4c"),
                    ("SVM MEL id","split_id_svm2000_mel_4c"),("SVM MEL navio","split_navio_svm2000_mel_4c"),("BiGRU id","split_id_bigru_lofar_4c"),("BiGRU navio","split_navio_bigru_lofar_4c")]:
        d=rec(exp); d=d[d.Target==c]
        rows[lab]=d.groupby("tipo").ok.mean().mul(100).round(0)
        rows[lab]["TOTAL"]=round(d.ok.mean()*100)
        if "navio" in lab or lab.startswith("CNN"):
            print(lab, "destino:", (d[~d.ok].pred.value_counts(normalize=True)*100).round(0).to_dict())
    print(pd.DataFrame(rows).to_string())
    print("n gravações/navios:", m[m.Target==c].groupby("tipo").agg(g=("Name","size"),n=("Ship ID","nunique")).to_dict())
