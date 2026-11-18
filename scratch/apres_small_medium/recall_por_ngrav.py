"""Recall de SMALL/MEDIUM pelo nº de gravações do navio: split por gravação (artigo) x split por navio, MLP LOFAR.
No split por gravação também por nº de gravações do MESMO navio no treino do fold."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, "notebooks/src")
import iara_results as R
from eval_ship_split_5x2cv import metadata, FOLDS_CSV
m = metadata(); folds = pd.read_csv(FOLDS_CSV, dtype={"Ship ID": str})
ngrav = m[m.Target < 2].groupby("Ship ID").size()
bins = [0, 1, 3, 9, 1000]; lab = ["1", "2-3", "4-9", "10+"]
out = []
for cond, exp in [("split_id", "split_id_mlp_lofar_4c"), ("split_navio", "split_navio_mlp_lofar_4c"),
                  ("split_navio", "augnavio_mlp_lofar_base_4c")]:
    P = R.file_predictions(exp)
    for k, p in P.items():
        f = folds[(folds.condicao == cond) & (folds.fold == k)]
        trn_ships = f[f.Subset == "trn"]["Ship ID"].value_counts()
        p = p[p.Target < 2].copy(); p["File"] = p.File.astype(int)
        p["ship"] = m.loc[p.File, "Ship ID"].values
        p["ngrav"] = p.ship.map(ngrav).values
        p["n_trn"] = p.ship.map(trn_ships).fillna(0).astype(int).values
        p["ok"] = p.Prediction == p.Target
        p["exp"] = exp; p["cond"] = cond; p["fold"] = k
        out.append(p)
D = pd.concat(out)
D["faixa"] = pd.cut(D.ngrav, bins, labels=lab)
D.to_csv("reports/small_medium/recall_por_ngrav.csv", index=False)
cls = {0: "SMALL", 1: "MEDIUM"}
for exp in D.exp.unique():
    d = D[D.exp == exp]
    print(f"\n=== {exp}  (recall % por gravação; navios = nº de navios distintos; rec/navio = média do recall de cada navio)")
    for c in [0, 1]:
        dc = d[d.Target == c]
        t = dc.groupby("faixa", observed=True).agg(navios=("ship", "nunique"), n_aval=("ok", "size"), recall=("ok", "mean"))
        t["rec_navio"] = dc.groupby(["faixa", "ship"], observed=True).ok.mean().groupby(level=0, observed=True).mean()
        t[["recall", "rec_navio"]] *= 100
        print(cls[c]); print(t.round(1).to_string())
    if exp.startswith("split_id"):
        print("-- split por gravação: pelo nº de gravações do mesmo navio NO TREINO do fold")
        d = d.assign(trn=pd.cut(d.n_trn, [-1, 0, 2, 8, 1000], labels=["0", "1-2", "3-8", "9+"]))
        for c in [0, 1]:
            t = d[d.Target == c].groupby("trn", observed=True).agg(navios=("ship", "nunique"), n_aval=("ok", "size"), recall=("ok", "mean"))
            t.recall *= 100; print(cls[c]); print(t.round(1).to_string())
