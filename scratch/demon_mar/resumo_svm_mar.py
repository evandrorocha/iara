"""Resumo do SVM m=6000 com augmentation dos LARGE com fundo de mar (scratch/demon_mar/svm_mar.py).
Médias de 10 folds, Δ vs base e F 5x2cv; comparação com a MLP. Saída: reports/demon_mar/svm_resumo.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "notebooks/src")
import iara_results as R
CN = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}
VAR = ["base", "dupL", "calmo", "agitado"]
MET = ["SP", "ACC"] + [f"R_{c}" for c in CN.values()]


def metricas(y, p):
    rec = np.array([np.mean(p[y == c] == c) for c in range(4)])
    sp = np.sqrt(rec.mean() * np.exp(np.mean(np.log(rec)))) * 100 if np.all(rec > 0) else 0.0
    return {"SP": sp, "ACC": rec.mean() * 100, **{f"R_{CN[c]}": r * 100 for c, r in enumerate(rec)}}


def exp(cond, rep, v):
    return f"svm6000_mar_{cond}_{rep}_4c" + ("" if v == "base" else f"_aug{v}")


rows, med = [], {}
for cond in ["split_navio", "split_id"]:
    for rep in ["lofar", "demon"]:
        F = {}
        for v in VAR:
            try:
                P = R.file_predictions(exp(cond, rep, v))
            except FileNotFoundError:
                continue
            if len(P) < 10:
                continue
            F[v] = pd.DataFrame({k: metricas(d.Target.values, d.Prediction.values) for k, d in P.items()}).T
            med[(cond, rep, v)] = F[v].mean()
        for v in VAR[1:]:
            if v in F and "base" in F:
                for m in MET:
                    rows.append({"split": cond, "rep": rep, "variante": v, "metrica": m, "delta": (F[v][m] - F["base"][m]).mean(),
                                 "folds_melhor": int((F[v][m] > F["base"][m]).sum()),
                                 "p": R.f_test_5x2(F[v][m].values, F["base"][m].values)[1]})
os.makedirs("reports/demon_mar", exist_ok=True)
T = pd.DataFrame(rows); T.to_csv("reports/demon_mar/svm_resumo.csv", index=False)
M = pd.DataFrame(med).T; M.to_csv("reports/demon_mar/svm_medias.csv")
pd.set_option("display.width", 220)
print(M.round(1).to_string())
print(T[T.metrica.isin(["SP", "ACC", "R_LARGE"])].round(3).to_string(index=False))
if len(T):
    print("menor p:", T.p.min().round(3), T.loc[T.p.idxmin()].to_dict())
