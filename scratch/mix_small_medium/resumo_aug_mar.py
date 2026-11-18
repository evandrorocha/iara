"""Resumo do augmentation dos LARGE com fundo de mar agitado (scratch/mix_small_medium/aug_mar.py).
Métricas por fold (por gravação, voto majoritário), Δ vs base e vs dupL, F 5x2cv, e acerto de LARGE por
estado do mar × CPA (split por navio). Saída: reports/small_medium/aug_mar_resumo.csv e aug_mar_large_mar.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "notebooks/src"); sys.path.insert(0, "scratch")
import iara_results as R
from eval_ship_split_5x2cv import metadata

CM = {0: "MIX_SMALL_MEDIUM", 1: "LARGE", 2: "BACKGROUND"}
VAR = ["base", "dupL", "agitado", "calmo"]


def exp(cond, v):
    return f"mix_small_medium_{cond}_mlp_lofar" + ("" if v == "base" else f"_aug{v}")


def metricas(y, p):
    rec = np.array([np.mean(p[y == c] == c) for c in range(3)])
    sp = np.sqrt(rec.mean() * np.exp(np.mean(np.log(rec)))) * 100 if np.all(rec > 0) else 0.0
    return {"SP": sp, "ACC": rec.mean() * 100, **{f"R_{CM[c]}": r * 100 for c, r in enumerate(rec)}}


def main():
    m = metadata()
    m["sea"] = pd.to_numeric(m["Sea state"], errors="coerce")
    rows, por_fold, prev = [], {}, {}
    for cond in ["split_navio", "split_id"]:
        for v in VAR:
            try:
                P = R.file_predictions(exp(cond, v))
            except FileNotFoundError:
                continue
            prev[(cond, v)] = P
            por_fold[(cond, v)] = pd.DataFrame({k: metricas(d.Target.values, d.Prediction.values) for k, d in P.items()}).T
        for v in VAR[1:]:
            if (cond, v) not in por_fold:
                continue
            A = por_fold[(cond, v)]
            for ref in ["base", "dupL"]:
                if ref == v or (cond, ref) not in por_fold:
                    continue
                B = por_fold[(cond, ref)]
                for met in ["SP", "ACC", "R_MIX_SMALL_MEDIUM", "R_LARGE", "R_BACKGROUND"]:
                    f, p = R.f_test_5x2(A[met].values, B[met].values)
                    rows.append({"split": cond, "variante": v, "vs": ref, "metrica": met, "valor": A[met].mean(),
                                 "ref": B[met].mean(), "delta": A[met].mean() - B[met].mean(),
                                 "folds_melhor": int((A[met].values > B[met].values).sum()), "F": f, "p": p})
    T = pd.DataFrame(rows)
    T.to_csv("reports/small_medium/aug_mar_resumo.csv", index=False)
    pd.set_option("display.width", 200)
    print("Médias por variante (10 folds):")
    print(pd.concat({f"{c}|{v}": f.mean() for (c, v), f in por_fold.items()}, axis=1).T.round(1).to_string())
    print("\nΔ e F 5x2cv:"); print(T.round(3).to_string(index=False))
    # LARGE por estado do mar x CPA, split por navio
    out = []
    for (cond, v), P in prev.items():
        if cond != "split_navio":
            continue
        d = pd.concat(P.values()); d = d[d.Target == 1].copy()
        d["File"] = d.File.astype(int)
        d = d.join(m[["sea", "Dataset"]], on="File")
        d["CPA"] = d.Dataset.isin(["A", "C"]).map({True: "com CPA", False: "sem CPA"})
        d["mar"] = np.where(d.sea <= 2, "calmo (0-2)", "agitado (3+)")
        g = d.assign(ok=d.Prediction == 1).groupby(["CPA", "mar"]).ok.mean().mul(100).rename(v)
        out.append(g)
    G = pd.concat(out, axis=1)
    G.to_csv("reports/small_medium/aug_mar_large_mar.csv")
    print("\nAcerto de LARGE (%, split por navio) por CPA × estado do mar:"); print(G.round(0).to_string())


if __name__ == "__main__":
    main()
