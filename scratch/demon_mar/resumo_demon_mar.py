"""Resumo do experimento DEMON + augmentation dos LARGE com fundo de mar (scratch/demon_mar/aug_mar_demon.py).
1. Métricas por fold (por gravação, voto majoritário), Δ vs base e vs dupL, F 5x2cv.
2. Acerto de LARGE por estado do mar × CPA (split por navio).
3. O ruído apaga as linhas de modulação? Contraste do DEMON médio de cada gravação LARGE (pico / mediana, em dB,
   em 0,4–25 Hz) no original e nas cópias misturadas, por razão sinal/fundo.
Saída: reports/demon_mar/{resumo,large_mar,contraste_copias}.csv"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "notebooks/src"); sys.path.insert(0, "scratch"); sys.path.insert(0, "scratch/demon_mar")
import iara_results as R
from eval_ship_split_5x2cv import metadata
from aug_mar_demon import exp_name, CACHE_DEMON

CM = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}
VAR = ["base", "dupL", "agitado", "calmo"]
MET = ["SP", "ACC"] + [f"R_{c}" for c in CM.values()]
OUT = "reports/demon_mar"


def metricas(y, p):
    rec = np.array([np.mean(p[y == c] == c) for c in range(4)])
    sp = np.sqrt(rec.mean() * np.exp(np.mean(np.log(rec)))) * 100 if np.all(rec > 0) else 0.0
    return {"SP": sp, "ACC": rec.mean() * 100, **{f"R_{CM[c]}": r * 100 for c, r in enumerate(rec)}}


def contraste(w):
    s = w.mean(0)
    return 20 * np.log10(s.max() / np.median(s))


def main():
    os.makedirs(OUT, exist_ok=True)
    m = metadata()
    m["sea"] = pd.to_numeric(m["Sea state"], errors="coerce")
    rows, por_fold, prev = [], {}, {}
    for cond in ["split_navio", "split_id"]:
        for v in VAR:
            try:
                P = R.file_predictions(exp_name(cond, v))
            except FileNotFoundError:
                continue
            prev[(cond, v)] = P
            por_fold[(cond, v)] = pd.DataFrame({k: metricas(d.Target.values, d.Prediction.values) for k, d in P.items()}).T
        for v in VAR[1:]:
            if (cond, v) not in por_fold:
                continue
            for ref in ["base", "dupL"]:
                if ref == v or (cond, ref) not in por_fold:
                    continue
                A, B = por_fold[(cond, v)], por_fold[(cond, ref)]
                for met in MET:
                    f, p = R.f_test_5x2(A[met].values, B[met].values)
                    rows.append({"split": cond, "variante": v, "vs": ref, "metrica": met, "valor": A[met].mean(),
                                 "ref": B[met].mean(), "delta": A[met].mean() - B[met].mean(),
                                 "folds_melhor": int((A[met].values > B[met].values).sum()), "F": f, "p": p})
    T = pd.DataFrame(rows)
    T.to_csv(f"{OUT}/resumo.csv", index=False)
    M = pd.concat({f"{c}|{v}": f.mean() for (c, v), f in por_fold.items()}, axis=1).T
    M.to_csv(f"{OUT}/medias.csv")
    pd.set_option("display.width", 220)
    print("Médias por variante (10 folds):"); print(M.round(1).to_string())
    if len(T):
        print("\nΔ e F 5x2cv (vs base):"); print(T[T.vs == "base"].round(3).to_string(index=False))
        print(f"\nmenor p: {T.p.min():.3f}")
    out, conf = [], []
    for (cond, v), P in prev.items():
        if cond != "split_navio":
            continue
        d = pd.concat(P.values())
        conf.append(pd.crosstab(d.Target.map(CM), d.Prediction.map(CM), normalize="index").mul(100).round(1).assign(variante=v))
        d = d[d.Target == 2].copy()
        d["File"] = d.File.astype(int)
        d = d.join(m[["sea", "Dataset"]], on="File")
        d["CPA"] = d.Dataset.isin(["A", "C"]).map({True: "com CPA", False: "sem CPA"})
        d["mar"] = np.where(d.sea <= 2, "calmo (0-2)", "agitado (3+)")
        out.append(d.assign(ok=d.Prediction == 2).groupby(["CPA", "mar"]).ok.mean().mul(100).rename(v))
    if out:
        G = pd.concat(out, axis=1); G.to_csv(f"{OUT}/large_mar.csv")
        print("\nAcerto de LARGE (%, split por navio) por CPA × estado do mar:"); print(G.round(0).to_string())
        C = pd.concat(conf); C.to_csv(f"{OUT}/confusao_split_navio.csv")
        print("\nMatriz de confusão (% da linha, split por navio):"); print(C.to_string())

    # 3. contraste das linhas de modulação
    cache = pickle.load(open(CACHE_DEMON, "rb"))["demon"]
    orig = pd.Series({f: contraste(w) for f, w in cache.items()}).rename("c").to_frame().join(m.Target)
    ref = {n: orig[orig.Target.isin(t)].c.median() for n, t in [("LARGE", [2]), ("SMALL+MEDIUM", [0, 1]), ("BACKGROUND", [3])]}
    print("\nContraste do DEMON médio (pico/mediana, dB), mediana por classe:", {k: round(v, 2) for k, v in ref.items()})
    rows = []
    for g in ["agitado", "calmo"]:
        p = f"data/iara_processed/cache_demon_mixbg_{g}.pkl"
        if not os.path.exists(p):
            continue
        mix = pickle.load(open(p, "rb"))
        for r in mix["sorteios"].itertuples():
            rows.append({"grupo": g, "ID": r.ID, "copia": r.copia, "rsf_db": r.rsf_db,
                         "c": contraste(mix["janelas"][(r.ID, r.copia)]), "c_orig": orig.c[r.ID]})
    if rows:
        D = pd.DataFrame(rows); D["d"] = D.c - D.c_orig
        D["abaixo_mix"] = D.c < ref["SMALL+MEDIUM"]
        D.to_csv(f"{OUT}/contraste_copias.csv", index=False)
        D["faixa"] = pd.cut(D.rsf_db, [0, 5, 10, 15, 20], include_lowest=True)
        print(f"LARGE originais com contraste abaixo da mediana SMALL+MEDIUM: "
              f"{(orig[orig.Target == 2].c < ref['SMALL+MEDIUM']).mean() * 100:.0f}%")
        t = D.groupby(["grupo", "faixa"], observed=True).agg(n=("d", "size"), c_orig=("c_orig", "median"), c=("c", "median"),
                                                             d=("d", "median"), abaixo_mix=("abaixo_mix", "mean"))
        t["abaixo_mix"] *= 100
        print(t.round(2).to_string())
        print(D.groupby("grupo").agg(d=("d", "median"), abaixo_mix=("abaixo_mix", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
