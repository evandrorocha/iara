"""
Teste SMALL × MEDIUM sem LARGE e sem BACKGROUND: um classificador DEDICADO separa as duas classes em navios novos?

Modelo: mesma MLP h32-16 do NB29 (run_mlp de scratch/eval_ship_split_nn_5x2cv.py), 2 saídas (0 = SMALL, 1 = MEDIUM),
LOFAR 5,12 s (cache_lofar.pkl) ou, com --rep demon, DEMON (cache_fusion_mel_demon.pkl['demon']). Folds: split_navio de reports/split_navio/folds.csv restrito a SMALL e MEDIUM
(nenhum navio em treino e teste ao mesmo tempo).
Variantes:
  todos       : todos os navios SMALL e MEDIUM
  sem_hangbo  : sem as dragas HANG BO 2002/2003 (Ship ID 237/238; 146 das 397 gravações MEDIUM)
Métricas:
  - por gravação (voto majoritário): recall SMALL, recall MEDIUM, acurácia balanceada;
  - por NAVIO: escore = média de P(MEDIUM) nas janelas de teste do navio; AUC entre navios por repetição
    (cada navio é testado uma vez por repetição) + p de Mann-Whitney;
  - permutação: re-treina com os rótulos SMALL/MEDIUM embaralhados ENTRE NAVIOS do treino (mesmo nº de navios por
    classe), teste com os rótulos verdadeiros -> distribuição nula da AUC por navio.
Saídas: results/trainings/multirep/small_vs_medium_split_navio_mlp_lofar[_semhangbo]
        reports/small_medium/small_vs_medium_{resumo,navios,permutacao}.csv
Rodar na raiz, no container iara-dev:  python scratch/apres_small_medium/small_vs_medium.py --perm 10
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from results_writer import ResultsWriter  # noqa: E402
from eval_ship_split_5x2cv import CACHES, FOLDS_CSV, fold_files, metadata  # noqa: E402
from eval_ship_split_nn_5x2cv import run_mlp, PARAMS, DEVICE  # noqa: E402

HANGBO = {"237", "238"}
OUT = "reports/small_medium"


def folds_sm(folds, ship, k, variante):
    tr = {f: t for f, t in fold_files(folds, "split_navio", k, "trn").items() if t in (0, 1)}
    te = {f: t for f, t in fold_files(folds, "split_navio", k, "test").items() if t in (0, 1)}
    if variante == "sem_hangbo":
        tr = {f: t for f, t in tr.items() if ship[f] not in HANGBO}
        te = {f: t for f, t in te.items() if ship[f] not in HANGBO}
    return tr, te


def por_navio(fte, yte, scores, ship):
    d = pd.DataFrame({"File": fte, "y": yte, "p": scores[:, 1]})
    d["ship"] = d.File.map(ship)
    return d.groupby("ship").agg(y=("y", "first"), p=("p", "mean"), n=("p", "size"))


def auc(y, s):
    u = st.mannwhitneyu(s[y == 1], s[y == 0], alternative="two-sided")
    return u.statistic / ((y == 1).sum() * (y == 0).sum()), u.pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=10)
    ap.add_argument("--rep", default="lofar", choices=["lofar", "demon"])
    args = ap.parse_args()
    print(f"Dispositivo: {DEVICE}", flush=True)
    folds = pd.read_csv(FOLDS_CSV, dtype={"Ship ID": str})
    m = metadata()
    ship = m["Ship ID"].astype(str).str.strip().to_dict()
    path = CACHES["lofar"][0] if args.rep == "lofar" else "data/iara_processed/cache_fusion_mel_demon.pkl"
    with open(path, "rb") as fh:
        cache = pickle.load(fh)
    if args.rep == "demon":
        cache = cache["demon"]
    suf = "" if args.rep == "lofar" else "_demon"
    os.makedirs(OUT, exist_ok=True)
    resumo, navios, perm = [], [], []
    for var in ["todos", "sem_hangbo"]:
        name = f"small_vs_medium_split_navio_mlp_{args.rep}" + ("" if var == "todos" else "_semhangbo")
        w = ResultsWriter(name, script=__file__, params={
            **PARAMS["mlp"], "representation": args.rep, "split": "split_navio", "num_classes": 2,
            "classes": {0: "SMALL", 1: "MEDIUM"}, "variante": var, "folds_csv": FOLDS_CSV,
            "seed": "torch/numpy = fold"})
        t0 = time.time()
        for k in range(10):
            tr, te = folds_sm(folds, ship, k, var)
            fte, yte, pred, scores = run_mlp(cache, tr, te, k, num_classes=2)
            w.save_fold(k, fte, yte, pred, train_files=tr, scores=scores)
            # voto por gravação
            d = pd.DataFrame({"File": fte, "y": yte, "pr": pred}).groupby("File").agg(y=("y", "first"),
                                                                                      pr=("pr", lambda s: s.mode().iloc[0]))
            rS, rM = (d.pr[d.y == 0] == 0).mean() * 100, (d.pr[d.y == 1] == 1).mean() * 100
            nv = por_navio(fte, yte, scores, ship).assign(variante=var, fold=k, rep=k // 2)
            navios.append(nv.reset_index())
            resumo.append({"variante": var, "fold": k, "rec_SMALL": rS, "rec_MEDIUM": rM, "bal_acc": (rS + rM) / 2,
                           "n_trn_navios": len({ship[f] for f in tr}), "n_te_navios": len(nv)})
            print(f"  {var} fold {k}: SMALL {rS:.1f} MEDIUM {rM:.1f} (navios teste {len(nv)})", flush=True)
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)
        # permutação (rótulos embaralhados entre navios do treino)
        for p in range(args.perm):
            rng = np.random.default_rng(10_000 + p)
            for k in range(10):
                tr, te = folds_sm(folds, ship, k, var)
                lab = pd.Series({ship[f]: t for f, t in tr.items()})
                lab = pd.Series(rng.permutation(lab.values), index=lab.index)
                trp = {f: int(lab[ship[f]]) for f in tr}
                fte, yte, pred, scores = run_mlp(cache, trp, te, k, num_classes=2)
                perm.append(por_navio(fte, yte, scores, ship).assign(variante=var, fold=k, rep=k // 2, perm=p).reset_index())
            print(f"  {var} permutação {p + 1}/{args.perm}", flush=True)
    R_ = pd.DataFrame(resumo); N = pd.concat(navios); P = pd.concat(perm) if perm else pd.DataFrame()
    R_.to_csv(f"{OUT}/small_vs_medium{suf}_resumo.csv", index=False)
    N.to_csv(f"{OUT}/small_vs_medium{suf}_navios.csv", index=False)
    if len(P):
        P.to_csv(f"{OUT}/small_vs_medium{suf}_permutacao.csv", index=False)
    print("\nPor gravação (média de 10 folds):")
    print(R_.groupby("variante")[["rec_SMALL", "rec_MEDIUM", "bal_acc"]].mean().round(1).to_string())
    for var, g in N.groupby("variante"):
        a = [auc(r.y.values, r.p.values) for _, r in g.groupby("rep")]
        nul = [auc(r.y.values, r.p.values)[0] for _, r in P[P.variante == var].groupby(["perm", "rep"])] if len(P) else []
        am = np.mean([x[0] for x in a])
        print(f"\n{var}: AUC por navio por repetição = {[round(x[0], 3) for x in a]} (média {am:.3f}); "
              f"p Mann-Whitney = {[f'{x[1]:.2g}' for x in a]}")
        if nul:
            nm = pd.Series(nul).groupby(np.arange(len(nul)) // 5).mean()  # média das 5 repetições por permutação
            print(f"   nulo (permutação): AUC média {nm.mean():.3f}, máx {nm.max():.3f}; "
                  f"p = {(1 + (nm >= am).sum()) / (1 + len(nm)):.3f}")


if __name__ == "__main__":
    main()
