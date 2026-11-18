"""
Curva de aprendizado pelo NÚMERO DE NAVIOS no treino (split por navio, MLP LOFAR base de aug_ship_split.py).

Em cada fold de split_navio, mantém no treino só uma fração f dos navios de uma ou mais classes (sorteio por fold e
repetição) e avalia no teste inteiro (navios nunca vistos). Cenários (--cenario):
  sm    : reduz SMALL e MEDIUM juntos (LARGE e BACKGROUND inteiros)  -> o recall SMALL/MEDIUM cresce com nº de navios?
  large : reduz só LARGE (controle) -> com ~40 navios, como SMALL/MEDIUM, LARGE cairia?
Frações: 0,25 / 0,5 / 0,75 / 1,0 (1,0 = base). 2 sorteios por fração e fold.

Saída: reports/small_medium/curva_navios_<cenario>.csv (uma linha por fold × fração × sorteio)
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from aug_ship_split import run  # noqa: E402
from eval_ship_split_5x2cv import CACHES, FOLDS_CSV, fold_files, metadata  # noqa: E402

import torch  # noqa: E402


def rec(y, p, c):
    return np.mean(p[y == c] == c) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cenario", default="sm")
    ap.add_argument("--fracs", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--draws", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(8)
    classes = {"sm": [0, 1], "large": [2]}[args.cenario]
    folds = pd.read_csv(FOLDS_CSV, dtype={"Ship ID": str})
    meta = metadata()
    ship_of = {f: (s if s != "-" else f"bg_{f}") for f, s in meta["Ship ID"].items()}
    with open(CACHES["lofar"][0], "rb") as fh:
        cache = pickle.load(fh)
    out = os.path.join(os.path.dirname(os.path.dirname(HERE)), "reports", "small_medium", f"curva_navios_{args.cenario}.csv")
    rows = []
    for k in range(10):
        tr_full, te = fold_files(folds, "split_navio", k, "trn"), fold_files(folds, "split_navio", k, "test")
        for f in map(float, args.fracs.split(",")):
            for d in range(args.draws if f < 1 else 1):
                rng = np.random.default_rng(1000 * k + 10 * d + int(f * 100))
                tr = dict(tr_full)
                n_ships = {}
                for c in [0, 1, 2]:
                    ships = sorted({ship_of[fid] for fid, t in tr_full.items() if t == c})
                    if c in classes and f < 1:
                        keep = set(rng.choice(ships, max(1, round(f * len(ships))), replace=False))
                        tr = {fid: t for fid, t in tr.items() if t != c or ship_of[fid] in keep}
                        ships = sorted(keep)
                    n_ships[c] = len(ships)
                fte, yte, pred, _ = run(cache, tr, te, k, "base", ship_of)
                # voto majoritário por gravação
                dfp = pd.DataFrame({"f": fte, "y": yte, "p": pred})
                g = dfp.groupby("f").agg(y=("y", "first"), p=("p", lambda s: s.value_counts().index[0]))
                y, p = g.y.values, g.p.values
                r = [rec(y, p, c) for c in range(4)]
                sp = np.sqrt(np.mean(r) * np.exp(np.mean(np.log(np.maximum(r, 1e-9))))) if min(r) > 0 else 0.0
                rows.append({"fold": k, "frac": f, "draw": d, "n_small": n_ships[0], "n_medium": n_ships[1],
                             "n_large": n_ships[2], "n_grav_trn": len(tr), "R_SMALL": r[0], "R_MEDIUM": r[1],
                             "R_LARGE": r[2], "R_BG": r[3], "ACC": np.mean(r), "SP": sp})
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
