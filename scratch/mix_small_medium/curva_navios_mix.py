"""
Curva de aprendizado pelo NÚMERO DE NAVIOS no treino, experimento mix_small_medium (MIX_SMALL_MEDIUM, LARGE, BACKGROUND).
Pergunta: com mais navios distintos, o modelo ainda melhoraria? (decide se vale investir em mais dados / pré-treino)

Mesma MLP h32-16 e folds de scratch/mix_small_medium/treino_mlp.py (split_navio). Em cada fold, mantém no treino só
uma fração f dos NAVIOS das classes de navio (MIX_SMALL_MEDIUM e LARGE juntas, cada classe reduzida na mesma
proporção; BACKGROUND inteiro) e avalia no teste inteiro (navios nunca vistos). Todas as gravações de um navio mantido
ficam no treino. Frações 0,125 / 0,25 / 0,5 / 0,75 (2 sorteios por fração e fold); f = 1 é o modelo base já treinado
(mix_small_medium_split_navio_mlp_lofar, mesma semente).

Saída: reports/small_medium/curva_navios_mix.csv (uma linha por fold × fração × sorteio)
Rodar na raiz, no container iara-dev:  python scratch/mix_small_medium/curva_navios_mix.py
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "notebooks", "src"))
from eval_ship_split_5x2cv import CACHES, FOLDS_CSV, fold_files, metadata  # noqa: E402
from eval_ship_split_nn_5x2cv import run_mlp  # noqa: E402
import iara_results as R  # noqa: E402

MAPA = {0: 0, 1: 0, 2: 1, 3: 2}
CM = {0: "MIX_SMALL_MEDIUM", 1: "LARGE", 2: "BACKGROUND"}
OUT = "reports/small_medium/curva_navios_mix.csv"


def metricas(fte, yte, pred):
    g = pd.DataFrame({"f": fte, "y": yte, "p": pred}).groupby("f").agg(
        y=("y", "first"), p=("p", lambda s: s.value_counts().index[0]))
    r = [np.mean(g.p[g.y == c] == c) * 100 for c in range(3)]
    sp = np.sqrt(np.mean(r) * np.exp(np.mean(np.log(np.maximum(r, 1e-9))))) if min(r) > 0 else 0.0
    return {**{f"R_{CM[c]}": r[c] for c in range(3)}, "ACC": np.mean(r), "SP": sp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", default="0.125,0.25,0.5,0.75")
    ap.add_argument("--draws", type=int, default=2)
    args = ap.parse_args()
    folds = pd.read_csv(FOLDS_CSV, dtype={"Ship ID": str})
    meta = metadata()
    ship_of = {f: str(s).strip() for f, s in meta["Ship ID"].items()}
    with open(CACHES["lofar"][0], "rb") as fh:
        cache = pickle.load(fh)
    rows = []
    # f = 1: modelo base já treinado (mesma MLP, mesma semente)
    P = R.file_predictions("mix_small_medium_split_navio_mlp_lofar")
    for k, d in P.items():
        tr = {f: MAPA[t] for f, t in fold_files(folds, "split_navio", k, "trn").items()}
        r = [np.mean(d.Prediction[d.Target == c] == c) * 100 for c in range(3)]
        sp = np.sqrt(np.mean(r) * np.exp(np.mean(np.log(r))))
        rows.append({"fold": k, "frac": 1.0, "draw": 0,
                     "n_mix": len({ship_of[f] for f, t in tr.items() if t == 0}),
                     "n_large": len({ship_of[f] for f, t in tr.items() if t == 1}),
                     "n_grav_trn": len(tr), **{f"R_{CM[c]}": r[c] for c in range(3)}, "ACC": np.mean(r), "SP": sp})
    for k in range(10):
        tr_full = {f: MAPA[t] for f, t in fold_files(folds, "split_navio", k, "trn").items()}
        te = {f: MAPA[t] for f, t in fold_files(folds, "split_navio", k, "test").items()}
        for frac in map(float, args.fracs.split(",")):
            for d in range(args.draws):
                rng = np.random.default_rng(1000 * k + 10 * d + int(frac * 1000))
                tr, n = dict(tr_full), {}
                for c in (0, 1):
                    ships = sorted({ship_of[f] for f, t in tr_full.items() if t == c})
                    keep = set(rng.choice(ships, max(1, round(frac * len(ships))), replace=False))
                    tr = {f: t for f, t in tr.items() if t != c or ship_of[f] in keep}
                    n[c] = len(keep)
                fte, yte, pred, _ = run_mlp(cache, tr, te, k, num_classes=3)
                rows.append({"fold": k, "frac": frac, "draw": d, "n_mix": n[0], "n_large": n[1], "n_grav_trn": len(tr),
                             **metricas(fte, yte, pred)})
                print({kk: (round(v, 1) if isinstance(v, float) else v) for kk, v in rows[-1].items()}, flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False)
    D = pd.DataFrame(rows)
    D.to_csv(OUT, index=False)
    g = D.groupby("frac")[["n_mix", "n_large", "R_MIX_SMALL_MEDIUM", "R_LARGE", "R_BACKGROUND", "ACC", "SP"]].mean()
    print(g.round(1).to_string())
    for m in ["ACC", "R_MIX_SMALL_MEDIUM", "R_LARGE"]:
        a = np.polyfit(np.log2(D.frac), D[m], 1)[0]
        print(f"{m}: +{a:.1f} pontos por duplicação do nº de navios")


if __name__ == "__main__":
    main()
