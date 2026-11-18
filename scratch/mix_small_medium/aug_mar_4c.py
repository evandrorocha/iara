"""
Augmentation dos LARGE com fundo de MAR AGITADO no problema ORIGINAL de 4 classes do IARA
(SMALL, MEDIUM, LARGE, BACKGROUND).

Mesma regra de scratch/mix_small_medium/aug_mar.py, só que com as classes originais:
mesma MLP h32-16 do NB29 (run_mlp), mesmos folds (reports/split_navio/folds.csv), teste sempre no LOFAR original.
Cada gravação LARGE do TREINO ganha UMA versão extra (ID + 100000, mesmo rótulo). Variantes:
  agitado : LOFAR do áudio LARGE + fundo real de mar agitado (estado >= 4, sem chuva) — cache_lofar_mixbg_agitado.pkl
  calmo   : idem com fundo de mar calmo (estado <= 2, sem chuva) — controle: é o mar agitado ou qualquer fundo?
  dupL    : cópia idêntica do LARGE — controle: só mais exemplos LARGE, sem ruído
Em agitado/calmo só valem cópias cujo fundo está no TREINO do fold (sem vazamento do BACKGROUND de teste).
Semente da escolha da cópia = 1000·fold + índice da variante (as mesmas cópias do aug_mar.py).

Comparar com: results/trainings/multirep/<split>_mlp_lofar_4c (MLP de 4 classes do NB29, sem augmentation)
Saída: results/trainings/multirep/<split>_mlp_lofar_4c_aug<variante>
Rodar na raiz, no container iara-dev:  python scratch/mix_small_medium/aug_mar_4c.py --variants agitado,calmo,dupL
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from results_writer import ResultsWriter  # noqa: E402
from eval_ship_split_5x2cv import CACHES, FOLDS_CSV, fold_files  # noqa: E402
from eval_ship_split_nn_5x2cv import run_mlp, PARAMS, DEVICE  # noqa: E402

CLASSES = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}
LARGE, BG = 2, 3
VARIANTS = ["dupL", "agitado", "calmo"]  # ordem fixa: define a semente (igual ao aug_mar.py)
OFF = 100000


def augment(cache, tr, variant, k, mix):
    rng = np.random.default_rng(1000 * k + VARIANTS.index(variant))
    extra = {}
    if mix is not None:
        bg_trn = {f for f, c in tr.items() if c == BG}
        sor = mix["sorteios"]
        valid = sor[sor.bg_id.isin(bg_trn)].groupby("ID").copia.apply(list).to_dict()
    for f, c in tr.items():
        if c != LARGE:
            continue
        if variant == "dupL":
            extra[f + OFF] = cache[f].copy()
            continue
        if f not in valid:
            continue
        j = valid[f][rng.integers(len(valid[f]))]
        y = mix["janelas"][(f, j)]
        assert len(y) == len(cache[f])
        extra[f + OFF] = y
    return {**cache, **extra}, {**tr, **{f: tr[f - OFF] for f in extra}}, len(extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="agitado,calmo,dupL")
    ap.add_argument("--splits", default="split_navio,split_id")
    args = ap.parse_args()
    print(f"Dispositivo: {DEVICE}", flush=True)
    folds = pd.read_csv(FOLDS_CSV, dtype={"Ship ID": str})
    with open(CACHES["lofar"][0], "rb") as fh:
        cache = pickle.load(fh)
    for v in args.variants.split(","):
        mix = None
        if v in ("agitado", "calmo"):
            with open(f"data/iara_processed/cache_lofar_mixbg_{v}.pkl", "rb") as fh:
                mix = pickle.load(fh)
        for cond in args.splits.split(","):
            name = f"{cond}_mlp_lofar_4c_aug{v}"
            w = ResultsWriter(name, script=__file__, params={
                **PARAMS["mlp"], "representation": "lofar", "split": cond, "num_classes": 4,
                "folds_csv": FOLDS_CSV, "seed": "torch/numpy = fold",
                "augmentation": v, "aug_regra": "1 versão extra por gravação LARGE de treino (ID + 100000)",
                "seed_aug": "1000*fold + índice da variante",
                "comparar_com": f"{cond}_mlp_lofar_4c"})
            cfg_path = os.path.join(w.exp_dir, "config.json")
            cfg = json.load(open(cfg_path, encoding="utf-8"))
            cfg["classes"] = CLASSES
            json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
            t0 = time.time()
            for k in range(10):
                tr = fold_files(folds, cond, k, "trn")
                te = fold_files(folds, cond, k, "test")
                c2, tr2, n = augment(cache, tr, v, k, mix)
                nL = sum(1 for c in tr.values() if c == LARGE)
                fte, yte, pred, scores = run_mlp(c2, tr2, te, k, num_classes=4)
                w.save_fold(k, fte, yte, pred, train_files=tr, scores=scores)
                print(f"  {name} fold {k}: {n}/{nL} LARGE de treino aumentados", flush=True)
            print(f"{name}: 10 folds em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
