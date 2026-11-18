"""
Experimento mix_small_medium + augmentation dos LARGE com fundo de MAR AGITADO.

Motivação: os LARGE do IARA foram gravados com mar mais calmo que os navios < 100 m, e a MLP do mix_small_medium erra
mais LARGE quando o mar está agitado (81% -> 69% de acerto com CPA; 54% -> 45% sem CPA). A ideia é mostrar ao modelo
LARGE "sujos" por ruído de mar agitado.

Mesma MLP h32-16 e mesmos folds de scratch/mix_small_medium/treino_mlp.py. Cada gravação LARGE do TREINO ganha UMA
versão extra (ID + 100000, mesmo rótulo). O teste é sempre o LOFAR original. Variantes:
  dupL    : cópia idêntica do LARGE (controle: mais janelas LARGE e peso de classe menor, sem ruído)
  agitado : LOFAR do áudio LARGE + fundo real de mar agitado (estado >= 4, sem chuva) — cache_lofar_mixbg_agitado.pkl
  calmo   : idem com fundo de mar calmo (estado <= 2, sem chuva) — controle: é o mar agitado ou qualquer fundo?
Em agitado/calmo só valem cópias cujo fundo está no TREINO do fold (sem vazamento do BACKGROUND de teste).
Semente da escolha da cópia = 1000·fold + índice da variante.

Saída: results/trainings/multirep/mix_small_medium_<split>_mlp_lofar_aug<variante>
Rodar na raiz, no container iara-dev:  python scratch/mix_small_medium/aug_mar.py --variants dupL,agitado,calmo
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

CLASSES = {0: "MIX_SMALL_MEDIUM", 1: "LARGE", 2: "BACKGROUND"}
MAPA = {0: 0, 1: 0, 2: 1, 3: 2}
VARIANTS = ["dupL", "agitado", "calmo"]
OFF = 100000


def augment(cache, tr, variant, k, mix):
    rng = np.random.default_rng(1000 * k + VARIANTS.index(variant))
    extra = {}
    if mix is not None:
        bg_trn = {f for f, c in tr.items() if c == 2}
        sor = mix["sorteios"]
        valid = sor[sor.bg_id.isin(bg_trn)].groupby("ID").copia.apply(list).to_dict()
    for f, c in tr.items():
        if c != 1:
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
    ap.add_argument("--variants", default=",".join(VARIANTS))
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
            name = f"mix_small_medium_{cond}_mlp_lofar_aug{v}"
            w = ResultsWriter(name, script=__file__, params={
                **PARAMS["mlp"], "representation": "lofar", "split": cond, "experimento": "mix_small_medium",
                "classes_mix_small_medium": CLASSES, "num_classes": 3, "folds_csv": FOLDS_CSV,
                "augmentation": v, "aug_regra": "1 versão extra por gravação LARGE de treino (ID + 100000)",
                "seed_aug": "1000*fold + índice da variante",
                "comparar_com": f"mix_small_medium_{cond}_mlp_lofar"})
            cfg_path = os.path.join(w.exp_dir, "config.json")
            cfg = json.load(open(cfg_path, encoding="utf-8"))
            cfg["classes"] = CLASSES
            json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
            t0 = time.time()
            for k in range(10):
                tr = {f: MAPA[t] for f, t in fold_files(folds, cond, k, "trn").items()}
                te = {f: MAPA[t] for f, t in fold_files(folds, cond, k, "test").items()}
                c2, tr2, n = augment(cache, tr, v, k, mix)
                nL = sum(1 for c in tr.values() if c == 1)
                fte, yte, pred, scores = run_mlp(c2, tr2, te, k, num_classes=3)
                w.save_fold(k, fte, yte, pred, train_files=tr, scores=scores)
                print(f"  {name} fold {k}: {n}/{nL} LARGE de treino aumentados", flush=True)
            print(f"{name}: 10 folds em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
