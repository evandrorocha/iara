"""
DEMON + augmentation dos LARGE com fundo de MAR AGITADO, problema original de 4 classes (SMALL, MEDIUM, LARGE,
BACKGROUND). Experimento separado dos de LOFAR (pedido do usuário, 16/11/2026).

Entrada: DEMON do pipeline (data/iara_processed/demon_features via cache_fusion_mel_demon.pkl['demon']):
janelas de 10,24 s a cada 5,12 s, 253 frequências de modulação (0,4–25 Hz) do envelope da banda 1–2 kHz.
Modelo: mesma MLP h32-16 do NB29 (run_mlp: dropout 0,2, BatchNorm, 35 épocas, Adam 1e-3, wd 1e-3, peso por classe,
StandardScaler no treino). Folds: reports/split_navio/folds.csv, nos dois splits (split_navio, split_id).
Previsão por gravação = voto majoritário das janelas.

Variantes (cada gravação LARGE do TREINO ganha UMA versão extra, ID + 100000; teste sempre no DEMON original):
  base    : sem augmentation (referência DEMON nestes folds)
  dupL    : cópia idêntica do LARGE (controle: só repetição)
  agitado : DEMON do áudio LARGE + fundo real de mar agitado (estado >= 4, sem chuva) — cache_demon_mixbg_agitado.pkl
  calmo   : idem com fundo de mar calmo (estado <= 2, sem chuva) — controle
As misturas são as MESMAS do LOFAR (scripts/extrair_demon_mixbg_mar.py lê os sorteios do LOFAR) e a escolha da cópia
usa a mesma semente (1000·fold + índice da variante), então cada LARGE recebe a mesma cópia que no LOFAR.
Em agitado/calmo só valem cópias cujo fundo está no TREINO do fold (sem vazamento do BACKGROUND de teste).

Saída: results/trainings/multirep/demon_mar_<split>_mlp_demon_4c[_aug<variante>]
Rodar na raiz, no container iara-dev:  python scratch/demon_mar/aug_mar_demon.py --variants base,dupL,agitado,calmo
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
from eval_ship_split_5x2cv import FOLDS_CSV, fold_files  # noqa: E402
from eval_ship_split_nn_5x2cv import run_mlp, PARAMS, DEVICE  # noqa: E402

CLASSES = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}
LARGE, BG = 2, 3
VARIANTS = ["dupL", "agitado", "calmo"]  # ordem fixa: define a semente (igual a aug_mar.py / aug_mar_4c.py)
OFF = 100000
CACHE_DEMON = "data/iara_processed/cache_fusion_mel_demon.pkl"


def exp_name(cond, v):
    return f"demon_mar_{cond}_mlp_demon_4c" + ("" if v == "base" else f"_aug{v}")


def augment(cache, tr, variant, k, mix):
    if variant == "base":
        return cache, tr, 0
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
        assert len(y) == len(cache[f]), (f, j, len(y), len(cache[f]))
        extra[f + OFF] = y
    return {**cache, **extra}, {**tr, **{f: tr[f - OFF] for f in extra}}, len(extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,dupL,agitado,calmo")
    ap.add_argument("--splits", default="split_navio,split_id")
    args = ap.parse_args()
    print(f"Dispositivo: {DEVICE}", flush=True)
    folds = pd.read_csv(FOLDS_CSV, dtype={"Ship ID": str})
    with open(CACHE_DEMON, "rb") as fh:
        cache = pickle.load(fh)["demon"]
    for v in args.variants.split(","):
        mix = None
        if v in ("agitado", "calmo"):
            with open(f"data/iara_processed/cache_demon_mixbg_{v}.pkl", "rb") as fh:
                mix = pickle.load(fh)
        for cond in args.splits.split(","):
            name = exp_name(cond, v)
            w = ResultsWriter(name, script=__file__, params={
                **PARAMS["mlp"], "representation": "demon", "split": cond, "num_classes": 4,
                "demon": "banda 1–2 kHz, envelope |x|, 50 Hz, STFT 512 (10,24 s, passo 5,12 s), 253 bins 0,4–25 Hz",
                "folds_csv": FOLDS_CSV, "seed": "torch/numpy = fold", "augmentation": v,
                "aug_regra": "1 versão extra por gravação LARGE de treino (ID + 100000)" if v != "base" else None,
                "seed_aug": "1000*fold + índice da variante", "comparar_com": exp_name(cond, "base")})
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
