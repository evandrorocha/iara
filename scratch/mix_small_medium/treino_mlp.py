"""
Experimento mix_small_medium: SMALL e MEDIUM (navios < 100 m) viram uma única classe, MIX_SMALL_MEDIUM.

Classes (rótulo gravado nos CSVs de resultado):
  0 = MIX_SMALL_MEDIUM  (SMALL + MEDIUM do IARA, comprimento < 100 m)
  1 = LARGE             (>= 100 m)
  2 = BACKGROUND        (subconjunto E)

Modelo: MLP h32-16 treinada DO ZERO para este experimento, com a mesma arquitetura e hiperparâmetros da MLP LOFAR do
NB29 (run_mlp de scratch/eval_ship_split_nn_5x2cv.py: dropout 0,2, BatchNorm, 35 épocas, Adam lr 1e-3, wd 1e-3,
CrossEntropy com peso por classe, StandardScaler no treino). Única mudança: as classes.
Folds: os mesmos do NB29 (reports/split_navio/folds.csv), nos dois splits:
  split_id    = divisão por gravação (a do artigo)
  split_navio = divisão por navio (nenhum navio em treino e teste ao mesmo tempo)

Saída: results/trainings/multirep/mix_small_medium_<split>_mlp_lofar
Rodar na raiz, no container iara-dev (GPU):  python scratch/mix_small_medium/treino_mlp.py
"""
import json
import os
import pickle
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from results_writer import ResultsWriter  # noqa: E402
from eval_ship_split_5x2cv import CACHES, FOLDS_CSV, fold_files  # noqa: E402
from eval_ship_split_nn_5x2cv import run_mlp, PARAMS, DEVICE  # noqa: E402

CLASSES = {0: "MIX_SMALL_MEDIUM", 1: "LARGE", 2: "BACKGROUND"}
MAPA = {0: 0, 1: 0, 2: 1, 3: 2}  # alvo do IARA (SMALL, MEDIUM, LARGE, BACKGROUND) -> mix_small_medium


def main():
    print(f"Dispositivo: {DEVICE}", flush=True)
    folds = pd.read_csv(FOLDS_CSV, dtype={"Ship ID": str})
    with open(CACHES["lofar"][0], "rb") as fh:
        cache = pickle.load(fh)
    for cond in ("split_navio", "split_id"):
        name = f"mix_small_medium_{cond}_mlp_lofar"
        w = ResultsWriter(name, script=__file__, params={
            **PARAMS["mlp"], "representation": "lofar", "split": cond, "experimento": "mix_small_medium",
            "classes_mix_small_medium": CLASSES, "mapa_alvo_iara": "SMALL,MEDIUM->0; LARGE->1; BACKGROUND->2",
            "folds_csv": FOLDS_CSV, "seed": "torch/numpy = fold", "num_classes": 3,
            "comparar_com": f"{cond}_mlp_lofar_4c (mesma MLP, classes do IARA)"})
        cfg_path = os.path.join(w.exp_dir, "config.json")
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        cfg["classes"] = CLASSES
        json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
        t0 = time.time()
        for k in range(10):
            tr = {f: MAPA[t] for f, t in fold_files(folds, cond, k, "trn").items()}
            te = {f: MAPA[t] for f, t in fold_files(folds, cond, k, "test").items()}
            fte, yte, pred, scores = run_mlp(cache, tr, te, k, num_classes=3)
            w.save_fold(k, fte, yte, pred, train_files=tr, scores=scores)
        print(f"{name}: 10 folds em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
