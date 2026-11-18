"""
SVM (Nystroem RBF m=6000 + SGD hinge, elasticnet l1 0,15) com o augmentation dos LARGE com fundo de mar,
no LOFAR e no DEMON, problema original de 4 classes. Pergunta: o SVM aproveita o augmentation que não ajudou a MLP?

Modelo: fit_predict de scratch/eval_ship_split_5x2cv.py (StandardScaler, Nystroem gamma = 1/(d·var), SGDClassifier
hinge, alpha = 1/(2n), class_weight balanced, early stopping) com n_components = 6000 (em vez de 2000).
Folds: reports/split_navio/folds.csv. Augmentation: o mesmo de aug_mar_demon.py / aug_mar_4c.py (mesmas cópias,
mesma semente 1000·fold + índice da variante; só fundos do treino do fold; teste sempre na representação original).

Saída: results/trainings/multirep/svm6000_mar_<split>_<rep>_4c[_aug<variante>]
Rodar na raiz, no container iara-dev:  python scratch/demon_mar/svm_mar.py --reps lofar,demon --splits split_navio
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from results_writer import ResultsWriter  # noqa: E402
from eval_ship_split_5x2cv import FOLDS_CSV, fold_files, matrices  # noqa: E402
from aug_mar_demon import augment, CLASSES, LARGE  # noqa: E402

M = 6000
CACHE = {"lofar": ("data/iara_processed/cache_lofar.pkl", None),
         "demon": ("data/iara_processed/cache_fusion_mel_demon.pkl", "demon")}


def exp_name(cond, rep, v):
    return f"svm{M}_mar_{cond}_{rep}_4c" + ("" if v == "base" else f"_aug{v}")


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler()
    Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
    ny = Nystroem(kernel="rbf", gamma=1.0 / (Xtr.shape[1] * Xtr.var()), n_components=M, random_state=42, n_jobs=8)
    Ztr = ny.fit_transform(Xtr).astype(np.float32)
    Zte = ny.transform(Xte).astype(np.float32)
    clf = SGDClassifier(loss="hinge", penalty="elasticnet", l1_ratio=0.15, alpha=1.0 / (2.0 * Xtr.shape[0]), max_iter=200,
                        tol=1e-3, early_stopping=True, n_iter_no_change=5, class_weight="balanced", random_state=42)
    clf.fit(Ztr, ytr)
    return clf.predict(Zte), clf.decision_function(Zte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="lofar,demon")
    ap.add_argument("--variants", default="base,dupL,agitado,calmo")
    ap.add_argument("--splits", default="split_navio,split_id")
    args = ap.parse_args()
    folds = pd.read_csv(FOLDS_CSV, dtype={"Ship ID": str})
    for cond in args.splits.split(","):
        for rep in args.reps.split(","):
            path, key = CACHE[rep]
            with open(path, "rb") as fh:
                cache = pickle.load(fh)
            cache = cache[key] if key else cache
            for v in args.variants.split(","):
                mix = None
                if v in ("agitado", "calmo"):
                    with open(f"data/iara_processed/cache_{rep}_mixbg_{v}.pkl", "rb") as fh:
                        mix = pickle.load(fh)
                name = exp_name(cond, rep, v)
                w = ResultsWriter(name, script=__file__, params={
                    "model": f"Nystroem(rbf, m={M}) + SGDClassifier(hinge, elasticnet l1 0.15, balanced)",
                    "representation": rep, "split": cond, "num_classes": 4, "folds_csv": FOLDS_CSV,
                    "augmentation": v, "seed_aug": "1000*fold + índice da variante"})
                cfg_path = os.path.join(w.exp_dir, "config.json")
                cfg = json.load(open(cfg_path, encoding="utf-8"))
                cfg["classes"] = CLASSES
                json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
                t0 = time.time()
                for k in range(10):
                    tr = fold_files(folds, cond, k, "trn")
                    te = fold_files(folds, cond, k, "test")
                    c2, tr2, n = augment(cache, tr, v, k, mix)
                    Xtr, ytr, _ = matrices(c2, tr2)
                    Xte, yte, fte = matrices(c2, te)
                    pred, scores = fit_predict(Xtr, ytr, Xte)
                    w.save_fold(k, fte, yte, pred, train_files=tr, scores=scores)
                    print(f"  {name} fold {k}: {n} LARGE aumentados ({time.time() - t0:.0f}s)", flush=True)
                print(f"{name}: 10 folds em {time.time() - t0:.0f}s", flush=True)
                del mix


if __name__ == "__main__":
    main()
