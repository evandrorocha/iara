"""
Data augmentation no LOFAR com SPLIT POR NAVIO para RF, SVM e BiGRU (complemento de aug_ship_split.py, que usa o MLP).

Os modelos e hiperparâmetros são exatamente os do NB29 (scratch/eval_ship_split_5x2cv.py: RF 100 árvores e SVM Nyström
m=2000 + SGD; scratch/eval_ship_split_nn_5x2cv.py: BiGRU + atenção, sequências 15/5, 15 épocas), com os mesmos folds
(reports/split_navio/folds.csv). Como RF e SVM não treinam por lotes, a augmentation é feita ANTES do treino:
cada gravação de treino ganha 1 versão aumentada, que entra como uma gravação extra (ID + 100000) com o mesmo rótulo.
O teste é sempre o LOFAR original.

Variantes (--variants):
  base     : só as gravações originais (igual ao NB29)
  dup      : original + cópia idêntica (controle: treino 2x maior sem nenhuma alteração)
  stretch  : eixo de frequência esticado por s ~ U(0,92; 1,08), um s por gravação
  mask     : 2 faixas de até 48 bins zeradas, as mesmas em todas as janelas da gravação
  mixnavio : mistura janela a janela com outra gravação da MESMA classe e de OUTRO navio (lambda = max(u, 1-u))
  ruido    : ruído gaussiano com desvio 0,3 × desvio da faixa no treino (equivale ao sigma 0,3 padronizado do MLP)
  bg       : uma cópia do lofar_mixbg (BACKGROUND real misturado ao áudio) cujo fundo está no TREINO do fold;
             gravações de BACKGROUND e gravações sem cópia válida não ganham versão extra
Semente da augmentation = 1000·fold + índice da variante.

Saída: results/trainings/multirep/augnavio_{rf,svm2000,bigru}_lofar_<variante>_4c
Rodar na raiz (container iara-dev):  python scratch/apres_small_medium/aug_ship_split_modelos.py --models rf,svm,bigru
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from results_writer import ResultsWriter  # noqa: E402
from eval_ship_split_5x2cv import CACHES, FOLDS_CSV, fold_files, matrices, metadata, fit_predict  # noqa: E402

VARIANTS = ["base", "dup", "stretch", "mask", "mixnavio", "ruido", "bg"]
OFF = 100000
MIX_PATH = "data/iara_processed/cache_lofar_mixbg.pkl"


def stretch_file(x, rng):
    s = rng.uniform(0.92, 1.08)
    d = x.shape[1]
    pos = np.arange(d) / s
    out = np.stack([np.interp(pos, np.arange(d), row, right=0.0) for row in x])
    return out.astype(np.float32)


def mask_file(x, rng, n=2, width=48):
    x = x.copy()
    d = x.shape[1]
    for _ in range(n):
        w = rng.integers(0, width + 1)
        f0 = rng.integers(0, d - w + 1)
        x[:, f0:f0 + w] = 0.0
    return x


def augment(cache, tr, variant, k, ship_of, mix=None):
    """Devolve (cache_aumentado, tr_aumentado) com as versões extras das gravações de treino."""
    if variant == "base":
        return cache, tr
    rng = np.random.default_rng(1000 * k + VARIANTS.index(variant))
    extra = {}
    if variant == "ruido":
        Xtr, _, _ = matrices(cache, tr)
        sd = Xtr.std(0).astype(np.float32)
    if variant == "mixnavio":
        by_class = {}
        for f, c in tr.items():
            by_class.setdefault(c, []).append(f)
    if variant == "bg":
        bg_trn = {f for f, c in tr.items() if c == 3}
        sor = mix["sorteios"]
        valid = sor[sor.bg_id.isin(bg_trn)].groupby("ID").copia.apply(list).to_dict()
    for f, c in tr.items():
        x = cache[f]
        if variant == "dup":
            y = x.copy()
        elif variant == "stretch":
            y = stretch_file(x, rng)
        elif variant == "mask":
            y = mask_file(x, rng)
        elif variant == "ruido":
            y = (x + 0.3 * sd * rng.standard_normal(x.shape)).astype(np.float32)
        elif variant == "mixnavio":
            cand = [g for g in by_class[c] if ship_of[g] != ship_of[f]]
            if not cand:
                continue
            p = cache[cand[rng.integers(len(cand))]]
            p = p[np.arange(len(x)) % len(p)]
            u = rng.uniform(size=(len(x), 1))
            lam = np.maximum(u, 1 - u)
            y = (lam * x + (1 - lam) * p).astype(np.float32)
        elif variant == "bg":
            if c == 3 or f not in valid:
                continue
            j = valid[f][rng.integers(len(valid[f]))]
            y = mix["janelas"][(f, j)]
            assert len(y) == len(x)
        else:
            raise ValueError(variant)
        extra[f + OFF] = y
    cache2 = {**cache, **extra}
    tr2 = {**tr, **{f: tr[f - OFF] for f in extra}}
    return cache2, tr2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rf,svm,bigru")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    folds = pd.read_csv(FOLDS_CSV, dtype={"Ship ID": str})
    meta = metadata()
    ship_of = {f: (s if s != "-" else f"bg_{f}") for f, s in meta["Ship ID"].items()}
    with open(CACHES["lofar"][0], "rb") as fh:
        cache = pickle.load(fh)
    variants = args.variants.split(",")
    mix = None
    if "bg" in variants:
        with open(MIX_PATH, "rb") as fh:
            mix = pickle.load(fh)
    for model in args.models.split(","):
        if model == "bigru":
            from eval_ship_split_nn_5x2cv import run_bigru, DEVICE
            print(f"BiGRU em {DEVICE}", flush=True)
        name_m = {"rf": "rf", "svm": "svm2000", "bigru": "bigru"}[model]
        for v in variants:
            w = ResultsWriter(f"augnavio_{name_m}_lofar_{v}_4c", script=__file__, params={
                "modelo": model, "base_nb29": f"split_navio_{name_m}_lofar_4c", "variant": v, "split": "split_navio",
                "folds_csv": FOLDS_CSV, "augmentation": "1 versão extra por gravação de treino (ID + 100000)",
                "seed_aug": "1000*fold + índice da variante"})
            t0 = time.time()
            for k in range(10):
                tr, te = fold_files(folds, "split_navio", k, "trn"), fold_files(folds, "split_navio", k, "test")
                c2, tr2 = augment(cache, tr, v, k, ship_of, mix)
                if model == "bigru":
                    fte, yte, pred, scores = run_bigru(c2, tr2, te, k)
                else:
                    Xtr, ytr, _ = matrices(c2, tr2)
                    Xte, yte, fte = matrices(cache, te)
                    pred, scores = fit_predict(model, Xtr, ytr, Xte)
                w.save_fold(k, fte, yte, pred, train_files=tr, scores=scores)
            print(f"{model} {v}: 10 folds em {time.time() - t0:.0f}s (treino com {len(tr2)} gravações no último fold)",
                  flush=True)


if __name__ == "__main__":
    main()
