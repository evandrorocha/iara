"""
Quanto a sobreposição de navios entre treino e teste infla o desempenho?

O 5x2cv oficial (artigo e pipeline iara) separa por gravação (`ID`): o mesmo navio (`Ship ID`) pode ter gravações
no treino e no teste. Este script compara, com o MESMO código e as mesmas sementes, dois particionamentos 5x2cv:

  * split_id    -> StratifiedGroupKFold(2) com grupos = ID       (controle: mesma regra do oficial)
  * split_navio -> StratifiedGroupKFold(2) com grupos = Ship ID  (nenhum navio em treino e teste ao mesmo tempo)

Nas duas condições: 5 repetições (random_state = 0..4) × 2 folds; treino = uma metade, teste = a outra metade inteira
(sem validação, pois RF e SVM não fazem early stopping; o treino tem o mesmo tamanho do oficial, ~853 gravações).
BACKGROUND não tem navio: cada gravação de BACKGROUND é o seu próprio grupo nas duas condições.

Modelos (mesmos hiperparâmetros dos scripts de origem, sobre os caches de 5 s / 5,12 s):
  * RF: 100 árvores, sem max_depth, class_weight='balanced', random_state=42 (generate_rf_predictions_and_analysis.py)
  * SVM: StandardScaler + Nyström RBF (m=2000, gamma = 1/(d·var)) + SGD hinge elasticnet (C=2, l1_ratio=0.15)
         (save_svm_all_representations.py)
Representações: MEL (cache_fusion_mel_demon['mel']), LOFAR (cache_lofar.pkl), HHT (cache_hht.pkl).

Os folds das duas condições são diferentes, então o F-test 5x2cv pareado NÃO se aplica entre condições; a comparação
entre condições é a diferença de médias, com o desvio entre folds de cada uma.

Saída: results/trainings/multirep/{split_id,split_navio}_{rf,svm2000}_{mel,lofar,hht}_4c
       reports/split_navio/folds.csv (definição dos folds, versionada)
Rodar na raiz do repositório (no container iara-dev):  python scratch/eval_ship_split_5x2cv.py [--models rf,svm]
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_writer import ResultsWriter  # noqa: E402

META = "src/iara/dataset_info/iara.csv"
FOLDS_CSV = "reports/split_navio/folds.csv"
CACHES = {"mel": ("data/iara_processed/cache_fusion_mel_demon.pkl", "mel"),
          "lofar": ("data/iara_processed/cache_lofar.pkl", None),
          "hht": ("data/iara_processed/cache_hht.pkl", None)}


def metadata():
    m = pd.read_csv(META, dtype={"Ship ID": str})
    m = m[m.Dataset.isin(list("ABCDE"))].copy()  # 1.706 gravações de UO (F, G, H não entram no 5x2cv)
    m["File"] = m.ID.astype(int)
    m["Ship ID"] = m["Ship ID"].str.strip()
    length = pd.to_numeric(m.Length, errors="coerce")
    m["Target"] = np.select([length < 50, length < 100, length >= 100], [0, 1, 2], default=3)
    m.loc[m.Dataset == "E", "Target"] = 3
    return m.set_index("File")


def make_folds(meta):
    rows = []
    groups = {"split_id": meta.index.astype(str).values,
              "split_navio": np.where(meta["Ship ID"] == "-", "bg_" + meta.index.astype(str), meta["Ship ID"]).astype(str)}
    for cond, g in groups.items():
        for rep in range(5):
            sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=rep)
            for half, (_, te) in enumerate(sgkf.split(meta.index, meta.Target.values, g)):
                k = 2 * rep + half
                rows.append(pd.DataFrame({"condicao": cond, "fold": k, "File": meta.index[te], "Subset": "test"}))
    folds = pd.concat(rows, ignore_index=True)
    # treino do fold k = teste do fold "irmão" (mesma repetição, outra metade)
    trn = folds.copy()
    trn["fold"] = trn.fold.values ^ 1
    trn["Subset"] = "trn"
    folds = pd.concat([folds, trn], ignore_index=True).sort_values(["condicao", "fold", "Subset", "File"])
    folds["Target"] = meta.loc[folds.File, "Target"].values
    folds["Ship ID"] = meta.loc[folds.File, "Ship ID"].values
    return folds


def check_folds(folds):
    for (cond, k), d in folds.groupby(["condicao", "fold"]):
        tr, te = d[d.Subset == "trn"], d[d.Subset == "test"]
        assert not set(tr.File) & set(te.File), (cond, k)
        assert len(tr) + len(te) == 1706, (cond, k, len(tr), len(te))
        shared = set(tr["Ship ID"]) & set(te["Ship ID"]) - {"-"}
        if cond == "split_navio":
            assert not shared, f"{cond} fold {k}: {len(shared)} navios em treino e teste"
    summary = folds[folds.Subset == "test"].groupby(["condicao", "fold"]).agg(
        n_test=("File", "size"), **{f"n_{c}": ("Target", lambda x, c=c: int((x == c).sum())) for c in range(4)})
    print(summary.to_string())
    for cond in folds.condicao.unique():
        d = folds[folds.condicao == cond]
        rate = []
        for k in range(10):
            dk = d[d.fold == k]
            tr_ships = set(dk[dk.Subset == "trn"]["Ship ID"]) - {"-"}
            te = dk[(dk.Subset == "test") & (dk.Target < 3)]
            rate.append(te["Ship ID"].isin(tr_ships).mean())
        print(f"{cond}: gravações de navio no teste cujo navio também está no treino = {np.mean(rate) * 100:.1f}%")


def fold_files(folds, cond, k, subset):
    d = folds[(folds.condicao == cond) & (folds.fold == k) & (folds.Subset == subset)]
    return dict(zip(d.File.astype(int), d.Target.astype(int)))


def matrices(cache, files):
    X, y, f = [], [], []
    for fid, t in files.items():
        w = cache[fid]
        X.append(w)
        y += [t] * len(w)
        f += [fid] * len(w)
    return np.vstack(X), np.array(y, dtype=np.int64), np.array(f, dtype=np.int64)


def fit_predict(model, Xtr, ytr, Xte):
    if model == "rf":
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)
        return clf.predict(Xte), clf.predict_proba(Xte)
    sc = StandardScaler()
    Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
    ny = Nystroem(kernel="rbf", gamma=1.0 / (Xtr.shape[1] * Xtr.var()), n_components=2000, random_state=42, n_jobs=4)
    Ztr, Zte = ny.fit_transform(Xtr), ny.transform(Xte)
    clf = SGDClassifier(loss="hinge", penalty="elasticnet", l1_ratio=0.15, alpha=1.0 / (2.0 * Xtr.shape[0]), max_iter=200,
                        tol=1e-3, early_stopping=True, n_iter_no_change=5, class_weight="balanced", random_state=42)
    clf.fit(Ztr, ytr)
    return clf.predict(Zte), clf.decision_function(Zte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rf,svm")
    ap.add_argument("--reps", default="mel,lofar,hht")
    args = ap.parse_args()

    meta = metadata()
    assert len(meta) == 1706
    folds = make_folds(meta)
    check_folds(folds)
    os.makedirs(os.path.dirname(FOLDS_CSV), exist_ok=True)
    folds.to_csv(FOLDS_CSV, index=False)

    for rep in args.reps.split(","):
        path, key = CACHES[rep]
        with open(path, "rb") as fh:
            cache = pickle.load(fh)
        cache = cache[key] if key else cache
        for model in args.models.split(","):
            tag = "rf" if model == "rf" else "svm2000"
            for cond in ("split_id", "split_navio"):
                name = f"{cond}_{tag}_{rep}_4c"
                w = ResultsWriter(name, script=__file__, params={
                    "model": model, "representation": rep, "split": cond,
                    "grouping": "ID" if cond == "split_id" else "Ship ID (BACKGROUND: ID)",
                    "cv": "StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=rep), rep = 0..4",
                    "folds_csv": FOLDS_CSV, "num_classes": 4})
                t0 = time.time()
                for k in range(10):
                    tr, te = fold_files(folds, cond, k, "trn"), fold_files(folds, cond, k, "test")
                    Xtr, ytr, _ = matrices(cache, tr)
                    Xte, yte, fte = matrices(cache, te)
                    pred, scores = fit_predict(model, Xtr, ytr, Xte)
                    w.save_fold(k, fte, yte, pred, train_files=tr, scores=scores)
                print(f"{name}: 10 folds em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
