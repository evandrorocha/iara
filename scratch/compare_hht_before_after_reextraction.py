"""
Compara cada experimento HHT antes e depois da re-extração do HHT com uma configuração única de EMD.

Antes: results/trainings/_backup_hht_iter_misto_20261015/<exp> (cache HHT com dois MAX_ITERATION).
Depois: results/trainings/multirep/<exp> (cache re-extraído, 8 IMFs, MAX_ITERATION=30).
Para cada experimento e modo (4c ou 3c): SP e recalls (média ± desvio nos 10 folds) e o F-test 5x2cv
(Alpaydin) entre as duas versões. Atenção: modelos sem semente fixa (MLP "noseed", BiGRU, CNN 2D de
eval_hht_models_5x2cv.py) também mudam por aleatoriedade do treino, que o F-test não separa da feature.

Uso (no container, na raiz): python scratch/compare_hht_before_after_reextraction.py [saida.csv]
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks", "src"))
import iara_results as R  # noqa: E402

BACKUP = os.path.join(R.ROOT, "results", "trainings", "_backup_hht_iter_misto_20261015")
METRICS = ["SP", "R_SMALL", "R_MEDIUM", "R_LARGE", "R_BACKGROUND"]


def fold_metrics_from(search_dirs, name, mode):
    R.SEARCH_DIRS[:] = search_dirs
    for fn in (R.fold_metrics, R.file_predictions):
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
    return R.fold_metrics(name, mode)


def main(out_csv=None):
    default_dirs = list(R.SEARCH_DIRS)
    rows = []
    for exp in sorted(os.listdir(BACKUP)):
        mode = "3c" if exp.endswith("_3c") else "4c"
        old = fold_metrics_from([BACKUP], exp, mode)
        try:
            new = fold_metrics_from(default_dirs, exp, mode)
        except FileNotFoundError as e:
            print("SEM VERSÃO NOVA:", exp, e)
            continue
        row = {"experimento": exp, "modo": mode}
        for m in METRICS:
            if m not in old or (mode == "3c" and m == "R_BACKGROUND"):
                continue
            row[f"{m} antes"] = round(old[m].mean(), 2)
            row[f"{m} depois"] = round(new[m].mean(), 2)
            row[f"Δ {m}"] = round(new[m].mean() - old[m].mean(), 2)
            row[f"p {m}"] = round(R.f_test_5x2(new[m].values, old[m].values)[1], 4)
        row["SP std antes"] = round(old["SP"].std(ddof=0), 2)
        row["SP std depois"] = round(new["SP"].std(ddof=0), 2)
        rows.append(row)
    R.SEARCH_DIRS[:] = default_dirs
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = ["experimento", "modo", "SP antes", "SP depois", "Δ SP", "p SP", "R_SMALL antes", "R_SMALL depois", "p R_SMALL",
            "R_BACKGROUND antes", "R_BACKGROUND depois", "p R_BACKGROUND"]
    print(df[[c for c in cols if c in df]].to_string(index=False))
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
