"""
Atualiza só a entrada 'hht' de data/iara_processed/{rf,mlp,svm,cnn}_all_representations_predictions.pkl
a partir das predições por janela re-treinadas depois da re-extração do HHT (8 IMFs, MAX_ITERATION=30).

As demais entradas (mel, lofar, demon, targets) ficam como estão. O voto por arquivo segue a convenção
desses pickles (np.argmax(np.bincount(...)): empate -> menor classe), a mesma de save_svm_all_representations.py
e generate_*_predictions_and_analysis.py. Confere que os arquivos de cada fold batem com os do pickle.

Uso (no container, na raiz): python scratch/update_hht_in_all_representations_pickles.py
"""
import os
import pickle

import numpy as np
import pandas as pd

PROC = "data/iara_processed"
MULTIREP = "results/trainings/multirep"
EXPS = {
    "rf": "rf_hht_n100_balanced_4c",
    "mlp": "mlp_hht_h32-16_e30_4c",
    "svm": "svm_nystroem_2000_hht_C2.0_elasticnet_l1r0.15_4c",
    "cnn": "cnn2d_hht_w32_b64_e20_4c",
}


def file_votes(exp, fold):
    df = pd.read_csv(os.path.join(MULTIREP, exp, "eval", f"fold_{fold}", f"{exp}_multiclass_test.csv"))
    return {int(f): int(np.argmax(np.bincount(g.Prediction.values, minlength=4))) for f, g in df.groupby("File")}


for fam, exp in EXPS.items():
    path = os.path.join(PROC, f"{fam}_all_representations_predictions.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    changed = 0
    for fold in range(10):
        new = file_votes(exp, fold)
        old = {int(k): int(v) for k, v in d["hht"][fold].items()}
        assert set(new) == set(old), f"{fam} fold {fold}: arquivos diferentes"
        changed += sum(new[k] != old[k] for k in new)
        d["hht"][fold] = new
    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{fam}: entrada 'hht' atualizada a partir de {exp} ({changed} predições arquivo x fold mudaram)")
