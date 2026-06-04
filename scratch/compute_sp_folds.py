"""
Computes SP (geometric mean of class recalls), ACC and F1 (micro)
from the per-fold multiclass_test.csv files.

Usage:
    python scratch/compute_sp_folds.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("results/trainings/tests/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C0.5/eval")

sp_list, acc_list, f1_list = [], [], []

for fold_i in range(10):
    csv_path = BASE / f"fold_{fold_i}" / "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C0.5_multiclass_test.csv"
    if not csv_path.exists():
        print(f"fold_{fold_i}: arquivo não encontrado, pulando")
        continue

    df = pd.read_csv(csv_path)
    # expected columns: File, Target, Prediction

    targets     = df["Target"].values
    predictions = df["Prediction"].values
    classes     = sorted(df["Target"].unique())

    # --- SP (geometric mean of per-class recall) ---
    recalls = []
    for cls in classes:
        mask  = targets == cls
        tp    = np.sum(predictions[mask] == cls)
        total = np.sum(mask)
        recalls.append(tp / total if total > 0 else 0.0)

    sp  = np.prod(recalls) ** (1.0 / len(recalls)) * 100

    # --- ACC (overall accuracy, window-level) ---
    acc = np.mean(targets == predictions) * 100

    # --- F1 micro ---
    # For micro-F1: TP_total / (TP_total + 0.5*(FP_total + FN_total))
    # In multiclass micro: F1 = accuracy (when all samples are labeled)
    tp_total = np.sum(targets == predictions)
    f1_micro = tp_total / len(targets) * 100  # micro-F1 == accuracy for fully-labeled multiclass

    sp_list.append(sp)
    acc_list.append(acc)
    f1_list.append(f1_micro)

    print(f"fold_{fold_i}:  SP={sp:.2f}%  ACC={acc:.2f}%  F1={f1_micro:.2f}%  (n_classes={len(classes)}, recalls={[f'{r*100:.1f}' for r in recalls]})")

print()
print(f"Mean  SP  = {np.mean(sp_list):.2f} ± {np.std(sp_list):.2f}")
print(f"Mean  ACC = {np.mean(acc_list):.2f} ± {np.std(acc_list):.2f}")
print(f"Mean  F1  = {np.mean(f1_list):.2f} ± {np.std(f1_list):.2f}")
print(f"(over {len(sp_list)} folds)")
