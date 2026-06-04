"""
Computes SP, ACC, F1 per fold from the g0.05 MEL experiment for comparison.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("results/trainings/tests/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0_g0.05/eval")
CLASS_NAMES = ["BACKGROUND", "SMALL", "MEDIUM", "LARGE"]

sp_list, acc_list = [], []

for fold_i in range(10):
    csv_path = BASE / f"fold_{fold_i}" / "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0_g0.05_multiclass_test.csv"
    if not csv_path.exists():
        print(f"fold_{fold_i}: not found")
        continue

    df = pd.read_csv(csv_path)
    targets = df["Target"].values
    predictions = df["Prediction"].values
    classes = sorted(df["Target"].unique())

    recalls = []
    for cls in classes:
        mask = targets == cls
        tp = np.sum(predictions[mask] == cls)
        recalls.append(tp / np.sum(mask) if np.sum(mask) > 0 else 0.0)

    sp  = np.prod(recalls) ** (1.0 / len(recalls)) * 100
    acc = np.mean(targets == predictions) * 100

    sp_list.append(sp)
    acc_list.append(acc)
    print(f"fold_{fold_i}:  SP={sp:.2f}%  ACC={acc:.2f}%  recalls={[f'{r*100:.1f}' for r in recalls]}")

print(f"\nMean SP  = {np.mean(sp_list):.2f} ± {np.std(sp_list):.2f}")
print(f"Mean ACC = {np.mean(acc_list):.2f} ± {np.std(acc_list):.2f}")
print(f"(over {len(sp_list)} folds)")
