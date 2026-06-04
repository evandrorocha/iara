import os
import pandas as pd
import collections
import numpy as np
import sys
sys.path.append(os.path.abspath('src'))

from iara.ml.metrics import Metric
from sklearn.metrics import confusion_matrix

EXP_NAME = "svm_nystroem_10000_lofar_elasticnet_l1r0.15"
BASE_DIR = f"results/trainings/tests/{EXP_NAME}/eval"

def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

completed_folds = []
for i in range(10):
    csv_path = os.path.join(BASE_DIR, f"fold_{i}", f"{EXP_NAME}_multiclass_test.csv")
    if os.path.exists(csv_path):
        completed_folds.append(i)

if not completed_folds:
    print("No completed folds found.")
    sys.exit(1)

print("=" * 70)
print(f"EVALUATING COMPLETED FOLDS FOR: {EXP_NAME}")
print("=" * 70)

win_accs, win_sps = [], []
aud_accs, aud_sps = [], []

class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}

for fold_i in completed_folds:
    csv_path = os.path.join(BASE_DIR, f"fold_{fold_i}", f"{EXP_NAME}_multiclass_test.csv")
    df = pd.read_csv(csv_path)
    
    # Window-level
    y_true_win = df['Target'].values
    y_pred_win = df['Prediction'].values
    win_acc = Metric.ACCURACY.compute(y_true_win, y_pred_win)
    win_sp = Metric.SP_INDEX.compute(y_true_win, y_pred_win)
    win_accs.append(win_acc)
    win_sps.append(win_sp)
    
    # Audio-level
    df_audio = df.groupby('File').agg({
        'Target': most_common_value,
        'Prediction': most_common_value
    }).reset_index()
    
    y_true_aud = df_audio['Target'].values
    y_pred_aud = df_audio['Prediction'].values
    aud_acc = Metric.ACCURACY.compute(y_true_aud, y_pred_aud)
    aud_sp = Metric.SP_INDEX.compute(y_true_aud, y_pred_aud)
    aud_accs.append(aud_acc)
    aud_sps.append(aud_sp)
    
    # Confusion Matrix for this fold
    labels = sorted(list(set(y_true_aud) | set(y_pred_aud)))
    cm = confusion_matrix(y_true_aud, y_pred_aud, labels=labels)
    class_names = [class_map.get(l, f"CLASS_{l}") for l in labels]
    
    print(f"\n---> FOLD {fold_i} RESULTS <---")
    print(f"  Window-level: ACC = {win_acc:.2f}% | SP = {win_sp:.2f}%")
    print(f"  Audio-level:  ACC = {aud_acc:.2f}% | SP = {aud_sp:.2f}%")
    print("\n  Confusion Matrix (Audio-level):")
    hdr = "    TRUE \\ PRED".ljust(16) + " | " + " | ".join(f"{name:<10}" for name in class_names) + " | TOTAL"
    print("    " + "-" * (len(hdr) - 4))
    print("    " + hdr)
    print("    " + "-" * (len(hdr) - 4))
    for idx, name in enumerate(class_names):
        row = cm[idx]
        total = sum(row)
        row_str = f"{name:<12} | " + " | ".join(f"{row[j]:<10}" for j in range(len(labels))) + f" | {total}"
        print("    " + row_str)
    print("=" * 70)

# Summary of completed folds
mean_win_acc, std_win_acc = np.mean(win_accs), np.std(win_accs)
mean_win_sp, std_win_sp = np.mean(win_sps), np.std(win_sps)
mean_aud_acc, std_aud_acc = np.mean(aud_accs), np.std(aud_accs)
mean_aud_sp, std_aud_sp = np.mean(aud_sps), np.std(aud_sps)

print("\n" + "=" * 70)
print(f"PARTIAL SUMMARY OVER {len(completed_folds)} FOLDS (Folds: {completed_folds})")
print("=" * 70)
print(f"Window-level Mean:")
print(f"  Accuracy (ACC): {mean_win_acc:.2f} ± {std_win_acc:.2f}%")
print(f"  SP Index (SP):  {mean_win_sp:.2f} ± {std_win_sp:.2f}%")
print("-" * 70)
print(f"Audio-level Mean:")
print(f"  Accuracy (ACC): {mean_aud_acc:.2f} ± {std_aud_acc:.2f}%")
print(f"  SP Index (SP):  {mean_aud_sp:.2f} ± {std_aud_sp:.2f}%")
print("=" * 70)
