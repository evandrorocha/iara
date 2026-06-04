import os
import pandas as pd
import collections
import numpy as np

import sys
sys.path.append(os.path.abspath('src'))

from iara.ml.metrics import Metric

EXP_NAME = "svm_nystroem_10000_lofar_elasticnet_l1r0.15"
csv_path = f"results/trainings/tests/{EXP_NAME}/eval/fold_1/{EXP_NAME}_multiclass_test.csv"

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    sys.exit(1)

df = pd.read_csv(csv_path)

# 1. Window-level Evaluation
y_true_win = df['Target'].values
y_pred_win = df['Prediction'].values

win_acc = Metric.ACCURACY.compute(y_true_win, y_pred_win)
win_sp = Metric.SP_INDEX.compute(y_true_win, y_pred_win)

# 2. Audio-level Evaluation (majority voting)
def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

df_audio = df.groupby('File').agg({
    'Target': most_common_value,
    'Prediction': most_common_value
}).reset_index()

y_true_aud = df_audio['Target'].values
y_pred_aud = df_audio['Prediction'].values

aud_acc = Metric.ACCURACY.compute(y_true_aud, y_pred_aud)
aud_sp = Metric.SP_INDEX.compute(y_true_aud, y_pred_aud)

print("=" * 60)
print(f"RESULTS FOR FOLD 1 - {EXP_NAME}")
print("=" * 60)
print(f"Window-level:")
print(f"  Accuracy (ACC): {win_acc:.2f}%")
print(f"  SP Index (SP):  {win_sp:.2f}%")
print("-" * 60)
print(f"Audio-level (Majority Vote):")
print(f"  Accuracy (ACC): {aud_acc:.2f}%")
print(f"  SP Index (SP):  {aud_sp:.2f}%")
print("=" * 60)

# Print confusion matrix for Audio-level
from sklearn.metrics import confusion_matrix
labels = sorted(list(set(y_true_aud) | set(y_pred_aud)))
cm = confusion_matrix(y_true_aud, y_pred_aud, labels=labels)

class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}
class_names = [class_map.get(l, f"CLASS_{l}") for l in labels]

print("\nConfusion Matrix (Audio-level):")
hdr = "TRUE \\ PRED".ljust(12) + " | " + " | ".join(f"{name:<10}" for name in class_names) + " | TOTAL"
print(hdr)
print("-" * len(hdr))
for i, name in enumerate(class_names):
    row = cm[i]
    total = sum(row)
    row_str = f"{name:<12} | " + " | ".join(f"{row[j]:<10}" for j in range(len(labels))) + f" | {total}"
    print(row_str)
print("=" * 60)
