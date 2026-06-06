import os
import pandas as pd
import collections
import numpy as np
import sys
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.abspath('src'))
from iara.ml.metrics import Metric

EXP_NAME = "svm_nystroem_4000_hybrid_cascade_pretrained_svm_nystroem_6000_lofar_elasticnet_l1r0.15_elasticnet_l1r0.15"
csv_path = f"results/trainings/tests/{EXP_NAME}/eval/fold_0/{EXP_NAME}_multiclass_test.csv"

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    sys.exit(1)

df = pd.read_csv(csv_path)

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
print(f"RESULTS FOR FOLD 0 (Audio-level Majority Vote)")
print(f"Model: {EXP_NAME}")
print("=" * 60)
print(f"Accuracy (ACC): {aud_acc:.2f}%")
print(f"SP Index (SP):  {aud_sp:.2f}%")
print("-" * 60)

report = classification_report(y_true_aud, y_pred_aud, target_names=["SMALL", "MEDIUM", "LARGE", "BACKGROUND"], digits=4)
print("Classification Report:")
print(report)
print("=" * 60)

labels = [0, 1, 2, 3]
cm = confusion_matrix(y_true_aud, y_pred_aud, labels=labels)
class_names = ["SMALL", "MEDIUM", "LARGE", "BACKGROUND"]

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
