import os
import collections
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from iara.ml.metrics import Metric

def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

exp_name = "svm_nystroem_4000_hybrid_cascade_pretrained_svm_nystroem_6000_lofar_elasticnet_l1r0.15_elasticnet_l1r0.15"
base_dir = os.path.abspath(f"results/trainings/tests/{exp_name}")
csv_path = os.path.join(base_dir, "eval", "fold_0", f"{exp_name}_multiclass_test.csv")

if not os.path.exists(csv_path):
    print("CSV de teste não encontrado.")
    exit(1)

df = pd.read_csv(csv_path)
df_audio = df.groupby('File').agg({
    'Target': most_common_value,
    'Prediction': most_common_value
}).reset_index()

targets = df_audio['Target'].values
predictions = df_audio['Prediction'].values

acc = Metric.ACCURACY.compute(targets, predictions)
sp = Metric.SP_INDEX.compute(targets, predictions)

cm = confusion_matrix(targets, predictions, labels=[0, 1, 2, 3])
recalls = {}
for cls in [0, 1, 2, 3]:
    r_sum = cm[cls].sum()
    recalls[cls] = (cm[cls, cls] / r_sum * 100) if r_sum > 0 else 0.0

print("\n" + "=" * 90)
print(f" EXPERIMENTO: {exp_name}")
print("=" * 90)
print(f"Acurácia Global : {acc:.2f}%")
print(f"Índice SP       : {sp:.2f}%")
print(f"Recall SMALL    : {recalls[0]:.1f}%")
print(f"Recall MEDIUM   : {recalls[1]:.1f}%")
print(f"Recall LARGE    : {recalls[2]:.1f}%")
print(f"Recall BACKGROUND: {recalls[3]:.1f}%")
print("=" * 90)
