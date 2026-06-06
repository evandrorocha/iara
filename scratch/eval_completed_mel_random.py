import os
import collections
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from iara.ml.metrics import Metric

def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}

# Target experiment
exp_name = "svm_nystroem_4000_log_melgram_cascade_1000_elasticnet_l1r0.15"
label = "Melgram Cascade Random (without KMeans)"

base_dir = os.path.abspath(f"results/trainings/tests/{exp_name}")
eval_dir = os.path.join(base_dir, "eval")

if not os.path.exists(eval_dir):
    print(f"\nExperimento {exp_name} ({label}) não possui diretório de avaliação.")
    exit(1)
    
folds = []
for i in range(10):
    fold_dir = os.path.join(eval_dir, f"fold_{i}")
    csv_path = os.path.join(fold_dir, f"{exp_name}_multiclass_test.csv")
    if os.path.exists(csv_path):
        folds.append(i)
        
print("\n" + "=" * 90)
print(f" EXPERIMENTO: {exp_name} ({label})")
print(f" Folds completos analisados: {folds}")
print("=" * 90)
print(f"{'Fold':<6} | {'Acurácia':<10} | {'SP Index':<10} | {'Recall S':<10} | {'Recall M':<10} | {'Recall L':<10} | {'Recall B':<10}")
print("-" * 90)

for i in folds:
    fold_dir = os.path.join(eval_dir, f"fold_{i}")
    csv_path = os.path.join(fold_dir, f"{exp_name}_multiclass_test.csv")
    
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
        
    print(f"{i:<6} | {acc:7.2f}% | {sp:7.2f}% | {recalls[0]:8.1f}% | {recalls[1]:8.1f}% | {recalls[2]:8.1f}% | {recalls[3]:8.1f}%")
print("=" * 90)
