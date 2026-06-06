import os
import collections
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from iara.ml.metrics import Metric

def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

def get_fold_metrics(exp_name, fold_idx):
    base_dir = os.path.abspath(f"results/trainings/tests/{exp_name}")
    csv_path = os.path.join(base_dir, "eval", f"fold_{fold_idx}", f"{exp_name}_multiclass_test.csv")
    if not os.path.exists(csv_path):
        return None
    
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
        
    return acc, sp, recalls

# Experiments
exp_random = "svm_nystroem_4000_log_melgram_cascade_1000_elasticnet_l1r0.15"
exp_kmeans = "svm_nystroem_4000_log_melgram_cascade_1000_kmeans_elasticnet_l1r0.15"

res_random = get_fold_metrics(exp_random, 0)
res_kmeans = get_fold_metrics(exp_kmeans, 0)

print("\n" + "=" * 90)
print(" COMPARISON OF FOLD 0: RANDOM (WITHOUT KMEANS) vs KMEANS")
print("=" * 90)
print(f"{'Métrica':<15} | {'Melgram Cascade Random (sem KMeans)':<40} | {'Melgram Cascade KMeans':<30}")
print("-" * 90)

if res_random and res_kmeans:
    acc_r, sp_r, recs_r = res_random
    acc_k, sp_k, recs_k = res_kmeans
    
    print(f"{'Acurácia':<15} | {acc_r:38.2f}% | {acc_k:28.2f}%")
    print(f"{'Índice SP':<15} | {sp_r:38.2f}% | {sp_k:28.2f}%")
    print(f"{'Recall SMALL':<15} | {recs_r[0]:38.1f}% | {recs_k[0]:28.1f}%")
    print(f"{'Recall MEDIUM':<15} | {recs_r[1]:38.1f}% | {recs_k[1]:28.1f}%")
    print(f"{'Recall LARGE':<15} | {recs_r[2]:38.1f}% | {recs_k[2]:28.1f}%")
    print(f"{'Recall BACKGROUND':<15} | {recs_r[3]:38.1f}% | {recs_k[3]:28.1f}%")
else:
    print("Erro ao carregar dados de um dos experimentos.")
print("=" * 90)
