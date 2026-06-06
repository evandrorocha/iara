import os
import collections
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from iara.ml.metrics import Metric

def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}

# Active experiment
exp_name = "svm_nystroem_4000_log_melgram_stratified_kmeans_weighted_ncw_S2.0_M1.0_L1.0_B1.0_elasticnet_l1r0.15"
label = "Melgram Stratified KMeans Weighted NCW S2/M1/L1/B1"

# We use relative paths because this will run inside the container's working directory /workspace/IARA
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
print(f" EXPERIMENTO ATIVO: {exp_name} ({label})")
print(f" Folds completos analisados: {folds}")
print("=" * 90)
print(f"{'Fold':<6} | {'Acurácia':<10} | {'SP Index':<10} | {'Recall S':<10} | {'Recall M':<10} | {'Recall L':<10} | {'Recall B':<10}")
print("-" * 90)

fold_accs = []
fold_sps = []
fold_recalls = {0: [], 1: [], 2: [], 3: []}

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
        fold_recalls[cls].append(recalls[cls])
        
    fold_accs.append(acc)
    fold_sps.append(sp)
    
    print(f"{i:<6} | {acc:7.2f}% | {sp:7.2f}% | {recalls[0]:8.1f}% | {recalls[1]:8.1f}% | {recalls[2]:8.1f}% | {recalls[3]:8.1f}%")
    
print("-" * 90)
if len(folds) > 0:
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    mean_sp = np.mean(fold_sps)
    std_sp = np.std(fold_sps)
    
    mean_r0 = np.mean(fold_recalls[0])
    std_r0 = np.std(fold_recalls[0])
    mean_r1 = np.mean(fold_recalls[1])
    std_r1 = np.std(fold_recalls[1])
    mean_r2 = np.mean(fold_recalls[2])
    std_r2 = np.std(fold_recalls[2])
    mean_r3 = np.mean(fold_recalls[3])
    std_r3 = np.std(fold_recalls[3])
    
    print(f"{'Média':<6} | {mean_acc:6.2f}±{std_acc:4.2f} | {mean_sp:6.2f}±{std_sp:4.2f} | {mean_r0:6.1f}±{std_r0:3.1f} | {mean_r1:6.1f}±{std_r1:3.1f} | {mean_r2:6.1f}±{std_r2:3.1f} | {mean_r3:6.1f}±{std_r3:3.1f}")
print("=" * 90)
