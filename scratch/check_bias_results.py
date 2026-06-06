import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import scipy.stats as scipy_stats

def most_common_value(series):
    return series.value_counts().idxmax()

def compute_sp_index(targets, predictions, labels=[0, 1, 2, 3]):
    cm = confusion_matrix(targets, predictions, labels=labels)
    row_sums = cm.sum(axis=1)
    detection_probabilities = []
    for i, r_sum in enumerate(row_sums):
        if r_sum > 0:
            detection_probabilities.append(cm[i, i] / r_sum)
        else:
            detection_probabilities.append(0.0)
    detection_probabilities = np.array(detection_probabilities)
    geometric_mean = scipy_stats.gmean(detection_probabilities)
    sp = np.sqrt(np.mean(detection_probabilities * geometric_mean)) * 100
    return sp

exp_name = "svm_nystroem_4000_log_melgram_cascade_1000_kmeans_elasticnet_l1r0.15"
eval_dir = os.path.abspath(f"results/trainings/tests/{exp_name}/eval")

folds = list(range(10))
biases = [0.0, 0.2, 0.5, 0.8]

print("=" * 110)
print(f" EVALUATION FOR CASCADED BIAS SHIFTS ON TEST SET")
print("=" * 110)
print(f"{'Bias':<6} | {'Fold':<6} | {'Accuracy':<10} | {'SP Index':<10} | {'Recall S':<10} | {'Recall M':<10} | {'Recall L':<10} | {'Recall B':<10}")
print("-" * 110)

for bias in biases:
    fold_accs = []
    fold_sps = []
    fold_recalls = {0: [], 1: [], 2: [], 3: []}
    
    available_folds = []
    for f in folds:
        fold_dir = os.path.join(eval_dir, f"fold_{f}")
        if bias == 0.0:
            csv_path = os.path.join(fold_dir, f"{exp_name}_multiclass_test.csv")
        else:
            bias_suffix = f"_cbs{bias:.4f}".rstrip('0').rstrip('.')
            csv_path = os.path.join(fold_dir, f"{exp_name}_multiclass_test{bias_suffix}.csv")
            
        if os.path.exists(csv_path):
            available_folds.append(f)
            df = pd.read_csv(csv_path)
            df_audio = df.groupby('File').agg({
                'Target': most_common_value,
                'Prediction': most_common_value
            }).reset_index()
            
            targets = df_audio['Target'].values
            predictions = df_audio['Prediction'].values
            
            acc = accuracy_score(targets, predictions) * 100
            sp = compute_sp_index(targets, predictions, labels=[0, 1, 2, 3])
            
            cm = confusion_matrix(targets, predictions, labels=[0, 1, 2, 3])
            recalls = {}
            for cls in [0, 1, 2, 3]:
                r_sum = cm[cls].sum()
                recalls[cls] = (cm[cls, cls] / r_sum * 100) if r_sum > 0 else 0.0
                fold_recalls[cls].append(recalls[cls])
                
            fold_accs.append(acc)
            fold_sps.append(sp)
            
            print(f"{bias:<6} | {f:<6} | {acc:7.2f}% | {sp:7.2f}% | {recalls[0]:8.1f}% | {recalls[1]:8.1f}% | {recalls[2]:8.1f}% | {recalls[3]:8.1f}%")
            
    if available_folds:
        print("-" * 110)
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
        print(f"{bias:<6} | {'MEAN':<6} | {mean_acc:6.2f}±{std_acc:4.2f} | {mean_sp:6.2f}±{std_sp:4.2f} | {mean_r0:6.1f}±{std_r0:3.1f} | {mean_r1:6.1f}±{std_r1:3.1f} | {mean_r2:6.1f}±{std_r2:3.1f} | {mean_r3:6.1f}±{std_r3:3.1f}")
        print("=" * 110)
