import os
import collections
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

EXP_NAME = "svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5"
BASE_DIR = os.path.abspath(f"results/trainings/tests/{EXP_NAME}")
EVAL_DIR = os.path.join(BASE_DIR, "eval")

def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

# Detect completed folds
completed_folds = []
for i_fold in range(10):
    fold_eval_dir = os.path.join(EVAL_DIR, f"fold_{i_fold}")
    csv_name = f"{EXP_NAME}_multiclass_test.csv"
    csv_path = os.path.join(fold_eval_dir, csv_name)
    if os.path.exists(csv_path):
        completed_folds.append(i_fold)

print("=" * 80)
print(f" PROGRESS REPORT FOR HYBRID EXPERIMENT: {EXP_NAME} ")
print(f" Completed Folds: {completed_folds} ")
print("=" * 80)

if not completed_folds:
    print("No folds have completed yet.")
    exit(0)

fold_accs = []
fold_sps = []

# Class map
class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}

# Store recall per class across folds
class_recalls = {cls: [] for cls in class_map.keys()}

all_targets = []
all_predictions = []

for i_fold in completed_folds:
    fold_eval_dir = os.path.join(EVAL_DIR, f"fold_{i_fold}")
    csv_name = f"{EXP_NAME}_multiclass_test.csv"
    csv_path = os.path.join(fold_eval_dir, csv_name)
    
    df = pd.read_csv(csv_path)
    # Apply majority vote by audio
    df_audio = df.groupby('File').agg({
        'Target': most_common_value,
        'Prediction': most_common_value
    }).reset_index()
    
    targets = df_audio['Target'].values
    predictions = df_audio['Prediction'].values
    
    all_targets.extend(targets)
    all_predictions.extend(predictions)
    
    # Calculate simple accuracy
    acc = np.mean(targets == predictions) * 100
    fold_accs.append(acc)
    
    # Calculate per-class recall for this fold
    cm = confusion_matrix(targets, predictions, labels=[0, 1, 2, 3])
    
    recalls = []
    for cls in range(4):
        total = sum(cm[cls])
        if total > 0:
            rec = (cm[cls][cls] / total) * 100
            class_recalls[cls].append(rec)
            recalls.append(rec)
        else:
            recalls.append(0.0)
            
    # Calculate SP Index (balanced accuracy) for this fold
    # SP = (mean(recalls) * GM(recalls)) ^ 0.5
    # where GM is geometric mean. Let's do standard SP Index:
    # SP = ( (1/K * sum(rec_k)) * (prod(rec_k))^(1/K) ) ^ 0.5
    mean_rec = np.mean(recalls)
    # Avoid zero product
    prod_rec = np.prod([max(r, 0.0001) for r in recalls])
    geom_rec = prod_rec ** (1.0 / 4)
    sp_index = np.sqrt(mean_rec * geom_rec)
    fold_sps.append(sp_index)
    
    print(f"Fold {i_fold}: ACC = {acc:.2f}%, SP Index = {sp_index:.2f}% | Recalls: " + 
          ", ".join(f"{class_map[c]}={cm[c][c]}/{sum(cm[c])} ({recalls[c]:.1f}%)" for c in range(4)))

print("\n" + "=" * 80)
print(" CONSOLIDATED STATISTICS FOR COMPLETED FOLDS ")
print("=" * 80)
print(f"  Mean Accuracy (ACC): {np.mean(fold_accs):.2f}% ± {np.std(fold_accs):.2f}%")
print(f"  Mean SP Index:       {np.mean(fold_sps):.2f}% ± {np.std(fold_sps):.2f}%")
print("-" * 80)
print("  Average Recall by Class:")
for cls, name in class_map.items():
    recs = class_recalls[cls]
    if recs:
        print(f"   * {name:<10}: {np.mean(recs):.2f}% ± {np.std(recs):.2f}%")
    else:
        print(f"   * {name:<10}: --")

print("=" * 80)

# Print cumulative confusion matrix for completed folds
cm_cum = confusion_matrix(all_targets, all_predictions, labels=[0, 1, 2, 3])
print("  CUMULATIVE CONFUSION MATRIX (BY AUDIO):")
header_str = f"True \\ Pred  | {'SMALL':<8} | {'MEDIUM':<8} | {'LARGE':<8} | {'BACKGR':<8} | TOTAL"
print(header_str)
print("-" * len(header_str))
for cls in range(4):
    row = cm_cum[cls]
    total = sum(row)
    print(f"{class_map[cls]:<12} | {row[0]:<8} | {row[1]:<8} | {row[2]:<8} | {row[3]:<8} | {total}")
print("=" * 80)
