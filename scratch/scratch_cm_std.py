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

# Class map
class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}

# Store normalized confusion matrices for each fold
fold_cms = []

for i_fold in range(10):
    fold_eval_dir = os.path.join(EVAL_DIR, f"fold_{i_fold}")
    csv_name = f"{EXP_NAME}_multiclass_test.csv"
    csv_path = os.path.join(fold_eval_dir, csv_name)
    
    if not os.path.exists(csv_path):
        continue
        
    df = pd.read_csv(csv_path)
    # Apply majority vote by audio
    df_audio = df.groupby('File').agg({
        'Target': most_common_value,
        'Prediction': most_common_value
    }).reset_index()
    
    targets = df_audio['Target'].values
    predictions = df_audio['Prediction'].values
    
    # 4x4 confusion matrix
    cm = confusion_matrix(targets, predictions, labels=[0, 1, 2, 3])
    
    # Normalize row-wise (convert counts to percentages)
    cm_norm = np.zeros((4, 4))
    for r in range(4):
        row_sum = sum(cm[r])
        if row_sum > 0:
            cm_norm[r] = (cm[r] / row_sum) * 100
        else:
            cm_norm[r] = 0.0
            
    fold_cms.append(cm_norm)

if not fold_cms:
    print("Error: No completed fold directories found.")
    exit(1)

# Stack them to shape (10, 4, 4)
stack_cms = np.stack(fold_cms, axis=0)

# Compute mean and standard deviation along fold axis (axis=0)
mean_cm = np.mean(stack_cms, axis=0)
std_cm = np.std(stack_cms, axis=0)

print("=" * 115)
print(f" MATRIZ DE CONFUSÃO NORMALIZADA COM DESVIO PADRÃO (%) — {EXP_NAME} ")
print("=" * 115)
true_pred = "True \\ Pred"
header_str = f"{true_pred:<14} | {'SMALL':<22} | {'MEDIUM':<22} | {'LARGE':<22} | {'BACKGROUND':<22}"
print(header_str)
print("-" * len(header_str))

for r in range(4):
    row_str = f"{class_map[r]:<14} | "
    cells = []
    for c in range(4):
        val_mean = mean_cm[r][c]
        val_std = std_cm[r][c]
        cells.append(f"{val_mean:6.2f}% ± {val_std:5.2f}%")
    row_str += " | ".join(cells)
    print(row_str)

print("=" * 115)
