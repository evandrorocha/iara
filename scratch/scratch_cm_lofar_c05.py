import os
import sys
import pandas as pd
import numpy as np
import collections
from sklearn.metrics import confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

exp_name = "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C0.5"
base_dir = f"results/trainings/tests/{exp_name}"
eval_dir = f"{base_dir}/eval"

if not os.path.exists(eval_dir):
    print(f"Directory {eval_dir} not found!")
    sys.exit(1)

all_targets = []
all_predictions = []

folds_found = 0
for fold in range(10):
    fold_dir = os.path.join(eval_dir, f"fold_{fold}")
    csv_path = os.path.join(fold_dir, f"{exp_name}_multiclass_test.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Audio aggregation (majority vote)
        df_audio = df.groupby('File').agg({
            'Target': lambda x: collections.Counter(x).most_common(1)[0][0],
            'Prediction': lambda x: collections.Counter(x).most_common(1)[0][0]
        }).reset_index()
        
        all_targets.extend(df_audio['Target'].values)
        all_predictions.extend(df_audio['Prediction'].values)
        folds_found += 1

targets = np.array(all_targets)
predictions = np.array(all_predictions)

cm = confusion_matrix(targets, predictions, labels=[0, 1, 2, 3])
class_names = ["SMALL", "MEDIUM", "LARGE", "BACKGROUND"]

# Print raw matrix
print(f"=== MATRIZ DE CONFUSÃO BRUTA ({folds_found} folds) ===")
header_title = "True \\ Pred"
print(f"{header_title:<12} | " + " | ".join(f"{name:<10}" for name in class_names))
print("-" * 60)
for i, name in enumerate(class_names):
    print(f"{name:<12} | " + " | ".join(f"{cm[i, j]:<10}" for j in range(4)))
print("-" * 60)

# Print normalized matrix (recall %)
print("\n=== MATRIZ DE CONFUSÃO NORMALIZADA (RECALL %) ===")
print(f"{header_title:<12} | " + " | ".join(f"{name:<10}" for name in class_names))
print("-" * 60)
for i, name in enumerate(class_names):
    row_sum = sum(cm[i])
    if row_sum > 0:
        row_str = " | ".join(f"{(cm[i, j]/row_sum)*100:8.2f}%" for j in range(4))
    else:
        row_str = " | ".join(f"{0.0:8.2f}%" for j in range(4))
    print(f"{name:<12} | {row_str}")
print("-" * 60)

# Print per-class recall
print("\nRECALL POR CLASSE:")
for i, name in enumerate(class_names):
    row_sum = sum(cm[i])
    rec = (cm[i, i]/row_sum)*100 if row_sum > 0 else 0.0
    print(f"   * {name:<10}: {rec:.2f}% ({cm[i, i]}/{row_sum})")
