import os
import sys
import pandas as pd
import numpy as np
import collections

# Ensure UTF-8 printing on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

models = {
    "MLP Mel (Baseline)": "mlp",
    "CNN Mel (Baseline)": "cnn",
    "SVM Nyström MEL (Best)": "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0",
    "SVM Nyström LOFAR (Best)": "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0"
}

def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

def compute_confusion_matrix(targets, predictions, num_classes=4):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, predictions):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm

class_names = ["SMALL", "MEDIUM", "LARGE", "BACKGROUND"]

print("=" * 80)
print("      COMPILING CONFUSION MATRICES & ACCURACIES FOR IARA EXPERIMENTS      ")
print("=" * 80)

for name, dir_name in models.items():
    base_dir = f"results/trainings/tests/{dir_name}"
    eval_dir = f"{base_dir}/eval"
    
    if not os.path.exists(eval_dir):
        print(f"\n[!] Error: Directory {eval_dir} not found. Skipping {name}.\n")
        continue
        
    all_targets = []
    all_predictions = []
    
    # We will compile across all 10 folds
    folds_found = 0
    for i_fold in range(10):
        fold_dir = os.path.join(eval_dir, f"fold_{i_fold}")
        if not os.path.exists(fold_dir):
            continue
            
        csv_files = [f for f in os.listdir(fold_dir) if f.endswith("_multiclass_test.csv")]
        if not csv_files:
            continue
            
        csv_path = os.path.join(fold_dir, csv_files[0])
        df = pd.read_csv(csv_path)
        
        # Audio aggregation: majority vote by File
        df_audio = df.groupby('File').agg({
            'Target': most_common_value,
            'Prediction': most_common_value
        }).reset_index()
        
        all_targets.extend(df_audio['Target'].values)
        all_predictions.extend(df_audio['Prediction'].values)
        folds_found += 1
        
    if folds_found == 0:
        print(f"\n[!] Error: No evaluation files found for {name} in {eval_dir}.\n")
        continue
        
    targets = np.array(all_targets)
    predictions = np.array(all_predictions)
    
    # Calculate simple accuracy
    correct = np.sum(targets == predictions)
    total = len(targets)
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    
    # Compute confusion matrix
    cm = compute_confusion_matrix(targets, predictions, num_classes=4)
    
    print("\n" + "=" * 70)
    print(f" MODEL: {name.upper()}")
    print(f" Compiled across {folds_found} folds | Total Audio Files: {total}")
    print(f" Simple Global Accuracy (by Audio): {accuracy:.2f}%")
    print("=" * 70)
    
    # Print Confusion Matrix Table
    true_pred_label = "TRUE \\ PRED"
    hdr = f"{true_pred_label:<14} | " + " | ".join(f"{c_name:<10}" for c_name in class_names) + " | TOTAL"
    print(hdr)
    print("-" * len(hdr))
    for i, c_name in enumerate(class_names):
        row = cm[i]
        row_total = sum(row)
        row_str = f"{c_name:<14} | " + " | ".join(f"{row[j]:<10}" for j in range(4)) + f" | {row_total:<5}"
        print(row_str)
    print("-" * len(hdr))
    
    # Print Recall / Accuracy per class
    print("\nPER-CLASS RECALL:")
    for i, c_name in enumerate(class_names):
        row = cm[i]
        row_total = sum(row)
        if row_total > 0:
            recall = (row[i] / row_total) * 100
            print(f"   * {c_name:<12}: {recall:.2f}% ({row[i]}/{row_total} correct)")
        else:
            print(f"   * {c_name:<12}: -- (No samples)")
    
    # Print Normalized Confusion Matrix (in %)
    print("\nNORMALIZED CONFUSION MATRIX (RECALL %):")
    print(hdr)
    print("-" * len(hdr))
    for i, c_name in enumerate(class_names):
        row = cm[i]
        row_total = sum(row)
        if row_total > 0:
            row_str = f"{c_name:<14} | " + " | ".join(f"{(row[j]/row_total)*100:<9.1f}%" for j in range(4)) + f" | 100.0%"
        else:
            row_str = f"{c_name:<14} | " + " | ".join(f"{0.0:<9.1f}%" for j in range(4)) + f" | 0.0%"
        print(row_str)
    print("-" * len(hdr))
    print("=" * 70)
