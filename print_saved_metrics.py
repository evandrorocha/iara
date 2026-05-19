import os
import collections
import pandas as pd
import numpy as np

import iara.ml.metrics as iara_metrics

import sys

def get_default_experiment():
    test_dir = os.path.abspath("results/trainings/tests")
    if not os.path.exists(test_dir):
        return "svm_nystroem_1000_lofar"
    subdirs = []
    for item in os.listdir(test_dir):
        item_path = os.path.join(test_dir, item)
        if os.path.isdir(item_path) and not item.startswith("2026") and item not in ("cnn", "mlp"):
            subdirs.append((item, os.path.getmtime(item_path)))
    if subdirs:
        subdirs.sort(key=lambda x: x[1], reverse=True)
        return subdirs[0][0]
    return "svm_nystroem_1000_lofar"

# Path to the active experiment folder
if len(sys.argv) > 1:
    EXP_NAME = sys.argv[1]
else:
    EXP_NAME = get_default_experiment()

BASE_DIR = os.path.abspath(f"results/trainings/tests/{EXP_NAME}")
EVAL_DIR = os.path.join(BASE_DIR, "eval")

def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

def compile_metrics(eval_subset="test", eval_strategy="by_audio"):
    grid = iara_metrics.GridCompiler()
    
    # We will compile all 10 folds
    for i_fold in range(10):
        fold_eval_dir = os.path.join(EVAL_DIR, f"fold_{i_fold}")
        csv_name = f"{EXP_NAME}_multiclass_{eval_subset}.csv"
        csv_path = os.path.join(fold_eval_dir, csv_name)
        
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} does not exist!")
            return None
            
        df = pd.read_csv(csv_path)
        
        # Apply majority vote if BY_AUDIO
        if eval_strategy == "by_audio":
            df = df.groupby('File').agg({
                'Target': most_common_value,
                'Prediction': most_common_value
            }).reset_index()
            
        targets = df['Target'].values
        predictions = df['Prediction'].values
        
        # Add to the GridCompiler
        grid.add(
            params={
                'eval_strategy': eval_strategy,
                'eval_subset': eval_subset,
            },
            i_fold=i_fold,
            target=targets,
            prediction=predictions
        )
        
    return grid

from sklearn.metrics import confusion_matrix

def print_confusion_matrix_and_class_stats(eval_subset="test", eval_strategy="by_audio"):
    all_targets = []
    all_predictions = []
    
    for i_fold in range(10):
        fold_eval_dir = os.path.join(EVAL_DIR, f"fold_{i_fold}")
        csv_name = f"{EXP_NAME}_multiclass_{eval_subset}.csv"
        csv_path = os.path.join(fold_eval_dir, csv_name)
        
        if not os.path.exists(csv_path):
            return
            
        df = pd.read_csv(csv_path)
        if eval_strategy == "by_audio":
            df = df.groupby('File').agg({
                'Target': most_common_value,
                'Prediction': most_common_value
            }).reset_index()
            
        all_targets.extend(df['Target'].values)
        all_predictions.extend(df['Prediction'].values)
        
    targets = np.array(all_targets)
    predictions = np.array(all_predictions)
    
    # Support both 3-class and 4-class collections by using actual labels present
    unique_labels = sorted(list(set(targets) | set(predictions)))
    cm = confusion_matrix(targets, predictions, labels=unique_labels)
    
    class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}
    class_names = [class_map.get(lbl, f"CLASS_{lbl}") for lbl in unique_labels]
    
    print("\n" + "=" * 70)
    print(f"      CONFUSION MATRIX & PER-CLASS RECALL ({eval_strategy.upper()})      ")
    print("=" * 70)
    
    # Print Confusion Matrix Header
    true_pred_label = "TRUE \\ PRED"
    hdr = f"{true_pred_label:<14} | " + " | ".join(f"{name:<8}" for name in class_names) + " | TOTAL"
    print(hdr)
    print("-" * len(hdr))
    for i, name in enumerate(class_names):
        row = cm[i]
        total = sum(row)
        row_str = f"{name:<14} | " + " | ".join(f"{row[j]:<8}" for j in range(len(unique_labels))) + f" | {total:<5}"
        print(row_str)
    print("-" * len(hdr))
    
    # Print Recall / Accuracy per class
    print("\nPER-CLASS ACCURACY (RECALL):")
    for i, name in enumerate(class_names):
        row = cm[i]
        total = sum(row)
        if total > 0:
            recall = (row[i] / total) * 100
            print(f"   * {name:<12}: {recall:.1f}% ({row[i]}/{total} correct)")
        else:
            print(f"   * {name:<12}: -- (No samples)")
    print("=" * 70)

if __name__ == "__main__":
    print("=" * 70)
    print(f"   COMPILING RESULTS FOR: {EXP_NAME} (BY AUDIO)   ")
    print("=" * 70)
    
    grid = compile_metrics(eval_subset="test", eval_strategy="by_audio")
    
    if grid:
        print(grid)
        print_confusion_matrix_and_class_stats(eval_subset="test", eval_strategy="by_audio")
    else:
        print("Failed to compile metrics. Make sure all folds are complete.")
