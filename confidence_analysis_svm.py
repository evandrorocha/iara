import os
import sys
import collections
import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import scipy.stats as scipy_stats

def get_default_experiment():
    test_dir = os.path.abspath("results/trainings/tests")
    if not os.path.exists(test_dir):
        return None
    subdirs = []
    for item in os.listdir(test_dir):
        item_path = os.path.join(test_dir, item)
        if os.path.isdir(item_path) and not item.startswith("2026") and item not in ("cnn", "mlp"):
            subdirs.append((item, os.path.getmtime(item_path)))
    if subdirs:
        subdirs.sort(key=lambda x: x[1], reverse=True)
        return subdirs[0][0]
    return None

if len(sys.argv) > 1:
    EXP_NAME = sys.argv[1]
else:
    EXP_NAME = get_default_experiment()

if not EXP_NAME:
    print("Error: No active experiment found under results/trainings/tests/!")
    sys.exit(1)

BASE_DIR = os.path.abspath(f"results/trainings/tests/{EXP_NAME}")
EVAL_DIR = os.path.join(BASE_DIR, "eval")

def compute_sp_index(target, prediction):
    if len(target) == 0:
        return 0.0
    labels = list(set(list(target)))
    cm = sk_metrics.confusion_matrix(target, prediction, labels=labels)
    # Handle classes with no samples or potential divide by zero
    sums = cm.sum(axis=1)
    sums[sums == 0] = 1
    detection_probabilities = cm.diagonal() / sums
    
    # Geometric mean
    if len(detection_probabilities) == 0:
        return 0.0
    geom_mean = scipy_stats.gmean(detection_probabilities)
    return np.sqrt(np.mean(detection_probabilities * geom_mean)) * 100

def compute_f1_score(target, prediction):
    if len(target) == 0:
        return 0.0
    return sk_metrics.f1_score(target, prediction, average='macro') * 100

def compute_accuracy(target, prediction):
    if len(target) == 0:
        return 0.0
    return sk_metrics.accuracy_score(target, prediction) * 100

# Thresholds to evaluate
thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]

print("=" * 80)
print(f"   CONFIDENCE ANALYSIS (BY FILE) FOR EXPERIMENT:".center(80))
exp_name_safe = EXP_NAME.replace('\uf02a', '*').encode('ascii', errors='replace').decode('ascii')
print(f"{exp_name_safe}".center(80))
print("=" * 80)

# Pre-load all files per fold and calculate their majority classes and ratios
fold_data = []

for i_fold in range(10):
    fold_eval_dir = os.path.join(EVAL_DIR, f"fold_{i_fold}")
    csv_name = f"{EXP_NAME}_multiclass_test.csv"
    csv_path = os.path.join(fold_eval_dir, csv_name)
    
    if not os.path.exists(csv_path):
        print(f"Error: Fold {i_fold} data not found at {csv_path}!")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    files_processed = []
    # Group by file to compute ratios
    for filename, group in df.groupby('File'):
        total_segments = len(group)
        targets = group['Target'].values
        predictions = group['Prediction'].values
        
        # Most common class
        target_counts = collections.Counter(targets)
        pred_counts = collections.Counter(predictions)
        
        majority_target = target_counts.most_common(1)[0][0]
        majority_pred, majority_pred_count = pred_counts.most_common(1)[0]
        
        ratio = majority_pred_count / total_segments
        
        files_processed.append({
            'file': filename,
            'target': majority_target,
            'prediction': majority_pred,
            'ratio': ratio
        })
        
    fold_data.append(files_processed)

# Now, evaluate for each threshold
results_table = []

for t in thresholds:
    fold_sps = []
    fold_accs = []
    fold_f1s = []
    fold_ratios = []
    
    for i_fold in range(10):
        files = fold_data[i_fold]
        total_files = len(files)
        
        # Filter files based on ratio constraint
        if t == 0.0:
            filtered_files = files
        else:
            filtered_files = [f for f in files if f['ratio'] >= t]
            
        classified_count = len(filtered_files)
        file_ratio = (classified_count / total_files) * 100 if total_files > 0 else 0.0
        
        targets = [f['target'] for f in filtered_files]
        predictions = [f['prediction'] for f in filtered_files]
        
        sp = compute_sp_index(targets, predictions)
        acc = compute_accuracy(targets, predictions)
        f1 = compute_f1_score(targets, predictions)
        
        fold_sps.append(sp)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_ratios.append(file_ratio)
        
    results_table.append({
        'threshold': t,
        'sp_mean': np.mean(fold_sps),
        'sp_std': np.std(fold_sps),
        'acc_mean': np.mean(fold_accs),
        'acc_std': np.std(fold_accs),
        'f1_mean': np.mean(fold_f1s),
        'f1_std': np.std(fold_f1s),
        'ratio_mean': np.mean(fold_ratios),
        'ratio_std': np.std(fold_ratios)
    })

# Print beautiful ASCII table
print(f"{'Threshold (t)':<15} | {'File Ratio (%)':<18} | {'SP (%)':<16} | {'ACC (%)':<16} | {'F1-Score (%)':<16}")
print("-" * 90)
for r in results_table:
    t_str = f"t >= {r['threshold']:.1f}" if r['threshold'] > 0 else "Sem restrição"
    print(f"{t_str:<15} | {r['ratio_mean']:5.2f} ± {r['ratio_std']:4.2f}  | {r['sp_mean']:5.2f} ± {r['sp_std']:4.2f}  | {r['acc_mean']:5.2f} ± {r['acc_std']:4.2f}  | {r['f1_mean']:5.2f} ± {r['f1_std']:4.2f}")
print("=" * 90)
print("OBSERVAÇÕES:")
print("1. File Ratio representação a porcentagem de arquivos válidos (não desconhecidos) avaliados.")
print("2. Conforme o limiar 't' aumenta, a confiança aumenta: apenas áudios com alta concordância")
print("   de janelas são classificados. O SP e a Acurácia tendem a subir acentuadamente!")
print("=" * 90)
