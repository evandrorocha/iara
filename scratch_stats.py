import os
import collections
import numpy as np
import pandas as pd

EXP_NAME = "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0"
BASE_DIR = os.path.abspath(f"results/trainings/tests/{EXP_NAME}")
EVAL_DIR = os.path.join(BASE_DIR, "eval")

t = 0.9

# We need to collect lists of coverages and accuracies for each class across the 10 folds
# fold_class_stats[fold_idx][class_idx] = { 'total': X, 'classified': Y, 'correct': Z }
fold_stats = []

for i_fold in range(10):
    fold_eval_dir = os.path.join(EVAL_DIR, f"fold_{i_fold}")
    csv_name = f"{EXP_NAME}_multiclass_test.csv"
    csv_path = os.path.join(fold_eval_dir, csv_name)
    
    if not os.path.exists(csv_path):
        continue
        
    df = pd.read_csv(csv_path)
    
    class_stats = {
        0: {'name': 'SMALL', 'classified': 0, 'correct': 0, 'total': 0},
        1: {'name': 'MEDIUM', 'classified': 0, 'correct': 0, 'total': 0},
        2: {'name': 'LARGE', 'classified': 0, 'correct': 0, 'total': 0},
        3: {'name': 'BACKGROUND', 'classified': 0, 'correct': 0, 'total': 0}
    }
    
    for filename, group in df.groupby('File'):
        total_segments = len(group)
        targets = group['Target'].values
        predictions = group['Prediction'].values
        
        target_counts = collections.Counter(targets)
        pred_counts = collections.Counter(predictions)
        
        majority_target = target_counts.most_common(1)[0][0]
        majority_pred, majority_pred_count = pred_counts.most_common(1)[0]
        
        ratio = majority_pred_count / total_segments
        
        class_stats[majority_target]['total'] += 1
        
        if ratio >= t:
            class_stats[majority_target]['classified'] += 1
            if majority_pred == majority_target:
                class_stats[majority_target]['correct'] += 1
                
    fold_stats.append(class_stats)

# Now calculate mean and std across the 10 folds for each class
classes = {0: 'SMALL', 1: 'MEDIUM', 2: 'LARGE', 3: 'BACKGROUND'}

print("=" * 110)
print(f"ANÁLISE DE CONFIANÇA POR CLASSE COM DESVIO PADRÃO (t >= 0.9)".center(110))
print("=" * 110)
print(f"{'Classe':<15} | {'Áudios Originais/Fold':<22} | {'Taxa de Cobertura (%)':<25} | {'Acurácia de Classe (%)':<25}")
print("-" * 110)

for cls_idx, name in classes.items():
    covs = []
    accs = []
    origs = []
    
    for fs in fold_stats:
        info = fs[cls_idx]
        orig = info['total']
        classif = info['classified']
        corr = info['correct']
        
        origs.append(orig)
        
        # Coverage per fold for this class
        cov = (classif / orig) * 100 if orig > 0 else 0.0
        covs.append(cov)
        
        # Accuracy per fold for this class if any were classified
        if classif > 0:
            acc = (corr / classif) * 100
            accs.append(acc)
            
    mean_orig = np.mean(origs)
    mean_cov = np.mean(covs)
    std_cov = np.std(covs)
    
    mean_acc = np.mean(accs) if len(accs) > 0 else 0.0
    std_acc = np.std(accs) if len(accs) > 0 else 0.0
    
    print(f"{name:<15} | {mean_orig:^22.1f} | {mean_cov:6.2f} ± {std_cov:5.2f} | {mean_acc:6.2f} ± {std_acc:5.2f}")

print("-" * 110)
# Overall stats
overall_covs = []
overall_accs = []
overall_origs = []

for fs in fold_stats:
    tot_orig = sum(fs[c]['total'] for c in classes)
    tot_class = sum(fs[c]['classified'] for c in classes)
    tot_corr = sum(fs[c]['correct'] for c in classes)
    
    overall_origs.append(tot_orig)
    overall_covs.append((tot_class / tot_orig) * 100 if tot_orig > 0 else 0.0)
    overall_accs.append((tot_corr / tot_class) * 100 if tot_class > 0 else 0.0)

print(f"{'TOTAL GLOBAL':<15} | {np.mean(overall_origs):^22.1f} | {np.mean(overall_covs):6.2f} ± {np.std(overall_covs):5.2f} | {np.mean(overall_accs):6.2f} ± {np.std(overall_accs):5.2f}")
print("=" * 110)
