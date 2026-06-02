import os
import pandas as pd
import numpy as np
import collections

# Inline grid metrics compiler logic from iara
class GridCompiler:
    def __init__(self):
        self.accs = []
        self.sps = []
    def add_metrics(self, targets, predictions):
        # Multi-class metrics (SP index and ACC)
        classes = sorted(list(set(targets)))
        # Recall per class
        recalls = []
        for cls in classes:
            mask = targets == cls
            if sum(mask) > 0:
                rec_val = np.sum((targets == cls) & (predictions == cls)) / np.sum(mask)
                recalls.append(rec_val)
        
        # SP index
        sp_index = (np.mean(recalls) * np.prod(recalls) ** (1 / len(classes))) ** 0.5 if recalls else 0.0
        # Accuracy
        acc = np.sum(targets == predictions) / len(targets)
        self.accs.append(acc * 100)
        self.sps.append(sp_index * 100)

def compile_exp(exp_name):
    base_dir = os.path.abspath(f"results/trainings/tests/{exp_name}")
    eval_dir = os.path.join(base_dir, "eval")
    if not os.path.exists(eval_dir):
        return None
        
    grid = GridCompiler()
    
    for i_fold in range(10):
        fold_eval_dir = os.path.join(eval_dir, f"fold_{i_fold}")
        csv_name = f"{exp_name}_multiclass_test.csv"
        csv_path = os.path.join(fold_eval_dir, csv_name)
        if not os.path.exists(csv_path):
            return None
            
        df = pd.read_csv(csv_path)
        # Apply majority vote by audio
        df = df.groupby('File').agg({
            'Target': lambda x: collections.Counter(x).most_common(1)[0][0],
            'Prediction': lambda x: collections.Counter(x).most_common(1)[0][0]
        }).reset_index()
        
        grid.add_metrics(df['Target'].values, df['Prediction'].values)
        
    return {
        'acc_mean': np.mean(grid.accs),
        'acc_std': np.std(grid.accs),
        'sp_mean': np.mean(grid.sps),
        'sp_std': np.std(grid.sps)
    }

test_dir = "results/trainings/tests"
results = {}
for item in os.listdir(test_dir):
    if item.startswith("svm_nystroem_") and os.path.isdir(os.path.join(test_dir, item)):
        res = compile_exp(item)
        if res:
            results[item] = res

print("\n" + "="*80)
print(f"{'EXPERIMENT NAME':<55} | {'ACC MEAN (%)':<12} | {'SP MEAN (%)':<12}")
print("="*80)
for k, v in sorted(results.items()):
    clean_k = k.encode('ascii', errors='ignore').decode('ascii')
    print(f"{clean_k:<55} | {v['acc_mean']:>5.2f}% +/- {v['acc_std']:>4.2f}% | {v['sp_mean']:>5.2f}% +/- {v['sp_std']:>4.2f}%")
print("="*80)
