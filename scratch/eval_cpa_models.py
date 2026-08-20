import os
import csv
import math
import numpy as np
from collections import defaultdict

id_to_dataset = {}
with open('src/iara/dataset_info/iara.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        fid = int(row['ID'])
        id_to_dataset[fid] = row['Dataset']

def eval_model_cpa(exp_dir):
    datasets = ['A', 'B', 'C', 'D', 'CPA_IN', 'CPA_OUT', 'ALL']
    fold_res = {ds: {'sp': [], 'acc': []} for ds in datasets}
    
    for fold in range(10):
        fold_dir = os.path.join(exp_dir, 'eval', f'fold_{fold}')
        if not os.path.exists(fold_dir):
            continue
        csvs = [f for f in os.listdir(fold_dir) if f.endswith('_test.csv') and 'cbs' not in f]
        if not csvs:
            continue
        
        ft, fp = {}, defaultdict(list)
        with open(os.path.join(fold_dir, csvs[0])) as f:
            for row in csv.DictReader(f):
                fid = int(row['File'])
                ft[fid] = int(row['Target'])
                fp[fid].append(int(row['Prediction']))
        
        for ds in datasets:
            cm = defaultdict(int)
            correct, total = 0, 0
            for fid, preds in fp.items():
                file_ds = id_to_dataset.get(fid, 'UNKNOWN')
                match = False
                if ds == 'ALL':
                    match = True
                elif ds == 'CPA_IN' and file_ds in ['A', 'C']:
                    match = True
                elif ds == 'CPA_OUT' and file_ds in ['B', 'D']:
                    match = True
                elif file_ds == ds:
                    match = True
                
                if match:
                    pred = max(set(preds), key=preds.count)
                    target = ft[fid]
                    cm[(target, pred)] += 1
                    total += 1
                    if target == pred:
                        correct += 1
            
            if total == 0:
                continue
            
            classes_present = sorted(list(set(t for (t, p) in cm.keys())))
            recs = []
            for c in classes_present:
                c_tot = sum(cm[(c, p)] for p in range(4))
                if c_tot > 0:
                    recs.append(cm[(c, c)] / c_tot)
            if recs:
                mean_r = sum(recs) / len(recs)
                gmean_r = math.exp(sum(math.log(max(r, 1e-9)) for r in recs) / len(recs))
                sp = math.sqrt(mean_r * gmean_r) * 100
                acc = (correct / total) * 100
                fold_res[ds]['sp'].append(sp)
                fold_res[ds]['acc'].append(acc)

    return fold_res

models = [
    ('MLP LOFAR (Baseline Artigo)', 'results/trainings/tests/mlp/20260520-175407'),
    ('CNN MEL (Artigo)', 'results/trainings/tests/cnn/20260520-163451'),
    ('MLP Híbrido Calibrado', 'results/trainings/tests/mlp_hybrid_pca64/20260604-034320'),
    ('SVM LOFAR 6000 (Base)', 'results/trainings/tests/svm_nystroem_6000_lofar_elasticnet_l1r0.15'),
    ('Cascata LOFAR 6000 + SL', 'results/trainings/tests/svm_nystroem_4000_hybrid_cascade_pretrained_sm6000pca64_smallalso_fbmc_large_sl6000pca8_lofarmc6000_nosm_elasticnet_l1r0.15')
]

print(f"{'Modelo':<28} | {'Near CPA (A)':<18} | {'Far CPA (C)':<18} | {'CPA IN (A+C)':<18} | {'CPA OUT (B+D)':<18} | {'Geral (5x2cv)':<18}")
print("-" * 130)
for label, path in models:
    if os.path.exists(path):
        res = eval_model_cpa(path)
        def fmt(ds):
            if not res[ds]['sp']:
                return "N/A"
            sp_m = np.mean(res[ds]['sp'])
            acc_m = np.mean(res[ds]['acc'])
            return f"{sp_m:4.1f}% (acc {acc_m:4.1f}%)"
        
        a_str = fmt('A')
        c_str = fmt('C')
        in_str = fmt('CPA_IN')
        out_str = fmt('CPA_OUT')
        all_str = fmt('ALL')
        print(f"{label:<28} | {a_str:<18} | {c_str:<18} | {in_str:<18} | {out_str:<18} | {all_str:<18}")
    else:
        print(f"{label:<28} | NOT FOUND ({path})")
