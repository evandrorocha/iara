import os
import pandas as pd
import collections
import numpy as np
import sys
from sklearn.metrics import recall_score

sys.path.append(os.path.abspath('src'))
from iara.ml.metrics import Metric

EXP_NAME = "svm_nystroem_4000_hybrid_cascade_pretrained_svm_nystroem_6000_lofar_elasticnet_l1r0.15_elasticnet_l1r0.15"

def most_common_value(series):
    return collections.Counter(series).most_common(1)[0][0]

results = []

for i_fold in range(10):
    csv_path = f"results/trainings/tests/{EXP_NAME}/eval/fold_{i_fold}/{EXP_NAME}_multiclass_test_ct0.55.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Aggregate to audio level
        df_audio = df.groupby('File').agg({
            'Target': most_common_value,
            'Prediction': most_common_value
        }).reset_index()
        
        y_true = df_audio['Target'].values
        y_pred = df_audio['Prediction'].values
        
        acc = Metric.ACCURACY.compute(y_true, y_pred)
        sp = Metric.SP_INDEX.compute(y_true, y_pred)
        
        recalls = recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None)
        
        results.append({
            'fold': i_fold,
            'acc': acc,
            'sp': sp,
            'r_small': recalls[0] * 100.0,
            'r_medium': recalls[1] * 100.0,
            'r_large': recalls[2] * 100.0,
            'r_bg': recalls[3] * 100.0
        })

if not results:
    print("No completed folds with t=0.55 found yet.")
    sys.exit(0)

df_results = pd.DataFrame(results)

print("=" * 80)
print(f"PARTIAL RESULTS FOR {EXP_NAME} (cascade_threshold = 0.55)")
print("=" * 80)
print(f"{'FOLD':<6} | {'ACC (%)':<8} | {'SP (%)':<8} | {'SMALL (%)':<10} | {'MEDIUM (%)':<10} | {'LARGE (%)':<10} | {'BG (%)':<10}")
print("-" * 80)
for _, r in df_results.iterrows():
    print(f"Fold {int(r['fold']):<1} | {r['acc']:<8.2f} | {r['sp']:<8.2f} | {r['r_small']:<10.2f} | {r['r_medium']:<10.2f} | {r['r_large']:<10.2f} | {r['r_bg']:<10.2f}")
print("-" * 80)

# Calculate average and standard deviation
if len(results) > 1:
    means = df_results.mean()
    stds = df_results.std()
    print(f"MEAN   | {means['acc']:<8.2f} | {means['sp']:<8.2f} | {means['r_small']:<10.2f} | {means['r_medium']:<10.2f} | {means['r_large']:<10.2f} | {means['r_bg']:<10.2f}")
    print(f"STD    | {stds['acc']:<8.2f} | {stds['sp']:<8.2f} | {stds['r_small']:<10.2f} | {stds['r_medium']:<10.2f} | {stds['r_large']:<10.2f} | {stds['r_bg']:<10.2f}")
else:
    r = results[0]
    print(f"MEAN   | {r['acc']:<8.2f} | {r['sp']:<8.2f} | {r['r_small']:<10.2f} | {r['r_medium']:<10.2f} | {r['r_large']:<10.2f} | {r['r_bg']:<10.2f}")
print("=" * 80)
