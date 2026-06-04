import pandas as pd
import numpy as np
import collections

N = 4
for fold_i in range(10):
    p_test = f"results/trainings/tests/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0/eval/fold_{fold_i}/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0_multiclass_test.csv"
    try:
        df = pd.read_csv(p_test)
    except Exception:
        continue
    df_a = df.groupby('File').agg(lambda x: collections.Counter(x).most_common(1)[0][0]).reset_index()
    cm_a = np.zeros((N, N))
    for t, p in zip(df_a['Target'], df_a['Prediction']):
        cm_a[t, p] += 1
    recalls = [cm_a[i, i] / cm_a[i].sum() * 100 for i in range(4)]
    print(f"Fold {fold_i} baseline recalls: {[f'{r:.1f}' for r in recalls]}")
