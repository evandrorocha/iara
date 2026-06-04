import os
import pandas as pd
import numpy as np
import sys
from iara.ml.metrics import Metric

sys.stdout.reconfigure(encoding='utf-8')

base_dir = "results/trainings/tests/svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C0.5/eval"
accs = []
sps = []
f1s = []

for fold in range(10):
    csv_path = os.path.join(base_dir, f"fold_{fold}", "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C0.5_multiclass_test_b0.32_0.27_0_0.csv")
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    targets = df['Target'].values
    predictions = df['Prediction'].values
    
    acc = Metric.ACCURACY.compute(targets, predictions)
    sp = Metric.SP_INDEX.compute(targets, predictions)
    f1 = Metric.MICRO_F1.compute(targets, predictions)
    
    accs.append(acc)
    sps.append(sp)
    f1s.append(f1)

if len(accs) > 0:
    print("=========================================================")
    print(" METRICAS JANELA (WINDOW) DO SVM LOFAR C=0.5 CALIBRADO   ")
    print("=========================================================")
    print(f"SP Index: {np.mean(sps):.2f} ± {np.std(sps, ddof=1):.2f}%")
    print(f"ACC:      {np.mean(accs):.2f} ± {np.std(accs, ddof=1):.2f}%")
    print(f"F1 Micro: {np.mean(f1s):.2f} ± {np.std(f1s, ddof=1):.2f}%")
    print("=========================================================")
else:
    print("Nenhum fold encontrado!")
