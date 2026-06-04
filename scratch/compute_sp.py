import pandas as pd
import numpy as np
import sys

csv_path = sys.argv[1]

df = pd.read_csv(csv_path)
# Assuming columns: File,Target,Prediction
# Compute per-class recall (sensitivity)
classes = df['Target'].unique()
recalls = []
for cls in classes:
    cls_df = df[df['Target'] == cls]
    tp = (cls_df['Prediction'] == cls).sum()
    fn = (cls_df['Prediction'] != cls).sum()
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    recalls.append(recall)
# Geometric mean of recalls
if len(recalls) == 0:
    sp = 0.0
else:
    sp = np.prod(recalls) ** (1.0 / len(recalls))
print(f"SP_INDEX: {sp:.4f}")
