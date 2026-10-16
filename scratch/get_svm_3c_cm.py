import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
import sklearn.metrics as sk_metrics
import scipy.stats as scipy_stats

def compute_sp_index(targets, preds, num_classes=3):
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    return sp, recalls, cm

cache_file = "data/iara_processed/cache_fusion_mel_demon.pkl"
if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    demon_cache = data["demon"]
else:
    demon_dir = "data/iara_processed/demon_features"
    demon_cache = {}
    for fname in os.listdir(demon_dir):
        if fname.endswith(".pkl"):
            fid = int(os.path.splitext(fname)[0])
            with open(os.path.join(demon_dir, fname), "rb") as f:
                demon_cache[fid] = pickle.load(f)["df"].values.astype(np.float32)

eval_base_dir = "results/trainings/tests/svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0/eval"
folds = []
for fold_idx in range(10):
    fold_dir = os.path.join(eval_base_dir, f"fold_{fold_idx}")
    trn_path = os.path.join(fold_dir, "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_trn.csv")
    tst_path = os.path.join(fold_dir, "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_test.csv")
    df_trn = pd.read_csv(trn_path)
    df_tst = pd.read_csv(tst_path)
    folds.append({
        "fold_idx": fold_idx,
        "train_files": df_trn.groupby("File")["Target"].first().to_dict(),
        "test_files": df_tst.groupby("File")["Target"].first().to_dict(),
    })

def build_matrices(cache, file_dict):
    X_list, y_list, fids = [], [], []
    for fid, target in file_dict.items():
        if target == 3 or fid not in cache:
            continue
        windows = cache[fid]
        X_list.append(windows)
        y_list.extend([target] * len(windows))
        fids.extend([fid] * len(windows))
    return np.vstack(X_list), np.array(y_list, dtype=np.int32), np.array(fids, dtype=np.int32)

total_cm = np.zeros((3, 3), dtype=np.int64)
for fold in folds:
    X_trn, y_trn, _ = build_matrices(demon_cache, fold["train_files"])
    X_tst, y_tst, test_fids = build_matrices(demon_cache, fold["test_files"])

    scaler = StandardScaler()
    X_trn = scaler.fit_transform(X_trn)
    X_tst = scaler.transform(X_tst)

    nystroem = Nystroem(kernel="rbf", gamma=1.0 / 253, n_components=4000, random_state=42 + fold["fold_idx"])
    X_trn_proj = nystroem.fit_transform(X_trn)
    X_tst_proj = nystroem.transform(X_tst)

    clf = SGDClassifier(loss="hinge", penalty="elasticnet", l1_ratio=0.15, alpha=1.0 / (2.0 * len(X_trn)),
                        class_weight="balanced", random_state=42, max_iter=1000, tol=1e-3)
    clf.fit(X_trn_proj, y_trn)
    preds = clf.predict(X_tst_proj)

    file_preds = defaultdict(list)
    file_targets = {}
    for fid, p, t in zip(test_fids, preds, y_tst):
        file_preds[fid].append(p)
        file_targets[fid] = t

    audio_preds = [int(np.argmax(np.bincount(file_preds[fid], minlength=3))) for fid in file_preds]
    audio_targets = [int(file_targets[fid]) for fid in file_preds]
    _, _, cm = compute_sp_index(audio_targets, audio_preds, num_classes=3)
    total_cm += cm

cm_norm = total_cm.astype(float) / total_cm.sum(axis=1)[:, np.newaxis] * 100
print("SVM Nystroem 3-Classes CM Norm (%):")
print(pd.DataFrame(np.round(cm_norm, 1), index=["SMALL", "MEDIUM", "LARGE"], columns=["SMALL", "MEDIUM", "LARGE"]))
