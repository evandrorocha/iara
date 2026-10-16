"""
Evaluation of Random Forest (100 trees, balanced) on Multimodal Fusion (MEL + DEMON - 509-D)
under the official 10-fold 5x2cv protocol with Ship ID isolation.
Outputs metrics for both 4-Class and 3-Class tasks.
"""

import os
import sys
import time
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sk_metrics
import scipy.stats as scipy_stats


def compute_sp_index(targets, preds, num_classes=4):
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    return sp, recalls, cm


def load_cached_features(cache_file="data/iara_processed/cache_fusion_mel_demon.pkl"):
    print(f"Carregando cache de {cache_file}...", flush=True)
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    return data["fused"]


def get_official_folds(eval_base_dir="results/trainings/tests/svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0/eval"):
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
    return folds


def build_fold_matrices(feature_cache, file_dict, exclude_target=None):
    X_list, y_list, file_ids = [], [], []
    for fid, target in file_dict.items():
        if exclude_target is not None and target == exclude_target:
            continue
        if fid not in feature_cache:
            continue
        windows = feature_cache[fid]
        X_list.append(windows)
        y_list.extend([target] * len(windows))
        file_ids.extend([fid] * len(windows))

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)
    file_ids = np.array(file_ids, dtype=np.int32)
    return X, y, file_ids


def evaluate_rf(feature_cache, folds, num_classes=4, n_estimators=100):
    class_names = ["SMALL", "MEDIUM", "LARGE", "AMBIENTAL"] if num_classes == 4 else ["SMALL", "MEDIUM", "LARGE"]
    exclude_target = 3 if num_classes == 3 else None
    problem_name = f"{num_classes}-Classes"

    print("\n" + "=" * 80, flush=True)
    print(f"AVALIAÇÃO: Random Forest (100 árvores) na FUSÃO MEL+DEMON [{problem_name}]", flush=True)
    print("=" * 80, flush=True)

    fold_accs, fold_sps, fold_recalls = [], [], []
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for fold in folds:
        fold_idx = fold["fold_idx"]
        t_start = time.time()

        X_train, y_train, _ = build_fold_matrices(feature_cache, fold["train_files"], exclude_target)
        X_test, y_test, test_fids = build_fold_matrices(feature_cache, fold["test_files"], exclude_target)

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        win_preds = rf.predict(X_test)

        file_preds = defaultdict(list)
        file_targets = {}
        for fid, pred, tgt in zip(test_fids, win_preds, y_test):
            file_preds[fid].append(pred)
            file_targets[fid] = tgt

        audio_preds = [int(np.argmax(np.bincount(file_preds[fid], minlength=num_classes))) for fid in file_preds]
        audio_targets = [int(file_targets[fid]) for fid in file_preds]

        acc = sk_metrics.accuracy_score(audio_targets, audio_preds) * 100
        sp, recalls, cm = compute_sp_index(audio_targets, audio_preds, num_classes=num_classes)

        fold_accs.append(acc)
        fold_sps.append(sp)
        fold_recalls.append(recalls)
        total_cm += cm

        elapsed = time.time() - t_start
        rec_str = " | ".join([f"{cn}: {r*100:.1f}%" for cn, r in zip(class_names, recalls)])
        print(f"  Fold {fold_idx} ({elapsed:.1f}s): Acc={acc:.2f}% | SP={sp:.2f}% | Recalls: {rec_str}", flush=True)

    mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
    mean_sp, std_sp = np.mean(fold_sps), np.std(fold_sps)
    mean_recs = np.mean(fold_recalls, axis=0) * 100

    print(f"\n>> RF FUSÃO MEL+DEMON [{problem_name}]: Acurácia = {mean_acc:.2f}% ± {std_acc:.2f}% | SP = {mean_sp:.2f}% ± {std_sp:.2f}%", flush=True)
    for cn, mr in zip(class_names, mean_recs):
        print(f"   Recall {cn:<10}: {mr:.2f}%", flush=True)

    cm_norm = total_cm.astype(float) / total_cm.sum(axis=1)[:, np.newaxis] * 100
    print("\nMatriz de Confusão Normalizada (%):", flush=True)
    print(pd.DataFrame(np.round(cm_norm, 1), index=class_names, columns=class_names), flush=True)

    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_sp": mean_sp,
        "std_sp": std_sp,
        "mean_recs": mean_recs,
        "cm_norm": cm_norm
    }


def main():
    cache_fused = load_cached_features()
    folds = get_official_folds()

    results = {}
    results["RF_FUSED_4C"] = evaluate_rf(cache_fused, folds, num_classes=4)
    results["RF_FUSED_3C"] = evaluate_rf(cache_fused, folds, num_classes=3)

    out_pkl = "data/iara_processed/results_rf_fusion.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResultados salvos com sucesso em {out_pkl}", flush=True)


if __name__ == "__main__":
    main()
