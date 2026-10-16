"""
Evaluation of pure DEMON features with SVM Nystroem on the official 5x2cv protocol.

This script evaluates:
1. Pure DEMON features (253 bins, 0.39 Hz - 25 Hz)
2. Exact 10-fold cross-validation with ship ID isolation matching LOFAR and MEL
3. SVM Nystroem (m=1000 and m=4000 landmark points, RBF kernel, ElasticNet penalty)
4. Both 4-Class (SMALL, MEDIUM, LARGE, AMBIENTAL) and 3-Class (SMALL, MEDIUM, LARGE)
5. Comprehensive metrics: Audio Accuracy, SP Index, Per-Class Recall, Confusion Matrices
"""

import os
import sys
import time
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
import sklearn.metrics as sk_metrics
import scipy.stats as scipy_stats

# Official IARA SP index calculation
def compute_sp_index(targets, preds, num_classes=4):
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    return sp, recalls, cm


def load_all_demon_features(demon_dir="data/iara_processed/demon_features"):
    """
    Load all DEMON feature DataFrames into memory for rapid cross-validation.
    Returns: dict mapping file_id (int) -> (numpy array of windows, shape (N, 253))
    """
    print(f"Carregando features DEMON da pasta {demon_dir}...")
    t0 = time.time()
    feature_cache = {}
    
    file_list = [f for f in os.listdir(demon_dir) if f.endswith(".pkl")]
    for fname in file_list:
        file_id = int(os.path.splitext(fname)[0])
        fpath = os.path.join(demon_dir, fname)
        with open(fpath, "rb") as f:
            data = pickle.load(f)
            df = data["df"]
            feature_cache[file_id] = df.values.astype(np.float32)

    elapsed = time.time() - t0
    print(f"Sucesso: {len(feature_cache)} arquivos carregados em {elapsed:.2f}s.")
    return feature_cache


def get_official_folds(eval_base_dir="results/trainings/tests/svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0/eval"):
    """
    Extract exact file IDs and targets for all 10 folds from official IARA records.
    """
    folds = []
    for fold_idx in range(10):
        fold_dir = os.path.join(eval_base_dir, f"fold_{fold_idx}")
        trn_path = os.path.join(fold_dir, "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_trn.csv")
        tst_path = os.path.join(fold_dir, "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_test.csv")
        
        df_trn = pd.read_csv(trn_path)
        df_tst = pd.read_csv(tst_path)

        train_files = df_trn.groupby("File")["Target"].first().to_dict()
        test_files = df_tst.groupby("File")["Target"].first().to_dict()

        folds.append({
            "fold_idx": fold_idx,
            "train_files": train_files, # file_id -> target
            "test_files": test_files,   # file_id -> target
        })
    return folds


def build_fold_matrices(feature_cache, file_dict, exclude_target=None):
    """
    Build window matrix X and labels y for a given set of file IDs.
    """
    X_list = []
    y_list = []
    file_ids = []

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


def evaluate_svm_nystroem(
    feature_cache,
    folds,
    n_components=4000,
    C=2.0,
    penalty="elasticnet",
    l1_ratio=0.15,
    normalize=True,
    use_pca=False,
    n_pca=64,
    num_classes=4,
    problem_name="4-Classes (SMALL, MED, LARGE, AMBIENTAL)",
):
    print("\n" + "=" * 80)
    print(f"AVALIAÇÃO DEMON ISOLADO: SVM Nyström ({problem_name})")
    print(f"Parâmetros: m={n_components} | C={C} | penalty={penalty} (l1_ratio={l1_ratio}) | norm={normalize} | PCA={use_pca} ({n_pca})")
    print("=" * 80)

    class_names = (
        ["SMALL", "MEDIUM", "LARGE", "AMBIENTAL"]
        if num_classes == 4
        else ["SMALL", "MEDIUM", "LARGE"]
    )
    exclude_target = 3 if num_classes == 3 else None

    fold_accuracies = []
    fold_sp_indices = []
    fold_recalls = []
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for fold in folds:
        fold_idx = fold["fold_idx"]
        t_start = time.time()

        X_train, y_train, _ = build_fold_matrices(
            feature_cache, fold["train_files"], exclude_target=exclude_target
        )
        X_test, y_test, test_fids = build_fold_matrices(
            feature_cache, fold["test_files"], exclude_target=exclude_target
        )

        # 1. Preprocessing (StandardScaler and optional PCA)
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if use_pca:
            pca = PCA(n_components=min(n_pca, X_train.shape[1]), random_state=42)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # 2. Nystroem kernel approximation
        gamma = 1.0 / (X_train.shape[1] * X_train.var())
        nystroem = Nystroem(
            kernel="rbf",
            gamma=gamma,
            n_components=n_components,
            random_state=42,
            n_jobs=-1,
        )
        X_train_trans = nystroem.fit_transform(X_train)
        X_test_trans = nystroem.transform(X_test)

        # 3. SGDClassifier (linear in Nystroem space = approximate RBF-SVM)
        alpha = 1.0 / (C * X_train.shape[0])
        clf = SGDClassifier(
            loss="hinge",
            penalty=penalty,
            l1_ratio=l1_ratio,
            alpha=alpha,
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train_trans, y_train)

        # 4. Window-level prediction
        win_preds = clf.predict(X_test_trans)

        # 5. Audio-level prediction (Majority voting)
        file_preds = defaultdict(list)
        file_targets = {}
        for fid, pred, tgt in zip(test_fids, win_preds, y_test):
            file_preds[fid].append(pred)
            file_targets[fid] = tgt

        audio_preds = []
        audio_targets = []
        for fid in file_preds:
            counts = np.bincount(file_preds[fid], minlength=num_classes)
            audio_preds.append(int(np.argmax(counts)))
            audio_targets.append(int(file_targets[fid]))

        audio_preds = np.array(audio_preds)
        audio_targets = np.array(audio_targets)

        # Metrics
        acc = sk_metrics.accuracy_score(audio_targets, audio_preds) * 100
        sp, recalls, cm = compute_sp_index(audio_targets, audio_preds, num_classes=num_classes)

        fold_accuracies.append(acc)
        fold_sp_indices.append(sp)
        fold_recalls.append(recalls)
        total_cm += cm

        elapsed = time.time() - t_start
        rec_str = " | ".join([f"{cn}: {r*100:.1f}%" for cn, r in zip(class_names, recalls)])
        print(f"  Fold {fold_idx} ({elapsed:.1f}s): Acc={acc:.2f}% | SP={sp:.2f}% | Recalls: {rec_str}")

    # Compile 10-fold statistics
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_sp = np.mean(fold_sp_indices)
    std_sp = np.std(fold_sp_indices)
    mean_recalls = np.mean(fold_recalls, axis=0) * 100
    std_recalls = np.std(fold_recalls, axis=0) * 100

    print("\n" + "-" * 80)
    print(f" RESUMO CONSOLIDADO (10 Folds - {problem_name}):")
    print(f"  Acurácia (Áudio): {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Índice SP Oficial: {mean_sp:.2f}% ± {std_sp:.2f}%")
    for cn, mr, sr in zip(class_names, mean_recalls, std_recalls):
        print(f"  Recall {cn:<10}: {mr:.2f}% ± {sr:.2f}%")
    print("-" * 80)

    print("\n Matriz de Confusão Acumulada (10 Folds):")
    print(total_cm)

    cm_norm = total_cm.astype(float) / total_cm.sum(axis=1)[:, np.newaxis] * 100
    print("\n Matriz de Confusão Normalizada (%):")
    df_cm = pd.DataFrame(np.round(cm_norm, 1), index=class_names, columns=class_names)
    print(df_cm)

    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_sp": mean_sp,
        "std_sp": std_sp,
        "mean_recalls": mean_recalls,
        "std_recalls": std_recalls,
        "total_cm": total_cm,
        "cm_norm": cm_norm,
        "class_names": class_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DEMON on 5x2cv")
    parser.add_argument("-m", "--n_components", type=int, default=4000, help="Nystroem landmark components")
    parser.add_argument("--no_norm", action="store_true", help="Disable StandardScaler")
    parser.add_argument("--pca", action="store_true", help="Apply PCA before Nystroem")
    parser.add_argument("--n_pca", type=int, default=64, help="PCA components")
    args = parser.parse_args()

    # 1. Load DEMON features
    cache = load_all_demon_features()

    # 2. Load official 10 folds
    folds = get_official_folds()
    print(f"Carregados 10 folds oficiais com isolamento estrito de navios.")

    # 3. Evaluate 4-Class Problem (SMALL, MEDIUM, LARGE, AMBIENTAL)
    res_4class = evaluate_svm_nystroem(
        cache,
        folds,
        n_components=args.n_components,
        normalize=not args.no_norm,
        use_pca=args.pca,
        n_pca=args.n_pca,
        num_classes=4,
        problem_name="4-Classes (SMALL, MEDIUM, LARGE, AMBIENTAL)",
    )

    # 4. Evaluate 3-Class Problem (SMALL, MEDIUM, LARGE)
    res_3class = evaluate_svm_nystroem(
        cache,
        folds,
        n_components=args.n_components,
        normalize=not args.no_norm,
        use_pca=args.pca,
        n_pca=args.n_pca,
        num_classes=3,
        problem_name="3-Classes (Embarcações: SMALL, MEDIUM, LARGE)",
    )

    # 5. Direct Comparative Summary
    print("\n" + "=" * 80)
    print("COMPARATIVO DIRETO DE DESEMPENHO NO 5x2cv (Votação Majoritária por Áudio)")
    print("=" * 80)
    print(f"{'Modelo':<25} | {'Acurácia':<18} | {'Índice SP':<18} | {'SMALL Rec':<12} | {'LARGE Rec':<12}")
    print("-" * 80)
    print(f"{'LOFAR SVM (4000)':<25} | 63.32% ± 2.01%     | 63.09% ± 2.06%     | 47.81%       | 74.91%")
    print(f"{'MEL SVM (4000)':<25} | 63.36% ± 1.00%     | 63.80% ± 1.12%     | 50.90%       | 69.16%")
    print(f"{'DEMON SVM (Puro)':<25} | {res_4class['mean_acc']:.2f}% ± {res_4class['std_acc']:.2f}%     | {res_4class['mean_sp']:.2f}% ± {res_4class['std_sp']:.2f}%     | {res_4class['mean_recalls'][0]:.2f}%       | {res_4class['mean_recalls'][2]:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
