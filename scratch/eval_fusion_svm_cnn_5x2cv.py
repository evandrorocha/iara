"""
Evaluation of Multimodal Fusion (MEL + DEMON) with:
1. SVM Nyström (m=6000, RBF kernel, ElasticNet penalty)
2. CNN 2D PyTorch (GPU RTX 5060 Ti, blocks 32 x D)

Evaluated under the official 10-fold 5x2cv protocol with strict Ship ID isolation.
Outputs metrics for both 4-Class and 3-Class tasks with Audio-Level Majority Voting.
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
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
import sklearn.metrics as sk_metrics
import scipy.stats as scipy_stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# SP Index calculation
def compute_sp_index(targets, preds, num_classes=4):
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    return sp, recalls, cm


def load_cached_features(cache_file="data/iara_processed/cache_fusion_mel_demon.pkl"):
    print(f"Carregando cache unificado de {cache_file}...", flush=True)
    t0 = time.time()
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    print(f"Sucesso: {len(data['fused'])} arquivos carregados em {time.time()-t0:.2f}s.", flush=True)
    return data["mel"], data["demon"], data["fused"]


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


def build_window_matrices(feature_cache, file_dict, exclude_target=None):
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


# -------------------------------------------------------------
# 1. SVM NYSTRÖM (m=6000) EVALUATION
# -------------------------------------------------------------
def evaluate_svm_nystroem(feature_cache, folds, experiment_name="SVM (m=6000)", num_classes=4, n_components=6000):
    class_names = ["SMALL", "MEDIUM", "LARGE", "AMBIENTAL"] if num_classes == 4 else ["SMALL", "MEDIUM", "LARGE"]
    exclude_target = 3 if num_classes == 3 else None
    problem_name = f"{num_classes}-Classes"
    in_dim = next(iter(feature_cache.values())).shape[1]

    print("\n" + "=" * 80, flush=True)
    print(f"AVALIAÇÃO: {experiment_name} [{problem_name}]", flush=True)
    print(f"Dimensão: {in_dim} | Landmarks: {n_components} | Kernel: RBF (gamma={1.0/in_dim:.6f}) | Penalidade: ElasticNet", flush=True)
    print("=" * 80, flush=True)

    fold_accs, fold_sps, fold_recalls = [], [], []
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for fold in folds:
        fold_idx = fold["fold_idx"]
        t_start = time.time()

        X_train, y_train, _ = build_window_matrices(feature_cache, fold["train_files"], exclude_target)
        X_test, y_test, test_fids = build_window_matrices(feature_cache, fold["test_files"], exclude_target)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        gamma = 1.0 / in_dim
        nystroem = Nystroem(kernel="rbf", gamma=gamma, n_components=n_components, random_state=42 + fold_idx, n_jobs=-1)
        X_train_proj = nystroem.fit_transform(X_train)
        X_test_proj = nystroem.transform(X_test)

        clf = SGDClassifier(
            loss="hinge",
            penalty="elasticnet",
            l1_ratio=0.15,
            alpha=1.0 / (2.0 * len(X_train)),
            class_weight="balanced",
            random_state=42,
            max_iter=1000,
            tol=1e-3
        )
        clf.fit(X_train_proj, y_train)
        win_preds = clf.predict(X_test_proj)

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

    print(f"\n>> {experiment_name} [{problem_name}]: Acurácia = {mean_acc:.2f}% ± {std_acc:.2f}% | SP = {mean_sp:.2f}% ± {std_sp:.2f}%", flush=True)
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


# -------------------------------------------------------------
# 2. CNN 2D PYTORCH EVALUATION
# -------------------------------------------------------------
def build_image_fold_data(feature_cache, file_dict, n_windows=32, exclude_target=None):
    images, labels, file_ids = [], [], []
    for fid, target in file_dict.items():
        if exclude_target is not None and target == exclude_target:
            continue
        if fid not in feature_cache:
            continue
        windows = feature_cache[fid] # (55, D)
        N = len(windows)
        if N < n_windows:
            pad_needed = n_windows - N
            padded = np.pad(windows, ((0, pad_needed), (0, 0)), mode="wrap")
            images.append(padded[np.newaxis, :, :])
            labels.append(target)
            file_ids.append(fid)
        else:
            stride = max(1, n_windows // 2)
            for start in range(0, N - n_windows + 1, stride):
                img = windows[start:start+n_windows]
                images.append(img[np.newaxis, :, :])
                labels.append(target)
                file_ids.append(fid)

    X = np.array(images, dtype=np.float32) # (M, 1, 32, D)
    y = np.array(labels, dtype=np.int64)
    file_ids = np.array(file_ids, dtype=np.int32)
    return X, y, file_ids


class PyTorch2DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def evaluate_cnn(feature_cache, folds, experiment_name="CNN 2D PyTorch", num_classes=4, epochs=25, batch_size=32, lr=1e-3):
    class_names = ["SMALL", "MEDIUM", "LARGE", "AMBIENTAL"] if num_classes == 4 else ["SMALL", "MEDIUM", "LARGE"]
    exclude_target = 3 if num_classes == 3 else None
    problem_name = f"{num_classes}-Classes"
    in_dim = next(iter(feature_cache.values())).shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80, flush=True)
    print(f"AVALIAÇÃO: {experiment_name} [{problem_name}]", flush=True)
    print(f"Dimensão de entrada: (1, 32, {in_dim}) | Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Arquitetura: Conv2D [64, 32] + GAP + FC [64, {num_classes}] | Epochs: {epochs} | Batch: {batch_size}", flush=True)
    print("=" * 80, flush=True)

    fold_accs, fold_sps, fold_recalls = [], [], []
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for fold in folds:
        fold_idx = fold["fold_idx"]
        t_start = time.time()

        X_train, y_train, _ = build_image_fold_data(feature_cache, fold["train_files"], n_windows=32, exclude_target=exclude_target)
        X_test, y_test, test_fids = build_image_fold_data(feature_cache, fold["test_files"], n_windows=32, exclude_target=exclude_target)

        mean = X_train.mean()
        std = X_train.std() + 1e-6
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        counts = np.bincount(y_train, minlength=num_classes)
        weights = len(y_train) / (num_classes * counts.astype(np.float32) + 1e-6)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
        N = len(X_train_t)

        model = PyTorch2DCNN(in_channels=1, num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model.train()
        for epoch in range(epochs):
            perm = torch.randperm(N, device=device)
            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                bx, by = X_train_t[idx], y_train_t[idx]
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            tst_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
            logits = model(tst_tensor)
            img_preds = torch.argmax(logits, dim=1).cpu().numpy()

        file_preds = defaultdict(list)
        file_targets = {}
        for fid, pred, tgt in zip(test_fids, img_preds, y_test):
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

    print(f"\n>> {experiment_name} [{problem_name}]: Acurácia = {mean_acc:.2f}% ± {std_acc:.2f}% | SP = {mean_sp:.2f}% ± {std_sp:.2f}%", flush=True)
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
    parser = argparse.ArgumentParser(description="Evaluate Fusion on SVM and CNN")
    parser.add_argument("--model", type=str, default="all", choices=["svm", "cnn", "all"])
    parser.add_argument("--n_components", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    cache_mel, cache_demon, cache_fused = load_cached_features()
    folds = get_official_folds()

    results = {}

    if args.model in ["svm", "all"]:
        print("\n" + "#" * 80, flush=True)
        print("INICIANDO AVALIAÇÃO: SVM NYSTRÖM (m=6000) NA FUSÃO MULTIMODAL (MEL + DEMON)", flush=True)
        print("#" * 80, flush=True)
        results["SVM_FUSED_4C"] = evaluate_svm_nystroem(cache_fused, folds, experiment_name="SVM (m=6000) FUSÃO MEL+DEMON", num_classes=4, n_components=args.n_components)
        results["SVM_FUSED_3C"] = evaluate_svm_nystroem(cache_fused, folds, experiment_name="SVM (m=6000) FUSÃO MEL+DEMON", num_classes=3, n_components=args.n_components)

    if args.model in ["cnn", "all"]:
        print("\n" + "#" * 80, flush=True)
        print("INICIANDO AVALIAÇÃO: CNN 2D PYTORCH NA FUSÃO MULTIMODAL (MEL + DEMON)", flush=True)
        print("#" * 80, flush=True)
        results["CNN_FUSED_4C"] = evaluate_cnn(cache_fused, folds, experiment_name="CNN 2D FUSÃO MEL+DEMON", num_classes=4, epochs=args.epochs)
        results["CNN_FUSED_3C"] = evaluate_cnn(cache_fused, folds, experiment_name="CNN 2D FUSÃO MEL+DEMON", num_classes=3, epochs=args.epochs)

    # Save results to pickle
    out_pkl = "data/iara_processed/results_fusion_svm_cnn.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResultados salvos com sucesso em {out_pkl}", flush=True)


if __name__ == "__main__":
    main()
