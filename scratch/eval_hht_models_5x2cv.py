"""
Avaliação Completa dos 4 Modelos no Dataset HHT (256 Bins):
1. SVM Nyström (m=6000, RBF, ElasticNet)
2. Random Forest (100 árvores, balanced)
3. MLP PyTorch (256 -> 32 -> 16 -> C, BatchNorm, Dropout)
4. CNN 2D PyTorch (Imagens 32x256 de Hilbert-Huang Spectrogram)

Protocolo Oficial 5x2cv (10 Folds com isolamento estrito por Ship ID).
Avaliação tanto para 4 Classes quanto para 3 Classes (sem ambiental).
Votação majoritária por áudio (Majority Voting).
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
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sk_metrics
import scipy.stats as scipy_stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# Cálculo oficial do Índice SP (Standardized Performance)
def compute_sp_index(targets, preds, num_classes=4):
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    return sp, recalls, cm


def load_all_hht_features(hht_dir="data/iara_processed/hht_features", cache_path="data/iara_processed/cache_hht.pkl"):
    """
    Carrega todas as features HHT (256 bins) em memória (com cache unificado).
    Retorna: dict mapping file_id (int) -> (numpy array of windows, shape (N, 256))
    """
    if os.path.exists(cache_path):
        print(f"\nCarregando features HHT unificadas de '{cache_path}'...", flush=True)
        t0 = time.time()
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"Sucesso: {len(cache)} arquivos carregados em {time.time()-t0:.2f}s.", flush=True)
        return cache

    print(f"\nCarregando features HHT individuais de '{hht_dir}'...", flush=True)
    t0 = time.time()
    feature_cache = {}
    
    file_list = [f for f in os.listdir(hht_dir) if f.endswith(".pkl")]
    for fname in file_list:
        file_id = int(os.path.splitext(fname)[0])
        fpath = os.path.join(hht_dir, fname)
        try:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
                df = data["df"]
                feature_cache[file_id] = df.values.astype(np.float32)
        except Exception as e:
            print(f"Erro ao ler {fpath}: {e}", flush=True)

    elapsed = time.time() - t0
    print(f"Sucesso: {len(feature_cache)} arquivos carregados em {elapsed:.2f}s.", flush=True)

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(feature_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cache unificado salvo com sucesso em '{cache_path}'.", flush=True)
    except Exception as e:
        print(f"Aviso ao salvar cache: {e}", flush=True)

    return feature_cache


def get_official_folds(eval_base_dir="results/trainings/tests/svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0/eval"):
    """
    Extrai partições oficiais dos 10 folds 5x2cv com isolamento por Ship ID.
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
            "train_files": train_files,
            "test_files": test_files,
        })
    return folds


def build_fold_matrices(feature_cache, file_dict, exclude_target=None):
    """
    Constrói matrizes de treino/teste por janela.
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


# ==============================================================================
# 1. SVM NYSTRÖM (m=6000)
# ==============================================================================
def evaluate_svm_nystroem(
    feature_cache,
    folds,
    n_components=6000,
    C=2.0,
    penalty="elasticnet",
    l1_ratio=0.15,
    num_classes=4,
):
    class_names = ["SMALL", "MEDIUM", "LARGE", "AMBIENTAL"] if num_classes == 4 else ["SMALL", "MEDIUM", "LARGE"]
    exclude_target = 3 if num_classes == 3 else None
    problem_name = f"{num_classes}-Classes"

    print("\n" + "=" * 80)
    print(f"AVALIAÇÃO HHT: SVM Nyström (m={n_components}, C={C}, penalty={penalty}) [{problem_name}]")
    print("=" * 80)

    fold_accuracies, fold_sp_indices, fold_recalls = [], [], []
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for fold in folds:
        fold_idx = fold["fold_idx"]
        t_start = time.time()

        X_train, y_train, _ = build_fold_matrices(feature_cache, fold["train_files"], exclude_target=exclude_target)
        X_test, y_test, test_fids = build_fold_matrices(feature_cache, fold["test_files"], exclude_target=exclude_target)

        # Standard Scaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Nystroem kernel approximation (RBF)
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

        # SGD Linear Classifier
        alpha = 1.0 / (C * X_train.shape[0])
        clf = SGDClassifier(
            loss="hinge",
            penalty=penalty,
            l1_ratio=l1_ratio,
            alpha=alpha,
            max_iter=200,
            tol=1e-3,
            early_stopping=True,
            n_iter_no_change=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train_trans, y_train)

        # Window predictions
        win_preds = clf.predict(X_test_trans)

        # Majority voting per audio
        file_preds = defaultdict(list)
        file_targets = {}
        for fid, pred, tgt in zip(test_fids, win_preds, y_test):
            file_preds[fid].append(pred)
            file_targets[fid] = tgt

        audio_preds = [int(np.argmax(np.bincount(file_preds[fid], minlength=num_classes))) for fid in file_preds]
        audio_targets = [int(file_targets[fid]) for fid in file_preds]

        acc = sk_metrics.accuracy_score(audio_targets, audio_preds) * 100
        sp, recalls, cm = compute_sp_index(audio_targets, audio_preds, num_classes=num_classes)

        fold_accuracies.append(acc)
        fold_sp_indices.append(sp)
        fold_recalls.append(recalls)
        total_cm += cm

        elapsed = time.time() - t_start
        rec_str = " | ".join([f"{cn}: {r*100:.1f}%" for cn, r in zip(class_names, recalls)])
        print(f"  Fold {fold_idx} ({elapsed:.1f}s): Acc={acc:.2f}% | SP={sp:.2f}% | Recalls: {rec_str}", flush=True)

    mean_acc, std_acc = np.mean(fold_accuracies), np.std(fold_accuracies)
    mean_sp, std_sp = np.mean(fold_sp_indices), np.std(fold_sp_indices)
    mean_recs = np.mean(fold_recalls, axis=0) * 100

    print(f"\n>> SVM Nyström HHT {problem_name}: Acurácia = {mean_acc:.2f}% ± {std_acc:.2f}% | SP = {mean_sp:.2f}% ± {std_sp:.2f}%", flush=True)
    for cn, mr in zip(class_names, mean_recs):
        print(f"   Recall {cn:<10}: {mr:.2f}%", flush=True)

    cm_norm = total_cm.astype(float) / total_cm.sum(axis=1)[:, np.newaxis] * 100
    print("\nMatriz de Confusão Normalizada (%):", flush=True)
    print(pd.DataFrame(np.round(cm_norm, 1), index=class_names, columns=class_names), flush=True)

    return {"mean_acc": mean_acc, "std_acc": std_acc, "mean_sp": mean_sp, "std_sp": std_sp, "mean_recs": mean_recs, "cm": total_cm, "cm_norm": cm_norm}


# ==============================================================================
# 2. RANDOM FOREST (100 árvores)
# ==============================================================================
def evaluate_random_forest(feature_cache, folds, num_classes=4, n_estimators=100):
    class_names = ["SMALL", "MEDIUM", "LARGE", "AMBIENTAL"] if num_classes == 4 else ["SMALL", "MEDIUM", "LARGE"]
    exclude_target = 3 if num_classes == 3 else None
    problem_name = f"{num_classes}-Classes"

    print("\n" + "=" * 80)
    print(f"AVALIAÇÃO HHT: Random Forest (n_estimators={n_estimators}, balanced) [{problem_name}]")
    print("=" * 80)

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

        # Majority voting per audio
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

    print(f"\n>> RF HHT {problem_name}: Acurácia = {mean_acc:.2f}% ± {std_acc:.2f}% | SP = {mean_sp:.2f}% ± {std_sp:.2f}%", flush=True)
    for cn, mr in zip(class_names, mean_recs):
        print(f"   Recall {cn:<10}: {mr:.2f}%", flush=True)

    cm_norm = total_cm.astype(float) / total_cm.sum(axis=1)[:, np.newaxis] * 100
    print("\nMatriz de Confusão Normalizada (%):", flush=True)
    print(pd.DataFrame(np.round(cm_norm, 1), index=class_names, columns=class_names), flush=True)

    return {"mean_acc": mean_acc, "std_acc": std_acc, "mean_sp": mean_sp, "std_sp": std_sp, "mean_recs": mean_recs, "cm": total_cm, "cm_norm": cm_norm}


# ==============================================================================
# 3. MLP PYTORCH (256 -> 32 -> 16 -> C)
# ==============================================================================
class PyTorchMLP(nn.Module):
    def __init__(self, in_features=256, num_classes=4, hidden=[32, 16], dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.BatchNorm1d(hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], num_classes)
        )

    def forward(self, x):
        return self.net(x)


def evaluate_mlp(feature_cache, folds, num_classes=4, epochs=30, batch_size=64, lr=1e-3):
    class_names = ["SMALL", "MEDIUM", "LARGE", "AMBIENTAL"] if num_classes == 4 else ["SMALL", "MEDIUM", "LARGE"]
    exclude_target = 3 if num_classes == 3 else None
    problem_name = f"{num_classes}-Classes"

    print("\n" + "=" * 80)
    print(f"AVALIAÇÃO HHT: MLP PyTorch [32, 16] (Batch={batch_size}, lr={lr}, epochs={epochs}) [{problem_name}]")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo PyTorch: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    fold_accs, fold_sps, fold_recalls = [], [], []
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for fold in folds:
        fold_idx = fold["fold_idx"]
        t_start = time.time()

        X_train, y_train, _ = build_fold_matrices(feature_cache, fold["train_files"], exclude_target)
        X_test, y_test, test_fids = build_fold_matrices(feature_cache, fold["test_files"], exclude_target)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        counts = np.bincount(y_train, minlength=num_classes)
        weights = len(y_train) / (num_classes * counts.astype(np.float32) + 1e-6)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

        trn_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)

        model = PyTorchMLP(in_features=X_train.shape[1], num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model.train()
        for epoch in range(epochs):
            for bx, by in trn_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            tst_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
            logits = model(tst_tensor)
            win_preds = torch.argmax(logits, dim=1).cpu().numpy()

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

    print(f"\n>> MLP HHT {problem_name}: Acurácia = {mean_acc:.2f}% ± {std_acc:.2f}% | SP = {mean_sp:.2f}% ± {std_sp:.2f}%", flush=True)
    for cn, mr in zip(class_names, mean_recs):
        print(f"   Recall {cn:<10}: {mr:.2f}%", flush=True)

    cm_norm = total_cm.astype(float) / total_cm.sum(axis=1)[:, np.newaxis] * 100
    print("\nMatriz de Confusão Normalizada (%):", flush=True)
    print(pd.DataFrame(np.round(cm_norm, 1), index=class_names, columns=class_names), flush=True)

    return {"mean_acc": mean_acc, "std_acc": std_acc, "mean_sp": mean_sp, "std_sp": std_sp, "mean_recs": mean_recs, "cm": total_cm, "cm_norm": cm_norm}


# ==============================================================================
# 4. CNN 2D PYTORCH (Imagens 32x256 de Hilbert-Huang Spectrogram)
# ==============================================================================
def build_image_fold_data(feature_cache, file_dict, n_windows=32, exclude_target=None):
    images = []
    labels = []
    file_ids = []

    for fid, target in file_dict.items():
        if exclude_target is not None and target == exclude_target:
            continue
        if fid not in feature_cache:
            continue
        windows = feature_cache[fid]  # shape (N, 256)
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

    X = np.array(images, dtype=np.float32)  # (M, 1, 32, 256)
    y = np.array(labels, dtype=np.int64)
    file_ids = np.array(file_ids, dtype=np.int32)
    return X, y, file_ids


class PyTorchHHTCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))  # (32, 256) -> (16, 128)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))  # (16, 128) -> (8, 64)
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


def evaluate_cnn(feature_cache, folds, num_classes=4, epochs=25, batch_size=32, lr=1e-3):
    class_names = ["SMALL", "MEDIUM", "LARGE", "AMBIENTAL"] if num_classes == 4 else ["SMALL", "MEDIUM", "LARGE"]
    exclude_target = 3 if num_classes == 3 else None
    problem_name = f"{num_classes}-Classes"

    print("\n" + "=" * 80)
    print(f"AVALIAÇÃO HHT: CNN 2D PyTorch [64, 32] (Imagens 32x256) [{problem_name}]")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo PyTorch: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

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

        trn_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)

        model = PyTorchHHTCNN(in_channels=1, num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model.train()
        for epoch in range(epochs):
            for bx, by in trn_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            tst_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
            logits = model(tst_tensor)
            slice_preds = torch.argmax(logits, dim=1).cpu().numpy()

        file_preds = defaultdict(list)
        file_targets = {}
        for fid, pred, tgt in zip(test_fids, slice_preds, y_test):
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

    print(f"\n>> CNN HHT {problem_name}: Acurácia = {mean_acc:.2f}% ± {std_acc:.2f}% | SP = {mean_sp:.2f}% ± {std_sp:.2f}%", flush=True)
    for cn, mr in zip(class_names, mean_recs):
        print(f"   Recall {cn:<10}: {mr:.2f}%", flush=True)

    cm_norm = total_cm.astype(float) / total_cm.sum(axis=1)[:, np.newaxis] * 100
    print("\nMatriz de Confusão Normalizada (%):", flush=True)
    print(pd.DataFrame(np.round(cm_norm, 1), index=class_names, columns=class_names), flush=True)

    return {"mean_acc": mean_acc, "std_acc": std_acc, "mean_sp": mean_sp, "std_sp": std_sp, "mean_recs": mean_recs, "cm": total_cm, "cm_norm": cm_norm}


# ==============================================================================
# MAIN BENCHMARK RUNNER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Avaliação dos 4 modelos em HHT (5x2cv)")
    parser.add_argument("--model", type=str, default="all", choices=["all", "svm", "rf", "mlp", "cnn"])
    parser.add_argument("--num_classes", type=int, default=0, choices=[0, 3, 4], help="0 for both 4 and 3 classes")
    args = parser.parse_args()

    feature_cache = load_all_hht_features()
    folds = get_official_folds()
    print(f"Partições dos 10 folds carregadas com sucesso.")

    tasks = [4, 3] if args.num_classes == 0 else [args.num_classes]
    out_pkl = "data/iara_processed/hht_benchmark_results.pkl"
    if os.path.exists(out_pkl):
        try:
            with open(out_pkl, "rb") as f:
                results = pickle.load(f)
            print(f"Resultados prévios carregados de {out_pkl}: {list(results.keys())}", flush=True)
        except Exception:
            results = {}
    else:
        results = {}

    for nc in tasks:
        print("\n" + "#" * 80, flush=True)
        print(f"INICIANDO BENCHMARK PARA {nc} CLASSES ({'COM' if nc == 4 else 'SEM'} AMBIENTAL)", flush=True)
        print("#" * 80, flush=True)

        # 1. Random Forest (100 árvores)
        if args.model in ["all", "rf"]:
            res_rf = evaluate_random_forest(feature_cache, folds, num_classes=nc, n_estimators=100)
            results[f"rf_{nc}c"] = res_rf
            with open(out_pkl, "wb") as f:
                pickle.dump(results, f)

        # 2. MLP PyTorch (256 -> 32 -> 16 -> C, GPU)
        if args.model in ["all", "mlp"]:
            res_mlp = evaluate_mlp(feature_cache, folds, num_classes=nc, epochs=30, batch_size=64)
            results[f"mlp_{nc}c"] = res_mlp
            with open(out_pkl, "wb") as f:
                pickle.dump(results, f)

        # 3. CNN 2D PyTorch (Imagens 32x256, GPU)
        if args.model in ["all", "cnn"]:
            res_cnn = evaluate_cnn(feature_cache, folds, num_classes=nc, epochs=25, batch_size=32)
            results[f"cnn_{nc}c"] = res_cnn
            with open(out_pkl, "wb") as f:
                pickle.dump(results, f)

        # 4. SVM Nyström (m=6000, RBF, ElasticNet)
        if args.model in ["all", "svm"]:
            res_svm = evaluate_svm_nystroem(feature_cache, folds, n_components=6000, C=2.0, num_classes=nc)
            results[f"svm_{nc}c"] = res_svm
            with open(out_pkl, "wb") as f:
                pickle.dump(results, f)

    print(f"\nResultados consolidados salvos em: {out_pkl}", flush=True)

    print("\n" + "=" * 80)
    print("TABELA RESUMO DE DESEMPENHO - HHT (256 BINS)")
    print("=" * 80)
    print(f"{'Modelo':<25} | {'Classes':<8} | {'Acurácia':<18} | {'Índice SP':<18}")
    print("-" * 80)
    for k, v in results.items():
        parts = k.split("_")
        mod_name = parts[0].upper()
        nc_str = parts[1]
        print(f"{mod_name:<25} | {nc_str:<8} | {v['mean_acc']:.2f}% ± {v['std_acc']:.2f}% | {v['mean_sp']:.2f}% ± {v['std_sp']:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
