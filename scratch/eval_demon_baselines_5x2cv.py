"""
Evaluation of DEMON features with Baseline Models:
- Random Forest (RF)
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)

Evaluated on the official 10-fold 5x2cv protocol (split by recording `ID`; the same ship may be in train and test).
Window-level predictions are saved in results/trainings/multirep/<exp>/ (results_writer.py).
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as sk_metrics
import scipy.stats as scipy_stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from results_writer import ResultsWriter

# SP Index calculation
def compute_sp_index(targets, preds, num_classes=4):
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    return sp, recalls, cm


def load_all_demon_features(demon_dir="data/iara_processed/demon_features"):
    print(f"Carregando features DEMON de {demon_dir}...")
    t0 = time.time()
    feature_cache = {}
    file_list = [f for f in os.listdir(demon_dir) if f.endswith(".pkl")]
    for fname in file_list:
        file_id = int(os.path.splitext(fname)[0])
        fpath = os.path.join(demon_dir, fname)
        with open(fpath, "rb") as f:
            data = pickle.load(f)
            feature_cache[file_id] = data["df"].values.astype(np.float32)
    print(f"Sucesso: {len(feature_cache)} arquivos carregados em {time.time()-t0:.2f}s.")
    return feature_cache


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


# -------------------------------------------------------------
# 1. RANDOM FOREST EVALUATION
# -------------------------------------------------------------
def evaluate_random_forest(feature_cache, folds, num_classes=4, n_estimators=100):
    class_names = ["SMALL", "MEDIUM", "LARGE", "AMBIENTAL"] if num_classes == 4 else ["SMALL", "MEDIUM", "LARGE"]
    exclude_target = 3 if num_classes == 3 else None
    problem_name = f"{num_classes}-Classes"

    print("\n" + "=" * 80)
    print(f"AVALIAÇÃO DEMON: Random Forest (n_estimators={n_estimators}, balanced) [{problem_name}]")
    print("=" * 80)
    writer = ResultsWriter(
        f"rf_demon_n{n_estimators}_balanced_{num_classes}c", script=__file__,
        params={"model": "RandomForestClassifier", "representation": "DEMON", "dim": 253,
                "n_estimators": n_estimators, "max_depth": None, "class_weight": "balanced",
                "random_state": 42, "num_classes": num_classes})

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
        writer.save_fold(fold_idx, test_fids, y_test, win_preds,
                         train_files={k: v for k, v in fold["train_files"].items() if v != exclude_target},
                         scores=rf.predict_proba(X_test))

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
        print(f"  Fold {fold_idx} ({elapsed:.1f}s): Acc={acc:.2f}% | SP={sp:.2f}% | Recalls: {rec_str}")

    mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
    mean_sp, std_sp = np.mean(fold_sps), np.std(fold_sps)
    mean_recs = np.mean(fold_recalls, axis=0) * 100

    print(f"\n>> RF DEMON {problem_name}: Acurácia = {mean_acc:.2f}% ± {std_acc:.2f}% | SP = {mean_sp:.2f}% ± {std_sp:.2f}%")
    for cn, mr in zip(class_names, mean_recs):
        print(f"   Recall {cn:<10}: {mr:.2f}%")

    cm_norm = total_cm.astype(float) / total_cm.sum(axis=1)[:, np.newaxis] * 100
    print("\nMatriz de Confusão Normalizada (%):")
    print(pd.DataFrame(np.round(cm_norm, 1), index=class_names, columns=class_names))

    return {"mean_acc": mean_acc, "std_acc": std_acc, "mean_sp": mean_sp, "std_sp": std_sp, "mean_recs": mean_recs}


# -------------------------------------------------------------
# 2. MLP EVALUATION (PyTorch Matching Silva et al. 2025)
# -------------------------------------------------------------
class PyTorchMLP(nn.Module):
    def __init__(self, in_features=253, num_classes=4, hidden=[32, 16], dropout=0.2):
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
    print(f"AVALIAÇÃO DEMON: MLP PyTorch [32, 16] (Batch={batch_size}, lr={lr}) [{problem_name}]")
    print("=" * 80)
    writer = ResultsWriter(
        f"mlp_demon_h32-16_e{epochs}_{num_classes}c", script=__file__,
        params={"model": "PyTorchMLP", "representation": "DEMON", "dim": 253, "hidden": [32, 16],
                "epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": 1e-3,
                "early_stopping": False, "scaler": "StandardScaler(treino)", "num_classes": num_classes})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
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

        # Class weights for CrossEntropyLoss
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
        writer.save_fold(fold_idx, test_fids, y_test, win_preds,
                         train_files={k: v for k, v in fold["train_files"].items() if v != exclude_target},
                         scores=torch.softmax(logits, dim=1).cpu().numpy())

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
        print(f"  Fold {fold_idx} ({elapsed:.1f}s): Acc={acc:.2f}% | SP={sp:.2f}% | Recalls: {rec_str}")

    mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
    mean_sp, std_sp = np.mean(fold_sps), np.std(fold_sps)
    mean_recs = np.mean(fold_recalls, axis=0) * 100

    print(f"\n>> MLP DEMON {problem_name}: Acurácia = {mean_acc:.2f}% ± {std_acc:.2f}% | SP = {mean_sp:.2f}% ± {std_sp:.2f}%")
    for cn, mr in zip(class_names, mean_recs):
        print(f"   Recall {cn:<10}: {mr:.2f}%")

    cm_norm = total_cm.astype(float) / total_cm.sum(axis=1)[:, np.newaxis] * 100
    print("\nMatriz de Confusão Normalizada (%):")
    print(pd.DataFrame(np.round(cm_norm, 1), index=class_names, columns=class_names))

    return {"mean_acc": mean_acc, "std_acc": std_acc, "mean_sp": mean_sp, "std_sp": std_sp, "mean_recs": mean_recs}


# -------------------------------------------------------------
# 3. CNN EVALUATION (2D Spectrogram/Gram on DEMON sequences)
# -------------------------------------------------------------
def build_image_fold_data(feature_cache, file_dict, n_windows=32, exclude_target=None):
    """
    Slices sequential windows of an audio file into (1, n_windows, 253) image tensors.
    """
    images = []
    labels = []
    file_ids = []

    for fid, target in file_dict.items():
        if exclude_target is not None and target == exclude_target:
            continue
        if fid not in feature_cache:
            continue
        windows = feature_cache[fid] # shape (N, 253)
        N = len(windows)
        if N < n_windows:
            # Pad with zeros or repeat
            pad_needed = n_windows - N
            padded = np.pad(windows, ((0, pad_needed), (0, 0)), mode="wrap")
            images.append(padded[np.newaxis, :, :])
            labels.append(target)
            file_ids.append(fid)
        else:
            # Stride of n_windows // 2
            stride = max(1, n_windows // 2)
            for start in range(0, N - n_windows + 1, stride):
                img = windows[start:start+n_windows]
                images.append(img[np.newaxis, :, :])
                labels.append(target)
                file_ids.append(fid)

    X = np.array(images, dtype=np.float32) # (M, 1, 32, 253)
    y = np.array(labels, dtype=np.int64)
    file_ids = np.array(file_ids, dtype=np.int32)
    return X, y, file_ids


class PyTorchDEMONCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)) # (32, 253) -> (16, 126)
        )
        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)) # (16, 126) -> (8, 63)
        )
        # Global Average Pooling + Linear
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
    print(f"AVALIAÇÃO DEMON: CNN 2D PyTorch [64, 32] (Images 32x253) [{problem_name}]")
    print("=" * 80)
    writer = ResultsWriter(
        f"cnn2d_demon_w32_b{batch_size}_e{epochs}_{num_classes}c", script=__file__,
        params={"model": "PyTorchDEMONCNN [64, 32]", "representation": "DEMON", "dim": 253, "image_windows": 32,
                "epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": 1e-3,
                "early_stopping": False, "normalization": "z-score escalar global (treino)",
                "num_classes": num_classes})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    fold_accs, fold_sps, fold_recalls = [], [], []
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for fold in folds:
        fold_idx = fold["fold_idx"]
        t_start = time.time()

        X_train, y_train, _ = build_image_fold_data(feature_cache, fold["train_files"], n_windows=32, exclude_target=exclude_target)
        X_test, y_test, test_fids = build_image_fold_data(feature_cache, fold["test_files"], n_windows=32, exclude_target=exclude_target)

        # Normalize features across dataset
        mean = X_train.mean()
        std = X_train.std() + 1e-6
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        counts = np.bincount(y_train, minlength=num_classes)
        weights = len(y_train) / (num_classes * counts.astype(np.float32) + 1e-6)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

        trn_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)

        model = PyTorchDEMONCNN(in_channels=1, num_classes=num_classes).to(device)
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

        model.eval()
        with torch.no_grad():
            tst_tensor = torch.tensor(X_test, device=device)
            logits = model(tst_tensor)
            img_preds = torch.argmax(logits, dim=1).cpu().numpy()
        writer.save_fold(fold_idx, test_fids, y_test, img_preds,
                         train_files={k: v for k, v in fold["train_files"].items() if v != exclude_target},
                         scores=torch.softmax(logits, dim=1).cpu().numpy())

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
        print(f"  Fold {fold_idx} ({elapsed:.1f}s): Acc={acc:.2f}% | SP={sp:.2f}% | Recalls: {rec_str}")

    mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
    mean_sp, std_sp = np.mean(fold_sps), np.std(fold_sps)
    mean_recs = np.mean(fold_recalls, axis=0) * 100

    print(f"\n>> CNN DEMON {problem_name}: Acurácia = {mean_acc:.2f}% ± {std_acc:.2f}% | SP = {mean_sp:.2f}% ± {std_sp:.2f}%")
    for cn, mr in zip(class_names, mean_recs):
        print(f"   Recall {cn:<10}: {mr:.2f}%")

    cm_norm = total_cm.astype(float) / total_cm.sum(axis=1)[:, np.newaxis] * 100
    print("\nMatriz de Confusão Normalizada (%):")
    print(pd.DataFrame(np.round(cm_norm, 1), index=class_names, columns=class_names))

    return {"mean_acc": mean_acc, "std_acc": std_acc, "mean_sp": mean_sp, "std_sp": std_sp, "mean_recs": mean_recs}


def main():
    parser = argparse.ArgumentParser(description="Evaluate DEMON on RF, MLP, CNN")
    parser.add_argument("--model", type=str, default="all", choices=["rf", "mlp", "cnn", "all"])
    args = parser.parse_args()

    cache = load_all_demon_features()
    folds = get_official_folds()

    results = {}

    if args.model in ["rf", "all"]:
        results["RF_4C"] = evaluate_random_forest(cache, folds, num_classes=4)
        results["RF_3C"] = evaluate_random_forest(cache, folds, num_classes=3)

    if args.model in ["mlp", "all"]:
        results["MLP_4C"] = evaluate_mlp(cache, folds, num_classes=4, epochs=25)
        results["MLP_3C"] = evaluate_mlp(cache, folds, num_classes=3, epochs=25)

    if args.model in ["cnn", "all"]:
        results["CNN_4C"] = evaluate_cnn(cache, folds, num_classes=4, epochs=20)
        results["CNN_3C"] = evaluate_cnn(cache, folds, num_classes=3, epochs=20)

    print("\n" + "=" * 90)
    print("QUADRO RESUMO: COMPARATIVO DEMON COM BASELINES DO ARTIGO ORIGINAL (SILVA ET AL. 2025)")
    print("=" * 90)
    print(f"{'Modelo':<20} | {'Representação':<15} | {'Acurácia 4C':<18} | {'Índice SP 4C':<18}")
    print("-" * 90)
    print(f"{'RF (Artigo)':<20} | {'MEL':<15} | {'62.64% ± 1.84%':<18} | {'62.22% ± 1.86%':<18}")
    print(f"{'RF (Artigo)':<20} | {'LOFAR':<15} | {'58.87% ± 1.68%':<18} | {'56.92% ± 1.69%':<18}")
    if "RF_4C" in results:
        print(f"{'RF (Novo)':<20} | {'DEMON':<15} | {results['RF_4C']['mean_acc']:.2f}% ± {results['RF_4C']['std_acc']:.2f}%     | {results['RF_4C']['mean_sp']:.2f}% ± {results['RF_4C']['std_sp']:.2f}%")
    print("-" * 90)
    print(f"{'MLP (Artigo)':<20} | {'MEL':<15} | {'64.51% ± 1.75%':<18} | {'63.38% ± 1.81%':<18}")
    print(f"{'MLP (Artigo)':<20} | {'LOFAR':<15} | {'67.48% ± 1.24%':<18} | {'66.51% ± 1.39%':<18}")
    if "MLP_4C" in results:
        print(f"{'MLP (Novo)':<20} | {'DEMON':<15} | {results['MLP_4C']['mean_acc']:.2f}% ± {results['MLP_4C']['std_acc']:.2f}%     | {results['MLP_4C']['mean_sp']:.2f}% ± {results['MLP_4C']['std_sp']:.2f}%")
    print("-" * 90)
    print(f"{'CNN (Artigo)':<20} | {'MEL':<15} | {'64.99% ± 2.09%':<18} | {'63.52% ± 2.26%':<18}")
    print(f"{'CNN (Artigo)':<20} | {'LOFAR':<15} | {'67.02% ± 1.78%':<18} | {'66.05% ± 1.90%':<18}")
    if "CNN_4C" in results:
        print(f"{'CNN (Novo)':<20} | {'DEMON':<15} | {results['CNN_4C']['mean_acc']:.2f}% ± {results['CNN_4C']['std_acc']:.2f}%     | {results['CNN_4C']['mean_sp']:.2f}% ± {results['CNN_4C']['std_sp']:.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
