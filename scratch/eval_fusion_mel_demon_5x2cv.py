"""
Evaluation of Multimodal Fusion: MEL + DEMON on the official 5x2cv protocol.

This script evaluates:
1. Pure MEL baseline (256 bands, aligned to 5.12s windows)
2. Pure DEMON baseline (253 modulation bins, 0.39 - 25 Hz)
3. Fused MEL + DEMON (509 features: 256 Mel + 253 DEMON)

All evaluated with PyTorch MLP on the NVIDIA RTX 5060 Ti GPU inside the Docker container,
using the exact 10 folds of 5x2cv (split by recording `ID`; the same ship may be in train and test)
on both 4-Class and 3-Class tasks. Window-level predictions are saved in results/trainings/multirep/<exp>/.
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
import sklearn.metrics as sk_metrics
import scipy.stats as scipy_stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from results_writer import ResultsWriter


def compute_sp_index(targets, preds, num_classes=4):
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    return sp, recalls, cm


def load_and_fuse_features(
    mel_dir="data/iara_processed/log_melgram_a68edc75124f6c2ae6f093281fb42a9a",
    demon_dir="data/iara_processed/demon_features",
    cache_file="data/iara_processed/cache_fusion_mel_demon.pkl"
):
    if os.path.exists(cache_file):
        print(f"Carregando cache unificado de {cache_file}...", flush=True)
        t0 = time.time()
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print(f"Cache carregado em {time.time()-t0:.2f}s.", flush=True)
        return data["mel"], data["demon"], data["fused"]

    print(f"Carregando e alinhando temporalmente MEL de {mel_dir} e DEMON de {demon_dir}...", flush=True)
    t0 = time.time()
    
    mel_files = {int(os.path.splitext(f)[0]): f for f in os.listdir(mel_dir) if f.endswith(".pkl")}
    demon_files = {int(os.path.splitext(f)[0]): f for f in os.listdir(demon_dir) if f.endswith(".pkl")}
    
    common_ids = sorted(set(mel_files.keys()).intersection(set(demon_files.keys())))
    print(f"Total de arquivos comuns: {len(common_ids)}", flush=True)
    
    cache_mel = {}
    cache_demon = {}
    cache_fused = {}

    for fid in common_ids:
        # Load MEL
        with open(os.path.join(mel_dir, mel_files[fid]), "rb") as f:
            d_mel = pickle.load(f)
            df_mel = d_mel["df"].values.astype(np.float32)
            tm_mel = np.array(d_mel["times"], dtype=np.float32)

        # Load DEMON
        with open(os.path.join(demon_dir, demon_files[fid]), "rb") as f:
            d_demon = pickle.load(f)
            df_demon = d_demon["df"].values.astype(np.float32)
            tm_demon = np.array(d_demon["times"], dtype=np.float32)

        # Temporal alignment: average MEL windows in [t - 5.12, t] for each DEMON window
        mel_aligned = []
        for t in tm_demon:
            mask = (tm_mel >= (t - 5.12)) & (tm_mel <= t)
            if np.any(mask):
                mel_aligned.append(df_mel[mask].mean(axis=0))
            else:
                nearest_idx = np.argmin(np.abs(tm_mel - t))
                mel_aligned.append(df_mel[nearest_idx])

        mel_aligned = np.array(mel_aligned, dtype=np.float32) # (N, 256)
        fused = np.hstack([mel_aligned, df_demon])            # (N, 509)

        cache_mel[fid] = mel_aligned
        cache_demon[fid] = df_demon
        cache_fused[fid] = fused

    elapsed = time.time() - t0
    print(f"Sucesso: {len(cache_fused)} arquivos processados e alinhados em {elapsed:.2f}s. Salvando cache...", flush=True)
    with open(cache_file, "wb") as f:
        pickle.dump({"mel": cache_mel, "demon": cache_demon, "fused": cache_fused}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Cache unificado salvo com sucesso.", flush=True)
    return cache_mel, cache_demon, cache_fused


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


class PyTorchFusionMLP(nn.Module):
    def __init__(self, in_features=509, num_classes=4, hidden=[64, 32], dropout=0.2):
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


def evaluate_mlp_experiment(
    feature_cache,
    folds,
    experiment_name="MEL + DEMON Fusion",
    num_classes=4,
    hidden=[64, 32],
    epochs=35,
    batch_size=64,
    lr=1e-3,
    dropout=0.2,
    rep_tag="mel-demon",
):
    class_names = ["SMALL", "MEDIUM", "LARGE", "AMBIENTAL"] if num_classes == 4 else ["SMALL", "MEDIUM", "LARGE"]
    exclude_target = 3 if num_classes == 3 else None
    problem_name = f"{num_classes}-Classes"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = next(iter(feature_cache.values())).shape[1]

    print("\n" + "=" * 80, flush=True)
    print(f"AVALIAÇÃO: {experiment_name} [{problem_name}]", flush=True)
    print(f"Dimensão: {in_dim} | Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Arquitetura: MLP {hidden} | Epochs: {epochs} | Batch: {batch_size} | lr: {lr} | Dropout: {dropout}", flush=True)
    print("=" * 80, flush=True)

    writer = ResultsWriter(
        f"mlp_{rep_tag}_h{'-'.join(map(str, hidden))}_e{epochs}_{num_classes}c", script=__file__,
        params={"model": "PyTorchFusionMLP", "representation": rep_tag, "dim": in_dim,
                "window": "5.12 s (MEL alinhado ao DEMON)", "hidden": hidden, "dropout": dropout,
                "epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": 1e-3,
                "early_stopping": False, "scaler": "StandardScaler(treino)", "num_classes": num_classes})

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

        # Place training tensors directly on GPU VRAM for ultra-fast training
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
        N = len(X_train_t)

        model = PyTorchFusionMLP(in_features=in_dim, num_classes=num_classes, hidden=hidden, dropout=dropout).to(device)
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
        print(f"  Fold {fold_idx} ({elapsed:.1f}s): Acc={acc:.2f}% | SP={sp:.2f}% | Recalls: {rec_str}", flush=True)

    mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
    mean_sp, std_sp = np.mean(fold_sps), np.std(fold_sps)
    mean_recs = np.mean(fold_recalls, axis=0) * 100

    print(f"\n>> {experiment_name} [{problem_name}]: Acurácia = {mean_acc:.2f}% ± {std_acc:.2f}% | SP = {mean_sp:.2f}% ± {std_sp:.2f}%", flush=True)
    for cn, mr in zip(class_names, mean_recs):
        print(f"   Recall {cn:<10}: {mr:.2f}%", flush=True)

    cm_norm = total_cm.astype(float) / total_cm.sum(axis=1)[:, np.newaxis] * 100
    print("\nMatriz de Confusão Normalizada (%):", flush=True)
    df_cm = pd.DataFrame(np.round(cm_norm, 1), index=class_names, columns=class_names)
    print(df_cm, flush=True)

    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_sp": mean_sp,
        "std_sp": std_sp,
        "mean_recs": mean_recs,
        "cm_norm": cm_norm,
        "total_cm": total_cm,
        "class_names": class_names
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Fusion: MEL + DEMON")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    cache_mel, cache_demon, cache_fused = load_and_fuse_features()
    folds = get_official_folds()

    print("\n" + "#" * 80, flush=True)
    print("INICIANDO COMPARAÇÃO CONTROLADA: MEL vs DEMON vs FUSÃO (MEL + DEMON)", flush=True)
    print("#" * 80, flush=True)

    results = {}

    # 1. MEL Isolado (256-D, 5.12s)
    results["MEL_4C"] = evaluate_mlp_experiment(
        cache_mel, folds, experiment_name="MEL Isolado (256-D)", rep_tag="mel5s", num_classes=4, hidden=[32, 16], epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
    results["MEL_3C"] = evaluate_mlp_experiment(
        cache_mel, folds, experiment_name="MEL Isolado (256-D)", rep_tag="mel5s", num_classes=3, hidden=[32, 16], epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )

    # 2. DEMON Isolado (253-D, 5.12s)
    results["DEMON_4C"] = evaluate_mlp_experiment(
        cache_demon, folds, experiment_name="DEMON Isolado (253-D)", rep_tag="demon", num_classes=4, hidden=[32, 16], epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
    results["DEMON_3C"] = evaluate_mlp_experiment(
        cache_demon, folds, experiment_name="DEMON Isolado (253-D)", rep_tag="demon", num_classes=3, hidden=[32, 16], epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )

    # 3. FUSÃO MULTIMODAL: MEL + DEMON (509-D)
    results["FUSED_4C"] = evaluate_mlp_experiment(
        cache_fused, folds, experiment_name="FUSÃO MULTIMODAL MEL + DEMON (509-D)", rep_tag="mel5s-demon", num_classes=4, hidden=[64, 32], epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
    results["FUSED_3C"] = evaluate_mlp_experiment(
        cache_fused, folds, experiment_name="FUSÃO MULTIMODAL MEL + DEMON (509-D)", rep_tag="mel5s-demon", num_classes=3, hidden=[64, 32], epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )

    # Quadro Resumo Final
    print("\n" + "=" * 90, flush=True)
    print("QUADRO RESUMO CONSOLIDADO: IMPACTO DA FUSÃO MULTIMODAL (MEL + DEMON)", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Configuração':<35} | {'Acurácia 4C':<18} | {'Índice SP 4C':<18} | {'Índice SP 3C':<18}", flush=True)
    print("-" * 90, flush=True)
    print(f"{'DEMON Isolado (253-D)':<35} | {results['DEMON_4C']['mean_acc']:.2f}% ± {results['DEMON_4C']['std_acc']:.2f}%     | {results['DEMON_4C']['mean_sp']:.2f}% ± {results['DEMON_4C']['std_sp']:.2f}%     | {results['DEMON_3C']['mean_sp']:.2f}% ± {results['DEMON_3C']['std_sp']:.2f}%", flush=True)
    print(f"{'MEL Isolado (256-D)':<35} | {results['MEL_4C']['mean_acc']:.2f}% ± {results['MEL_4C']['std_acc']:.2f}%     | {results['MEL_4C']['mean_sp']:.2f}% ± {results['MEL_4C']['std_sp']:.2f}%     | {results['MEL_3C']['mean_sp']:.2f}% ± {results['MEL_3C']['std_sp']:.2f}%", flush=True)
    print(f"{'FUSÃO MULTIMODAL: MEL + DEMON (509-D)':<35} | {results['FUSED_4C']['mean_acc']:.2f}% ± {results['FUSED_4C']['std_acc']:.2f}%     | {results['FUSED_4C']['mean_sp']:.2f}% ± {results['FUSED_4C']['std_sp']:.2f}%     | {results['FUSED_3C']['mean_sp']:.2f}% ± {results['FUSED_3C']['std_sp']:.2f}%", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
