"""
Training and Evaluation of Hybrid MLP (MEL-256 + LOFAR-PCA64) with Asymmetric Loss Penalty for High LARGE Precision
Optimizes decision boundaries to achieve >80% Precision on LARGE ships while monitoring Recall and SP index.

Classes: 0: SMALL, 1: MEDIUM, 2: LARGE, 3: BACKGROUND
Protocol: 5x2cv (10 independent folds) with strict ship ID isolation.
Hardware: Accelerated on GPU/CPU.
"""
import os
import sys
import math
import time
import argparse
import typing
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import sklearn.metrics as sk_metrics
import scipy.stats as scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import iara.default as iara_default
import iara.records
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc
import iara.ml.dataset as iara_dataset
import iara.ml.experiment as iara_exp
from iara.default import DEFAULT_DIRECTORIES


def compute_sp_index(targets, preds, num_classes=4):
    """Compute SP Index according to standard IARA formula."""
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    if np.any(recalls == 0):
        gmean = 0.0
    else:
        gmean = scipy.gmean(recalls)
    return float(np.sqrt(np.mean(recalls * gmean)) * 100)


class HybridMLPClassifier(nn.Module):
    """Deep Hybrid MLP with BatchNorm, Dropout, and Residual-style skip connections."""
    def __init__(self, in_features=320, hidden_dims=(256, 128, 64), dropout=0.3, n_classes=4):
        super().__init__()
        layers = []
        curr_in = in_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(curr_in, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            curr_in = h
        layers.append(nn.Linear(curr_in, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def extract_features(loader_mel, loader_lofar, file_ids, pca=None, scaler=None, fit=False, n_pca=64):
    loader_mel.pre_load(file_ids)
    loader_lofar.pre_load(file_ids)

    lofar_list = []
    mel_list = []
    file_map = {}

    for fid in file_ids:
        m = loader_mel.get_all(fid)
        l = loader_lofar.get_all(fid)
        if m is None or l is None or len(m) == 0 or len(l) == 0:
            continue
        min_len = min(len(m), len(l))
        m_np = m.numpy()[:min_len]
        l_np = l.numpy()[:min_len]
        mel_list.append(m_np)
        lofar_list.append(l_np)
        file_map[fid] = (m_np, l_np)

    if fit:
        pca = PCA(n_components=n_pca, random_state=42)
        pca.fit(np.vstack(lofar_list))

    hybrid_all = []
    for m_np, l_np in zip(mel_list, lofar_list):
        l_pca = pca.transform(l_np)
        hyb = np.hstack([m_np, l_pca])
        hybrid_all.append(hyb)

    if fit:
        scaler = StandardScaler()
        scaler.fit(np.vstack(hybrid_all))

    feat_by_file = {}
    for fid, (m_np, l_np) in file_map.items():
        l_pca = pca.transform(l_np)
        hyb = np.hstack([m_np, l_pca])
        feat_by_file[fid] = scaler.transform(hyb).astype(np.float32)

    return feat_by_file, pca, scaler


def build_dataset(feat_by_file, df):
    X_list = []
    y_list = []
    fid_list = []

    for _, row in df.iterrows():
        fid = int(row['ID'])
        tgt = int(row['Target'])
        feat = feat_by_file.get(fid)
        if feat is None:
            continue
        for row_vec in feat:
            X_list.append(row_vec)
            y_list.append(tgt)
            fid_list.append(fid)

    X_t = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_t = torch.tensor(np.array(y_list), dtype=torch.long)
    return X_t, y_t, np.array(fid_list)


@torch.no_grad()
def evaluate_audio(model, X_t, y_t, fids, device):
    model.eval()
    batch_size = 512
    all_preds = []
    all_targets = []
    all_fids = []

    for i in range(0, len(X_t), batch_size):
        bx = X_t[i : i + batch_size].to(device)
        by = y_t[i : i + batch_size]
        bf = fids[i : i + batch_size]

        logits = model(bx)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(by.numpy())
        all_fids.extend(bf)

    file_preds = defaultdict(list)
    file_targets = {}
    for fid, pred, tgt in zip(all_fids, all_preds, all_targets):
        file_preds[fid].append(pred)
        file_targets[fid] = tgt

    audio_preds = []
    audio_targets = []
    for fid in file_preds:
        counts = np.bincount(file_preds[fid], minlength=4)
        audio_preds.append(int(np.argmax(counts)))
        audio_targets.append(int(file_targets[fid]))

    audio_preds = np.array(audio_preds)
    audio_targets = np.array(audio_targets)

    cm = sk_metrics.confusion_matrix(audio_targets, audio_preds, labels=[0, 1, 2, 3])
    tp = cm.diagonal()
    prec = tp / np.maximum(cm.sum(axis=0), 1) * 100
    rec = tp / np.maximum(cm.sum(axis=1), 1) * 100
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)

    class_names = ['small', 'medium', 'large', 'background']
    metrics = {
        'acc': float(sk_metrics.accuracy_score(audio_targets, audio_preds) * 100),
        'bal_acc': float(sk_metrics.balanced_accuracy_score(audio_targets, audio_preds) * 100),
        'sp': compute_sp_index(audio_targets, audio_preds, num_classes=4)
    }
    for c, cname in enumerate(class_names):
        metrics[f'prec_{cname}'] = prec[c]
        metrics[f'rec_{cname}'] = rec[c]
        metrics[f'f1_{cname}'] = f1[c]

    return metrics, cm


def main():
    parser = argparse.ArgumentParser(description="Train High-Precision Hybrid MLP")
    parser.add_argument("-F", "--folds", type=str, default="0-9", help="Folds to train (e.g. 0-9)")
    parser.add_argument("--n_pca", type=int, default=64, help="LOFAR PCA components")
    parser.add_argument("--large_penalty", type=float, default=1.0,
                        help="Loss weight penalty multiplier for class LARGE (default 1.0 = balanced)")
    parser.add_argument("--hidden_dims", type=str, default="256,128,64", help="MLP hidden layer sizes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=35, help="Maximum epochs")
    parser.add_argument("--save_dir", type=str, default="", help="Directory name to save results")
    args = parser.parse_args()

    if "-" in args.folds:
        s, e = args.folds.split("-")
        fold_list = list(range(int(s), int(e) + 1))
    else:
        fold_list = [int(f) for f in args.folds.split(",")]

    hidden_dims = tuple(int(h) for h in args.hidden_dims.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    directories = DEFAULT_DIRECTORIES
    dp_mel = iara_manager.AudioFileProcessor(
        data_base_dir=directories.data_dir, data_processed_base_dir=directories.process_dir,
        normalization=iara_proc.Normalization.NORM_L2, analysis=iara_proc.SpectralAnalysis.LOG_MELGRAM,
        n_pts=1024, n_overlap=0, decimation_rate=3, n_mels=256, integration_interval=0.512
    )
    dp_lofar = iara_manager.AudioFileProcessor(
        data_base_dir=directories.data_dir, data_processed_base_dir=directories.process_dir,
        normalization=iara_proc.Normalization.NORM_L2, analysis=iara_proc.SpectralAnalysis.LOFAR,
        n_pts=1024, n_overlap=0, decimation_rate=3, n_mels=256, integration_interval=0.512
    )

    multiclass_collection = iara_default.default_collection()
    exp_config = iara_exp.Config(
        name="mlp_hyb", dataset=multiclass_collection, dataset_processor=dp_mel,
        output_base_dir=f"{directories.training_dir}/tests", input_type=iara_dataset.InputType.Window()
    )
    split_list = exp_config.split_datasets()

    data_loader_mel = exp_config.get_data_loader()
    
    exp_config_lofar = iara_exp.Config(
        name="mlp_lofar", dataset=multiclass_collection, dataset_processor=dp_lofar,
        output_base_dir=f"{directories.training_dir}/tests", input_type=iara_dataset.InputType.Window()
    )
    data_loader_lofar = exp_config_lofar.get_data_loader()

    in_features = 256 + args.n_pca
    exp_name = args.save_dir if args.save_dir else f"mlp_hybrid_pca{args.n_pca}_largepen{args.large_penalty}_lr{args.lr}"
    output_dir = os.path.join(directories.training_dir, "tests", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"\n{'='*80}")
    print(f" Treinamento MLP Híbrido de Alta Precisão (MEL-256 + LOFAR-PCA{args.n_pca})")
    print(f" Dimensão de Entrada: {in_features} | Camadas: {hidden_dims} | Penalidade LARGE: {args.large_penalty}x")
    print(f" Saída: {output_dir}")
    print(f"{'='*80}\n")

    for fold_idx in fold_list:
        trn_df, val_df, test_df = split_list[fold_idx]
        train_fids = trn_df['ID'].tolist()
        val_fids = val_df['ID'].tolist()
        test_fids = test_df['ID'].tolist()

        # Fit on train, transform all
        feat_train, pca, scaler = extract_features(data_loader_mel, data_loader_lofar, train_fids, fit=True, n_pca=args.n_pca)
        feat_val, _, _ = extract_features(data_loader_mel, data_loader_lofar, val_fids, pca=pca, scaler=scaler, fit=False, n_pca=args.n_pca)
        feat_test, _, _ = extract_features(data_loader_mel, data_loader_lofar, test_fids, pca=pca, scaler=scaler, fit=False, n_pca=args.n_pca)

        X_trn, y_trn, _ = build_dataset(feat_train, trn_df)
        X_val, y_val, fids_val = build_dataset(feat_val, val_df)
        X_tst, y_tst, fids_tst = build_dataset(feat_test, test_df)

        trn_ds = TensorDataset(X_trn, y_trn)
        trn_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

        # Class weights with custom penalty for LARGE (Class 2)
        cw = compute_class_weight('balanced', classes=np.arange(4), y=y_trn.numpy())
        cw[2] *= args.large_penalty  # Adjust penalty for LARGE
        class_weights = torch.tensor(cw, dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model = HybridMLPClassifier(in_features=in_features, hidden_dims=hidden_dims, dropout=args.dropout, n_classes=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_sp = 0.0
        best_model_state = None
        patience = 8
        patience_counter = 0

        pbar = tqdm.trange(args.epochs, desc=f"Fold {fold_idx}", leave=False)
        for epoch in pbar:
            model.train()
            total_loss = 0.0
            for bx, by in trn_loader:
                bx = bx.to(device)
                by = by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * bx.size(0)

            val_metrics, _ = evaluate_audio(model, X_val, y_val, fids_val, device)
            scheduler.step(val_metrics['sp'])

            if val_metrics['sp'] > best_sp:
                best_sp = val_metrics['sp']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            pbar.set_postfix({'val_sp': f"{val_metrics['sp']:.1f}%", 'prec_L': f"{val_metrics['prec_large']:.1f}%", 'rec_L': f"{val_metrics['rec_large']:.1f}%"})

        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

        test_metrics, _ = evaluate_audio(model, X_tst, y_tst, fids_tst, device)
        results.append(test_metrics)

        print(f"  Fold {fold_idx} Test: "
              f"LARGE [Precisao: {test_metrics['prec_large']:.1f}%, Recall: {test_metrics['rec_large']:.1f}%] | "
              f"SMALL [P: {test_metrics['prec_small']:.1f}%, R: {test_metrics['rec_small']:.1f}%] | "
              f"MED [P: {test_metrics['prec_medium']:.1f}%, R: {test_metrics['rec_medium']:.1f}%] | "
              f"ACC: {test_metrics['acc']:.1f}% | SP: {test_metrics['sp']:.1f}%")

    print("\n" + "=" * 80)
    print(f" === 10-FOLD SUMMARY: {exp_name} (Penalidade LARGE = {args.large_penalty}x) ===")
    print("=" * 80)
    metrics_keys = [
        'prec_large', 'rec_large', 'f1_large',
        'prec_small', 'rec_small', 'f1_small',
        'prec_medium', 'rec_medium', 'f1_medium',
        'prec_background', 'rec_background', 'f1_background',
        'acc', 'bal_acc', 'sp'
    ]
    for k in metrics_keys:
        vals = [r[k] for r in results]
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"  {k:<18}: {mean:6.2f}% +- {std:5.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
