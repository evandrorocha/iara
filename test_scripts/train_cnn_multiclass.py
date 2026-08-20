"""
Training and Evaluation of Multiclass 2D CNN (Standard 1-Channel vs. Hybrid 2-Channel MEL+LOFAR)
Modes:
  --mode mel    : Standard 1-Channel Log-Melgram (1 x 32 x 256) [DEFAULT]
  --mode lofar  : Standard 1-Channel LOFAR Spectrogram (1 x 32 x 1024)
  --mode hybrid : 2-Channel Hybrid Image (2 x 32 x 256) combining MEL (Channel 0) + LOFAR-256 (Channel 1)

Classes: 0: SMALL, 1: MEDIUM, 2: LARGE, 3: BACKGROUND
Protocol: 5x2cv (10 independent folds) with strict ship ID isolation.
Hardware: Accelerated by NVIDIA CUDA on GPU.
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
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import sklearn.metrics as sk_metrics
import scipy.stats as scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import iara.default as iara_default
import iara.records
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc
import iara.ml.dataset as iara_dataset
import iara.ml.experiment as iara_exp
import iara.ml.models.cnn as iara_cnn
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


class CNNImageDataset(Dataset):
    """
    Dataset creating 2D image patches:
    - 1-channel: (1, n_windows, n_freq)
    - 2-channel: (2, n_windows, n_freq)
    """
    def __init__(self, file_features_map, file_ids, targets, n_windows=32, stride=16):
        self.samples = []
        self.targets = []
        self.file_ids = []

        for fid, tgt in zip(file_ids, targets):
            feat = file_features_map.get(fid)
            if feat is None:
                continue

            # feat is (Channels, Total_Windows, Freq_Bins)
            n_ch, total_w, freq_dim = feat.shape

            if total_w < n_windows:
                pad = np.zeros((n_ch, n_windows - total_w, freq_dim), dtype=np.float32)
                img = np.concatenate([feat, pad], axis=1)
                self.samples.append(torch.tensor(img, dtype=torch.float32))
                self.targets.append(tgt)
                self.file_ids.append(fid)
                continue

            n_patches = (total_w - n_windows) // stride + 1
            for i in range(n_patches):
                start = i * stride
                patch = feat[:, start : start + n_windows, :]
                self.samples.append(torch.tensor(patch, dtype=torch.float32))
                self.targets.append(tgt)
                self.file_ids.append(fid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx], self.file_ids[idx]


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for X, y, _ in dataloader:
        X = X.to(device)
        y = y.to(device).long()

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_multiclass(model, dataloader, criterion, device, num_classes=4):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_file_ids = []

    for X, y, fids in dataloader:
        X = X.to(device)
        y_dev = y.to(device).long()

        logits = model(X)
        loss = criterion(logits, y_dev)

        preds = torch.argmax(logits, dim=1).cpu().numpy()

        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

        all_preds.extend(preds)
        all_targets.extend(y.numpy().flatten())
        all_file_ids.extend(fids.numpy().flatten() if isinstance(fids, torch.Tensor) else fids)

    file_preds = defaultdict(list)
    file_targets = {}
    for fid, pred, tgt in zip(all_file_ids, all_preds, all_targets):
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

    class_names = ['small', 'medium', 'large', 'background']
    metrics = {
        'loss': total_loss / max(total_samples, 1),
        'acc': float(sk_metrics.accuracy_score(audio_targets, audio_preds) * 100),
        'bal_acc': float(sk_metrics.balanced_accuracy_score(audio_targets, audio_preds) * 100),
        'macro_f1': float(sk_metrics.f1_score(audio_targets, audio_preds, average='macro', zero_division=0) * 100),
        'sp': compute_sp_index(audio_targets, audio_preds, num_classes=num_classes)
    }

    cm = sk_metrics.confusion_matrix(audio_targets, audio_preds, labels=list(range(num_classes)))
    for c, cname in enumerate(class_names):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        
        rec = float(tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        prec = float(tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        metrics[f'rec_{cname}'] = rec
        metrics[f'prec_{cname}'] = prec
        metrics[f'f1_{cname}'] = f1

    return metrics


def build_features_map(mode, data_loader_mel, data_loader_lofar, file_ids_train, file_ids_all):
    """
    Build scaled feature tensors per file ID.
    - 'mel': shape (1, n_windows, 256)
    - 'lofar': shape (1, n_windows, 1024)
    - 'hybrid': shape (2, n_windows, 256) [Channel 0: MEL-256, Channel 1: LOFAR-256 (pooled)]
    """
    if mode in ('mel', 'hybrid'):
        data_loader_mel.pre_load(file_ids_all)
    if mode in ('lofar', 'hybrid'):
        data_loader_lofar.pre_load(file_ids_all)

    # Scaler 1: MEL
    scaler_mel = None
    if mode in ('mel', 'hybrid'):
        train_mel = [data_loader_mel.get_all(fid).numpy() for fid in file_ids_train if data_loader_mel.get_all(fid) is not None]
        scaler_mel = StandardScaler()
        scaler_mel.fit(np.vstack(train_mel))

    # Scaler 2: LOFAR
    scaler_lofar = None
    if mode in ('lofar', 'hybrid'):
        train_lofar = []
        for fid in file_ids_train:
            frames = data_loader_lofar.get_all(fid)
            if frames is not None and len(frames) > 0:
                l_np = frames.numpy()
                if mode == 'hybrid':
                    # Pool 1024 bins down to 256 bins to match MEL width
                    l_t = torch.tensor(l_np).unsqueeze(1) # (N, 1, 1024)
                    l_pooled = F.adaptive_avg_pool1d(l_t, 256).squeeze(1).numpy()
                    train_lofar.append(l_pooled)
                else:
                    train_lofar.append(l_np)
        scaler_lofar = StandardScaler()
        scaler_lofar.fit(np.vstack(train_lofar))

    features_map = {}
    for fid in file_ids_all:
        if mode == 'mel':
            m = data_loader_mel.get_all(fid)
            if m is None or len(m) == 0: continue
            scaled_m = scaler_mel.transform(m.numpy()).astype(np.float32)
            features_map[fid] = np.expand_dims(scaled_m, axis=0) # (1, T, 256)

        elif mode == 'lofar':
            l = data_loader_lofar.get_all(fid)
            if l is None or len(l) == 0: continue
            scaled_l = scaler_lofar.transform(l.numpy()).astype(np.float32)
            features_map[fid] = np.expand_dims(scaled_l, axis=0) # (1, T, 1024)

        elif mode == 'hybrid':
            m = data_loader_mel.get_all(fid)
            l = data_loader_lofar.get_all(fid)
            if m is None or l is None or len(m) == 0 or len(l) == 0: continue

            min_len = min(len(m), len(l))
            m_np = m.numpy()[:min_len]
            l_np = l.numpy()[:min_len]

            # Pool LOFAR to 256 bins
            l_t = torch.tensor(l_np).unsqueeze(1)
            l_pooled = F.adaptive_avg_pool1d(l_t, 256).squeeze(1).numpy()

            scaled_m = scaler_mel.transform(m_np).astype(np.float32)
            scaled_l = scaler_lofar.transform(l_pooled).astype(np.float32)

            # Stack into 2 channels: (2, T, 256)
            hybrid_img = np.stack([scaled_m, scaled_l], axis=0)
            features_map[fid] = hybrid_img

    return features_map


class Conv2DClassifier(nn.Module):
    """Configurable 2D CNN Architecture for Spectrogram Images."""
    def __init__(self, in_channels=1, n_windows=32, n_freq=256, conv_channels=(32, 64, 128), kernel_size=5, dense_units=128, dropout=0.3, n_classes=4):
        super().__init__()
        conv_layers = []
        in_ch = in_channels
        pad = (kernel_size - 1) // 2

        for out_ch in conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=dropout * 0.5)
            ])
            in_ch = out_ch

        self.conv_net = nn.Sequential(*conv_layers)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        flat_dim = conv_channels[-1] * 4 * 4

        self.dense_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flat_dim, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_units, n_classes)
        )

    def forward(self, x):
        # x: (B, Channels, Windows, Freq)
        feat = self.conv_net(x)
        pooled = self.global_pool(feat)
        flat = torch.flatten(pooled, start_dim=1)
        logits = self.dense_net(flat)
        return logits


def main():
    parser = argparse.ArgumentParser(description="Train Multiclass 2D CNN (Standard vs. Hybrid)")
    parser.add_argument("-m", "--mode", type=str, default="mel", choices=["mel", "lofar", "hybrid"],
                        help="Input mode: 'mel' (1-channel MEL-256) [DEFAULT], 'lofar' (1-channel LOFAR-1024), or 'hybrid' (2-channel MEL+LOFAR)")
    parser.add_argument("-F", "--folds", type=str, default="0-9", help="Folds to train (e.g. 0-9)")
    parser.add_argument("--n_windows", type=int, default=32, help="Number of windows per 2D patch (~16s)")
    parser.add_argument("--stride", type=int, default=16, help="Stride between patches (~8s)")
    parser.add_argument("--conv_channels", type=str, default="32,64,128", help="Conv2D channels, comma-separated")
    parser.add_argument("--kernel_size", type=int, default=5, help="Conv2D kernel size")
    parser.add_argument("--dense_units", type=int, default=128, help="Dense classification units")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=35, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay L2 regularization")
    parser.add_argument("--save_dir", type=str, default="", help="Directory name to save models and CSVs")
    args = parser.parse_args()

    if "-" in args.folds:
        s, e = args.folds.split("-")
        fold_list = list(range(int(s), int(e) + 1))
    else:
        fold_list = [int(f) for f in args.folds.split(",")]

    conv_ch = tuple(int(c) for c in args.conv_channels.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    directories = DEFAULT_DIRECTORIES
    dp_mel = iara_manager.AudioFileProcessor(
        data_base_dir=directories.data_dir,
        data_processed_base_dir=directories.process_dir,
        normalization=iara_proc.Normalization.NORM_L2,
        analysis=iara_proc.SpectralAnalysis.LOG_MELGRAM,
        n_pts=1024, n_overlap=0, decimation_rate=3, n_mels=256,
        integration_interval=0.512
    )
    dp_lofar = iara_manager.AudioFileProcessor(
        data_base_dir=directories.data_dir,
        data_processed_base_dir=directories.process_dir,
        normalization=iara_proc.Normalization.NORM_L2,
        analysis=iara_proc.SpectralAnalysis.LOFAR,
        n_pts=1024, n_overlap=0, decimation_rate=3, n_mels=256,
        integration_interval=0.512
    )

    multiclass_collection = iara_default.default_collection()
    exp_config_mel = iara_exp.Config(
        name="cnn_dataset",
        dataset=multiclass_collection,
        dataset_processor=dp_mel,
        output_base_dir=f"{directories.training_dir}/tests",
        input_type=iara_dataset.InputType.Window()
    )
    split_list = exp_config_mel.split_datasets()
    data_loader_mel = exp_config_mel.get_data_loader()

    exp_config_lofar = iara_exp.Config(
        name="cnn_dataset",
        dataset=multiclass_collection,
        dataset_processor=dp_lofar,
        output_base_dir=f"{directories.training_dir}/tests",
        input_type=iara_dataset.InputType.Window()
    )
    data_loader_lofar = exp_config_lofar.get_data_loader()

    in_channels = 2 if args.mode == "hybrid" else 1
    freq_bins = 1024 if args.mode == "lofar" else 256

    exp_name = args.save_dir if args.save_dir else f"cnn_{args.mode}_w{args.n_windows}_s{args.stride}_conv{args.conv_channels}_lr{args.lr}"
    output_dir = os.path.join(directories.training_dir, "tests", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"\n{'='*80}")
    print(f" Treinamento MULTICLASSE CNN 2D — Modo: {args.mode.upper()} ({in_channels} Canais x {args.n_windows}x{freq_bins})")
    print(f" Convoluções: {conv_ch} | Kernel: {args.kernel_size} | Dense: {args.dense_units} | LR: {args.lr} | Dropout: {args.dropout}")
    print(f" Saída: {output_dir}")
    print(f"{'='*80}\n")

    for fold_idx in fold_list:
        print(f"\n>>> [Fold {fold_idx}/10] Preparando tensores 2D (Modo: {args.mode.upper()})...")
        trn_df, val_df, test_df = split_list[fold_idx]

        train_fids = trn_df['ID'].tolist()
        val_fids = val_df['ID'].tolist()
        test_fids = test_df['ID'].tolist()
        all_fids = train_fids + val_fids + test_fids

        feat_map = build_features_map(
            mode=args.mode,
            data_loader_mel=data_loader_mel,
            data_loader_lofar=data_loader_lofar,
            file_ids_train=train_fids,
            file_ids_all=all_fids
        )

        train_ds = CNNImageDataset(feat_map, train_fids, trn_df['Target'].tolist(), n_windows=args.n_windows, stride=args.stride)
        val_ds = CNNImageDataset(feat_map, val_fids, val_df['Target'].tolist(), n_windows=args.n_windows, stride=args.stride)
        test_ds = CNNImageDataset(feat_map, test_fids, test_df['Target'].tolist(), n_windows=args.n_windows, stride=args.stride)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        trn_targets = np.array(train_ds.targets)
        cw = compute_class_weight('balanced', classes=np.arange(4), y=trn_targets)
        class_weights = torch.tensor(cw, dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model = Conv2DClassifier(
            in_channels=in_channels,
            n_windows=args.n_windows,
            n_freq=freq_bins,
            conv_channels=conv_ch,
            kernel_size=args.kernel_size,
            dense_units=args.dense_units,
            dropout=args.dropout,
            n_classes=4
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_loss = float('inf')
        best_model_state = None
        patience = 8
        patience_counter = 0

        pbar = tqdm.trange(args.epochs, desc=f"Fold {fold_idx}", leave=False)
        for epoch in pbar:
            trn_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate_multiclass(model, val_loader, criterion, device, num_classes=4)
            scheduler.step(val_metrics['loss'])

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            pbar.set_postfix({'trn_l': f"{trn_loss:.3f}", 'val_l': f"{val_metrics['loss']:.3f}", 'val_sp': f"{val_metrics['sp']:.1f}%"})

        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

        test_metrics = evaluate_multiclass(model, test_loader, criterion, device, num_classes=4)
        results.append(test_metrics)

        # Save model and CSV evaluation for this fold
        fold_eval_dir = os.path.join(output_dir, "eval", f"fold_{fold_idx}")
        fold_model_dir = os.path.join(output_dir, "model", f"fold_{fold_idx}")
        os.makedirs(fold_eval_dir, exist_ok=True)
        os.makedirs(fold_model_dir, exist_ok=True)

        torch.save(best_model_state, os.path.join(fold_model_dir, "cnn_model.pt"))

        # Save audio predictions CSV
        model.eval()
        csv_rows = []
        with torch.no_grad():
            for X, y, fids in test_loader:
                X = X.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                for fid, pred, tgt in zip(fids, preds, y.numpy().flatten()):
                    fid_val = fid.item() if isinstance(fid, torch.Tensor) else fid
                    csv_rows.append({'File': fid_val, 'Target': tgt, 'Prediction': pred})
        
        csv_df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(fold_eval_dir, f"{exp_name}_test.csv")
        csv_df.to_csv(csv_path, index=False)

        print(f"  Fold {fold_idx} Test: "
              f"SMALL [P: {test_metrics['prec_small']:.1f}%, R: {test_metrics['rec_small']:.1f}%] | "
              f"LARGE [P: {test_metrics['prec_large']:.1f}%, R: {test_metrics['rec_large']:.1f}%] | "
              f"ACC: {test_metrics['acc']:.1f}% | SP: {test_metrics['sp']:.1f}%")

    print("\n" + "=" * 80)
    print(f" === 10-FOLD SUMMARY: {exp_name} ({args.mode.upper()}) ===")
    print("=" * 80)
    summary_rows = []
    metrics_keys = [
        'prec_small', 'rec_small', 'f1_small',
        'prec_medium', 'rec_medium', 'f1_medium',
        'prec_large', 'rec_large', 'f1_large',
        'prec_background', 'rec_background', 'f1_background',
        'acc', 'bal_acc', 'macro_f1', 'sp'
    ]
    for k in metrics_keys:
        vals = [r[k] for r in results]
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"  {k:<18}: {mean:6.2f}% +- {std:5.2f}%")
        summary_rows.append({'metric': k, 'mean': mean, 'std': std})
    print("=" * 80 + "\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    print(f"[Done] Resultados e modelos salvos em: {output_dir}\n")


if __name__ == "__main__":
    main()
