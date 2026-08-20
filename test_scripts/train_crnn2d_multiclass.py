"""
Training and Evaluation of a 4-Class High-Resolution 2D-CRNN Model
Input: Full 2D LOFAR Spectrogram (Time x 1024 Bins).
Architecture: Conv2D (5x5) Feature Extractor -> High-Resolution Frequency Pooling (32 bins) -> Bi-GRU (128 units) -> Temporal Attention -> Classification Head (4 Classes).

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
import iara.ml.models.crnn2d as iara_crnn2d
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


def apply_spec_augment(X: torch.Tensor, time_mask_max: int = 2, freq_mask_max: int = 32, noise_std: float = 0.0) -> torch.Tensor:
    """Apply 2D SpecAugment and Gaussian Noise Jitter to training batch."""
    B, C, T, D = X.shape
    X_aug = X.clone()

    if noise_std > 0:
        X_aug = X_aug + torch.randn_like(X_aug) * noise_std

    if time_mask_max > 0:
        for i in range(B):
            t_len = torch.randint(1, time_mask_max + 1, (1,)).item()
            t_start = torch.randint(0, max(1, T - t_len + 1), (1,)).item()
            X_aug[i, :, t_start : t_start + t_len, :] = 0.0

    if freq_mask_max > 0:
        for i in range(B):
            f_len = torch.randint(1, freq_mask_max + 1, (1,)).item()
            f_start = torch.randint(0, max(1, D - f_len + 1), (1,)).item()
            X_aug[i, :, :, f_start : f_start + f_len] = 0.0

    return X_aug


def train_epoch(model, dataloader, optimizer, criterion, device, spec_augment=False, noise_jitter=0.0, time_mask_max=2, freq_mask_max=32):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for X, y, _ in dataloader:
        if spec_augment or noise_jitter > 0:
            X = apply_spec_augment(
                X,
                time_mask_max=time_mask_max if spec_augment else 0,
                freq_mask_max=freq_mask_max if spec_augment else 0,
                noise_std=noise_jitter
            )

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

    # 0: SMALL, 1: MEDIUM, 2: LARGE, 3: BACKGROUND
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


class SpectrogramSequenceDataset(Dataset):
    def __init__(self, file_features_map, file_ids, targets, seq_len=20, stride=10):
        self.samples = []
        self.targets = []
        self.file_ids = []

        for fid, tgt in zip(file_ids, targets):
            feat = file_features_map.get(fid)
            if feat is None:
                continue

            n_windows = len(feat)
            if n_windows < seq_len:
                pad = np.zeros((seq_len - n_windows, feat.shape[1]), dtype=np.float32)
                seq = np.vstack([feat, pad])
                self.samples.append(torch.tensor(seq, dtype=torch.float32).unsqueeze(0))
                self.targets.append(tgt)
                self.file_ids.append(fid)
                continue

            n_seqs = (n_windows - seq_len) // stride + 1
            for i in range(n_seqs):
                start = i * stride
                seq = feat[start : start + seq_len]
                self.samples.append(torch.tensor(seq, dtype=torch.float32).unsqueeze(0))
                self.targets.append(tgt)
                self.file_ids.append(fid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx], self.file_ids[idx]


def build_scaled_spectrograms(loader, file_ids_train, file_ids_all):
    loader.pre_load(file_ids_all)

    train_list = []
    for fid in file_ids_train:
        frames = loader.get_all(fid)
        if frames is not None and len(frames) > 0:
            train_list.append(frames.numpy())

    X_train = np.vstack(train_list)
    scaler = StandardScaler()
    scaler.fit(X_train)

    features_map = {}
    for fid in file_ids_all:
        frames_t = loader.get_all(fid)
        if frames_t is None or len(frames_t) == 0:
            continue
        features_map[fid] = scaler.transform(frames_t.numpy()).astype(np.float32)

    return features_map


def main():
    parser = argparse.ArgumentParser(description="Train 4-Class 2D-CRNN Model")
    parser.add_argument("-F", "--folds", type=str, default="0-9", help="Folds to train (e.g. 0-9)")
    parser.add_argument("--analysis", type=str, default="LOFAR", choices=["LOFAR", "LOG_MELGRAM"], help="Spectral representation")
    parser.add_argument("--seq_len", type=int, default=20, help="Number of windows per sequence (~10s)")
    parser.add_argument("--stride", type=int, default=10, help="Stride between sequences (~5s)")
    parser.add_argument("--conv_channels", type=str, default="32,64,128", help="Conv2D channels, comma-separated")
    parser.add_argument("--conv_kernel", type=int, default=5, help="Conv2D kernel size")
    parser.add_argument("--freq_pool_size", type=int, default=32, help="Adaptive frequency pooling size (e.g. 32 or 64)")
    parser.add_argument("--gru_hidden", type=int, default=128, help="GRU hidden size")
    parser.add_argument("--gru_layers", type=int, default=2, help="Number of GRU layers")
    parser.add_argument("--pooling", type=str, default="attention", choices=["mean", "attention", "last", "max"])
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay L2 regularization")
    parser.add_argument("--spec_augment", action="store_true", default=False, help="Enable 2D SpecAugment")
    parser.add_argument("--noise_jitter", type=float, default=0.0, help="Gaussian noise std")
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
    spectral_analysis = iara_proc.SpectralAnalysis.LOFAR if args.analysis == "LOFAR" else iara_proc.SpectralAnalysis.LOG_MELGRAM
    freq_bins = 1024 if args.analysis == "LOFAR" else 256

    dp = iara_manager.AudioFileProcessor(
        data_base_dir=directories.data_dir,
        data_processed_base_dir=directories.process_dir,
        normalization=iara_proc.Normalization.NORM_L2,
        analysis=spectral_analysis,
        n_pts=1024,
        n_overlap=0,
        decimation_rate=3,
        n_mels=256,
        integration_interval=0.512
    )

    multiclass_collection = iara_default.default_collection()

    exp_name = args.save_dir if args.save_dir else f"crnn2d_multiclass_{args.analysis.lower()}_s{args.stride}_fpool{args.freq_pool_size}_h{args.gru_hidden}_lr{args.lr}"
    output_dir = os.path.join(directories.training_dir, "tests", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    exp_config = iara_exp.Config(
        name=exp_name,
        dataset=multiclass_collection,
        dataset_processor=dp,
        output_base_dir=f"{directories.training_dir}/tests",
        input_type=iara_dataset.InputType.Window()
    )

    split_list = exp_config.split_datasets()
    data_loader = exp_config.get_data_loader()

    results = []

    print(f"\n{'='*75}")
    print(f" Treinamento MULTICLASSE 2D-CRNN HD (Conv2D + FreqPool-{args.freq_pool_size} + Bi-GRU {args.gru_hidden} + Attention)")
    print(f" Espectrograma: {args.analysis} ({freq_bins} bins) | SeqLen: {args.seq_len} ({args.seq_len*0.512:.1f}s) | Stride: {args.stride}")
    print(f" Convoluções: {conv_ch} | Kernel: {args.conv_kernel} | GRU: {args.gru_hidden}x{args.gru_layers} | LR: {args.lr}")
    print(f" Saída: {output_dir}")
    print(f"{'='*75}\n")

    for fold_idx in fold_list:
        print(f"\n>>> [Fold {fold_idx}/10] Preparando espectrogramas 4-Classes ({args.analysis})...")
        trn_df, val_df, test_df = split_list[fold_idx]

        train_fids = trn_df['ID'].tolist()
        val_fids = val_df['ID'].tolist()
        test_fids = test_df['ID'].tolist()
        all_fids = train_fids + val_fids + test_fids

        feat_map = build_scaled_spectrograms(
            data_loader,
            file_ids_train=train_fids,
            file_ids_all=all_fids
        )

        train_ds = SpectrogramSequenceDataset(
            feat_map, train_fids, trn_df['Target'].tolist(),
            seq_len=args.seq_len, stride=args.stride
        )
        val_ds = SpectrogramSequenceDataset(
            feat_map, val_fids, val_df['Target'].tolist(),
            seq_len=args.seq_len, stride=args.stride
        )
        test_ds = SpectrogramSequenceDataset(
            feat_map, test_fids, test_df['Target'].tolist(),
            seq_len=args.seq_len, stride=args.stride
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        # Compute balanced class weights
        trn_targets = np.array(train_ds.targets)
        cw = compute_class_weight('balanced', classes=np.arange(4), y=trn_targets)
        class_weights = torch.tensor(cw, dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model = iara_crnn2d.CRNN2D(
            input_freq_bins=freq_bins,
            conv_channels=conv_ch,
            conv_kernel_size=args.conv_kernel,
            freq_pool_size=args.freq_pool_size,
            gru_hidden_size=args.gru_hidden,
            gru_num_layers=args.gru_layers,
            bidirectional=True,
            dropout=args.dropout,
            pooling=args.pooling,
            n_targets=4,
            fc_hidden=64
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_loss = float('inf')
        best_model_state = None
        patience = 8
        patience_counter = 0

        pbar = tqdm.trange(args.epochs, desc=f"Fold {fold_idx}", leave=False)
        for epoch in pbar:
            trn_loss = train_epoch(
                model, train_loader, optimizer, criterion, device,
                spec_augment=args.spec_augment,
                noise_jitter=args.noise_jitter
            )
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

        torch.save(best_model_state, os.path.join(fold_model_dir, "crnn2d_model.pt"))

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

        print(f"  Fold {fold_idx} Test Results: "
              f"SMALL: {test_metrics['rec_small']:.1f}% | MED: {test_metrics['rec_medium']:.1f}% | "
              f"LARGE: {test_metrics['rec_large']:.1f}% | BG: {test_metrics['rec_background']:.1f}% | "
              f"ACC: {test_metrics['acc']:.1f}% | SP: {test_metrics['sp']:.1f}%")

    print("\n" + "=" * 75)
    print(f" === 10-FOLD SUMMARY: {exp_name} ===")
    print("=" * 75)
    summary_rows = []
    metrics_keys = [
        'rec_small', 'prec_small', 'f1_small',
        'rec_medium', 'prec_medium', 'f1_medium',
        'rec_large', 'prec_large', 'f1_large',
        'rec_background', 'prec_background', 'f1_background',
        'acc', 'bal_acc', 'macro_f1', 'sp'
    ]
    for k in metrics_keys:
        vals = [r[k] for r in results]
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"  {k:<18}: {mean:6.2f}% +- {std:5.2f}%")
        summary_rows.append({'metric': k, 'mean': mean, 'std': std})
    print("=" * 75 + "\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    print(f"[Done] Resultados e modelos salvos em: {output_dir}\n")


if __name__ == "__main__":
    main()
