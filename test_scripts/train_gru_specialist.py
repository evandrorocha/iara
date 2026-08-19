"""
Training and Evaluation of a Binary GRU Specialist (SMALL vs. LARGE)
using temporal sequences of spectral windows with Bi-GRU and Temporal Attention.

Protocol: 5x2cv (10 independent folds) with strict ship ID isolation.
Data: Log-Melgram (MEL, 256 bands) formatted as temporal sequences (T=20 windows, ~10s).
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
import iara.ml.models.gru as iara_gru
from iara.default import DEFAULT_DIRECTORIES


class SmallLargeFilter:
    """Keep only SMALL (length < 50m) and LARGE (length >= 100m) ships."""
    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        lengths = pd.to_numeric(input_df['Length'], errors='coerce')
        return input_df[(lengths < 50) | (lengths >= 100)].copy()


def classify_row_small_large(row) -> int:
    """Map SMALL (0) -> 0, LARGE (2) -> 1."""
    base = iara_default.Target.classify_row(row)
    if base == 2:
        return 1  # LARGE
    return 0      # SMALL


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for X, y, _ in dataloader:
        X = X.to(device)
        y = y.to(device).float().unsqueeze(1)

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
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_file_ids = []

    for X, y, fids in dataloader:
        X = X.to(device)
        y_float = y.to(device).float().unsqueeze(1)

        logits = model(X)
        loss = criterion(logits, y_float)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)

        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

        all_preds.extend(preds)
        all_targets.extend(y.numpy().flatten())
        all_file_ids.extend(fids.numpy().flatten() if isinstance(fids, torch.Tensor) else fids)

    # Audio-level aggregation (majority vote)
    file_preds = defaultdict(list)
    file_targets = {}
    for fid, pred, tgt in zip(all_file_ids, all_preds, all_targets):
        file_preds[fid].append(pred)
        file_targets[fid] = tgt

    audio_preds = []
    audio_targets = []
    for fid in file_preds:
        # Majority vote
        counts = np.bincount(file_preds[fid], minlength=2)
        audio_preds.append(int(np.argmax(counts)))
        audio_targets.append(int(file_targets[fid]))

    audio_preds = np.array(audio_preds)
    audio_targets = np.array(audio_targets)

    # Compute binary metrics
    # 0 = SMALL, 1 = LARGE
    tp0 = int(np.sum((audio_targets == 0) & (audio_preds == 0)))
    fn0 = int(np.sum((audio_targets == 0) & (audio_preds == 1)))
    fp0 = int(np.sum((audio_targets == 1) & (audio_preds == 0)))

    tp1 = int(np.sum((audio_targets == 1) & (audio_preds == 1)))
    fn1 = int(np.sum((audio_targets == 1) & (audio_preds == 0)))
    fp1 = int(np.sum((audio_targets == 0) & (audio_preds == 1)))

    rec0 = tp0 / (tp0 + fn0) * 100 if (tp0 + fn0) > 0 else 0.0
    prec0 = tp0 / (tp0 + fp0) * 100 if (tp0 + fp0) > 0 else 0.0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0

    rec1 = tp1 / (tp1 + fn1) * 100 if (tp1 + fn1) > 0 else 0.0
    prec1 = tp1 / (tp1 + fp1) * 100 if (tp1 + fp1) > 0 else 0.0
    f1_1 = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0

    acc = np.mean(audio_preds == audio_targets) * 100
    bal_acc = (rec0 + rec1) / 2
    sp = math.sqrt(rec0 * rec1)

    metrics = {
        'loss': total_loss / max(total_samples, 1),
        'rec_small': rec0,
        'prec_small': prec0,
        'f1_small': f1_0,
        'rec_large': rec1,
        'prec_large': prec1,
        'f1_large': f1_1,
        'acc': acc,
        'bal_acc': bal_acc,
        'sp': sp,
        'tp0': tp0, 'fn0': fn0, 'fp0': fp0,
        'tp1': tp1, 'fn1': fn1, 'fp1': fp1,
    }
    return metrics


class SequenceAudioDataset(Dataset):
    """
    Dataset that returns sequences of T consecutive windows along with file_id and target.
    """
    def __init__(self, data_loader, file_ids, targets, seq_len=20, stride=10):
        self.samples = []
        self.targets = []
        self.file_ids = []

        data_loader.pre_load(file_ids)

        for fid, tgt in zip(file_ids, targets):
            data_df = data_loader.get_all(fid)
            if data_df is None or len(data_df) < seq_len:
                # If audio is shorter than seq_len, pad or take all
                if data_df is not None and len(data_df) > 0:
                    pad = torch.zeros(seq_len - len(data_df), data_df.shape[1])
                    seq = torch.cat([data_df, pad], dim=0)
                    self.samples.append(seq)
                    self.targets.append(tgt)
                    self.file_ids.append(fid)
                continue

            n_seqs = (len(data_df) - seq_len) // stride + 1
            for i in range(n_seqs):
                start = i * stride
                seq = data_df[start : start + seq_len]
                self.samples.append(seq)
                self.targets.append(tgt)
                self.file_ids.append(fid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx], self.file_ids[idx]


def main():
    parser = argparse.ArgumentParser(description="Train GRU SL Specialist")
    parser.add_argument("-F", "--folds", type=str, default="0-9", help="Folds to train (e.g. 0-9)")
    parser.add_argument("--seq_len", type=int, default=20, help="Number of windows per sequence (~10s)")
    parser.add_argument("--stride", type=int, default=10, help="Stride between sequences (~5s)")
    parser.add_argument("--hidden_size", type=int, default=64, help="GRU hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GRU layers")
    parser.add_argument("--pooling", type=str, default="attention", choices=["mean", "attention", "last", "max"])
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
    args = parser.parse_args()

    if "-" in args.folds:
        s, e = args.folds.split("-")
        fold_list = list(range(int(s), int(e) + 1))
    else:
        fold_list = [int(f) for f in args.folds.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    directories = DEFAULT_DIRECTORIES
    dp_mel = iara_manager.AudioFileProcessor(
        data_base_dir=directories.data_dir,
        data_processed_base_dir=directories.process_dir,
        normalization=iara_proc.Normalization.NORM_L2,
        analysis=iara_proc.SpectralAnalysis.LOG_MELGRAM,
        n_pts=1024,
        n_overlap=0,
        decimation_rate=3,
        n_mels=256,
        integration_interval=0.512
    )

    binary_collection = iara.records.CustomCollection(
        collection=iara.records.Collection.OS,
        target=iara.records.GenericTarget(
            n_targets=2,
            function=classify_row_small_large,
            include_others=False
        ),
        filters=SmallLargeFilter(),
        only_sample=False
    )

    exp_config = iara_exp.Config(
        name="gru_specialist_small_large",
        dataset=binary_collection,
        dataset_processor=dp_mel,
        output_base_dir=f"{directories.training_dir}/tests",
        input_type=iara_dataset.InputType.Window()
    )

    split_list = exp_config.split_datasets()
    data_loader = exp_config.get_data_loader()

    results = []

    print(f"\n{'='*75}")
    print(f" Treinamento do Especialista GRU (SMALL vs. LARGE)")
    print(f" Config: seq_len={args.seq_len} ({args.seq_len*0.512:.1f}s), hidden={args.hidden_size}, "
          f"layers={args.num_layers}, pooling={args.pooling}, lr={args.lr}")
    print(f"{'='*75}\n")

    for fold_idx in fold_list:
        print(f"\n>>> [Fold {fold_idx}/10] Iniciando preparação de dados...")
        trn_df, val_df, test_df = split_list[fold_idx]

        train_ds = SequenceAudioDataset(
            data_loader,
            trn_df['ID'].tolist(),
            trn_df['Target'].tolist(),
            seq_len=args.seq_len,
            stride=args.stride
        )
        val_ds = SequenceAudioDataset(
            data_loader,
            val_df['ID'].tolist(),
            val_df['Target'].tolist(),
            seq_len=args.seq_len,
            stride=args.stride
        )
        test_ds = SequenceAudioDataset(
            data_loader,
            test_df['ID'].tolist(),
            test_df['Target'].tolist(),
            seq_len=args.seq_len,
            stride=args.stride
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        # Compute pos_weight for BCE loss
        n_pos = sum(train_ds.targets)
        n_neg = len(train_ds.targets) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model = iara_gru.GRU(
            input_dim=256,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            bidirectional=True,
            dropout=args.dropout,
            pooling=args.pooling,
            n_targets=1,
            fc_hidden=32
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_loss = float('inf')
        best_model_state = None
        patience = 8
        patience_counter = 0

        pbar = tqdm.trange(args.epochs, desc=f"Fold {fold_idx}", leave=False)
        for epoch in pbar:
            trn_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_metrics['loss'])

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            pbar.set_postfix({'trn_l': f"{trn_loss:.3f}", 'val_l': f"{val_metrics['loss']:.3f}", 'val_f1_s': f"{val_metrics['f1_small']:.1f}%"})

        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

        test_metrics = evaluate(model, test_loader, criterion, device)
        results.append(test_metrics)

        print(f"  Fold {fold_idx} Test Results: "
              f"SMALL [Rec: {test_metrics['rec_small']:.1f}%, Prec: {test_metrics['prec_small']:.1f}%, F1: {test_metrics['f1_small']:.1f}%] | "
              f"LARGE [Rec: {test_metrics['rec_large']:.1f}%, Prec: {test_metrics['prec_large']:.1f}%, F1: {test_metrics['f1_large']:.1f}%] | "
              f"ACC: {test_metrics['acc']:.1f}% | SP: {test_metrics['sp']:.1f}%")

    # 10-fold Summary
    print("\n" + "=" * 75)
    print(" === 10-FOLD SUMMARY: GRU SPECIALIST (SMALL vs. LARGE) ===")
    print("=" * 75)
    metrics_keys = ['rec_small', 'prec_small', 'f1_small', 'rec_large', 'prec_large', 'f1_large', 'acc', 'bal_acc', 'sp']
    for k in metrics_keys:
        vals = [r[k] for r in results]
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"  {k:<12}: {mean:6.2f}% +- {std:5.2f}%")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
