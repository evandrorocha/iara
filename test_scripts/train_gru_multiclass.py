"""
Training and Evaluation of a 4-Class Multiclass GRU Model
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
from sklearn.decomposition import PCA
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
import iara.ml.metrics as iara_metrics
import iara.ml.models.gru as iara_gru
from iara.default import DEFAULT_DIRECTORIES


def compute_sp_index(targets, preds, num_classes=4):
    """Compute SP Index according to standard IARA formula."""
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    
    if np.any(recalls == 0):
        # If any class has 0 recall, geometric mean is 0
        gmean = 0.0
    else:
        gmean = scipy.gmean(recalls)
    return float(np.sqrt(np.mean(recalls * gmean)) * 100)


def apply_spec_augment(X: torch.Tensor, time_mask_max: int = 2, freq_mask_max: int = 16, noise_std: float = 0.0) -> torch.Tensor:
    """Apply SpecAugment and Gaussian Noise Jitter to training batch."""
    B, T, D = X.shape
    X_aug = X.clone()

    if noise_std > 0:
        X_aug = X_aug + torch.randn_like(X_aug) * noise_std

    if time_mask_max > 0:
        for i in range(B):
            t_len = torch.randint(1, time_mask_max + 1, (1,)).item()
            t_start = torch.randint(0, max(1, T - t_len + 1), (1,)).item()
            X_aug[i, t_start : t_start + t_len, :] = 0.0

    if freq_mask_max > 0:
        for i in range(B):
            f_len = torch.randint(1, freq_mask_max + 1, (1,)).item()
            f_start = torch.randint(0, max(1, D - f_len + 1), (1,)).item()
            X_aug[i, :, f_start : f_start + f_len] = 0.0

    return X_aug


def train_epoch(model, dataloader, optimizer, criterion, device, spec_augment=False, noise_jitter=0.0, time_mask_max=2, freq_mask_max=16):
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

    # Class names: 0: SMALL, 1: MEDIUM, 2: LARGE, 3: BACKGROUND
    class_names = ['small', 'medium', 'large', 'background']
    metrics = {
        'loss': total_loss / max(total_samples, 1),
        'acc': float(sk_metrics.accuracy_score(audio_targets, audio_preds) * 100),
        'bal_acc': float(sk_metrics.balanced_accuracy_score(audio_targets, audio_preds) * 100),
        'macro_f1': float(sk_metrics.f1_score(audio_targets, audio_preds, average='macro', zero_division=0) * 100),
        'sp': compute_sp_index(audio_targets, audio_preds, num_classes=num_classes)
    }

    # Per-class metrics
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


class MulticlassSequenceDataset(Dataset):
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
                self.samples.append(torch.tensor(seq, dtype=torch.float32))
                self.targets.append(tgt)
                self.file_ids.append(fid)
                continue

            n_seqs = (n_windows - seq_len) // stride + 1
            for i in range(n_seqs):
                start = i * stride
                seq = feat[start : start + seq_len]
                self.samples.append(torch.tensor(seq, dtype=torch.float32))
                self.targets.append(tgt)
                self.file_ids.append(fid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx], self.file_ids[idx]


def build_transformed_features(loader_mel, loader_lofar, file_ids_train, file_ids_all, n_pca=32, lofar_only=False, mel_only=False):
    """
    Transform files into feature map {fid: (n_windows, input_dim)}.
    """
    if not lofar_only:
        loader_mel.pre_load(file_ids_all)
    if not mel_only:
        loader_lofar.pre_load(file_ids_all)

    if n_pca > 0 and not lofar_only and not mel_only:
        lofar_train_list = []
        for fid in file_ids_train:
            lof = loader_lofar.get_all(fid)
            if lof is not None and len(lof) > 0:
                lofar_train_list.append(lof.numpy())
        X_lofar_train = np.vstack(lofar_train_list)
        pca = PCA(n_components=n_pca, random_state=42)
        pca.fit(X_lofar_train)
    else:
        pca = None

    # Collect training data for StandardScaler
    train_list = []
    for fid in file_ids_train:
        if mel_only:
            mel_t = loader_mel.get_all(fid)
            if mel_t is None or len(mel_t) == 0: continue
            feat = mel_t.numpy()
        elif lofar_only:
            lof_t = loader_lofar.get_all(fid)
            if lof_t is None or len(lof_t) == 0: continue
            feat = lof_t.numpy()
        else:
            mel_t = loader_mel.get_all(fid)
            lof_t = loader_lofar.get_all(fid)
            if mel_t is None or lof_t is None or len(mel_t) == 0 or len(lof_t) == 0: continue
            mel = mel_t.numpy()
            lof = lof_t.numpy()
            min_len = min(len(mel), len(lof))
            if pca is not None:
                lof_pca = pca.transform(lof[:min_len])
                feat = np.hstack([mel[:min_len], lof_pca])
            else:
                feat = np.hstack([mel[:min_len], lof[:min_len]])
        train_list.append(feat)

    X_train = np.vstack(train_list)
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Transform all files
    features_map = {}
    for fid in file_ids_all:
        if mel_only:
            mel_t = loader_mel.get_all(fid)
            if mel_t is None or len(mel_t) == 0: continue
            feat = mel_t.numpy()
        elif lofar_only:
            lof_t = loader_lofar.get_all(fid)
            if lof_t is None or len(lof_t) == 0: continue
            feat = lof_t.numpy()
        else:
            mel_t = loader_mel.get_all(fid)
            lof_t = loader_lofar.get_all(fid)
            if mel_t is None or lof_t is None or len(mel_t) == 0 or len(lof_t) == 0: continue
            mel = mel_t.numpy()
            lof = lof_t.numpy()
            min_len = min(len(mel), len(lof))
            if min_len == 0: continue
            if pca is not None:
                lof_pca = pca.transform(lof[:min_len])
                feat = np.hstack([mel[:min_len], lof_pca])
            else:
                feat = np.hstack([mel[:min_len], lof[:min_len]])
        features_map[fid] = scaler.transform(feat).astype(np.float32)

    return features_map


def main():
    parser = argparse.ArgumentParser(description="Train Multiclass 4-Class GRU Model")
    parser.add_argument("-F", "--folds", type=str, default="0-9", help="Folds to train (e.g. 0-9)")
    parser.add_argument("--seq_len", type=int, default=20, help="Number of windows per sequence (~10s)")
    parser.add_argument("--stride", type=int, default=10, help="Stride between sequences (~5s)")
    parser.add_argument("--n_pca", type=int, default=32, help="Number of LOFAR PCA components")
    parser.add_argument("--lofar_only", action="store_true", help="Use full LOFAR only")
    parser.add_argument("--mel_only", action="store_true", help="Use MEL only")
    parser.add_argument("--hidden_size", type=int, default=64, help="GRU hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GRU layers")
    parser.add_argument("--pooling", type=str, default="attention", choices=["mean", "attention", "last", "max"])
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay L2 regularization")
    parser.add_argument("--spec_augment", action="store_true", default=False, help="Enable SpecAugment")
    parser.add_argument("--noise_jitter", type=float, default=0.0, help="Gaussian noise std (0.0 = disabled)")
    parser.add_argument("--save_dir", type=str, default="", help="Directory name to save models and CSVs")
    args = parser.parse_args()

    if "-" in args.folds:
        s, e = args.folds.split("-")
        fold_list = list(range(int(s), int(e) + 1))
    else:
        fold_list = [int(f) for f in args.folds.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

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
    dp_lofar = iara_manager.AudioFileProcessor(
        data_base_dir=directories.data_dir,
        data_processed_base_dir=directories.process_dir,
        normalization=iara_proc.Normalization.NORM_L2,
        analysis=iara_proc.SpectralAnalysis.LOFAR,
        n_pts=1024,
        n_overlap=0,
        decimation_rate=3,
        n_mels=256,
        integration_interval=0.512
    )

    multiclass_collection = iara_default.default_collection()

    if args.mel_only:
        mode_str = "mel256"
        input_dim = 256
    elif args.lofar_only:
        mode_str = "lofar1024"
        input_dim = 1024
    elif args.n_pca == 0:
        mode_str = "hybrid1280_mel256_lofar1024"
        input_dim = 1280
    else:
        mode_str = f"hybrid_mel256_lofarPCA{args.n_pca}"
        input_dim = 256 + args.n_pca

    exp_name = args.save_dir if args.save_dir else f"gru_multiclass_{mode_str}_s{args.stride}_h{args.hidden_size}_l{args.num_layers}_lr{args.lr}"
    output_dir = os.path.join(directories.training_dir, "tests", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    exp_config_mel = iara_exp.Config(
        name=exp_name,
        dataset=multiclass_collection,
        dataset_processor=dp_mel,
        output_base_dir=f"{directories.training_dir}/tests",
        input_type=iara_dataset.InputType.Window()
    )
    exp_config_lofar = iara_exp.Config(
        name=exp_name,
        dataset=multiclass_collection,
        dataset_processor=dp_lofar,
        output_base_dir=f"{directories.training_dir}/tests",
        input_type=iara_dataset.InputType.Window()
    )

    split_list = exp_config_mel.split_datasets()
    data_loader_mel = exp_config_mel.get_data_loader()
    data_loader_lofar = exp_config_lofar.get_data_loader()

    results = []

    print(f"\n{'='*75}")
    print(f" Treinamento MULTICLASSE (4 Classes) com GRU ({mode_str} = {input_dim} dimensões)")
    print(f" Config: seq_len={args.seq_len} ({args.seq_len*0.512:.1f}s), stride={args.stride}, hidden={args.hidden_size}, "
          f"layers={args.num_layers}, dropout={args.dropout}, lr={args.lr}, wd={args.weight_decay}, batch={args.batch_size}")
    print(f" Saída: {output_dir}")
    print(f"{'='*75}\n")

    for fold_idx in fold_list:
        print(f"\n>>> [Fold {fold_idx}/10] Preparando features 4-Classes ({mode_str})...")
        trn_df, val_df, test_df = split_list[fold_idx]

        train_fids = trn_df['ID'].tolist()
        val_fids = val_df['ID'].tolist()
        test_fids = test_df['ID'].tolist()
        all_fids = train_fids + val_fids + test_fids

        feat_map = build_transformed_features(
            data_loader_mel, data_loader_lofar,
            file_ids_train=train_fids,
            file_ids_all=all_fids,
            n_pca=args.n_pca,
            lofar_only=args.lofar_only,
            mel_only=args.mel_only
        )

        train_ds = MulticlassSequenceDataset(
            feat_map, train_fids, trn_df['Target'].tolist(),
            seq_len=args.seq_len, stride=args.stride
        )
        val_ds = MulticlassSequenceDataset(
            feat_map, val_fids, val_df['Target'].tolist(),
            seq_len=args.seq_len, stride=args.stride
        )
        test_ds = MulticlassSequenceDataset(
            feat_map, test_fids, test_df['Target'].tolist(),
            seq_len=args.seq_len, stride=args.stride
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        # Compute balanced class weights for CrossEntropyLoss
        trn_targets = np.array(train_ds.targets)
        unique_classes = np.unique(trn_targets)
        cw = compute_class_weight('balanced', classes=np.arange(4), y=trn_targets)
        class_weights = torch.tensor(cw, dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model = iara_gru.GRU(
            input_dim=input_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
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

        torch.save(best_model_state, os.path.join(fold_model_dir, "gru_model.pt"))

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
