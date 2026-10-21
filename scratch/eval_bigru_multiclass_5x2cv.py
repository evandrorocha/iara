"""
Treinamento e Avaliação do BiGRU com Mecanismo de Atenção Temporal (Temporal Attention)
no Protocolo Oficial 5x2cv (10 Folds, particionamento por gravação `ID`; o mesmo navio pode estar em treino e teste).
Predições por sequência salvas em results/trainings/multirep/<exp>/ (results_writer.py).

Suporta:
- Entradas: DEMON (253-D), HHT (256-D), MEL (256-D), MEL+DEMON (509-D), MEL+HHT (512-D)
- Classificação: 4 Classes (SMALL, MEDIUM, LARGE, BACKGROUND) ou 3 Classes (SMALL, MEDIUM, LARGE)
- Hardware: Aceleração por GPU NVIDIA CUDA (RTX 5060 Ti)
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
from sklearn.utils.class_weight import compute_class_weight
import sklearn.metrics as sk_metrics
import scipy.stats as scipy_stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def compute_sp_index(targets, preds, num_classes=4):
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    return sp, recalls, cm


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(16, hidden_dim // 2), 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)
        return context


class BiGRUAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        n_targets: int = 4,
        fc_hidden: int = 32
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.attention = TemporalAttention(out_dim)
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, fc_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(fc_hidden),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_targets)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        context = self.attention(out)
        logits = self.classifier(context)
        return logits


class SequenceDataset(Dataset):
    def __init__(self, samples, targets, file_ids):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.file_ids = file_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx], self.file_ids[idx]


def build_sequences_for_files(cache, file_dict, scaler=None, fit_scaler=False, seq_len=15, stride=5, num_classes=4):
    """
    Slices each audio file into sequences of shape (seq_len, D).
    """
    if fit_scaler:
        all_windows = []
        for fid, tgt in file_dict.items():
            if fid in cache:
                all_windows.append(cache[fid])
        if len(all_windows) > 0:
            X_all = np.vstack(all_windows)
            scaler = StandardScaler()
            scaler.fit(X_all)
        else:
            scaler = StandardScaler()

    samples = []
    targets = []
    file_ids = []

    for fid, tgt in file_dict.items():
        if fid not in cache:
            continue
        raw_feat = cache[fid]
        if scaler is not None:
            feat = scaler.transform(raw_feat)
        else:
            feat = raw_feat

        n_windows = len(feat)
        if n_windows < seq_len:
            pad = np.zeros((seq_len - n_windows, feat.shape[1]), dtype=np.float32)
            seq = np.vstack([feat, pad])
            samples.append(seq)
            targets.append(tgt)
            file_ids.append(fid)
        else:
            n_seqs = (n_windows - seq_len) // stride + 1
            for i in range(n_seqs):
                start = i * stride
                seq = feat[start : start + seq_len]
                samples.append(seq)
                targets.append(tgt)
                file_ids.append(fid)

    samples = np.array(samples, dtype=np.float32)
    targets = np.array(targets, dtype=np.int64)
    return samples, targets, file_ids, scaler


def get_official_folds(eval_base_dir="results/trainings/tests/svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0/eval", num_classes=4):
    folds = []
    for fold_idx in range(10):
        fold_dir = os.path.join(eval_base_dir, f"fold_{fold_idx}")
        trn_path = os.path.join(fold_dir, "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_trn.csv")
        tst_path = os.path.join(fold_dir, "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_test.csv")
        
        df_trn = pd.read_csv(trn_path)
        df_tst = pd.read_csv(tst_path)

        train_files = df_trn.groupby("File")["Target"].first().to_dict()
        test_files = df_tst.groupby("File")["Target"].first().to_dict()

        if num_classes == 3:
            train_files = {fid: tgt for fid, tgt in train_files.items() if tgt != 3}
            test_files = {fid: tgt for fid, tgt in test_files.items() if tgt != 3}

        folds.append({
            "fold_idx": fold_idx,
            "train_files": train_files,
            "test_files": test_files,
        })
    return folds


def train_and_eval_bigru(cache, num_classes=4, epochs=20, batch_size=128, lr=3e-4, device='cuda', seq_len=15, stride=5,
                         rep_tag="rep"):
    from results_writer import ResultsWriter
    folds = get_official_folds(num_classes=num_classes)
    sample_fid = list(cache.keys())[0]
    input_dim = cache[sample_fid].shape[1]

    writer = ResultsWriter(
        f"bigru_att_{rep_tag}_h64_s{seq_len}-{stride}_e{epochs}_{num_classes}c", script=__file__,
        params={"model": "BiGRUAttention", "representation": rep_tag, "dim": input_dim, "hidden_size": 64,
                "num_layers": 2, "bidirectional": True, "dropout": 0.3, "fc_hidden": 32,
                "seq_len_frames": seq_len, "seq_stride_frames": stride, "frame_step_s": 5.12,
                "epochs": epochs, "batch_size": batch_size, "lr": lr, "optimizer": "AdamW(wd=1e-3)",
                "scheduler": "CosineAnnealingLR", "class_weight": "balanced", "early_stopping": False,
                "num_classes": num_classes})

    fold_metrics = []

    for fold in folds:
        fold_idx = fold["fold_idx"]
        trn_dict = fold["train_files"]
        tst_dict = fold["test_files"]

        # Build sequence datasets
        X_trn, y_trn, fids_trn, scaler = build_sequences_for_files(
            cache, trn_dict, fit_scaler=True, seq_len=seq_len, stride=stride, num_classes=num_classes
        )
        X_tst, y_tst, fids_tst, _ = build_sequences_for_files(
            cache, tst_dict, scaler=scaler, fit_scaler=False, seq_len=seq_len, stride=stride, num_classes=num_classes
        )

        trn_dataset = SequenceDataset(X_trn, y_trn, fids_trn)
        tst_dataset = SequenceDataset(X_tst, y_tst, fids_tst)

        trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        tst_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # Class weights
        classes_present = np.unique(y_trn)
        weights = compute_class_weight('balanced', classes=classes_present, y=y_trn)
        class_weights_t = torch.zeros(num_classes, dtype=torch.float32)
        for c, w in zip(classes_present, weights):
            class_weights_t[c] = float(w)
        class_weights_t = class_weights_t.to(device)

        model = BiGRUAttention(
            input_dim=input_dim,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            n_targets=num_classes,
            fc_hidden=32
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Train
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y, _ in trn_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        # Evaluate test sequences and aggregate by file ID
        model.eval()
        file_preds = defaultdict(list)
        file_targets = {}
        seq_fids, seq_targets, seq_preds, seq_scores = [], [], [], []

        with torch.no_grad():
            for batch_X, batch_y, batch_fids in tst_loader:
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                targets = batch_y.numpy()
                seq_fids.extend(np.asarray(batch_fids).tolist())
                seq_targets.extend(targets.tolist())
                seq_preds.extend(preds.tolist())
                seq_scores.append(torch.softmax(logits, dim=1).cpu().numpy())

                for fid, pred, tgt in zip(batch_fids, preds, targets):
                    file_preds[fid].append(pred)
                    file_targets[fid] = tgt

        writer.save_fold(fold_idx, seq_fids, seq_targets, seq_preds, train_files=trn_dict,
                         scores=np.concatenate(seq_scores))

        audio_preds = []
        audio_targets = []
        for fid in file_preds:
            counts = np.bincount(file_preds[fid], minlength=num_classes)
            audio_preds.append(int(np.argmax(counts)))
            audio_targets.append(int(file_targets[fid]))

        audio_preds = np.array(audio_preds)
        audio_targets = np.array(audio_targets)

        sp, recalls, cm = compute_sp_index(audio_targets, audio_preds, num_classes=num_classes)
        acc = float(sk_metrics.accuracy_score(audio_targets, audio_preds) * 100)
        bal_acc = float(sk_metrics.balanced_accuracy_score(audio_targets, audio_preds) * 100)

        res = {
            'fold': fold_idx,
            'sp': sp,
            'acc': acc,
            'bal_acc': bal_acc,
            'rec_small': float(recalls[0] * 100),
            'rec_medium': float(recalls[1] * 100),
            'rec_large': float(recalls[2] * 100),
        }
        if num_classes == 4:
            res['rec_background'] = float(recalls[3] * 100)

        fold_metrics.append(res)
        print(f"Fold {fold_idx}: SP = {sp:.2f}%, Bal Acc = {bal_acc:.2f}%, Recalls: " + 
              ", ".join([f"C{c}={recalls[c]*100:.1f}%" for c in range(num_classes)]), flush=True)

    df_metrics = pd.DataFrame(fold_metrics)
    summary = {
        'sp_mean': df_metrics['sp'].mean(),
        'sp_std': df_metrics['sp'].std(),
        'bal_acc_mean': df_metrics['bal_acc'].mean(),
        'bal_acc_std': df_metrics['bal_acc'].std(),
        'acc_mean': df_metrics['acc'].mean(),
        'acc_std': df_metrics['acc'].std(),
        'rec_small_mean': df_metrics['rec_small'].mean(),
        'rec_medium_mean': df_metrics['rec_medium'].mean(),
        'rec_large_mean': df_metrics['rec_large'].mean(),
    }
    if num_classes == 4:
        summary['rec_background_mean'] = df_metrics['rec_background'].mean()

    return summary, df_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', type=str, default='all', choices=['demon', 'hht', 'mel', 'mel_demon', 'mel_hht', 'all'])
    parser.add_argument('--classes', type=int, default=0, choices=[0, 3, 4], help="0 for both 3 and 4 classes")
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== BiGRU MULTICLASS BENCHMARK (5x2cv - 10 FOLDS) ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load caches
    print("\nCarregando caches de features...")
    cache_fusion = pickle.load(open('data/iara_processed/cache_fusion_mel_demon.pkl', 'rb'))
    cache_hht_raw = pickle.load(open('data/iara_processed/cache_hht.pkl', 'rb'))

    mel_cache = cache_fusion['mel']
    demon_cache = cache_fusion['demon']
    fused_mel_demon = cache_fusion['fused']
    
    # Align HHT with MEL keys
    hht_cache = {}
    fused_mel_hht = {}
    for fid in mel_cache:
        if fid in cache_hht_raw:
            hht_w = cache_hht_raw[fid]
            mel_w = mel_cache[fid]
            min_len = min(len(mel_w), len(hht_w))
            hht_cache[fid] = hht_w[:min_len]
            fused_mel_hht[fid] = np.hstack([mel_w[:min_len], hht_w[:min_len]])

    representations = {}
    if args.rep in ['demon', 'all']:
        representations['DEMON (253-D)'] = demon_cache
    if args.rep in ['hht', 'all']:
        representations['HHT (256-D)'] = hht_cache
    if args.rep in ['mel', 'all']:
        representations['MEL (256-D)'] = mel_cache
    if args.rep in ['mel_demon', 'all']:
        representations['FUSÃO MEL + DEMON (509-D)'] = fused_mel_demon
    if args.rep in ['mel_hht', 'all']:
        representations['FUSÃO MEL + HHT (512-D)'] = fused_mel_hht

    class_settings = [4, 3] if args.classes == 0 else [args.classes]

    all_results = []

    for n_cls in class_settings:
        print(f"\n{'='*75}")
        print(f"--- EXPERIMENTO: {n_cls} CLASSES (BiGRU com Atenção Temporal) ---")
        print(f"{'='*75}")

        for rep_name, cache_dict in representations.items():
            print(f"\n>>> Avaliando {rep_name} ({n_cls} Classes)...")
            t0 = time.time()
            rep_tag = {'DEMON (253-D)': 'demon', 'HHT (256-D)': 'hht', 'MEL (256-D)': 'mel5s',
                       'FUSÃO MEL + DEMON (509-D)': 'mel5s-demon', 'FUSÃO MEL + HHT (512-D)': 'mel5s-hht'}[rep_name]
            summary, df_m = train_and_eval_bigru(cache_dict, num_classes=n_cls, epochs=args.epochs, device=device,
                                                 rep_tag=rep_tag)
            elapsed = time.time() - t0

            print(f"Concluído em {elapsed:.1f}s.")
            print(f"==> SP: {summary['sp_mean']:.2f}% ± {summary['sp_std']:.2f}%, Bal Acc: {summary['bal_acc_mean']:.2f}% ± {summary['bal_acc_std']:.2f}%")
            print(f"==> Recalls: Small={summary['rec_small_mean']:.1f}%, Med={summary['rec_medium_mean']:.1f}%, Large={summary['rec_large_mean']:.1f}%" +
                  (f", BG={summary['rec_background_mean']:.1f}%" if n_cls == 4 else ""))

            res_row = {
                'Classes': f"{n_cls} Classes",
                'Representacao': rep_name,
                'SP (%)': f"{summary['sp_mean']:.2f} ± {summary['sp_std']:.2f}",
                'Bal Acc (%)': f"{summary['bal_acc_mean']:.2f} ± {summary['bal_acc_std']:.2f}",
                'Recall SMALL (%)': f"{summary['rec_small_mean']:.1f}%",
                'Recall MEDIUM (%)': f"{summary['rec_medium_mean']:.1f}%",
                'Recall LARGE (%)': f"{summary['rec_large_mean']:.1f}%",
                'Tempo (s)': f"{elapsed:.1f}"
            }
            if n_cls == 4:
                res_row['Recall BG (%)'] = f"{summary['rec_background_mean']:.1f}%"
            all_results.append(res_row)

    print("\n" + "="*80)
    print("TABELA RESUMO FINAL — BiGRU MULTICLASSE (5x2cv)")
    print("="*80)
    df_final = pd.DataFrame(all_results)
    print(df_final.to_string(index=False))

    # Save results to pickle and markdown
    out_pkl = "data/iara_processed/bigru_multiclass_results.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResultados salvos com sucesso em '{out_pkl}'.")


if __name__ == "__main__":
    main()
