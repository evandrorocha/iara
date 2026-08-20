"""
Training and Evaluation of SVM Nystroem (m=6000) on 3-Class Problem (SMALL, MEDIUM, LARGE)
Excludes class BACKGROUND to directly evaluate vessel hull and engine classification.

Protocol: 5x2cv (10 independent folds) with strict ship ID isolation.
Features: Hybrid (MEL-256 + LOFAR-PCA64, in_dim=320)
Classifier: Nystroem (m=6000) + SGDClassifier (ElasticNet penalty)
"""
import os
import sys
import math
import time
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
import sklearn.metrics as sk_metrics
import scipy.stats as scipy
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import iara.default as iara_default
import iara.records
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc
import iara.ml.dataset as iara_dataset
import iara.ml.experiment as iara_exp
from iara.default import DEFAULT_DIRECTORIES


def compute_sp_index(targets, preds, num_classes=3):
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    if np.any(recalls == 0):
        gmean = 0.0
    else:
        gmean = scipy.gmean(recalls)
    return float(np.sqrt(np.mean(recalls * gmean)) * 100)


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


def build_dataset_3class(feat_by_file, df):
    """Filters out Target == 3 (BACKGROUND) and retains only 0, 1, 2."""
    X_list = []
    y_list = []
    fid_list = []

    for _, row in df.iterrows():
        fid = int(row['ID'])
        tgt = int(row['Target'])
        if tgt == 3:  # Skip BACKGROUND
            continue
        feat = feat_by_file.get(fid)
        if feat is None:
            continue
        for row_vec in feat:
            X_list.append(row_vec)
            y_list.append(tgt)
            fid_list.append(fid)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int), np.array(fid_list, dtype=int)


def evaluate_audio_level(model, nystroem, X_tst, y_tst, fids_tst):
    X_trans = nystroem.transform(X_tst)
    win_preds = model.predict(X_trans)

    file_preds = defaultdict(list)
    file_targets = {}
    for fid, pred, tgt in zip(fids_tst, win_preds, y_tst):
        file_preds[fid].append(pred)
        file_targets[fid] = tgt

    audio_preds = []
    audio_targets = []
    for fid in file_preds:
        counts = np.bincount(file_preds[fid], minlength=3)
        audio_preds.append(int(np.argmax(counts)))
        audio_targets.append(int(file_targets[fid]))

    audio_preds = np.array(audio_preds)
    audio_targets = np.array(audio_targets)

    cm = sk_metrics.confusion_matrix(audio_targets, audio_preds, labels=[0, 1, 2])
    tp = cm.diagonal()
    prec = tp / np.maximum(cm.sum(axis=0), 1) * 100
    rec = tp / np.maximum(cm.sum(axis=1), 1) * 100
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)

    class_names = ['small', 'medium', 'large']
    metrics = {
        'acc': float(sk_metrics.accuracy_score(audio_targets, audio_preds) * 100),
        'bal_acc': float(sk_metrics.balanced_accuracy_score(audio_targets, audio_preds) * 100),
        'sp': compute_sp_index(audio_targets, audio_preds, num_classes=3)
    }
    for c, cname in enumerate(class_names):
        metrics[f'prec_{cname}'] = prec[c]
        metrics[f'rec_{cname}'] = rec[c]
        metrics[f'f1_{cname}'] = f1[c]

    return metrics, cm, audio_targets, audio_preds


def main():
    parser = argparse.ArgumentParser(description="Train 3-Class Hybrid SVM Nystroem")
    parser.add_argument("-F", "--folds", type=str, default="0-9", help="Folds to train (e.g. 0-9)")
    parser.add_argument("-m", "--n_components", type=int, default=6000, help="Nystroem components")
    parser.add_argument("--n_pca", type=int, default=64, help="LOFAR PCA components")
    parser.add_argument("--alpha", type=float, default=1e-4, help="SGD alpha")
    parser.add_argument("--l1_ratio", type=float, default=0.15, help="ElasticNet l1 ratio")
    args = parser.parse_args()

    if "-" in args.folds:
        s, e = args.folds.split("-")
        fold_list = list(range(int(s), int(e) + 1))
    else:
        fold_list = [int(f) for f in args.folds.split(",")]

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
        name="svm_3class", dataset=multiclass_collection, dataset_processor=dp_mel,
        output_base_dir=f"{directories.training_dir}/tests", input_type=iara_dataset.InputType.Window()
    )
    split_list = exp_config.split_datasets()

    data_loader_mel = exp_config.get_data_loader()
    exp_config_lofar = iara_exp.Config(
        name="svm_lofar", dataset=multiclass_collection, dataset_processor=dp_lofar,
        output_base_dir=f"{directories.training_dir}/tests", input_type=iara_dataset.InputType.Window()
    )
    data_loader_lofar = exp_config_lofar.get_data_loader()

    results = []
    total_cm = np.zeros((3, 3), dtype=int)

    print("\n" + "=" * 80)
    print(f" Treinamento SVM Nystroem Híbrido de 3 Classes (SMALL vs. MEDIUM vs. LARGE)")
    print(f" Nystroem: m={args.n_components} | Features: MEL-256 + LOFAR-PCA{args.n_pca} (320-D)")
    print("=" * 80 + "\n")

    for fold_idx in fold_list:
        t0 = time.time()
        trn_df, val_df, test_df = split_list[fold_idx]
        train_fids = trn_df['ID'].tolist()
        test_fids = test_df['ID'].tolist()

        feat_train, pca, scaler = extract_features(data_loader_mel, data_loader_lofar, train_fids, fit=True, n_pca=args.n_pca)
        feat_test, _, _ = extract_features(data_loader_mel, data_loader_lofar, test_fids, pca=pca, scaler=scaler, fit=False, n_pca=args.n_pca)

        X_trn, y_trn, _ = build_dataset_3class(feat_train, trn_df)
        X_tst, y_tst, fids_tst = build_dataset_3class(feat_test, test_df)

        nystroem = Nystroem(kernel='rbf', n_components=args.n_components, random_state=42, n_jobs=-1)
        X_trn_trans = nystroem.fit_transform(X_trn)

        clf = SGDClassifier(
            loss='modified_huber', penalty='elasticnet', l1_ratio=args.l1_ratio,
            alpha=args.alpha, class_weight='balanced', random_state=42, max_iter=1000, tol=1e-3, n_jobs=-1
        )
        clf.fit(X_trn_trans, y_trn)

        metrics, cm, _, _ = evaluate_audio_level(clf, nystroem, X_tst, y_tst, fids_tst)
        results.append(metrics)
        total_cm += cm
        dt = time.time() - t0

        print(f"  Fold {fold_idx} ({dt:.1f}s): "
              f"SMALL [P: {metrics['prec_small']:.1f}%, R: {metrics['rec_small']:.1f}%] | "
              f"MED [P: {metrics['prec_medium']:.1f}%, R: {metrics['rec_medium']:.1f}%] | "
              f"LARGE [P: {metrics['prec_large']:.1f}%, R: {metrics['rec_large']:.1f}%] | "
              f"ACC: {metrics['acc']:.1f}% | SP: {metrics['sp']:.1f}%")

    print("\n" + "=" * 80)
    print(f" === 10-FOLD SUMMARY: SVM Nystroem Híbrido (m={args.n_components}, 3 Classes) ===")
    print("=" * 80)
    metrics_keys = [
        'prec_small', 'rec_small', 'f1_small',
        'prec_medium', 'rec_medium', 'f1_medium',
        'prec_large', 'rec_large', 'f1_large',
        'acc', 'bal_acc', 'sp'
    ]
    for k in metrics_keys:
        vals = [r[k] for r in results]
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"  {k:<16}: {mean:6.2f}% +- {std:5.2f}%")
    print("=" * 80)

    print("\n=== MATRIZ DE CONFUSAO ABSOLUTA (10 FOLDS - 3 CLASSES) ===")
    print(total_cm)
    print("\n=== MATRIZ DE CONFUSAO NORMALIZADA (% RECALL REAL) ===")
    cm_norm = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis] * 100
    print(np.round(cm_norm, 1))

    tp = total_cm.diagonal()
    prec = tp / total_cm.sum(axis=0) * 100
    rec = tp / total_cm.sum(axis=1) * 100
    f1 = 2 * prec * rec / (prec + rec)
    print("\n=== METRICAS CONSOLIDADAS POR CLASSE ===")
    for c, cn in [(0, 'SMALL'), (1, 'MEDIUM'), (2, 'LARGE')]:
        print(f"  {cn:<10}: Precisao = {prec[c]:5.2f}% | Recall = {rec[c]:5.2f}% | F1 = {f1[c]:5.2f}%")


if __name__ == "__main__":
    main()
