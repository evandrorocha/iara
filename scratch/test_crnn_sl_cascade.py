"""
Test Cascade: Stage 1 = 2D-CRNN Multiclass HD, Stage 2 = SVM SL Specialist (m=6000)
Tested with confidence thresholds on MEDIUM and/or LARGE/SMALL predictions.
"""
import os, sys, glob, math
import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import scipy.stats as scipy
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_scripts')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import iara.records
import iara.default as iara_default
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc
import iara.ml.dataset as iara_dataset
import iara.ml.experiment as iara_exp
import iara.ml.models.base_model as iara_model
from iara.default import DEFAULT_DIRECTORIES

CRNN_EVAL_DIR = "results/trainings/tests/crnn2d_multiclass_lofar_s10_fpool32_h128_lr0.0002/eval"
SL_DIR = "results/trainings/tests/svm_nystroem_6000_hybrid_binary_small_large_pca8_elasticnet_l1r0.15"

def compute_sp_index(targets, preds, num_classes=4):
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    if np.any(recalls == 0):
        gmean = 0.0
    else:
        gmean = scipy.gmean(recalls)
    return float(np.sqrt(np.mean(recalls * gmean)) * 100)

def load_sl_specialist(fold_idx):
    model_dir = os.path.join(SL_DIR, "model", f"fold_{fold_idx}")
    if not os.path.exists(model_dir):
        model_dir = os.path.join(SL_DIR, f"fold_{fold_idx}")
    pkls = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    return iara_model.BaseModel.load(os.path.join(model_dir, pkls[0]))

def get_sl_probs(sl_model, windows_hybrid):
    """Compute per-window probabilities [P(SMALL), P(LARGE)] from SVM SL."""
    X = windows_hybrid.copy()
    if hasattr(sl_model, 'n_mel_features'):
        n_mel = sl_model.n_mel_features
        X_lofar_pca = sl_model.pca_trans.transform(X[:, n_mel:])
        X = np.hstack([X[:, :n_mel], X_lofar_pca])
        if sl_model.normalize and sl_model.scaler is not None:
            X = sl_model.scaler.transform(X)
    else:
        if getattr(sl_model, 'normalize', False) and getattr(sl_model, 'scaler', None) is not None:
            X = sl_model.scaler.transform(X)
        if getattr(sl_model, 'pca', False) and getattr(sl_model, 'pca_trans', None) is not None:
            X = sl_model.pca_trans.transform(X)

    X_final = sl_model.nystroem.transform(X)
    scores = sl_model.sgd.decision_function(X_final)
    scores_binary = scores[:, 0] if len(scores.shape) > 1 else scores
    probs_large = 1.0 / (1.0 + np.exp(-scores_binary))
    probs_small = 1.0 - probs_large
    return np.mean(probs_small), np.mean(probs_large)

def main():
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
    exp_config_mel = iara_exp.Config(name="test", dataset=multiclass_collection, dataset_processor=dp_mel, output_base_dir="scratch", input_type=iara_dataset.InputType.Window())
    exp_config_lofar = iara_exp.Config(name="test", dataset=multiclass_collection, dataset_processor=dp_lofar, output_base_dir="scratch", input_type=iara_dataset.InputType.Window())

    split_list = exp_config_mel.split_datasets()
    data_loader_mel = exp_config_mel.get_data_loader()
    data_loader_lofar = exp_config_lofar.get_data_loader()

    print("\n" + "="*80)
    print(" === TESTE DA CASCATA: 2D-CRNN (1º Estágio) + SVM SL Especialista (2º Estágio) ===")
    print("="*80 + "\n")

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    for thresh in thresholds:
        fold_results = []
        for fold_idx in range(10):
            crnn_csv = glob.glob(f"{CRNN_EVAL_DIR}/fold_{fold_idx}/*.csv")[0]
            crnn_df = pd.read_csv(crnn_csv)
            crnn_audio = crnn_df.groupby('File').agg(lambda x: x.value_counts().index[0]).reset_index()

            sl_model = load_sl_specialist(fold_idx)
            test_fids = crnn_audio['File'].tolist()
            data_loader_mel.pre_load(test_fids)
            data_loader_lofar.pre_load(test_fids)

            final_preds = []
            targets = []

            for _, row in crnn_audio.iterrows():
                fid = int(row['File'])
                tgt = int(row['Target'])
                pred_crnn = int(row['Prediction'])
                targets.append(tgt)

                # Se a 2D-CRNN previu MEDIUM (1)
                if pred_crnn == 1:
                    m = data_loader_mel.get_all(fid)
                    l = data_loader_lofar.get_all(fid)
                    if m is None or l is None or len(m) == 0 or len(l) == 0:
                        final_preds.append(pred_crnn)
                        continue

                    min_l = min(len(m), len(l))
                    hyb_windows = np.hstack([m.numpy()[:min_l], l.numpy()[:min_l]])
                    p_small, p_large = get_sl_probs(sl_model, hyb_windows)

                    if p_small >= thresh:
                        final_preds.append(0)  # Reverte para SMALL
                    elif p_large >= thresh:
                        final_preds.append(2)  # Reverte para LARGE
                    else:
                        final_preds.append(1)  # Mantém MEDIUM
                else:
                    final_preds.append(pred_crnn)

            final_preds = np.array(final_preds)
            targets = np.array(targets)

            cm = sk_metrics.confusion_matrix(targets, final_preds, labels=[0, 1, 2, 3])
            rec = cm.diagonal() / cm.sum(axis=1) * 100
            acc = sk_metrics.accuracy_score(targets, final_preds) * 100
            sp = compute_sp_index(targets, final_preds, num_classes=4)

            fold_results.append({
                'rec_small': rec[0], 'rec_medium': rec[1], 'rec_large': rec[2], 'rec_bg': rec[3],
                'acc': acc, 'sp': sp
            })

        rec_s = np.mean([r['rec_small'] for r in fold_results])
        rec_m = np.mean([r['rec_medium'] for r in fold_results])
        rec_l = np.mean([r['rec_large'] for r in fold_results])
        rec_bg = np.mean([r['rec_bg'] for r in fold_results])
        acc_m = np.mean([r['acc'] for r in fold_results])
        sp_m = np.mean([r['sp'] for r in fold_results])

        print(f"Confiança >= {thresh*100:4.1f}% | SMALL: {rec_s:5.1f}% | MED: {rec_m:5.1f}% | LARGE: {rec_l:5.1f}% | BG: {rec_bg:5.1f}% | ACC: {acc_m:5.1f}% | SP: {sp_m:5.2f}%")

if __name__ == "__main__":
    main()
