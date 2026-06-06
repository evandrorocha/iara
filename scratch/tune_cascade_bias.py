import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys
import os
import torch
import scipy.stats as scipy_stats
from sklearn.metrics import confusion_matrix, accuracy_score
import collections

CLASS_NAMES = ["SMALL", "MEDIUM", "LARGE", "BACKGROUND"]
N = 4

exp_name = "svm_nystroem_4000_log_melgram_cascade_1000_elasticnet_l1r0.15"
BASE_DIR = Path(f"results/trainings/tests/{exp_name}")

def sp_from_cm(cm):
    """Geometric mean of per-class recall."""
    recalls = []
    for i in range(cm.shape[0]):
        total = cm[i].sum()
        recalls.append(cm[i, i] / total if total > 0 else 0.0)
    return np.prod(recalls) ** (1.0 / len(recalls)) * 100

def get_consensus_vote(preds):
    return collections.Counter(preds).most_common(1)[0][0]

# Import IARA modules
sys.path.insert(0, "src")
sys.path.insert(0, ".")
import iara.ml.experiment as iara_exp
import iara.default as iara_default
import iara.ml.dataset as iara_dataset
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc
from iara.default import DEFAULT_DIRECTORIES

# Inject classes into __main__ so pickle can find them
import test_scripts.svm as svm
sys.modules['__main__'].SVMNystroemCascaded = svm.SVMNystroemCascaded
sys.modules['__main__'].SVMNystroemLocal = svm.SVMNystroemLocal
sys.modules['__main__'].KMeansNystroem = svm.KMeansNystroem
sys.modules['__main__'].StratifiedKMeansNystroem = svm.StratifiedKMeansNystroem

# Setup the exact same data processor
dp = iara_manager.AudioFileProcessor(
    data_base_dir=DEFAULT_DIRECTORIES.data_dir,
    data_processed_base_dir=DEFAULT_DIRECTORIES.process_dir,
    normalization=iara_proc.Normalization.NORM_L2,
    analysis=iara_proc.SpectralAnalysis.LOG_MELGRAM,
    n_pts=1024,
    n_overlap=0,
    decimation_rate=3,
    n_mels=256,
    integration_interval=0.512
)

input_type = iara_dataset.InputType.Window()

config = iara_exp.Config(
    name=exp_name,
    dataset=iara_default.default_collection(),
    dataset_processor=dp,
    output_base_dir=f"{DEFAULT_DIRECTORIES.training_dir}/tests",
    input_type=input_type
)

print("Splitting datasets and collecting unique file IDs...", flush=True)
datasets = config.split_datasets()

# Gather all unique file IDs we will need to load
all_needed_fids = set()
valid_folds = []
for fold_i in range(10):
    pkl_path = BASE_DIR / "model" / f"fold_{fold_i}" / f"{exp_name}_multiclass.pkl"
    if not pkl_path.exists():
        continue
    valid_folds.append(fold_i)
    trn_df, val_df, test_df = datasets[fold_i]
    all_needed_fids.update(val_df["ID"].tolist())
    all_needed_fids.update(test_df["ID"].tolist())

print(f"Active completed folds: {valid_folds}")
print(f"Total unique files to load: {len(all_needed_fids)}", flush=True)

# Pre-load raw features into memory
raw_features_cache = {}
full_df = config.dataset.to_df()
id_to_target = dict(zip(full_df["ID"], full_df["Target"]))

print("Loading raw features into memory...", flush=True)
for idx, fid in enumerate(sorted(all_needed_fids)):
    df_feat, _ = dp.get_data(fid)
    raw_features_cache[fid] = df_feat.values.astype(np.float32)
    if (idx + 1) % 200 == 0 or (idx + 1) == len(all_needed_fids):
        print(f"  Loaded {idx + 1}/{len(all_needed_fids)} files...", flush=True)

# Define bias range
bias_range = np.arange(0.0, 1.51, 0.1)

# Cache predictions of each file's windows under different biases
prediction_cache = {fold_i: {bias: {"val": {}, "test": {}} for bias in bias_range} for fold_i in valid_folds}

print("\nEvaluating models and testing biases...", flush=True)
for fold_i in valid_folds:
    print(f"--- FOLD {fold_i} ---", flush=True)
    
    # Load model
    pkl_path = BASE_DIR / "model" / f"fold_{fold_i}" / f"{exp_name}_multiclass.pkl"
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
        
    trn_df, val_df, test_df = datasets[fold_i]
    val_file_ids  = val_df["ID"].tolist()
    test_file_ids = test_df["ID"].tolist()
    
    def get_multiclass_and_spec_scores(file_ids):
        results = {}
        for fid in file_ids:
            X_np = raw_features_cache[fid]
            data = torch.from_numpy(X_np)
            
            with torch.no_grad():
                preds_mc = model.multiclass_model.forward(data).cpu().numpy()
            
            medium_mask = (preds_mc == 1)
            spec_scores = None
            if medium_mask.any():
                X_med = X_np[medium_mask]
                data_med = torch.from_numpy(X_med)
                
                # Get raw decision scores from specialist model
                X_in = data_med.view(data_med.size(0), -1).cpu().numpy()
                if model.specialist_model.normalize:
                    X_in = model.specialist_model.scaler.transform(X_in)
                if model.specialist_model.pca:
                    X_in = model.specialist_model.pca_trans.transform(X_in)
                X_t = model.specialist_model.nystroem.transform(X_in)
                spec_scores = model.specialist_model.sgd.decision_function(X_t)
                
            results[fid] = (preds_mc, spec_scores, medium_mask)
        return results

    print("  Pre-computing predictions and scores...", flush=True)
    val_scores_info = get_multiclass_and_spec_scores(val_file_ids)
    test_scores_info = get_multiclass_and_spec_scores(test_file_ids)
    
    print("  Sweeping biases...", flush=True)
    for bias in bias_range:
        # Evaluate validation set
        for fid in val_file_ids:
            preds_mc, spec_scores, medium_mask = val_scores_info[fid]
            preds = preds_mc.copy()
            if medium_mask.any() and spec_scores is not None:
                spec_preds = (spec_scores > bias).astype(int)
                if len(spec_preds.shape) > 1:
                    spec_preds = spec_preds.flatten()
                preds[medium_mask] = spec_preds
            
            pred_audio = get_consensus_vote(preds)
            prediction_cache[fold_i][bias]["val"][fid] = pred_audio
            
        # Evaluate test set
        for fid in test_file_ids:
            preds_mc, spec_scores, medium_mask = test_scores_info[fid]
            preds = preds_mc.copy()
            if medium_mask.any() and spec_scores is not None:
                spec_preds = (spec_scores > bias).astype(int)
                if len(spec_preds.shape) > 1:
                    spec_preds = spec_preds.flatten()
                preds[medium_mask] = spec_preds
            
            pred_audio = get_consensus_vote(preds)
            prediction_cache[fold_i][bias]["test"][fid] = pred_audio

# Report results for each bias value averaged across folds
print("\n" + "=" * 110)
print(f" RESULTS GRID FOR CASCADE BIAS TUNING")
print("=" * 110)
print(f"{'Bias':<6} | {'Val SP':<10} | {'Test ACC':<10} | {'Test SP':<10} | {'Recall S':<10} | {'Recall M':<10} | {'Recall L':<10} | {'Recall B':<10}")
print("-" * 110)

best_val_sp = -1.0
best_bias = 0.0
best_test_metrics = None

for bias in bias_range:
    val_sps = []
    test_accs = []
    test_sps = []
    test_recalls = {0: [], 1: [], 2: [], 3: []}
    
    for fold_i in valid_folds:
        trn_df, val_df, test_df = datasets[fold_i]
        
        # Validation
        val_targets = [id_to_target[fid] for fid in val_df["ID"]]
        val_preds = [prediction_cache[fold_i][bias]["val"][fid] for fid in val_df["ID"]]
        val_cm = confusion_matrix(val_targets, val_preds, labels=[0, 1, 2, 3])
        val_sps.append(sp_from_cm(val_cm))
        
        # Test
        test_targets = [id_to_target[fid] for fid in test_df["ID"]]
        test_preds = [prediction_cache[fold_i][bias]["test"][fid] for fid in test_df["ID"]]
        test_accs.append(accuracy_score(test_targets, test_preds) * 100)
        
        test_cm = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2, 3])
        test_sps.append(sp_from_cm(test_cm))
        
        for cls in range(N):
            r_sum = test_cm[cls].sum()
            rec = (test_cm[cls, cls] / r_sum * 100) if r_sum > 0 else 0.0
            test_recalls[cls].append(rec)
            
    mean_val_sp = np.mean(val_sps)
    mean_test_acc = np.mean(test_accs)
    std_test_acc = np.std(test_accs)
    mean_test_sp = np.mean(test_sps)
    std_test_sp = np.std(test_sps)
    mean_r0 = np.mean(test_recalls[0])
    std_r0 = np.std(test_recalls[0])
    mean_r1 = np.mean(test_recalls[1])
    std_r1 = np.std(test_recalls[1])
    mean_r2 = np.mean(test_recalls[2])
    std_r2 = np.std(test_recalls[2])
    mean_r3 = np.mean(test_recalls[3])
    std_r3 = np.std(test_recalls[3])
    
    if mean_val_sp > best_val_sp:
        best_val_sp = mean_val_sp
        best_bias = bias
        best_test_metrics = (mean_test_acc, std_test_acc, mean_test_sp, std_test_sp, mean_r0, std_r0, mean_r1, std_r1, mean_r2, std_r2, mean_r3, std_r3)
        
    print(f"{bias:5.1f}  | {mean_val_sp:8.2f}% | {mean_test_acc:7.2f}% | {mean_test_sp:7.2f}% | {mean_r0:8.1f}% | {mean_r1:8.1f}% | {mean_r2:8.1f}% | {mean_r3:8.1f}%")

print("-" * 110)
print(f"Best Bias chosen by Val SP: {best_bias:.1f} (Val SP: {best_val_sp:.2f}%)")
if best_test_metrics:
    acc, sd_acc, sp, sd_sp, r0, sd_r0, r1, sd_r1, r2, sd_r2, r3, sd_r3 = best_test_metrics
    print(f"Best Bias Test Performance:")
    print(f"  * Accuracy:             {acc:.2f}% ± {sd_acc:.2f}%")
    print(f"  * SP Index:             {sp:.2f}% ± {sd_sp:.2f}%")
    print(f"  * Recall SMALL:         {r0:.1f}% ± {sd_r0:.1f}%")
    print(f"  * Recall MEDIUM:        {r1:.1f}% ± {sd_r1:.1f}%")
    print(f"  * Recall LARGE:         {r2:.1f}% ± {sd_r2:.1f}%")
    print(f"  * Recall BACKGROUND:    {r3:.1f}% ± {sd_r3:.1f}%")
print("=" * 110)
