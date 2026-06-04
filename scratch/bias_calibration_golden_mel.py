"""
Bias calibration for Golden MEL (SVM m=4000, C=2.0, ElasticNet).

Pipeline for each fold:
  1. Load saved .pkl model
  2. Run decision_function on the val CSV file IDs → get raw scores
  3. Grid-search biases [b_BG, b_SM, b_ME, b_LA] that maximize SP on val set (AUDIO level)
  4. Apply best biases to test set → compute SP/ACC on test set (AUDIO level)

Optimized version:
  - Pre-loads all raw features in memory once to avoid repeated disk I/O.
  - Batches Nystroem transforms to speed up vector calculations.
  - Fixes b_BG = 0.0 to search 3 independent relative biases, saving 11x search space.
  - Performs consensus voting (majority vote) inside grid-search to maximize AUDIO-level SP.
  - Evaluates and reports AUDIO-level SP and Accuracy.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import pickle
import sys
import os

CLASS_NAMES = ["BACKGROUND", "SMALL", "MEDIUM", "LARGE"]
N = 4

BASE_GOLDEN = Path("results/trainings/tests/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0/eval")

def sp_from_cm(cm):
    """Geometric mean of per-class recall."""
    recalls = []
    for i in range(cm.shape[0]):
        total = cm[i].sum()
        recalls.append(cm[i, i] / total if total > 0 else 0.0)
    return np.prod(recalls) ** (1.0 / len(recalls)) * 100

def load_predictions(fold_i, subset):
    """Load Target/Prediction CSV for a given fold and subset (val or test)."""
    exp = "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0"
    p = BASE_GOLDEN / f"fold_{fold_i}" / f"{exp}_multiclass_{subset}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)

# ─── Load Configuration and Dataset ───────────────────────────────────────────
sys.path.insert(0, "src")
import iara.ml.experiment as iara_exp
import iara.default as iara_default
import iara.ml.dataset as iara_dataset
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc
from iara.default import DEFAULT_DIRECTORIES

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

exp_name = "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0"
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
    # Check if predictions exist first
    df_val  = load_predictions(fold_i, "val")
    df_test = load_predictions(fold_i, "test")
    if df_val is None or df_test is None:
        print(f"fold_{fold_i}: missing prediction files, skip")
        continue
    valid_folds.append(fold_i)
    trn_df, val_df, test_df = datasets[fold_i]
    all_needed_fids.update(val_df["ID"].tolist())
    all_needed_fids.update(test_df["ID"].tolist())

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

# Function to get scores and slices using cached features and batch transform
def get_scores_and_slices_batched(file_ids, model, batch_size=100):
    all_scores = []
    slices = []
    cur = 0
    
    for i in range(0, len(file_ids), batch_size):
        batch_fids = file_ids[i : i + batch_size]
        batch_Xs = [raw_features_cache[fid] for fid in batch_fids]
        batch_lengths = [len(x) for x in batch_Xs]
        X_batch = np.vstack(batch_Xs)
        
        # Transform and get decision scores
        X_t = model.nystroem.transform(X_batch)
        scores = model.sgd.decision_function(X_t)
        
        all_scores.append(scores)
        for length in batch_lengths:
            slices.append((cur, cur + length))
            cur += length
            
    return np.vstack(all_scores), slices

# ─── Grid Search and Calibration ──────────────────────────────────────────────
# We fix b_BG = 0.0 and search relative biases of b_SM, b_ME, b_LA.
bias_range = np.arange(-0.8, 0.81, 0.1)
bias_range = np.round(bias_range, 2)

best_biases_list = []
val_sp_list, test_sp_list, test_acc_list = [], [], []

print(f"\nBias grid range: {bias_range[0]:.1f} to {bias_range[-1]:.1f} step 0.1 (relative to BG=0.0)")
print(f"Starting grid search over {len(bias_range)**3} combinations per fold (AUDIO level)...\n", flush=True)

for fold_i in valid_folds:
    print(f"--- FOLD {fold_i} ---", flush=True)
    
    # Load model
    pkl_path = Path(f"results/trainings/tests/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0/model/fold_{fold_i}/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0_multiclass.pkl")
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
        
    trn_df, val_df, test_df = datasets[fold_i]
    val_file_ids  = val_df["ID"].tolist()
    test_file_ids = test_df["ID"].tolist()
    
    # Get scores and slice mapping
    val_scores, val_slices = get_scores_and_slices_batched(val_file_ids, model)
    test_scores, test_slices = get_scores_and_slices_batched(test_file_ids, model)
    
    val_audio_targets = np.array([id_to_target[fid] for fid in val_file_ids])
    test_audio_targets = np.array([id_to_target[fid] for fid in test_file_ids])
    
    class_indices = model.sgd.classes_
    
    # Fast grid search using numpy and slice bincount
    best_sp, best_bias = -1, [0.0, 0.0, 0.0, 0.0]
    
    # We fix b_bg = 0.0
    for b_sm, b_me, b_la in product(bias_range, repeat=3):
        biases = np.array([0.0, b_sm, b_me, b_la])
        bias_vector = np.array([biases[c] if c < len(biases) else 0.0 for c in class_indices])
        
        # Shift scores and predict window classes
        shifted = val_scores + bias_vector
        window_preds = class_indices[np.argmax(shifted, axis=1)]
        
        # Audio-level consensus voting
        audio_preds = np.array([np.argmax(np.bincount(window_preds[start:end], minlength=N)) for start, end in val_slices])
        
        # Fast confusion matrix using np.bincount
        flat_idx = val_audio_targets * N + audio_preds
        cm = np.bincount(flat_idx, minlength=N*N).reshape(N, N)
        sp = sp_from_cm(cm)
        
        if sp > best_sp:
            best_sp = sp
            best_bias = [0.0, b_sm, b_me, b_la]
            
    # Apply best bias to test set
    best_bias_vec = np.array([best_bias[c] if c < len(best_bias) else 0.0 for c in class_indices])
    test_shifted = test_scores + best_bias_vec
    test_window_preds = class_indices[np.argmax(test_shifted, axis=1)]
    
    test_audio_preds = np.array([np.argmax(np.bincount(test_window_preds[start:end], minlength=N)) for start, end in test_slices])
    
    flat_idx_test = test_audio_targets * N + test_audio_preds
    cm_test = np.bincount(flat_idx_test, minlength=N*N).reshape(N, N)
    test_sp  = sp_from_cm(cm_test)
    test_acc = np.mean(test_audio_targets == test_audio_preds) * 100
    
    val_sp_list.append(best_sp)
    test_sp_list.append(test_sp)
    test_acc_list.append(test_acc)
    best_biases_list.append(best_bias)
    
    recalls_cal = [cm_test[i, i] / cm_test[i].sum() * 100 if cm_test[i].sum() > 0 else 0.0 for i in range(N)]
    print(f"  Best val SP={best_sp:.2f}% with biases BG={best_bias[0]:.1f}, SM={best_bias[1]:.1f}, ME={best_bias[2]:.1f}, LA={best_bias[3]:.1f}")
    print(f"  Test Audio SP={test_sp:.2f}%  ACC={test_acc:.2f}%  recalls={[f'{r:.1f}' for r in recalls_cal]}", flush=True)

print()
print(f"Mean val  SP  = {np.mean(val_sp_list):.2f} ± {np.std(val_sp_list):.2f}", flush=True)
print(f"Mean test SP  = {np.mean(test_sp_list):.2f} ± {np.std(test_sp_list):.2f}", flush=True)
print(f"Mean test ACC = {np.mean(test_acc_list):.2f} ± {np.std(test_acc_list):.2f}", flush=True)
print(f"(over {len(test_sp_list)} folds)", flush=True)
mean_biases = np.mean(best_biases_list, axis=0)
print(f"Mean best biases: BG={mean_biases[0]:.2f}, SM={mean_biases[1]:.2f}, ME={mean_biases[2]:.2f}, LA={mean_biases[3]:.2f}", flush=True)
