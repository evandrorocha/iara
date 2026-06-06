import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import pickle
import sys
import os

CLASS_NAMES = ["SMALL", "MEDIUM", "LARGE", "BACKGROUND"]
N = 4

exp_name = "svm_nystroem_4000_log_melgram_stratified_kmeans_weighted_ncw_S2.0_M1.0_L1.0_B1.0_elasticnet_l1r0.15"
BASE_DIR = Path(f"results/trainings/tests/{exp_name}")

def sp_from_cm(cm):
    """Geometric mean of per-class recall."""
    recalls = []
    for i in range(cm.shape[0]):
        total = cm[i].sum()
        recalls.append(cm[i, i] / total if total > 0 else 0.0)
    return np.prod(recalls) ** (1.0 / len(recalls)) * 100

def load_predictions(fold_i, subset):
    p = BASE_DIR / "eval" / f"fold_{fold_i}" / f"{exp_name}_multiclass_{subset}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)

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
sys.modules['__main__'].SVMNystroemLocal = svm.SVMNystroemLocal
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
    df_val  = load_predictions(fold_i, "val")
    df_test = load_predictions(fold_i, "test")
    if df_val is None or df_test is None:
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
# We fix b_B = 0.0 and search relative biases of b_S, b_M, b_L.
bias_S_range = np.arange(0.0, 1.81, 0.05)
bias_M_range = np.arange(-0.6, 0.61, 0.05)
bias_L_range = np.arange(-0.6, 0.61, 0.05)

best_biases_list = []
val_sp_list, test_sp_list, test_acc_list = [], [], []

# Record test metrics before and after calibration to compare
pre_cal_sp_list = []
pre_cal_acc_list = []
pre_cal_recalls_list = []
post_cal_recalls_list = []

print(f"\nStarting grid search (S in [0, 1.8], M/L in [-0.6, 0.6]) relative to BG=0.0...\n", flush=True)

for fold_i in valid_folds:
    print(f"--- FOLD {fold_i} ---", flush=True)
    
    # Load model
    pkl_path = BASE_DIR / "model" / f"fold_{fold_i}" / f"{exp_name}_multiclass.pkl"
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
    
    # --- 1. Compute Pre-Calibration Metrics on Test Set ---
    pre_window_preds = class_indices[np.argmax(test_scores, axis=1)]
    pre_test_audio_preds = np.array([np.argmax(np.bincount(pre_window_preds[start:end], minlength=N)) for start, end in test_slices])
    pre_flat_idx_test = test_audio_targets * N + pre_test_audio_preds
    pre_cm_test = np.bincount(pre_flat_idx_test, minlength=N*N).reshape(N, N)
    pre_test_sp = sp_from_cm(pre_cm_test)
    pre_test_acc = np.mean(test_audio_targets == pre_test_audio_preds) * 100
    pre_recalls = [pre_cm_test[i, i] / pre_cm_test[i].sum() * 100 if pre_cm_test[i].sum() > 0 else 0.0 for i in range(N)]
    
    pre_cal_sp_list.append(pre_test_sp)
    pre_cal_acc_list.append(pre_test_acc)
    pre_cal_recalls_list.append(pre_recalls)
    
    print(f"  [Antes Calibração] Test SP: {pre_test_sp:.2f}% | ACC: {pre_test_acc:.2f}% | Recalls: S={pre_recalls[0]:.1f}%, M={pre_recalls[1]:.1f}%, L={pre_recalls[2]:.1f}%, B={pre_recalls[3]:.1f}%")
    
    # --- 2. Grid Search on Val Set ---
    best_sp, best_bias = -1, [0.0, 0.0, 0.0, 0.0]
    
    for b_s, b_m, b_l in product(bias_S_range, bias_M_range, bias_L_range):
        biases = np.array([b_s, b_m, b_l, 0.0]) # b_B is fixed to 0.0
        bias_vector = np.array([biases[c] for c in class_indices])
        
        # Shift scores and predict window classes
        shifted = val_scores + bias_vector
        window_preds = class_indices[np.argmax(shifted, axis=1)]
        
        # Audio-level consensus voting
        audio_preds = np.array([np.argmax(np.bincount(window_preds[start:end], minlength=N)) for start, end in val_slices])
        
        flat_idx = val_audio_targets * N + audio_preds
        cm = np.bincount(flat_idx, minlength=N*N).reshape(N, N)
        sp = sp_from_cm(cm)
        
        if sp > best_sp:
            best_sp = sp
            best_bias = [b_s, b_m, b_l, 0.0]
            
    # --- 3. Apply Best Biases to Test Set ---
    best_bias_vec = np.array([best_bias[c] for c in class_indices])
    test_shifted = test_scores + best_bias_vec
    test_window_preds = class_indices[np.argmax(test_shifted, axis=1)]
    
    test_audio_preds = np.array([np.argmax(np.bincount(test_window_preds[start:end], minlength=N)) for start, end in test_slices])
    
    flat_idx_test = test_audio_targets * N + test_audio_preds
    cm_test = np.bincount(flat_idx_test, minlength=N*N).reshape(N, N)
    test_sp  = sp_from_cm(cm_test)
    test_acc = np.mean(test_audio_targets == test_audio_preds) * 100
    post_recalls = [cm_test[i, i] / cm_test[i].sum() * 100 if cm_test[i].sum() > 0 else 0.0 for i in range(N)]
    
    val_sp_list.append(best_sp)
    test_sp_list.append(test_sp)
    test_acc_list.append(test_acc)
    best_biases_list.append(best_bias)
    post_cal_recalls_list.append(post_recalls)
    
    print(f"  [Melhor Bias Val]  S={best_bias[0]:.2f}, M={best_bias[1]:.2f}, L={best_bias[2]:.2f}, B={best_bias[3]:.2f} (Val SP: {best_sp:.2f}%)")
    print(f"  [Após Calibração]  Test SP: {test_sp:.2f}% | ACC: {test_acc:.2f}% | Recalls: S={post_recalls[0]:.1f}%, M={post_recalls[1]:.1f}%, L={post_recalls[2]:.1f}%, B={post_recalls[3]:.1f}%\n", flush=True)

# --- Consolidated Report ---
print("=" * 90)
print(" RELATÓRIO CONSOLIDADO: CALIBRAÇÃO DE BIAS NO EXPERIMENTO ATIVO ")
print("=" * 90)
print(f"Média Sem Calibração (Test):")
print(f"  * Acurácia Global (ACC): {np.mean(pre_cal_acc_list):.2f}% ± {np.std(pre_cal_acc_list):.2f}%")
print(f"  * Índice SP:             {np.mean(pre_cal_sp_list):.2f}% ± {np.std(pre_cal_sp_list):.2f}%")
pre_recalls_avg = np.mean(pre_cal_recalls_list, axis=0)
print(f"  * Recalls por Classe:")
print(f"    - SMALL (S):           {pre_recalls_avg[0]:.2f}%")
print(f"    - MEDIUM (M):          {pre_recalls_avg[1]:.2f}%")
print(f"    - LARGE (L):           {pre_recalls_avg[2]:.2f}%")
print(f"    - BACKGROUND (B):      {pre_recalls_avg[3]:.2f}%")
print("-" * 90)
print(f"Média Com Calibração (Test):")
print(f"  * Acurácia Global (ACC): {np.mean(test_acc_list):.2f}% ± {np.std(test_acc_list):.2f}%")
print(f"  * Índice SP:             {np.mean(test_sp_list):.2f}% ± {np.std(test_sp_list):.2f}%")
post_recalls_avg = np.mean(post_cal_recalls_list, axis=0)
print(f"  * Recalls por Classe:")
print(f"    - SMALL (S):           {post_recalls_avg[0]:.2f}%  (Ganho: {post_recalls_avg[0]-pre_recalls_avg[0]:+.2f}%)")
print(f"    - MEDIUM (M):          {post_recalls_avg[1]:.2f}%  (Ganho: {post_recalls_avg[1]-pre_recalls_avg[1]:+.2f}%)")
print(f"    - LARGE (L):           {post_recalls_avg[2]:.2f}%  (Ganho: {post_recalls_avg[2]-pre_recalls_avg[2]:+.2f}%)")
print(f"    - BACKGROUND (B):      {post_recalls_avg[3]:.2f}%  (Ganho: {post_recalls_avg[3]-pre_recalls_avg[3]:+.2f}%)")
print("-" * 90)
mean_biases = np.mean(best_biases_list, axis=0)
print(f"Biases ótimos médios: S={mean_biases[0]:.2f}, M={mean_biases[1]:.2f}, L={mean_biases[2]:.2f}, B={mean_biases[3]:.2f}")
print("=" * 90)
