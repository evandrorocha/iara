"""
Ship-level cascade evaluation (correct architecture).

For each fold:
  1. Get ship-level LOFAR-MC predictions (from existing CSVs if available,
     otherwise run the multiclass from scratch on all windows)
  2. Identify ships classified as LARGE by the multiclass
  3. Load hybrid features ONLY for those LARGE ships
  4. Run SL specialist on those windows → majority vote → final prediction
  5. Non-LARGE ships keep the multiclass majority-vote result

This avoids the window-level routing artifact where MEDIUM ships lose their
majority because the SL specialist converts per-window LARGE→SMALL.

Usage:
  python scratch/eval_cascade_by_ship.py
"""

import sys, os, math, csv as csv_mod
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import iara.ml.experiment as iara_exp
import iara.ml.dataset as iara_dataset
import iara.ml.models.base_model as iara_model
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc

# ── configuration (mirrors svm.py) ─────────────────────────────────────────

CASCADE_DIR = (
    "results/trainings/tests/"
    "svm_nystroem_4000_hybrid_cascade_pretrained_sm6000pca64_smallalso_fbmc"
    "_large_sl6000pca8_lofarmc6000_nosm_elasticnet_l1r0.15"
)
LOFAR_MC_DIR   = "results/trainings/tests/svm_nystroem_6000_lofar_elasticnet_l1r0.15"
SL_DIR         = "results/trainings/tests/svm_nystroem_6000_hybrid_binary_small_large_pca8_elasticnet_l1r0.15"
CASCADE_BIAS   = 0.0
N_FOLDS        = 10

# ── data pipeline (same as svm.py main()) ──────────────────────────────────

class _Dirs:
    data_dir = "data"
    process_dir = "data_processed"
    training_dir = "results/trainings"

dirs = _Dirs()
norm = iara_proc.Normalization.NORM_L2

dp_mel = iara_manager.AudioFileProcessor(
    data_base_dir=dirs.data_dir,
    data_processed_base_dir=dirs.process_dir,
    normalization=norm,
    analysis=iara_proc.SpectralAnalysis.LOG_MELGRAM,
    n_pts=1024, n_overlap=0, decimation_rate=3, n_mels=256,
    integration_interval=0.512
)
dp_lofar = iara_manager.AudioFileProcessor(
    data_base_dir=dirs.data_dir,
    data_processed_base_dir=dirs.process_dir,
    normalization=norm,
    analysis=iara_proc.SpectralAnalysis.LOFAR,
    n_pts=1024, n_overlap=0, decimation_rate=3, n_mels=256,
    integration_interval=0.512
)

# ── helpers ────────────────────────────────────────────────────────────────

def majority_vote(labels):
    counts = np.bincount(np.array(labels, dtype=int), minlength=4)
    return int(np.argmax(counts))


def get_window_file_ids(dataset):
    """Build per-window file_id array matching get_samples() order."""
    ids = []
    for i, fid in enumerate(dataset.file_ids):
        n = dataset.limit_ids[i + 1] - dataset.limit_ids[i]
        ids.extend([fid] * n)
    return np.array(ids)


def load_lofar_mc(fold_idx):
    model_dir = os.path.join(LOFAR_MC_DIR, "model", f"fold_{fold_idx}")
    pkls = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    mc_pkls = [f for f in pkls if "multiclass" in f] or pkls
    return iara_model.BaseModel.load(os.path.join(model_dir, mc_pkls[0]))


def load_sl_specialist(fold_idx):
    model_dir = os.path.join(SL_DIR, "model", f"fold_{fold_idx}")
    if not os.path.exists(model_dir):
        model_dir = os.path.join(SL_DIR, f"fold_{fold_idx}")
    pkls = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    return iara_model.BaseModel.load(os.path.join(model_dir, pkls[0]))


def run_stage1(mc_model, windows_lofar):
    """Run LOFAR multiclass on LOFAR feature slice, return numpy predictions."""
    preds = mc_model.forward(torch.tensor(windows_lofar, dtype=torch.float32))
    return preds.cpu().numpy()


def run_sl_specialist(sl_model, windows_hybrid, bias=0.0):
    """Run SL specialist on windows; returns numpy array of predictions in {0, 2}.

    Replicates SVMNystroemHybridCascaded._evaluate_specialist() with
    allowed_classes=[0, 2] (SMALL/LARGE). Score > 0 → LARGE (2), else SMALL (0).
    """
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

    scores_binary = (scores[:, 0] if len(scores.shape) > 1 else scores) + bias
    preds = np.where(scores_binary > 0, 2, 0)   # LARGE=2, SMALL=0
    return preds


def read_lofar_mc_csv(fold_idx):
    """Read existing LOFAR-MC window-level CSV for fold_idx.
    Returns {file_id: {'target': int, 'preds': [int,...]}} or None if not found.
    """
    fold_dir = os.path.join(LOFAR_MC_DIR, "eval", f"fold_{fold_idx}")
    if not os.path.exists(fold_dir):
        return None
    csvs = [f for f in os.listdir(fold_dir)
            if "test" in f and f.endswith(".csv")]
    if not csvs:
        return None
    ships = {}
    with open(os.path.join(fold_dir, csvs[0])) as f:
        for row in csv_mod.DictReader(f):
            fid = int(row["File"])
            if fid not in ships:
                ships[fid] = {"target": int(row["Target"]), "preds": []}
            ships[fid]["preds"].append(int(row["Prediction"]))
    print(f"  [shortcut] Using existing LOFAR-MC CSV ({len(ships)} ships)")
    return ships


def ship_predictions_from_dataset(fold_idx, dataset):
    """Run LOFAR-MC from scratch on all windows and return per-ship predictions."""
    mc_model = load_lofar_mc(fold_idx)
    samples  = dataset.get_samples().numpy()
    targets  = dataset.get_targets().numpy()
    file_ids = get_window_file_ids(dataset)
    lofar_slice = samples[:, 256:]
    window_preds = run_stage1(mc_model, lofar_slice)
    ships = {}
    for fid in np.unique(file_ids):
        mask = (file_ids == fid)
        ships[int(fid)] = {
            "target": int(targets[mask][0]),
            "preds":  window_preds[mask].tolist()
        }
    return ships


def evaluate_fold_fast(fold_idx, ships, data_loader, input_type):
    """Fast path: CSV already loaded; load features only for LARGE-predicted ships."""
    ship_stage1 = {fid: majority_vote(info["preds"]) for fid, info in ships.items()}
    large_ids   = sorted(fid for fid, pred in ship_stage1.items() if pred == 2)
    print(f"  LARGE ships (stage-1): {len(large_ids)} / {len(ships)}")

    sl_model = load_sl_specialist(fold_idx)

    # Load features ONLY for LARGE-predicted ships (avoids loading corrupted files)
    large_dataset = iara_dataset.AudioDataset(data_loader, input_type, large_ids)
    large_samples  = large_dataset.get_samples().numpy()
    large_file_ids = get_window_file_ids(large_dataset)

    ship_preds   = {}
    ship_targets = {}

    for fid, info in ships.items():
        tgt = info["target"]
        if fid in large_ids:
            mask = (large_file_ids == fid)
            if mask.any():
                sl_out = run_sl_specialist(sl_model, large_samples[mask], bias=CASCADE_BIAS)
                final  = majority_vote(sl_out)
            else:
                final = ship_stage1[fid]
        else:
            final = ship_stage1[fid]
        ship_preds[fid]   = final
        ship_targets[fid] = tgt

    return ship_preds, ship_targets


def evaluate_fold_full(fold_idx, dataset):
    """Full path: run LOFAR-MC from scratch on all windows, then SL on LARGE ships."""
    ships = ship_predictions_from_dataset(fold_idx, dataset)

    ship_stage1 = {fid: majority_vote(info["preds"]) for fid, info in ships.items()}
    large_ids   = sorted(fid for fid, pred in ship_stage1.items() if pred == 2)
    print(f"  LARGE ships (stage-1): {len(large_ids)} / {len(ships)}")

    sl_model = load_sl_specialist(fold_idx)
    samples  = dataset.get_samples().numpy()
    file_ids = get_window_file_ids(dataset)

    ship_preds   = {}
    ship_targets = {}

    for fid, info in ships.items():
        tgt = info["target"]
        if fid in large_ids:
            mask = (file_ids == fid)
            if mask.any():
                sl_out = run_sl_specialist(sl_model, samples[mask], bias=CASCADE_BIAS)
                final  = majority_vote(sl_out)
            else:
                final = ship_stage1[fid]
        else:
            final = ship_stage1[fid]
        ship_preds[fid]   = final
        ship_targets[fid] = tgt

    return ship_preds, ship_targets


def compute_metrics(ship_preds, ship_targets):
    cm = defaultdict(int)
    correct = total = 0
    for fid in ship_preds:
        p = ship_preds[fid]
        t = ship_targets[fid]
        cm[(t, p)] += 1
        correct += (p == t)
        total += 1
    rec = {c: cm[(c, c)] / max(sum(cm[(c, p)] for p in range(4)), 1) for c in range(4)}
    recs = [rec[c] for c in range(4)]
    gm  = math.exp(sum(math.log(max(r, 1e-9)) for r in recs) / 4)
    sp  = math.sqrt(sum(recs) / 4 * gm) * 100
    acc = correct / total * 100
    return rec, acc, sp


# ── main ───────────────────────────────────────────────────────────────────

def main():
    import sys
    import test_scripts.svm as svm_mod
    import iara.default as iara_default
    # PKL files were saved with module name 'svm' (Docker run); fix for local load
    sys.modules['svm'] = svm_mod

    dp = svm_mod.HybridAudioFileProcessor(dp_mel, dp_lofar)

    config = iara_exp.Config(
        name="eval_cascade_by_ship",
        dataset=iara_default.default_collection(),
        dataset_processor=dp,
        output_base_dir=CASCADE_DIR,
        input_type=iara_dataset.InputType.Window()
    )

    id_list = config.split_datasets()
    data_loader = config.get_data_loader()

    all_rec = {c: [] for c in range(4)}
    all_acc, all_sp = [], []

    for fold_idx in range(N_FOLDS):
        print(f"\n--- Fold {fold_idx} ---")
        _, _, test_set = id_list[fold_idx]
        test_ids = test_set['ID'].tolist()

        ships = read_lofar_mc_csv(fold_idx)
        if ships is not None:
            # Fast path: CSV exists → load only LARGE ships' features
            ship_preds, ship_targets = evaluate_fold_fast(
                fold_idx, ships, data_loader, config.input_type)
        else:
            # Full path: load all test features, run multiclass from scratch
            print(f"  [full run] No LOFAR-MC CSV found, running multiclass...")
            dataset = iara_dataset.AudioDataset(
                data_loader, config.input_type, test_ids)
            ship_preds, ship_targets = evaluate_fold_full(fold_idx, dataset)

        rec, acc, sp = compute_metrics(ship_preds, ship_targets)

        for c in range(4):
            all_rec[c].append(rec[c] * 100)
        all_acc.append(acc)
        all_sp.append(sp)

        print(f"  SMALL={rec[0]*100:.1f}%  MED={rec[1]*100:.1f}%  "
              f"LARGE={rec[2]*100:.1f}%  BG={rec[3]*100:.1f}%  SP={sp:.2f}%")

    def st(v):
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / len(v))
        return mu, sd

    print("\n=== 10-fold summary (ship-level cascade) ===")
    for c, cn in [(0, 'SMALL'), (1, 'MEDIUM'), (2, 'LARGE'), (3, 'BG')]:
        r = st(all_rec[c])
        print(f"  {cn:<8}: {r[0]:.1f}% (+-{r[1]:.1f})")
    a = st(all_acc)
    s = st(all_sp)
    print(f"  ACC    : {a[0]:.2f}% (+-{a[1]:.2f})")
    print(f"  SP     : {s[0]:.2f}% (+-{s[1]:.2f})")


if __name__ == "__main__":
    main()
