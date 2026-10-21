"""
Gera as predições completas do SVM Nystroem para todas as 4 representações (MEL, LOFAR, HHT, DEMON)
nos 10 folds oficiais 5x2cv (particionamento por gravação `ID`; o mesmo navio pode estar em treino e teste).
Otimizado com OPENBLAS_NUM_THREADS=4 e m=2000 para execução rápida.
Salva em: data/iara_processed/svm_all_representations_predictions.pkl
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import pickle
import time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from results_writer import ResultsWriter

def mode(x):
    return Counter(x).most_common(1)[0][0]

def get_official_folds(eval_base_dir="results/trainings/tests/svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0/eval"):
    folds = []
    for fold_idx in range(10):
        fold_dir = os.path.join(eval_base_dir, f"fold_{fold_idx}")
        trn_path = os.path.join(fold_dir, "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_trn.csv")
        tst_path = os.path.join(fold_dir, "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_test.csv")
        df_trn = pd.read_csv(trn_path)
        df_tst = pd.read_csv(tst_path)
        folds.append({
            "fold_idx": fold_idx,
            "train_files": df_trn.groupby("File")["Target"].first().to_dict(),
            "test_files": df_tst.groupby("File")["Target"].first().to_dict(),
        })
    return folds

def build_fold_matrices(feature_cache, file_dict):
    X_list, y_list, file_ids = [], [], []
    for fid, target in file_dict.items():
        if fid not in feature_cache:
            continue
        windows = feature_cache[fid]
        X_list.append(windows)
        y_list.extend([target] * len(windows))
        file_ids.extend([fid] * len(windows))
    return np.vstack(X_list), np.array(y_list, dtype=np.int64), np.array(file_ids, dtype=np.int64)

def run_svm_predictions(feature_cache, folds, n_components=2000, C=2.0, l1_ratio=0.15, writer=None):
    preds_by_fold = {}
    for fold in folds:
        fold_idx = fold["fold_idx"]
        t0 = time.time()
        X_train, y_train, _ = build_fold_matrices(feature_cache, fold["train_files"])
        X_test, y_test, test_fids = build_fold_matrices(feature_cache, fold["test_files"])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        gamma = 1.0 / (X_train.shape[1] * X_train.var())
        nystroem = Nystroem(kernel="rbf", gamma=gamma, n_components=n_components, random_state=42, n_jobs=4)
        X_train_trans = nystroem.fit_transform(X_train)
        X_test_trans = nystroem.transform(X_test)

        alpha = 1.0 / (C * X_train.shape[0])
        clf = SGDClassifier(
            loss="hinge", penalty="elasticnet", l1_ratio=l1_ratio,
            alpha=alpha, max_iter=200, tol=1e-3, early_stopping=True,
            n_iter_no_change=5, class_weight="balanced", random_state=42
        )
        clf.fit(X_train_trans, y_train)

        win_preds = clf.predict(X_test_trans)
        if writer is not None:
            writer.save_fold(fold_idx, test_fids, y_test, win_preds, train_files=fold["train_files"],
                             scores=clf.decision_function(X_test_trans))
        file_preds = defaultdict(list)
        for fid, pred in zip(test_fids, win_preds):
            file_preds[fid].append(pred)

        preds_by_fold[fold_idx] = {int(fid): int(np.argmax(np.bincount(file_preds[fid], minlength=4))) for fid in file_preds}
        print(f"Fold {fold_idx+1}/10 processado em {time.time()-t0:.1f}s")
    return preds_by_fold

def main():
    print("Iniciando compilação das predições do SVM nos 10 folds (5x2cv)...")
    folds = get_official_folds()

    p_mel = "results/trainings/tests/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0"
    p_lofar = "results/trainings/tests/svm_nystroem_6000_lofar_elasticnet_l1r0.15"

    mel_preds = {}
    lofar_preds = {}
    targets = {}

    for i in range(10):
        f_mel = f"{p_mel}/eval/fold_{i}/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0_multiclass_test.csv"
        df_m = pd.read_csv(f_mel).groupby("File").agg({"Target": mode, "Prediction": mode}).reset_index()
        mel_preds[i] = {int(k): int(v) for k, v in zip(df_m["File"], df_m["Prediction"])}
        targets[i] = {int(k): int(v) for k, v in zip(df_m["File"], df_m["Target"])}

        f_lof = f"{p_lofar}/eval/fold_{i}/svm_nystroem_6000_lofar_elasticnet_l1r0.15_multiclass_test.csv"
        df_l = pd.read_csv(f_lof).groupby("File").agg({"Target": mode, "Prediction": mode}).reset_index()
        lofar_preds[i] = {int(k): int(v) for k, v in zip(df_l["File"], df_l["Prediction"])}

    print("\n--- 1. Executando SVM Nystroem no HHT ---")
    with open("data/iara_processed/cache_hht.pkl", "rb") as f:
        cache_hht = pickle.load(f)
    hht_preds = run_svm_predictions(cache_hht, folds, n_components=2000, C=2.0, writer=ResultsWriter(
        "svm_nystroem_2000_hht_C2.0_elasticnet_l1r0.15_4c", script=__file__,
        params={"model": "Nystroem(rbf)+SGDClassifier(hinge)", "representation": "HHT", "dim": 256,
                "n_components": 2000, "C": 2.0, "penalty": "elasticnet", "l1_ratio": 0.15, "max_iter": 200, "early_stopping": True,
                "class_weight": "balanced", "gamma": "1/(n_features*var)", "random_state": 42, "num_classes": 4}))

    print("\n--- 2. Executando SVM Nystroem no DEMON ---")
    with open("data/iara_processed/cache_fusion_mel_demon.pkl", "rb") as f:
        cache_fusion = pickle.load(f)
    demon_preds = run_svm_predictions(cache_fusion["demon"], folds, n_components=2000, C=2.0, writer=ResultsWriter(
        "svm_nystroem_2000_demon_C2.0_elasticnet_l1r0.15_4c", script=__file__,
        params={"model": "Nystroem(rbf)+SGDClassifier(hinge)", "representation": "DEMON", "dim": 253,
                "n_components": 2000, "C": 2.0, "penalty": "elasticnet", "l1_ratio": 0.15, "max_iter": 200, "early_stopping": True,
                "class_weight": "balanced", "gamma": "1/(n_features*var)", "random_state": 42, "num_classes": 4}))

    out_data = {
        'mel': mel_preds,
        'lofar': lofar_preds,
        'hht': hht_preds,
        'demon': demon_preds,
        'targets': targets
    }

    out_file = "data/iara_processed/svm_all_representations_predictions.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(out_data, f)
    print(f"\nPredições salvas com sucesso em '{out_file}'!")

if __name__ == "__main__":
    main()
