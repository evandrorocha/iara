"""
Script de Análise Detalhada de Erros na Classe SMALL para o SVM
Compara as predições do SVM nas 4 representações: MEL, LOFAR, HHT e DEMON.
Objetivo: Verificar se os áudios SMALL que sofrem confusão são SEMPRE OS MESMOS ou se os erros mudam com a entrada.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier

from results_writer import ResultsWriter
import sklearn.metrics as sk_metrics

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
        train_files = df_trn.groupby("File")["Target"].first().to_dict()
        test_files = df_tst.groupby("File")["Target"].first().to_dict()
        folds.append({
            "fold_idx": fold_idx,
            "train_files": train_files,
            "test_files": test_files,
        })
    return folds

def build_fold_matrices(feature_cache, file_dict):
    X_list = []
    y_list = []
    file_ids = []
    for fid, target in file_dict.items():
        if fid not in feature_cache:
            continue
        windows = feature_cache[fid]
        X_list.append(windows)
        y_list.extend([target] * len(windows))
        file_ids.extend([fid] * len(windows))
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)
    file_ids = np.array(file_ids, dtype=np.int64)
    return X, y, file_ids

def run_svm_predictions(feature_cache, folds, n_components=4000, C=2.0, l1_ratio=0.15, writer=None):
    preds_by_fold = {}
    for fold in folds:
        fold_idx = fold["fold_idx"]
        X_train, y_train, _ = build_fold_matrices(feature_cache, fold["train_files"])
        X_test, y_test, test_fids = build_fold_matrices(feature_cache, fold["test_files"])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        gamma = 1.0 / (X_train.shape[1] * X_train.var())
        nystroem = Nystroem(kernel="rbf", gamma=gamma, n_components=n_components, random_state=42, n_jobs=-1)
        X_train_trans = nystroem.fit_transform(X_train)
        X_test_trans = nystroem.transform(X_test)

        alpha = 1.0 / (C * X_train.shape[0])
        clf = SGDClassifier(
            loss="hinge", penalty="elasticnet", l1_ratio=l1_ratio,
            alpha=alpha, max_iter=200, tol=1e-3, early_stopping=True,
            n_iter_no_change=5, class_weight="balanced", random_state=42, n_jobs=-1
        )
        clf.fit(X_train_trans, y_train)

        win_preds = clf.predict(X_test_trans)
        if writer is not None:
            writer.save_fold(fold_idx, test_fids, y_test, win_preds, train_files=fold["train_files"],
                             scores=clf.decision_function(X_test_trans))
        file_preds = defaultdict(list)
        file_targets = {}
        for fid, pred, tgt in zip(test_fids, win_preds, y_test):
            file_preds[fid].append(pred)
            file_targets[fid] = tgt

        audio_preds = {fid: int(np.argmax(np.bincount(file_preds[fid], minlength=4))) for fid in file_preds}
        preds_by_fold[fold_idx] = audio_preds
    return preds_by_fold

def main():
    folds = get_official_folds()

    # 1. Carregar predições salvas do SVM MEL
    print("Carregando predições de SVM MEL...")
    p_mel = "results/trainings/tests/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0"
    mel_preds = {}
    targets_by_fold = {}
    for i in range(10):
        fpath = os.path.join(p_mel, f"eval/fold_{i}/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0_multiclass_test.csv")
        df = pd.read_csv(fpath).groupby("File").agg({"Target": mode, "Prediction": mode}).reset_index()
        mel_preds[i] = dict(zip(df["File"], df["Prediction"]))
        targets_by_fold[i] = dict(zip(df["File"], df["Target"]))

    # 2. Carregar predições salvas do SVM LOFAR
    print("Carregando predições de SVM LOFAR...")
    p_lofar = "results/trainings/tests/svm_nystroem_6000_lofar_elasticnet_l1r0.15"
    lofar_preds = {}
    for i in range(10):
        fpath = os.path.join(p_lofar, f"eval/fold_{i}/svm_nystroem_6000_lofar_elasticnet_l1r0.15_multiclass_test.csv")
        df = pd.read_csv(fpath).groupby("File").agg({"Target": mode, "Prediction": mode}).reset_index()
        lofar_preds[i] = dict(zip(df["File"], df["Prediction"]))

    # 3. Rodar predições de SVM HHT
    print("Carregando cache HHT e gerando predições SVM HHT...")
    with open("data/iara_processed/cache_hht.pkl", "rb") as f:
        cache_hht = pickle.load(f)
    hht_preds = run_svm_predictions(cache_hht, folds, n_components=4000, C=2.0, writer=ResultsWriter(
        "svm_nystroem_4000_hht_C2.0_elasticnet_l1r0.15_4c", script=__file__,
        params={"model": "Nystroem(rbf)+SGDClassifier(hinge)", "representation": "HHT", "dim": 256,
                "n_components": 4000, "C": 2.0, "penalty": "elasticnet", "l1_ratio": 0.15, "max_iter": 200,
                "early_stopping": True, "class_weight": "balanced", "gamma": "1/(n_features*var)",
                "random_state": 42, "num_classes": 4}))

    # 4. Rodar predições de SVM DEMON
    print("Carregando cache DEMON e gerando predições SVM DEMON...")
    with open("data/iara_processed/cache_fusion_mel_demon.pkl", "rb") as f:
        cache_fusion = pickle.load(f)
    cache_demon = cache_fusion["demon"]
    demon_preds = run_svm_predictions(cache_demon, folds, n_components=4000, C=2.0, writer=ResultsWriter(
        "svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_4c", script=__file__,
        params={"model": "Nystroem(rbf)+SGDClassifier(hinge)", "representation": "DEMON", "dim": 253,
                "n_components": 4000, "C": 2.0, "penalty": "elasticnet", "l1_ratio": 0.15, "max_iter": 200,
                "early_stopping": True, "class_weight": "balanced", "gamma": "1/(n_features*var)",
                "random_state": 42, "num_classes": 4}))

    # Consolidar comparação sobre todos os áudios reais de SMALL (Target == 0)
    records = []
    for fold_idx in range(10):
        t_dict = targets_by_fold[fold_idx]
        for fid, tgt in t_dict.items():
            if tgt == 0:  # SMALL
                p_m = mel_preds[fold_idx].get(fid, -1)
                p_l = lofar_preds[fold_idx].get(fid, -1)
                p_h = hht_preds[fold_idx].get(fid, -1)
                p_d = demon_preds[fold_idx].get(fid, -1)

                records.append({
                    "fold": fold_idx,
                    "file_id": fid,
                    "target": 0,
                    "pred_mel": p_m,
                    "pred_lofar": p_l,
                    "pred_hht": p_h,
                    "pred_demon": p_d,
                    "hit_mel": p_m == 0,
                    "hit_lofar": p_l == 0,
                    "hit_hht": p_h == 0,
                    "hit_demon": p_d == 0,
                    "num_hits": (p_m == 0) + (p_l == 0) + (p_h == 0) + (p_d == 0)
                })

    df = pd.DataFrame(records)
    total_small = len(df)
    print("\n" + "=" * 80)
    print(f"ANÁLISE COMPARATIVA DE ERROS EM SMALL NO SVM (Total Avaliado: {total_small} áudios)")
    print("=" * 80)

    print("\n--- 1. DESEMPENHO INDIVIDUAL NO SMALL ---")
    print(f"  Acertos SVM MEL  : {df['hit_mel'].sum():4d} / {total_small} ({df['hit_mel'].mean()*100:.2f}%)")
    print(f"  Acertos SVM LOFAR: {df['hit_lofar'].sum():4d} / {total_small} ({df['hit_lofar'].mean()*100:.2f}%)")
    print(f"  Acertos SVM HHT  : {df['hit_hht'].sum():4d} / {total_small} ({df['hit_hht'].mean()*100:.2f}%)")
    print(f"  Acertos SVM DEMON: {df['hit_demon'].sum():4d} / {total_small} ({df['hit_demon'].mean()*100:.2f}%)")

    print("\n--- 2. CONCORDÂNCIA E COMPLEMENTARIDADE DE ACERTOS/ERROS ---")
    hit_counts = df['num_hits'].value_counts().sort_index()
    for hits, count in hit_counts.items():
        pct = count / total_small * 100
        desc = ""
        if hits == 4: desc = "-> Acertado por TODAS as 4 representações"
        elif hits == 3: desc = "-> Acertado por 3 representações"
        elif hits == 2: desc = "-> Acertado por 2 representações"
        elif hits == 1: desc = "-> Acertado por APENAS 1 representação (exclusivo!)"
        elif hits == 0: desc = "-> NENHUMA acertou (erro incorrigível/comum)"
        print(f"  {hits} modelos acertaram: {count:4d} áudios ({pct:5.2f}%) {desc}")

    union_hits = (df['num_hits'] > 0).sum()
    print(f"\n  >>> UNIÃO TOTAL (Pelo menos um acertou): {union_hits:4d} / {total_small} ({union_hits/total_small*100:.2f}%)")
    print(f"  >>> ERRO COMUM (Todos erraram):          {hit_counts[0]:4d} / {total_small} ({hit_counts[0]/total_small*100:.2f}%)")

    print("\n--- 3. ACERTOS EXCLUSIVOS (Onde só uma entrada salva o áudio) ---")
    only_mel = ((df['hit_mel']) & (~df['hit_lofar']) & (~df['hit_hht']) & (~df['hit_demon'])).sum()
    only_lofar = ((~df['hit_mel']) & (df['hit_lofar']) & (~df['hit_hht']) & (~df['hit_demon'])).sum()
    only_hht = ((~df['hit_mel']) & (~df['hit_lofar']) & (df['hit_hht']) & (~df['hit_demon'])).sum()
    only_demon = ((~df['hit_mel']) & (~df['hit_lofar']) & (~df['hit_hht']) & (df['hit_demon'])).sum()

    print(f"  Salvos EXCLUSIVAMENTE pelo MEL  : {only_mel:4d} áudios ({only_mel/total_small*100:.2f}%)")
    print(f"  Salvos EXCLUSIVAMENTE pelo LOFAR: {only_lofar:4d} áudios ({only_lofar/total_small*100:.2f}%)")
    print(f"  Salvos EXCLUSIVAMENTE pelo HHT  : {only_hht:4d} áudios ({only_hht/total_small*100:.2f}%)")
    print(f"  Salvos EXCLUSIVAMENTE pelo DEMON: {only_demon:4d} áudios ({only_demon/total_small*100:.2f}%)")

    print("\n--- 4. PARA ONDE O SMALL VAI QUANDO CADA ENTRADA ERRA? (PADRÃO DE CONFUSÃO) ---")
    class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "AMBIENTAL"}
    for mod_name, col in [("MEL", "pred_mel"), ("LOFAR", "pred_lofar"), ("HHT", "pred_hht"), ("DEMON", "pred_demon")]:
        errs = df[df[col] != 0][col].map(class_map).value_counts()
        print(f"\n  Padrão de Erro no {mod_name} (Total Erros: {(df[col] != 0).sum()}):")
        for cls, cnt in errs.items():
            print(f"    -> Confundido com {cls:<10}: {cnt:4d} ({cnt/(df[col] != 0).sum()*100:5.2f}%)")

if __name__ == "__main__":
    main()
