"""
Script de Análise Detalhada de Complementaridade, Consenso e Efeito Oráculo para a Classe MEDIUM (Classe 1).
Compara as 4 representações no SVM Nyström sob o protocolo oficial 5x2cv (10 folds):
- MEL
- LOFAR
- HHT
- DEMON
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier

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
        for fid, pred in zip(test_fids, win_preds):
            file_preds[fid].append(pred)

        preds_by_fold[fold_idx] = {fid: int(np.argmax(np.bincount(file_preds[fid], minlength=4))) for fid in file_preds}
    return preds_by_fold

def main():
    folds = get_official_folds()

    p_mel = "results/trainings/tests/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0"
    mel_preds, targets_by_fold = {}, {}
    for i in range(10):
        fpath = os.path.join(p_mel, f"eval/fold_{i}/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0_multiclass_test.csv")
        df = pd.read_csv(fpath).groupby("File").agg({"Target": mode, "Prediction": mode}).reset_index()
        mel_preds[i] = dict(zip(df["File"], df["Prediction"]))
        targets_by_fold[i] = dict(zip(df["File"], df["Target"]))

    p_lofar = "results/trainings/tests/svm_nystroem_6000_lofar_elasticnet_l1r0.15"
    lofar_preds = {}
    for i in range(10):
        fpath = os.path.join(p_lofar, f"eval/fold_{i}/svm_nystroem_6000_lofar_elasticnet_l1r0.15_multiclass_test.csv")
        df = pd.read_csv(fpath).groupby("File").agg({"Target": mode, "Prediction": mode}).reset_index()
        lofar_preds[i] = dict(zip(df["File"], df["Prediction"]))

    with open("data/iara_processed/cache_hht.pkl", "rb") as f:
        cache_hht = pickle.load(f)
    hht_preds = run_svm_predictions(cache_hht, folds, n_components=4000, C=2.0, writer=ResultsWriter(
        "svm_nystroem_4000_hht_C2.0_elasticnet_l1r0.15_4c", script=__file__,
        params={"model": "Nystroem(rbf)+SGDClassifier(hinge)", "representation": "HHT", "dim": 256,
                "n_components": 4000, "C": 2.0, "penalty": "elasticnet", "l1_ratio": 0.15, "max_iter": 200,
                "early_stopping": True, "class_weight": "balanced", "gamma": "1/(n_features*var)",
                "random_state": 42, "num_classes": 4}))

    with open("data/iara_processed/cache_fusion_mel_demon.pkl", "rb") as f:
        cache_fusion = pickle.load(f)
    demon_preds = run_svm_predictions(cache_fusion["demon"], folds, n_components=4000, C=2.0, writer=ResultsWriter(
        "svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_4c", script=__file__,
        params={"model": "Nystroem(rbf)+SGDClassifier(hinge)", "representation": "DEMON", "dim": 253,
                "n_components": 4000, "C": 2.0, "penalty": "elasticnet", "l1_ratio": 0.15, "max_iter": 200,
                "early_stopping": True, "class_weight": "balanced", "gamma": "1/(n_features*var)",
                "random_state": 42, "num_classes": 4}))

    # Filtrar para MEDIUM (Target == 1)
    records = []
    for fold_idx in range(10):
        t_dict = targets_by_fold[fold_idx]
        for fid, tgt in t_dict.items():
            if tgt == 1:  # MEDIUM
                p_m = mel_preds[fold_idx].get(fid, -1)
                p_l = lofar_preds[fold_idx].get(fid, -1)
                p_h = hht_preds[fold_idx].get(fid, -1)
                p_d = demon_preds[fold_idx].get(fid, -1)

                h_m = (p_m == 1)
                h_l = (p_l == 1)
                h_h = (p_h == 1)
                h_d = (p_d == 1)
                n_hits = h_m + h_l + h_h + h_d

                records.append({
                    "fold": fold_idx,
                    "file_id": fid,
                    "pred_mel": p_m,
                    "pred_lofar": p_l,
                    "pred_hht": p_h,
                    "pred_demon": p_d,
                    "hit_mel": h_m,
                    "hit_lofar": h_l,
                    "hit_hht": h_h,
                    "hit_demon": h_d,
                    "num_hits": n_hits
                })

    df = pd.DataFrame(records)
    total_med = len(df)
    total_correct = (df["num_hits"] > 0).sum()

    print("=" * 80)
    print(f"ANÁLISE DE COMPLEMENTARIDADE E CONSENSO PARA A CLASSE MEDIUM ({total_med} ÁUDIOS)")
    print("=" * 80)

    print("\n--- 1. DESEMPENHO INDIVIDUAL NO MEDIUM ---")
    print(f"  Acertos SVM MEL  : {df['hit_mel'].sum():4d} / {total_med} ({df['hit_mel'].mean()*100:.2f}%)")
    print(f"  Acertos SVM LOFAR: {df['hit_lofar'].sum():4d} / {total_med} ({df['hit_lofar'].mean()*100:.2f}%)")
    print(f"  Acertos SVM HHT  : {df['hit_hht'].sum():4d} / {total_med} ({df['hit_hht'].mean()*100:.2f}%)")
    print(f"  Acertos SVM DEMON: {df['hit_demon'].sum():4d} / {total_med} ({df['hit_demon'].mean()*100:.2f}%)")

    print("\n--- 2. TETO TEÓRICO DO ORÁCULO E ERRO COMUM ---")
    print(f"  >>> ORÁCULO (Pelo menos 1 acertou): {total_correct:4d} / {total_med} ({total_correct/total_med*100:.2f}%)")
    print(f"  >>> ERRO COMUM (Todos erraram)    : {total_med - total_correct:4d} / {total_med} ({(total_med - total_correct)/total_med*100:.2f}%)")

    print("\n--- 3. GRAU DE CONSENSO SOBRE O TOTAL (1.490 áudios) ---")
    c4 = (df["num_hits"] == 4).sum()
    c3 = (df["num_hits"] == 3).sum()
    c2 = (df["num_hits"] == 2).sum()
    c1 = (df["num_hits"] == 1).sum()
    c0 = (df["num_hits"] == 0).sum()

    print(f"  * 4 Entradas (Unanimidade / Independente): {c4:4d} ({c4/total_med*100:5.2f}%)")
    print(f"  * 3 Entradas (Alto Consenso)             : {c3:4d} ({c3/total_med*100:5.2f}%)")
    print(f"  * 2 Entradas (Consenso Dividido)         : {c2:4d} ({c2/total_med*100:5.2f}%)")
    print(f"  * 1 Entrada  (Frágil / Exclusivo)        : {c1:4d} ({c1/total_med*100:5.2f}%)")
    print(f"  * 0 Entradas (Erro Comum)                : {c0:4d} ({c0/total_med*100:5.2f}%)")

    print("\n--- 4. RESGATES EXCLUSIVOS (Apenas 1 entrada salva o áudio) ---")
    only_mel = ((df['hit_mel']) & (~df['hit_lofar']) & (~df['hit_hht']) & (~df['hit_demon'])).sum()
    only_lofar = ((~df['hit_mel']) & (df['hit_lofar']) & (~df['hit_hht']) & (~df['hit_demon'])).sum()
    only_hht = ((~df['hit_mel']) & (~df['hit_lofar']) & (df['hit_hht']) & (~df['hit_demon'])).sum()
    only_demon = ((~df['hit_mel']) & (~df['hit_lofar']) & (~df['hit_hht']) & (df['hit_demon'])).sum()

    print(f"  Salvos EXCLUSIVAMENTE pelo MEL  : {only_mel:4d} ({only_mel/total_med*100:5.2f}%)")
    print(f"  Salvos EXCLUSIVAMENTE pelo LOFAR: {only_lofar:4d} ({only_lofar/total_med*100:5.2f}%)")
    print(f"  Salvos EXCLUSIVAMENTE pelo HHT  : {only_hht:4d} ({only_hht/total_med*100:5.2f}%)")
    print(f"  Salvos EXCLUSIVAMENTE pelo DEMON: {only_demon:4d} ({only_demon/total_med*100:5.2f}%)")

    print("\n--- 5. PADRÃO DE CONFUSÃO EM MEDIUM (QUANDO CADA ENTRADA ERRA, PARA ONDE VAI?) ---")
    class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "AMBIENTAL"}
    for mod_name, col in [("MEL", "pred_mel"), ("LOFAR", "pred_lofar"), ("HHT", "pred_hht"), ("DEMON", "pred_demon")]:
        errs = df[df[col] != 1][col].map(class_map).value_counts()
        total_errs = (df[col] != 1).sum()
        print(f"\n  Padrão de Erro no {mod_name} (Total Erros: {total_errs}):")
        for cls, cnt in errs.items():
            print(f"    -> Confundido com {cls:<10}: {cnt:4d} ({cnt/total_errs*100:5.2f}%)")

    print("\n--- 6. EFEITO REVERSO A PARTIR DO MEL (900 certos, 590 errados) ---")
    mel_hits = df[df["hit_mel"]]
    mel_errs = df[~df["hit_mel"]]

    for name, hit_col in [("HHT", "hit_hht"), ("LOFAR", "hit_lofar"), ("DEMON", "hit_demon")]:
        keeps = mel_hits[hit_col].sum()
        loses = (~mel_hits[hit_col]).sum()
        rescues = mel_errs[hit_col].sum()
        saldo = rescues - loses
        print(f"\n  Se chavear de MEL para {name}:")
        print(f"    - Mantém do MEL          : {keeps:4d} ({keeps/len(mel_hits)*100:5.2f}%)")
        print(f"    - PERDE acertos do MEL   : {loses:4d} ({loses/len(mel_hits)*100:5.2f}%) -> Efeito Reverso")
        print(f"    - GANHA dos erros do MEL : {rescues:4d} ({rescues/len(mel_errs)*100:5.2f}%) -> Resgate")
        print(f"    - Saldo Líquido          : {saldo:+4d} áudios")

    print("\n--- 7. VOTAÇÃO POR MAIORIA / CONSENSO EM MEDIUM ---")
    print(f"  * Unanimidade (4 de 4)    : {c4:4d} ({c4/total_med*100:5.2f}% de recall)")
    print(f"  * Maioria Estrita (>= 3/4): {c4+c3:4d} ({(c4+c3)/total_med*100:5.2f}% de recall)")
    print(f"  * Maioria Simples (>= 2/4): {c4+c3+c2:4d} ({(c4+c3+c2)/total_med*100:5.2f}% de recall)")
    print(f"  * Oráculo (>= 1/4)        : {total_correct:4d} ({total_correct/total_med*100:5.2f}% de recall)")

    # Salvar resultados brutos para uso na geração do notebook e gráficos
    out_pkl = "data/iara_processed/medium_complementarity_analysis.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "df": df,
            "total_med": total_med,
            "total_correct": total_correct,
            "c4": c4, "c3": c3, "c2": c2, "c1": c1, "c0": c0,
            "only_mel": only_mel, "only_lofar": only_lofar, "only_hht": only_hht, "only_demon": only_demon,
        }, f)
    print(f"\nDados salvos com sucesso em '{out_pkl}'.")

if __name__ == "__main__":
    main()
