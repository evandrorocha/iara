"""
Script para analisar o Consenso em SMALL e o Risco de 'Efeito Reverso' do Roteador / Oráculo.
Responde:
1. Em quantos áudios SMALL a predição estava correta independente da entrada (4/4), em 3, em 2, em 1?
2. Se usarmos MEL como baseline, quantos áudios certos perderíamos ao chavear erroneamente para HHT, LOFAR ou DEMON?
3. Quantos novos áudios cada entrada ganha (resgates) vs perde (risco de perda)?
4. Desempenho de regras de votação por consenso (unanimidade, maioria >=3, >=2, >=1).
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier

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

def run_svm_predictions(feature_cache, folds, n_components=4000, C=2.0, l1_ratio=0.15):
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
    hht_preds = run_svm_predictions(cache_hht, folds, n_components=4000, C=2.0)

    with open("data/iara_processed/cache_fusion_mel_demon.pkl", "rb") as f:
        cache_fusion = pickle.load(f)
    demon_preds = run_svm_predictions(cache_fusion["demon"], folds, n_components=4000, C=2.0)

    # Coletar predições de TODOS os áudios reais de SMALL (Target == 0)
    records = []
    for fold_idx in range(10):
        t_dict = targets_by_fold[fold_idx]
        for fid, tgt in t_dict.items():
            if tgt == 0:  # SMALL
                p_m = mel_preds[fold_idx].get(fid, -1)
                p_l = lofar_preds[fold_idx].get(fid, -1)
                p_h = hht_preds[fold_idx].get(fid, -1)
                p_d = demon_preds[fold_idx].get(fid, -1)

                h_m = (p_m == 0)
                h_l = (p_l == 0)
                h_h = (p_h == 0)
                h_d = (p_d == 0)
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
    total_small = len(df)
    total_correct = (df["num_hits"] > 0).sum()

    print("=" * 80)
    print(f"ANÁLISE DE CONSENSO E ROBUSTEZ DA CLASSE SMALL ({total_small} ÁUDIOS)")
    print("=" * 80)

    print("\n--- 1. GRAU DE CONSENSO SOBRE O TOTAL DE SMALL (1.780 áudios) ---")
    c4 = (df["num_hits"] == 4).sum()
    c3 = (df["num_hits"] == 3).sum()
    c2 = (df["num_hits"] == 2).sum()
    c1 = (df["num_hits"] == 1).sum()
    c0 = (df["num_hits"] == 0).sum()

    print(f"  * 4 Entradas Corretas (INDEPENDENTE DA ENTRADA) : {c4:4d} áudios ({c4/total_small*100:5.2f}%)")
    print(f"  * 3 Entradas Corretas (Alto Consenso)           : {c3:4d} áudios ({c3/total_small*100:5.2f}%)")
    print(f"  * 2 Entradas Corretas (Consenso Parcial/Dividido): {c2:4d} áudios ({c2/total_small*100:5.2f}%)")
    print(f"  * 1 Entrada Correta (Frágil / Resgate Exclusivo): {c1:4d} áudios ({c1/total_small*100:5.2f}%)")
    print(f"  * 0 Entradas Corretas (Erro Comum / Incorrigível): {c0:4d} áudios ({c0/total_small*100:5.2f}%)")

    print("\n--- 2. PROPORÇÃO DE CONSENSO DENTRO DOS 1.495 ÁUDIOS CERTOS (ORÁCULO) ---")
    print(f"  * Consenso Total (4/4)      : {c4:4d} / {total_correct} ({c4/total_correct*100:5.2f}%)")
    print(f"  * Alta Robustez (3 ou 4)    : {c4+c3:4d} / {total_correct} ({(c4+c3)/total_correct*100:5.2f}%)")
    print(f"  * Pelo Menos 2 Entradas (2+): {c4+c3+c2:4d} / {total_correct} ({(c4+c3+c2)/total_correct*100:5.2f}%)")
    print(f"  * Dependência Crítica (Só 1): {c1:4d} / {total_correct} ({c1/total_correct*100:5.2f}%) -> RISCO MÁXIMO!")

    print("\n--- 3. O 'EFEITO REVERSO' EM RELAÇÃO AO BASELINE MEL (903 acertos, 50.73%) ---")
    mel_hits = df[df["hit_mel"]]
    mel_errs = df[~df["hit_mel"]]

    # Se trocarmos cegamente para HHT:
    hht_keeps_mel = mel_hits["hit_hht"].sum()
    hht_loses_mel = (~mel_hits["hit_hht"]).sum()
    hht_rescues_mel = mel_errs["hit_hht"].sum()

    # Se trocarmos cegamente para LOFAR:
    lofar_keeps_mel = mel_hits["hit_lofar"].sum()
    lofar_loses_mel = (~mel_hits["hit_lofar"]).sum()
    lofar_rescues_mel = mel_errs["hit_lofar"].sum()

    # Se trocarmos cegamente para DEMON:
    demon_keeps_mel = mel_hits["hit_demon"].sum()
    demon_loses_mel = (~mel_hits["hit_demon"]).sum()
    demon_rescues_mel = mel_errs["hit_demon"].sum()

    print(f"  Base MEL: {len(mel_hits)} certos | {len(mel_errs)} errados")
    print("\n  Se o roteador chavear de MEL para HHT:")
    print(f"    - Mantém do MEL          : {hht_keeps_mel:4d} ({hht_keeps_mel/len(mel_hits)*100:5.2f}%)")
    print(f"    - PERDE acertos do MEL   : {hht_loses_mel:4d} ({hht_loses_mel/len(mel_hits)*100:5.2f}%) -> EFEITO REVERSO!")
    print(f"    - GANHA dos erros do MEL : {hht_rescues_mel:4d} ({hht_rescues_mel/len(mel_errs)*100:5.2f}%) -> RESGATE!")
    print(f"    - Saldo Líquido          : {hht_rescues_mel - hht_loses_mel:+4d} áudios")

    print("\n  Se o roteador chavear de MEL para LOFAR:")
    print(f"    - Mantém do MEL          : {lofar_keeps_mel:4d} ({lofar_keeps_mel/len(mel_hits)*100:5.2f}%)")
    print(f"    - PERDE acertos do MEL   : {lofar_loses_mel:4d} ({lofar_loses_mel/len(mel_hits)*100:5.2f}%) -> EFEITO REVERSO!")
    print(f"    - GANHA dos erros do MEL : {lofar_rescues_mel:4d} ({lofar_rescues_mel/len(mel_errs)*100:5.2f}%) -> RESGATE!")
    print(f"    - Saldo Líquido          : {lofar_rescues_mel - lofar_loses_mel:+4d} áudios")

    print("\n  Se o roteador chavear de MEL para DEMON:")
    print(f"    - Mantém do MEL          : {demon_keeps_mel:4d} ({demon_keeps_mel/len(mel_hits)*100:5.2f}%)")
    print(f"    - PERDE acertos do MEL   : {demon_loses_mel:4d} ({demon_loses_mel/len(mel_hits)*100:5.2f}%) -> EFEITO REVERSO!")
    print(f"    - GANHA dos erros do MEL : {demon_rescues_mel:4d} ({demon_rescues_mel/len(mel_errs)*100:5.2f}%) -> RESGATE!")
    print(f"    - Saldo Líquido          : {demon_rescues_mel - demon_loses_mel:+4d} áudios")

    print("\n--- 4. VOTAÇÃO POR MAIORIA / CONSENSO (ENSEMBLE SEM ROTEADOR) ---")
    print(f"  * Unanimidade (4 de 4)    : {c4:4d} áudios ({c4/total_small*100:5.2f}% de recall)")
    print(f"  * Maioria Estrita (>= 3/4): {c4+c3:4d} áudios ({(c4+c3)/total_small*100:5.2f}% de recall)")
    print(f"  * Maioria Simples (>= 2/4): {c4+c3+c2:4d} áudios ({(c4+c3+c2)/total_small*100:5.2f}% de recall)")
    print(f"  * Oráculo (>= 1/4)        : {total_correct:4d} áudios ({total_correct/total_small*100:5.2f}% de recall)")

if __name__ == "__main__":
    main()
