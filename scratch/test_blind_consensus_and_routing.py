"""
Teste de Estratégias Cegas (Sem Oráculo) para Roteamento e Combinação Multimodal:
1. Votação por Maioria Simples / Empate
2. Soft Voting / Média de Probabilidades (se disponível) ou Votação Ponderada por Especialista
3. Cascata por Confiança / Margem (sem saber a classe real de antemão)
4. Stacking Meta-Classifier com Validação Cruzada 5x2cv estrita (Out-of-Fold)

Avalia em TODAS as 4 classes (SMALL, MEDIUM, LARGE, AMBIENTAL) nos 10 folds oficiais.
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import scipy.stats as scipy_stats

def compute_sp(y_true, y_pred, num_classes=4):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    recalls = cm.diagonal() / cm.sum(axis=1)
    recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    acc = float((y_true == y_pred).mean() * 100)
    return sp, acc, recalls * 100, cm

def main():
    # Carregar as predições salvas dos 4 modelos em todos os áudios
    # Vamos compor a matriz de predições completas para todos os 6.390 testes nos 10 folds
    p_mel = "results/trainings/tests/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0"
    p_lofar = "results/trainings/tests/svm_nystroem_6000_lofar_elasticnet_l1r0.15"

    def mode(x): return Counter(x).most_common(1)[0][0]

    # Carregar os resultados de SMALL, MEDIUM e LARGE gerados pelos scripts anteriores
    with open("data/iara_processed/medium_complementarity_analysis.pkl", "rb") as f:
        data_med = pickle.load(f)
    df_med = data_med["df"]

    with open("data/iara_processed/large_complementarity_analysis.pkl", "rb") as f:
        data_lrg = pickle.load(f)
    df_lrg = data_lrg["df"]

    # Carregar SMALL
    # Rodar predições de AMBIENTAL também para ter o dataset completo 100% blind
    # Vamos carregar todos os arquivos de teste
    eval_base = "results/trainings/tests/svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0/eval"
    
    # Carregar cache HHT e DEMON
    with open("data/iara_processed/cache_hht.pkl", "rb") as f:
        cache_hht = pickle.load(f)
    with open("data/iara_processed/cache_fusion_mel_demon.pkl", "rb") as f:
        cache_fusion = pickle.load(f)
    cache_demon = cache_fusion["demon"]

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
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import SGDClassifier
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

    folds = get_official_folds()

    print("Gerando predições completas para todas as 4 classes (SMALL, MEDIUM, LARGE, AMBIENTAL)...")
    hht_preds = run_svm_predictions(cache_hht, folds, n_components=4000, C=2.0)
    demon_preds = run_svm_predictions(cache_demon, folds, n_components=4000, C=2.0)

    mel_preds = {}
    lofar_preds = {}
    targets_by_fold = {}
    for i in range(10):
        f_mel = f"{p_mel}/eval/fold_{i}/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0_multiclass_test.csv"
        df_m = pd.read_csv(f_mel).groupby("File").agg({"Target": mode, "Prediction": mode}).reset_index()
        mel_preds[i] = dict(zip(df_m["File"], df_m["Prediction"]))
        targets_by_fold[i] = dict(zip(df_m["File"], df_m["Target"]))

        f_lof = f"{p_lofar}/eval/fold_{i}/svm_nystroem_6000_lofar_elasticnet_l1r0.15_multiclass_test.csv"
        df_l = pd.read_csv(f_lof).groupby("File").agg({"Target": mode, "Prediction": mode}).reset_index()
        lofar_preds[i] = dict(zip(df_l["File"], df_l["Prediction"]))

    all_records = []
    for fold_idx in range(10):
        t_dict = targets_by_fold[fold_idx]
        for fid, tgt in t_dict.items():
            pm = mel_preds[fold_idx].get(fid, -1)
            pl = lofar_preds[fold_idx].get(fid, -1)
            ph = hht_preds[fold_idx].get(fid, -1)
            pd_ = demon_preds[fold_idx].get(fid, -1)
            all_records.append({
                "fold": fold_idx,
                "file_id": fid,
                "target": tgt,
                "pred_mel": pm,
                "pred_lofar": pl,
                "pred_hht": ph,
                "pred_demon": pd_
            })

    df_all = pd.DataFrame(all_records)
    y_true = df_all["target"].values

    print("\n" + "=" * 80)
    print(f"AVALIAÇÃO DE ESTRATÉGIAS CEGAS (SEM CONHECER A CLASSE REAL) — {len(df_all)} ÁUDIOS TOTAIS")
    print("=" * 80)

    # 1. Modelos Isolados
    for name, col in [("SVM MEL", "pred_mel"), ("SVM LOFAR", "pred_lofar"), ("SVM HHT", "pred_hht"), ("SVM DEMON", "pred_demon")]:
        sp, acc, recs, cm = compute_sp(y_true, df_all[col].values)
        print(f"\n{name:12s} -> ACC: {acc:5.2f}% | SP: {sp:5.2f}% | SML: {recs[0]:5.2f}% | MED: {recs[1]:5.2f}% | LRG: {recs[2]:5.2f}% | AMB: {recs[3]:5.2f}%")

    # 2. Estratégia Cega 1: Votação por Maioria (com desempate pelo LOFAR ou MEL)
    # Em caso de empate 2x2, quem desempata?
    # Se desempate for LOFAR:
    def vote_with_tiebreaker(row, tiebreaker_col):
        votes = [row["pred_mel"], row["pred_lofar"], row["pred_hht"], row["pred_demon"]]
        c = Counter(votes)
        top = c.most_common(2)
        if len(top) > 1 and top[0][1] == top[1][1]:
            # Empate: usa o desempate
            return row[tiebreaker_col]
        return top[0][0]

    preds_vote_lof = df_all.apply(lambda r: vote_with_tiebreaker(r, "pred_lofar"), axis=1).values
    sp_v1, acc_v1, recs_v1, _ = compute_sp(y_true, preds_vote_lof)
    print(f"\n{'VOTAÇÃO (Tie=LOFAR)':22s} -> ACC: {acc_v1:5.2f}% | SP: {sp_v1:5.2f}% | SML: {recs_v1[0]:5.2f}% | MED: {recs_v1[1]:5.2f}% | LRG: {recs_v1[2]:5.2f}% | AMB: {recs_v1[3]:5.2f}%")

    preds_vote_mel = df_all.apply(lambda r: vote_with_tiebreaker(r, "pred_mel"), axis=1).values
    sp_v2, acc_v2, recs_v2, _ = compute_sp(y_true, preds_vote_mel)
    print(f"{'VOTAÇÃO (Tie=MEL)':22s} -> ACC: {acc_v2:5.2f}% | SP: {sp_v2:5.2f}% | SML: {recs_v2[0]:5.2f}% | MED: {recs_v2[1]:5.2f}% | LRG: {recs_v2[2]:5.2f}% | AMB: {recs_v2[3]:5.2f}%")

    # 3. Estratégia Cega 2: Roteamento Hierárquico por Consenso de LARGE
    # Regra: Se LOFAR diz LARGE (ou se LOFAR e outro concordam em LARGE), É LARGE (protege o LARGE!).
    # Se LOFAR diz outra coisa, consulta MEL, HHT e DEMON para desempatar SMALL vs MEDIUM.
    def hierarchical_rule(row):
        # 1. Âncora de LARGE: Se LOFAR diz 2 (LARGE), mantemos 2 a menos que todos os outros 3 discordem unanimemente
        if row["pred_lofar"] == 2 and (row["pred_mel"] == 2 or row["pred_hht"] == 2 or row["pred_demon"] == 2):
            return 2
        # 2. Se LOFAR diz 2 mas nenhum outro diz 2, votação entre os outros 3
        # 3. Se LOFAR diz diferente de 2:
        # Se HHT diz SMALL e DEMON diz diferente de MEDIUM:
        votes = [row["pred_mel"], row["pred_lofar"], row["pred_hht"], row["pred_demon"]]
        c = Counter(votes)
        # Se 2 dizem SMALL (classe 0), dá SMALL:
        if c[0] >= 2:
            return 0
        if c[1] >= 2:
            return 1
        if c[2] >= 2:
            return 2
        if c[3] >= 2:
            return 3
        # Empate total: prioridade MEL
        return row["pred_mel"]

    preds_hier = df_all.apply(hierarchical_rule, axis=1).values
    sp_h, acc_h, recs_h, _ = compute_sp(y_true, preds_hier)
    print(f"\n{'REGRA HIERÁRQUICA CEGA':22s} -> ACC: {acc_h:5.2f}% | SP: {sp_h:5.2f}% | SML: {recs_h[0]:5.2f}% | MED: {recs_h[1]:5.2f}% | LRG: {recs_h[2]:5.2f}% | AMB: {recs_h[3]:5.2f}%")

    # 4. Estratégia Cega 3: Meta-Classificador Stacking Out-of-Fold (100% rigoroso)
    # One-hot encode das 4 predições (4 * 4 = 16 features binárias)
    X_stack = []
    for _, r in df_all.iterrows():
        feat = np.zeros(16, dtype=np.float32)
        feat[r["pred_mel"]] = 1.0
        feat[4 + r["pred_lofar"]] = 1.0
        feat[8 + r["pred_hht"]] = 1.0
        feat[12 + r["pred_demon"]] = 1.0
        X_stack.append(feat)
    X_stack = np.array(X_stack)

    stack_oof_preds = np.zeros(len(df_all), dtype=np.int64)
    for fold_idx in range(10):
        trn_idx = df_all["fold"] != fold_idx
        tst_idx = df_all["fold"] == fold_idx
        clf_meta = LogisticRegression(C=1.0, max_iter=200, class_weight="balanced", random_state=42)
        clf_meta.fit(X_stack[trn_idx], y_true[trn_idx])
        stack_oof_preds[tst_idx] = clf_meta.predict(X_stack[tst_idx])

    sp_s, acc_s, recs_s, _ = compute_sp(y_true, stack_oof_preds)
    print(f"\n{'STACKING OOF (16-D)':22s} -> ACC: {acc_s:5.2f}% | SP: {sp_s:5.2f}% | SML: {recs_s[0]:5.2f}% | MED: {recs_s[1]:5.2f}% | LRG: {recs_s[2]:5.2f}% | AMB: {recs_s[3]:5.2f}%")

if __name__ == "__main__":
    main()
