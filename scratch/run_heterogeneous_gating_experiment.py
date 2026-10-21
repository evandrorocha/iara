"""
Experimento Oficial: Sistema Gating Front-End Heterogêneo (Heterogeneous Mixture of Experts - H-MoE)
Protocolo Oficial: 10 Folds 5x2cv (Silva et al., 2025) com Treino Estritamente Disjunto (trn_fold = fold ^ 1).

Metodologia de Descoberta Automática (Nested Selection):
Para cada fold k (k=0..9):
  1. No conjunto de treino estritamente disjunto da mesma repetição (trn_fold = fold ^ 1, 0.0% overlap),
     avalia-se o SP isolado de cada candidato (RF, MLP, SVM) em cada uma das 4 representações (MEL, LOFAR, HHT, DEMON).
  2. Seleciona-se o especialista ótimo para cada representação:
     M*(r) = argmax_M SP_train(r, M).
  3. Treina-se o RF Competence Router (max_depth=3, min_samples_leaf=5, 7 meta-features físicas)
     para prever a probabilidade de acerto dos 4 especialistas selecionados:
     P(Especialista_{r, M*(r)} acertar | x_meta), combinada com o prior empírico de SP do treino.
  4. No conjunto de teste do fold k, o router avalia x_meta e direciona o áudio para
     o especialista com maior escore de competência estimado.
"""
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import scipy.stats as scipy_stats
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def compute_sp(y_true, y_pred, num_classes=4):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    recalls = cm.diagonal() / (cm.sum(axis=1) + 1e-12)
    recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100)
    acc = float((y_true == y_pred).mean() * 100)
    return sp, bal_acc, acc, recalls * 100, cm

def main():
    print("=" * 100)
    print("EXPERIMENTO H-MoE: GATING FRONT-END COM ESPECIALISTAS HETEROGÊNEOS")
    print("Protocolo Oficial: 10 Folds 5x2cv (Silva et al., 2025) com Treino Disjunto (trn_fold = fold ^ 1)")
    print("=" * 100)

    # 1. Carregar predições cacheadas de todas as arquiteturas
    files = {
        'RF': 'data/iara_processed/rf_all_representations_predictions.pkl',
        'SVM': 'data/iara_processed/svm_all_representations_predictions.pkl',
        'MLP': 'data/iara_processed/mlp_all_representations_predictions.pkl',
        'CNN': 'data/iara_processed/cnn_all_representations_predictions.pkl',
    }
    data = {k: pickle.load(open(v, 'rb')) for k, v in files.items()}

    with open('data/iara_processed/cache_gating_metafeatures.pkl', 'rb') as f:
        meta_dict = pickle.load(f)

    feature_cols = ['crest_factor', 'spectral_flatness', 'mod_index', 'flux_std', 'high_low_ratio', 'rms_db', 'zcr']
    reps = ['mel', 'lofar', 'hht', 'demon']
    candidate_models = ['RF', 'MLP', 'SVM']  # CNN foi fraca em LOFAR/HHT/DEMON, mantemos RF, MLP, SVM

    res_homog_rf = []
    res_hetero_auto = []
    res_hetero_expert_fixed = []
    res_super_oracle = []
    
    routes_auto_counter = Counter()
    routes_fixed_counter = Counter()
    selected_models_per_fold = {r: [] for r in reps}

    # Configuração Heterogênea Especializada Fixa (Informed by Physics):
    # MEL: RF (protege Small), LOFAR: MLP (máximo SP e harmônicos diesel), HHT: RF (não colapsa), DEMON: MLP
    fixed_expert_map = {'mel': 'RF', 'lofar': 'MLP', 'hht': 'RF', 'demon': 'MLP'}

    for fold in range(10):
        tst_targets = data['RF']['targets'][fold]
        fids_test = sorted(list(tst_targets.keys()))
        y_test = np.array([tst_targets[fid] for fid in fids_test])

        # --- A) Descoberta Automática de Modelos no Treino Estritamente Disjunto (trn_fold = fold ^ 1) ---
        trn_fold = fold ^ 1
        trn_fids_raw = sorted(list(data['RF']['targets'][trn_fold].keys()))
        assert len(set(fids_test).intersection(trn_fids_raw)) == 0, f"Leakage detectado entre fold {fold} e {trn_fold}!"
        train_fids_all = [(trn_fold, fid) for fid in trn_fids_raw if fid in meta_dict]

        best_model_per_rep = {}
        for r in reps:
            best_sp = -1.0
            best_m = candidate_models[0]
            for m in candidate_models:
                y_tr_true = []
                y_tr_pred = []
                for tf, fid in train_fids_all:
                    y_tr_true.append(data[m]['targets'][tf][fid])
                    y_tr_pred.append(data[m][r][tf][fid])
                sp_m, _, _, _, _ = compute_sp(np.array(y_tr_true), np.array(y_tr_pred))
                if sp_m > best_sp:
                    best_sp = sp_m
                    best_m = m
            best_model_per_rep[r] = best_m
            selected_models_per_fold[r].append(best_m)

        # --- B) Super-Oráculo Teórico (16 Especialistas: 4 modelos x 4 representações) ---
        all_preds = []
        for m in ['RF', 'SVM', 'MLP', 'CNN']:
            for r in reps:
                all_preds.append([data[m][r][fold][fid] for fid in fids_test])
        all_preds = np.array(all_preds)
        y_oracle = []
        for i, yt in enumerate(y_test):
            cands = all_preds[:, i]
            y_oracle.append(yt if yt in cands else cands[0])
        res_super_oracle.append(compute_sp(y_test, np.array(y_oracle)))

        # --- C) Treinamento do Roteador de Competência no Fold Disjunto (trn_fold = fold ^ 1) ---
        X_train, y_comp_auto, y_comp_fixed, y_comp_rf = [], [], [], []
        y_tr_true_list = []
        cands_tr_auto, cands_tr_fixed, cands_tr_rf = [], [], []
        for tf, fid in train_fids_all:
            yt = data['RF']['targets'][tf][fid]
            y_tr_true_list.append(yt)
            X_train.append([meta_dict[fid][c] for c in feature_cols])
            
            cands_tr_auto.append([data[best_model_per_rep[r]][r][tf][fid] for r in reps])
            cands_tr_fixed.append([data[fixed_expert_map[r]][r][tf][fid] for r in reps])
            cands_tr_rf.append([data['RF'][r][tf][fid] for r in reps])

            # Auto: modelo descoberto por representação
            y_comp_auto.append([
                int(data[best_model_per_rep[r]][r][tf][fid] == yt)
                for r in reps
            ])
            # Fixo especializado (MEL:RF, LOFAR:MLP, HHT:RF, DEMON:MLP)
            y_comp_fixed.append([
                int(data[fixed_expert_map[r]][r][tf][fid] == yt)
                for r in reps
            ])
            # RF Homogêneo
            y_comp_rf.append([
                int(data['RF'][r][tf][fid] == yt)
                for r in reps
            ])

        X_train = np.array(X_train)
        y_tr_true_arr = np.array(y_tr_true_list)
        cands_tr_auto = np.array(cands_tr_auto)
        cands_tr_fixed = np.array(cands_tr_fixed)
        cands_tr_rf = np.array(cands_tr_rf)
        y_comp_auto = np.array(y_comp_auto)
        y_comp_fixed = np.array(y_comp_fixed)
        y_comp_rf = np.array(y_comp_rf)

        sp_prior_auto = np.array([compute_sp(y_tr_true_arr, cands_tr_auto[:, m])[0] / 100.0 for m in range(4)])
        sp_prior_fixed = np.array([compute_sp(y_tr_true_arr, cands_tr_fixed[:, m])[0] / 100.0 for m in range(4)])
        sp_prior_rf = np.array([compute_sp(y_tr_true_arr, cands_tr_rf[:, m])[0] / 100.0 for m in range(4)])

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        # Treinar os 3 roteadores regularizados
        clfs_auto = [
            RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train_s, y_comp_auto[:, m])
            for m in range(4)
        ]
        clfs_fixed = [
            RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train_s, y_comp_fixed[:, m])
            for m in range(4)
        ]
        clfs_rf = [
            RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train_s, y_comp_rf[:, m])
            for m in range(4)
        ]

        # Inferência no conjunto de teste disjunto
        X_test = np.array([[meta_dict[fid][c] if fid in meta_dict else 0.0 for c in feature_cols] for fid in fids_test])
        X_test_s = scaler.transform(X_test)

        probs_auto = np.column_stack([clfs_auto[m].predict_proba(X_test_s)[:, 1] for m in range(4)])
        routes_auto = np.argmax(0.90 * probs_auto + 0.10 * sp_prior_auto[None, :], axis=1)

        probs_fixed = np.column_stack([clfs_fixed[m].predict_proba(X_test_s)[:, 1] for m in range(4)])
        routes_fixed = np.argmax(0.90 * probs_fixed + 0.10 * sp_prior_fixed[None, :], axis=1)

        probs_rf = np.column_stack([clfs_rf[m].predict_proba(X_test_s)[:, 1] for m in range(4)])
        routes_rf = np.argmax(0.90 * probs_rf + 0.10 * sp_prior_rf[None, :], axis=1)

        # Predições Finais
        y_pred_auto, y_pred_fixed, y_pred_rf = [], [], []
        for i in range(len(fids_test)):
            fid = fids_test[i]
            # Auto
            r_auto = reps[routes_auto[i]]
            m_auto = best_model_per_rep[r_auto]
            routes_auto_counter[f"{r_auto.upper()}:{m_auto}"] += 1
            y_pred_auto.append(data[m_auto][r_auto][fold][fid])
            
            # Fixo
            r_fixed = reps[routes_fixed[i]]
            m_fixed = fixed_expert_map[r_fixed]
            routes_fixed_counter[f"{r_fixed.upper()}:{m_fixed}"] += 1
            y_pred_fixed.append(data[m_fixed][r_fixed][fold][fid])

            # RF Homogêneo
            r_rf = reps[routes_rf[i]]
            y_pred_rf.append(data['RF'][r_rf][fold][fid])

        res_hetero_auto.append(compute_sp(y_test, np.array(y_pred_auto)))
        res_hetero_expert_fixed.append(compute_sp(y_test, np.array(y_pred_fixed)))
        res_homog_rf.append(compute_sp(y_test, np.array(y_pred_rf)))

    def format_row(name, res_list):
        sps = [r[0] for r in res_list]
        bal_accs = [r[1] for r in res_list]
        accs = [r[2] for r in res_list]
        recs = np.mean([r[3] for r in res_list], axis=0)
        return {
            'Arquitetura': name,
            'SP (%)': f"{np.mean(sps):.2f} ± {np.std(sps):.2f}",
            'Bal Acc (%)': f"{np.mean(bal_accs):.2f} ± {np.std(bal_accs):.2f}",
            'Acc (%)': f"{np.mean(accs):.2f} ± {np.std(accs):.2f}",
            'Recall SMALL (%)': f"{recs[0]:.1f}%",
            'Recall MEDIUM (%)': f"{recs[1]:.1f}%",
            'Recall LARGE (%)': f"{recs[2]:.1f}%",
            'Recall BG (%)': f"{recs[3]:.1f}%",
        }

    rows = [
        format_row('1. Baseline MEL Isolado (RF)', [
            compute_sp(
                np.array([data['RF']['targets'][f][fid] for fid in sorted(list(data['RF']['targets'][f].keys()))]),
                np.array([data['RF']['mel'][f][fid] for fid in sorted(list(data['RF']['targets'][f].keys()))])
            )
            for f in range(10)
        ]),
        format_row('2. Baseline LOFAR Isolado (MLP)', [
            compute_sp(
                np.array([data['MLP']['targets'][f][fid] for fid in sorted(list(data['MLP']['targets'][f].keys()))]),
                np.array([data['MLP']['lofar'][f][fid] for fid in sorted(list(data['MLP']['targets'][f].keys()))])
            )
            for f in range(10)
        ]),
        format_row('3. Gated RF Homogêneo (MEL, LOFAR, HHT, DEMON todos em RF)', res_homog_rf),
        format_row('4. H-MoE Automático (Nested CV Model Selection)', res_hetero_auto),
        format_row('5. H-MoE Especializado (MEL:RF, LOFAR:MLP, HHT:RF, DEM:MLP)', res_hetero_expert_fixed),
        format_row('6. Super-Oráculo Teórico (16 Especialistas: 4 Modelos x 4 Entradas)', res_super_oracle)
    ]

    df_results = pd.DataFrame(rows)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\n" + "=" * 100)
    print("RESULTADOS FINAIS COMPARATIVOS: H-MoE VS. GATING HOMOGÊNEO")
    print("=" * 100)
    print(df_results)

    print("\nModelos selecionados automaticamente por representação nos 10 folds:")
    for r in reps:
        print(f"  {r.upper()}: {Counter(selected_models_per_fold[r])}")

    print("\nDistribuição de Rotas no H-MoE Especializado:")
    tot_routes = sum(routes_fixed_counter.values())
    for k, v in routes_fixed_counter.most_common():
        print(f"  {k}: {v} áudios ({v/tot_routes*100:.1f}%)")

    # Salvar resultados em pickle
    out_path = 'data/iara_processed/results_heterogeneous_gating.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump({
            'df': df_results,
            'routes_fixed': routes_fixed_counter,
            'routes_auto': routes_auto_counter,
            'selected_models_per_fold': selected_models_per_fold,
            'res_hetero_expert_fixed': res_hetero_expert_fixed,
            'res_super_oracle': res_super_oracle
        }, f)
    print(f"\nResultados salvos com sucesso em {out_path}!")

if __name__ == '__main__':
    main()
