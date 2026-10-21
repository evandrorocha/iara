"""
Avaliação do Roteador de Competência (RF Competence Router) aplicado a TODOS os Modelos:
1. MLP
2. Random Forest (RF)
3. CNN 2D
Protocolo Oficial: 10 Folds 5x2cv (Silva et al., 2025) com Treino Estritamente Disjunto (trn_fold = fold ^ 1).
"""
import os
import pickle
import time
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

def evaluate_model_family(model_name, pkl_name, meta_dict, feature_cols):
    with open('data/iara_processed/' + pkl_name, 'rb') as f:
        pred_data = pickle.load(f)

    res_mel = []
    res_lofar = []
    res_hht = []
    res_demon = []
    res_gated = []
    res_oracle = []

    routes_counter = Counter()

    for fold in range(10):
        tst_targets = pred_data['targets'][fold]
        tst_mel = pred_data['mel'][fold]
        tst_lofar = pred_data['lofar'][fold]
        tst_hht = pred_data['hht'][fold]
        tst_demon = pred_data['demon'][fold]

        fids_test = sorted(list(tst_targets.keys()))
        y_test = np.array([tst_targets[fid] for fid in fids_test])
        y_mel = np.array([tst_mel[fid] for fid in fids_test])
        y_lofar = np.array([tst_lofar[fid] for fid in fids_test])
        y_hht = np.array([tst_hht[fid] for fid in fids_test])
        y_demon = np.array([tst_demon[fid] for fid in fids_test])

        res_mel.append(compute_sp(y_test, y_mel))
        res_lofar.append(compute_sp(y_test, y_lofar))
        res_hht.append(compute_sp(y_test, y_hht))
        res_demon.append(compute_sp(y_test, y_demon))

        # Oráculo Teórico
        y_o = []
        for i, yt in enumerate(y_test):
            cand = [y_mel[i], y_lofar[i], y_hht[i], y_demon[i]]
            y_o.append(yt if yt in cand else y_mel[i])
        res_oracle.append(compute_sp(y_test, np.array(y_o)))

        # Em 5x2cv, apenas o fold complementar da mesma repetição (trn_fold = fold ^ 1) é disjunto do teste
        trn_fold = fold ^ 1
        trn_t = pred_data['targets'][trn_fold]
        trn_m = pred_data['mel'][trn_fold]
        trn_l = pred_data['lofar'][trn_fold]
        trn_h = pred_data['hht'][trn_fold]
        trn_d = pred_data['demon'][trn_fold]

        trn_fids = sorted(list(trn_t.keys()))
        assert len(set(fids_test).intersection(trn_fids)) == 0, f"Leakage detectado entre fold {fold} e {trn_fold}!"

        X_train, y_comp, y_tr_true, cands_tr = [], [], [], []
        for fid in trn_fids:
            if fid not in meta_dict:
                continue
            yt = trn_t[fid]
            X_train.append([meta_dict[fid][c] for c in feature_cols])
            y_tr_true.append(yt)
            cands_tr.append([trn_m[fid], trn_l[fid], trn_h[fid], trn_d[fid]])
            y_comp.append([
                int(trn_m[fid] == yt),
                int(trn_l[fid] == yt),
                int(trn_h[fid] == yt),
                int(trn_d[fid] == yt)
            ])

        X_train = np.array(X_train)
        y_comp = np.array(y_comp)
        y_tr_true = np.array(y_tr_true)
        cands_tr = np.array(cands_tr)

        # Prior empírico de SP calculado estritamente no fold de treino disjunto (trn_fold = fold ^ 1)
        sp_prior_tr = np.array([compute_sp(y_tr_true, cands_tr[:, m])[0] / 100.0 for m in range(4)])

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        # 4 Classificadores de Competência regularizados para generalização inter-gravação (639 amostras de treino)
        clfs = [
            RandomForestClassifier(
                n_estimators=50, max_depth=3, min_samples_leaf=5, random_state=42
            ).fit(X_train_s, y_comp[:, m])
            for m in range(4)
        ]

        # Inferência no conjunto de teste disjunto
        X_test = np.array([[meta_dict[fid][c] if fid in meta_dict else 0.0 for c in feature_cols] for fid in fids_test])
        X_test_s = scaler.transform(X_test)
        probs = np.column_stack([clfs[m].predict_proba(X_test_s)[:, 1] for m in range(4)])
        scores = 0.90 * probs + 0.10 * sp_prior_tr[None, :]
        best_routes = np.argmax(scores, axis=1)

        y_pred_gated = []
        for i, r in enumerate(best_routes):
            r_name = ['MEL', 'LOFAR', 'HHT', 'DEMON'][r]
            routes_counter[r_name] += 1
            if r == 0: y_pred_gated.append(y_mel[i])
            elif r == 1: y_pred_gated.append(y_lofar[i])
            elif r == 2: y_pred_gated.append(y_hht[i])
            else: y_pred_gated.append(y_demon[i])

        y_pred_gated = np.array(y_pred_gated)
        res_gated.append(compute_sp(y_test, y_pred_gated))

    def aggregate(res_list):
        sps = [r[0] for r in res_list]
        bal_accs = [r[1] for r in res_list]
        accs = [r[2] for r in res_list]
        recs = np.mean([r[3] for r in res_list], axis=0)
        return {
            'SP (%)': f"{np.mean(sps):.2f} ± {np.std(sps):.2f}",
            'Bal Acc (%)': f"{np.mean(bal_accs):.2f} ± {np.std(bal_accs):.2f}",
            'Acc (%)': f"{np.mean(accs):.2f} ± {np.std(accs):.2f}",
            'Recall SMALL (%)': f"{recs[0]:.1f}%",
            'Recall MEDIUM (%)': f"{recs[1]:.1f}%",
            'Recall LARGE (%)': f"{recs[2]:.1f}%",
            'Recall BG (%)': f"{recs[3]:.1f}%"
        }

    df_summary = pd.DataFrame([
        {'Configuração': '1. Baseline MEL Isolado', **aggregate(res_mel)},
        {'Configuração': '2. Baseline LOFAR Isolado', **aggregate(res_lofar)},
        {'Configuração': '3. Baseline HHT Isolado', **aggregate(res_hht)},
        {'Configuração': '4. Baseline DEMON Isolado', **aggregate(res_demon)},
        {'Configuração': f'5. GATED {model_name} (RF Competence)', **aggregate(res_gated)},
        {'Configuração': f'6. Teto do Oráculo Teórico ({model_name})', **aggregate(res_oracle)}
    ])

    return df_summary, routes_counter

def main():
    print("=" * 100)
    print("AVALIAÇÃO DO ROTEADOR POR COMPETÊNCIA (RF COMPETENCE ROUTER) EM MLP, RF E CNN")
    print("Protocolo Oficial: 10 Folds 5x2cv (Silva et al., 2025) com Treino Disjunto (trn_fold = fold ^ 1)")
    print("=" * 100)

    with open('data/iara_processed/cache_gating_metafeatures.pkl', 'rb') as f:
        meta_dict = pickle.load(f)
    feature_cols = ['crest_factor', 'spectral_flatness', 'mod_index', 'flux_std', 'high_low_ratio', 'rms_db', 'zcr']

    models = [
        ('MLP', 'mlp_all_representations_predictions.pkl'),
        ('RANDOM FOREST', 'rf_all_representations_predictions.pkl'),
        ('CNN 2D', 'cnn_all_representations_predictions.pkl')
    ]

    all_results = {}

    for model_name, pkl_file in models:
        print(f"\n>>> EXECUTANDO GATING PARA O MODELO: {model_name}...")
        t0 = time.time()
        df_res, routes = evaluate_model_family(model_name, pkl_file, meta_dict, feature_cols)
        elapsed = time.time() - t0
        all_results[model_name] = {'df': df_res, 'routes': routes}
        print(f"Concluído em {elapsed:.1f}s!")
        print(f"\n--- TABELA DE RESULTADOS: {model_name} (10 FOLDS 5x2cv) ---")
        print(df_res.to_string(index=False))

        tot_r = sum(routes.values())
        print(f"\n--- Distribuição de Rotas no {model_name} ---")
        for r, c in routes.most_common():
            print(f"  * Rota {r:<6}: {c:4d} áudios ({c/tot_r*100:5.1f}%)")
        print("-" * 100)

    # Salvar resultados consolidados
    out_pkl = 'data/iara_processed/results_competence_gating_all_models.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResultados salvos com sucesso em '{out_pkl}'!")

if __name__ == '__main__':
    main()
