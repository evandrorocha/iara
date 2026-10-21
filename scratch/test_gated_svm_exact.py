"""
Avaliação Rigorosa do Gating Front-End aplicado DIRETAMENTE ao SVM Nystroem
nos 10 Folds Oficiais 5x2cv com Isolamento de Navios.

Compara:
1. SVM MEL Isolado
2. SVM LOFAR Isolado
3. SVM DEMON Isolado
4. SVM HHT Isolado
5. Gating Front-End Físico (Physics-Informed)
6. Gating Front-End com Salvaguarda Suave (Soft Safeguard - protege MEDIUM)
7. Gating Front-End Aprendido (Data-Driven Router)
8. Oráculo Teórico do SVM (Teto Máximo de Complementaridade)
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import scipy.stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

def compute_metrics(y_true, y_pred, num_classes=4):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    recalls = cm.diagonal() / (cm.sum(axis=1) + 1e-12)
    recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100)
    acc = float((y_true == y_pred).mean() * 100)
    res = {
        'sp': sp,
        'bal_acc': bal_acc,
        'acc': acc,
        'rec_small': float(recalls[0] * 100),
        'rec_medium': float(recalls[1] * 100),
        'rec_large': float(recalls[2] * 100),
        'cm': cm
    }
    if num_classes == 4:
        res['rec_bg'] = float(recalls[3] * 100)
    return res

def main():
    print("=" * 85)
    print("AVALIAÇÃO OFICIAL DO GATING FRONT-END APLICADO AO SVM (NYSTRÖM RBF)")
    print("Protocolo Oficial: 5x2cv (10 Folds com Isolamento Estrito por Ship ID)")
    print("=" * 85)

    meta_path = 'data/iara_processed/cache_gating_metafeatures.pkl'
    svm_pred_path = 'data/iara_processed/svm_all_representations_predictions.pkl'

    if not os.path.exists(meta_path) or not os.path.exists(svm_pred_path):
        print(f"Aguardando arquivos: meta={os.path.exists(meta_path)}, svm={os.path.exists(svm_pred_path)}")
        return

    with open(meta_path, 'rb') as f:
        meta_dict = pickle.load(f)
    with open(svm_pred_path, 'rb') as f:
        pred_data = pickle.load(f)

    feature_cols = ['crest_factor', 'spectral_flatness', 'mod_index', 'flux_std', 'high_low_ratio', 'rms_db', 'zcr']

    res_mel, res_lofar, res_demon, res_hht = [], [], [], []
    res_physics, res_safeguard, res_learned, res_oracle = [], [], [], []

    routes_p = Counter()
    routes_s = Counter()
    routes_l = Counter()

    for fold_idx in range(10):
        tst_targets = pred_data['targets'][fold_idx]
        tst_mel = pred_data['mel'][fold_idx]
        tst_lofar = pred_data['lofar'][fold_idx]
        tst_hht = pred_data['hht'][fold_idx]
        tst_demon = pred_data['demon'][fold_idx]

        fids_test = sorted(list(tst_targets.keys()))
        y_test = np.array([tst_targets[fid] for fid in fids_test])

        y_mel = np.array([tst_mel[fid] for fid in fids_test])
        y_lofar = np.array([tst_lofar[fid] for fid in fids_test])
        y_hht = np.array([tst_hht[fid] for fid in fids_test])
        y_demon = np.array([tst_demon[fid] for fid in fids_test])

        res_mel.append(compute_metrics(y_test, y_mel))
        res_lofar.append(compute_metrics(y_test, y_lofar))
        res_demon.append(compute_metrics(y_test, y_demon))
        res_hht.append(compute_metrics(y_test, y_hht))

        # 1. Oráculo Teórico do SVM
        y_oracle = []
        for i, yt in enumerate(y_test):
            candidates = [y_mel[i], y_lofar[i], y_hht[i], y_demon[i]]
            y_oracle.append(yt if yt in candidates else y_mel[i])
        res_oracle.append(compute_metrics(y_test, np.array(y_oracle)))

        # 2. Gating Físico Padrão
        y_pred_p = []
        for i, fid in enumerate(fids_test):
            feat = meta_dict.get(fid, None)
            if feat is None:
                r = 'mel'
            else:
                if feat['high_low_ratio'] > 0.8 and feat['mod_index'] > 1.15:
                    r = 'demon'
                elif feat['crest_factor'] > 5.0 and feat['high_low_ratio'] < 0.5:
                    r = 'lofar'
                elif feat['flux_std'] > 0.35:
                    r = 'hht'
                else:
                    r = 'mel'
            routes_p[r] += 1
            mapping = {'demon': y_demon[i], 'lofar': y_lofar[i], 'hht': y_hht[i], 'mel': y_mel[i]}
            y_pred_p.append(mapping[r])
        res_physics.append(compute_metrics(y_test, np.array(y_pred_p)))

        # 3. Gating com Salvaguarda Suave (Soft Safeguard - protege MEDIUM)
        # Sinais com alta/baixa moderada (0.4 <= high_low_ratio <= 1.0) e modulação moderada
        # são mantidos em MEL, evitando falsos acionamentos de DEMON
        y_pred_s = []
        for i, fid in enumerate(fids_test):
            feat = meta_dict.get(fid, None)
            if feat is None:
                r = 'mel'
            else:
                # Disparo restrito de DEMON: apenas se alta frequência for claramente dominante (> 1.2)
                # ou modulação de cavitação for muito evidente (> 1.25)
                if feat['high_low_ratio'] > 1.2 and feat['mod_index'] > 1.20:
                    r = 'demon'
                # Disparo restrito de LOFAR: fortes raias harmônicas (< 250 Hz) e baixo ruído de cavitação
                elif feat['crest_factor'] > 5.2 and feat['high_low_ratio'] < 0.45:
                    r = 'lofar'
                # Disparo de HHT: alta variabilidade de fluxo
                elif feat['flux_std'] > 0.40:
                    r = 'hht'
                # Fallback seguro para MEL (excelente e balanceado para MEDIUM)
                else:
                    r = 'mel'
            routes_s[r] += 1
            mapping = {'demon': y_demon[i], 'lofar': y_lofar[i], 'hht': y_hht[i], 'mel': y_mel[i]}
            y_pred_s.append(mapping[r])
        res_safeguard.append(compute_metrics(y_test, np.array(y_pred_s)))

        # 4. Gating Aprendido Supervisionado (Treinado no Out-of-Fold)
        X_train_meta = []
        y_train_route = []
        for trn_fold in range(10):
            if trn_fold == fold_idx:
                continue
            trn_t = pred_data['targets'][trn_fold]
            trn_m = pred_data['mel'][trn_fold]
            trn_l = pred_data['lofar'][trn_fold]
            trn_h = pred_data['hht'][trn_fold]
            trn_d = pred_data['demon'][trn_fold]

            for fid, yt in trn_t.items():
                if fid not in meta_dict:
                    continue
                feat_vec = [meta_dict[fid][c] for c in feature_cols]
                h_m = (trn_m[fid] == yt)
                h_l = (trn_l[fid] == yt)
                h_h = (trn_h[fid] == yt)
                h_d = (trn_d[fid] == yt)

                # Priorização acústica inteligente:
                if yt == 0 and h_d:
                    route_opt = 0 # demon
                elif yt == 2 and h_l:
                    route_opt = 1 # lofar
                elif yt == 1 and h_m:
                    route_opt = 3 # mel protege medium
                elif h_m:
                    route_opt = 3 # mel
                elif h_l:
                    route_opt = 1 # lofar
                elif h_h and not h_m:
                    route_opt = 2 # hht
                elif h_d:
                    route_opt = 0 # demon
                else:
                    route_opt = 3 # fallback mel

                X_train_meta.append(feat_vec)
                y_train_route.append(route_opt)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_meta)
        router_clf = RandomForestClassifier(n_estimators=40, max_depth=5, class_weight='balanced', random_state=42)
        router_clf.fit(X_train_scaled, y_train_route)

        X_test_meta = [[meta_dict[fid][c] if fid in meta_dict else 0.0 for c in feature_cols] for fid in fids_test]
        X_test_scaled = scaler.transform(X_test_meta)
        test_routes = router_clf.predict(X_test_scaled)

        y_pred_l = []
        for i, r in enumerate(test_routes):
            r_name = ['demon', 'lofar', 'hht', 'mel'][r]
            routes_l[r_name] += 1
            mapping = {0: y_demon[i], 1: y_lofar[i], 2: y_hht[i], 3: y_mel[i]}
            y_pred_l.append(mapping[r])
        res_learned.append(compute_metrics(y_test, np.array(y_pred_l)))

    def aggregate(res_list):
        return {
            'SP (%)': f"{np.mean([r['sp'] for r in res_list]):.2f} ± {np.std([r['sp'] for r in res_list]):.2f}",
            'Bal Acc (%)': f"{np.mean([r['bal_acc'] for r in res_list]):.2f} ± {np.std([r['bal_acc'] for r in res_list]):.2f}",
            'Recall SMALL (%)': f"{np.mean([r['rec_small'] for r in res_list]):.1f}%",
            'Recall MEDIUM (%)': f"{np.mean([r['rec_medium'] for r in res_list]):.1f}%",
            'Recall LARGE (%)': f"{np.mean([r['rec_large'] for r in res_list]):.1f}%",
            'Recall BG (%)': f"{np.mean([r['rec_bg'] for r in res_list]):.1f}%",
        }

    summary_rows = [
        {'Modelo / Estratégia': '1. SVM MEL Isolado', **aggregate(res_mel)},
        {'Modelo / Estratégia': '2. SVM LOFAR Isolado', **aggregate(res_lofar)},
        {'Modelo / Estratégia': '3. SVM DEMON Isolado', **aggregate(res_demon)},
        {'Modelo / Estratégia': '4. SVM HHT Isolado', **aggregate(res_hht)},
        {'Modelo / Estratégia': '5. Gating Front-End Físico', **aggregate(res_physics)},
        {'Modelo / Estratégia': '6. Gating com Salvaguarda Suave', **aggregate(res_safeguard)},
        {'Modelo / Estratégia': '7. Gating Front-End Aprendido', **aggregate(res_learned)},
        {'Modelo / Estratégia': '8. Oráculo Teórico do SVM', **aggregate(res_oracle)},
    ]

    df_res = pd.DataFrame(summary_rows)
    print("\n" + "=" * 95)
    print("QUADRO COMPARATIVO DO SISTEMA GATED-SVM (10 FOLDS 5x2cv)")
    print("=" * 95)
    print(df_res.to_string(index=False))

    out_file = 'data/iara_processed/svm_gating_benchmark_results.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump({
            'df_res': df_res,
            'res_mel': res_mel,
            'res_lofar': res_lofar,
            'res_demon': res_demon,
            'res_hht': res_hht,
            'res_physics': res_physics,
            'res_safeguard': res_safeguard,
            'res_learned': res_learned,
            'res_oracle': res_oracle,
            'routes_physics': routes_p,
            'routes_safeguard': routes_s,
            'routes_learned': routes_l
        }, f)
    print(f"\nResultados gravados em {out_file} com sucesso!")

if __name__ == '__main__':
    main()
