"""
Implementação e Validação do Sistema Gated-SVM (Front-End Gating + Roteamento de Entradas)
sob o protocolo oficial 5x2cv (10 Folds estratificados com isolamento estrito de navios).

Testa:
1. Gating Baseado em Regras Físicas Calibradas (Physics-Informed Gating)
2. Gating Aprendido / Supervisionado (Data-Driven Decision Gating)
Compara com:
- Baseline Isolado (MEL)
- Baselines Individuais (LOFAR, DEMON, HHT)
- Teto do Oráculo Teórico
"""
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import scipy.stats as scipy_stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

def compute_sp_metrics(y_true, y_pred, num_classes=4):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    recalls = cm.diagonal() / (cm.sum(axis=1) + 1e-12)
    recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100)
    acc = float((y_true == y_pred).mean() * 100)
    return {
        'sp': sp,
        'bal_acc': bal_acc,
        'acc': acc,
        'rec_small': float(recalls[0] * 100),
        'rec_medium': float(recalls[1] * 100),
        'rec_large': float(recalls[2] * 100),
        'rec_bg': float(recalls[3] * 100) if num_classes == 4 else 0.0,
        'cm': cm
    }

def main():
    print("=" * 80)
    print("IMPLEMENTAÇÃO DO GATING FRONT-END + SISTEMA GATED MULTIMODAL")
    print("Protocolo Oficial: 5x2cv (10 Folds com Isolamento de Navios)")
    print("=" * 80)

    # 1. Carregar meta-características acústicas físicas extraídas dos WAVs
    meta_path = 'data/iara_processed/cache_gating_metafeatures.pkl'
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Arquivo {meta_path} não encontrado.")
    with open(meta_path, 'rb') as f:
        meta_dict = pickle.load(f)
    print(f"Meta-características físicas carregadas: {len(meta_dict)} áudios.")

    # 2. Carregar predições das 4 representações
    # Usaremos os resultados consolidados das representações (RF e MLP possuem all_df completo nos 10 folds)
    with open('data/iara_processed/rf_all_representations_predictions.pkl', 'rb') as f:
        pred_data = pickle.load(f)

    # pred_data tem: 'mel', 'lofar', 'hht', 'demon', 'targets' por fold (0 a 9)
    print("Predições das 4 representações (MEL, LOFAR, HHT, DEMON) carregadas com sucesso.")

    feature_cols = ['crest_factor', 'spectral_flatness', 'mod_index', 'flux_std', 'high_low_ratio', 'rms_db', 'zcr']

    # Avaliações fold a fold
    results_baseline_mel = []
    results_baseline_lofar = []
    results_gating_physics = []
    results_gating_learned = []
    results_oracle = []

    route_counts_physics = Counter()
    route_counts_learned = Counter()

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

        # Baseline MEL & LOFAR
        m_mel = compute_sp_metrics(y_test, y_mel)
        m_lofar = compute_sp_metrics(y_test, y_lofar)
        results_baseline_mel.append(m_mel)
        results_baseline_lofar.append(m_lofar)

        # Oráculo Teórico (se pelo menos 1 acertar, acerta)
        y_oracle = []
        for i, yt in enumerate(y_test):
            candidates = [y_mel[i], y_lofar[i], y_hht[i], y_demon[i]]
            if yt in candidates:
                y_oracle.append(yt)
            else:
                y_oracle.append(y_mel[i])
        y_oracle = np.array(y_oracle)
        results_oracle.append(compute_sp_metrics(y_test, y_oracle))

        # -------------------------------------------------------------
        # Estratégia A: Gating Baseado em Física (Physics-Informed Gating)
        # -------------------------------------------------------------
        # Regras de roteamento acústico sem conhecer o rótulo:
        y_pred_physics = []
        for i, fid in enumerate(fids_test):
            feat = meta_dict.get(fid, None)
            if feat is None:
                route = 'mel'
            else:
                # 1. Indicador de cavitação / alta rotação de hélice (DEMON)
                # Lanchas Small têm alta energia em alta frequência (high_low_ratio > 0.8) e modulação
                if feat['high_low_ratio'] > 0.8 and feat['mod_index'] > 1.15:
                    route = 'demon'
                # 2. Indicador de raias harmônicas diesel estreitas (LOFAR)
                # Cargueiros Large e rebocadores têm alto crest factor em graves (< 250 Hz)
                elif feat['crest_factor'] > 5.0 and feat['high_low_ratio'] < 0.5:
                    route = 'lofar'
                # 3. Indicador de manobra / aceleração / não-estacionaridade (HHT)
                elif feat['flux_std'] > 0.35:
                    route = 'hht'
                # 4. Caso padrão: ruído contínuo de banda larga (MEL)
                else:
                    route = 'mel'

            route_counts_physics[route] += 1
            if route == 'demon':
                y_pred_physics.append(y_demon[i])
            elif route == 'lofar':
                y_pred_physics.append(y_lofar[i])
            elif route == 'hht':
                y_pred_physics.append(y_hht[i])
            else:
                y_pred_physics.append(y_mel[i])

        y_pred_physics = np.array(y_pred_physics)
        results_gating_physics.append(compute_sp_metrics(y_test, y_pred_physics))

        # -------------------------------------------------------------
        # Estratégia B: Gating Supervisionado Aprendido (Treinado no Fold)
        # -------------------------------------------------------------
        # Coletar dados de treino (todos os folds exceto o fold atual)
        X_train_meta = []
        y_train_route = []

        for trn_fold in range(10):
            if trn_fold == fold_idx:
                continue
            trn_targets = pred_data['targets'][trn_fold]
            trn_mel = pred_data['mel'][trn_fold]
            trn_lofar = pred_data['lofar'][trn_fold]
            trn_hht = pred_data['hht'][trn_fold]
            trn_demon = pred_data['demon'][trn_fold]

            for fid, yt in trn_targets.items():
                if fid not in meta_dict:
                    continue
                feat_vec = [meta_dict[fid][c] for c in feature_cols]
                
                # Definir a rota ótima para o treino:
                # Se DEMON acertou e era Small -> rota 0 (demon)
                # Se LOFAR acertou e era Large -> rota 1 (lofar)
                # Se HHT acertou e os outros erraram -> rota 2 (hht)
                # Se MEL acertou -> rota 3 (mel)
                h_mel = (trn_mel[fid] == yt)
                h_lofar = (trn_lofar[fid] == yt)
                h_hht = (trn_hht[fid] == yt)
                h_demon = (trn_demon[fid] == yt)

                if yt == 0 and h_demon:
                    route_opt = 0 # demon
                elif yt == 2 and h_lofar:
                    route_opt = 1 # lofar
                elif h_hht and not h_mel:
                    route_opt = 2 # hht
                elif h_mel:
                    route_opt = 3 # mel
                elif h_lofar:
                    route_opt = 1 # lofar
                elif h_demon:
                    route_opt = 0 # demon
                else:
                    route_opt = 3 # default mel

                X_train_meta.append(feat_vec)
                y_train_route.append(route_opt)

        X_train_meta = np.array(X_train_meta)
        y_train_route = np.array(y_train_route)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_meta)

        # Treinar um classificador de roteamento leve (Decision Tree rasa ou Random Forest)
        router_clf = RandomForestClassifier(n_estimators=30, max_depth=5, class_weight='balanced', random_state=42)
        router_clf.fit(X_train_scaled, y_train_route)

        # Inferência no conjunto de teste
        X_test_meta = []
        for fid in fids_test:
            if fid in meta_dict:
                X_test_meta.append([meta_dict[fid][c] for c in feature_cols])
            else:
                X_test_meta.append([0.0]*len(feature_cols))
        X_test_scaled = scaler.transform(X_test_meta)

        test_routes = router_clf.predict(X_test_scaled)
        y_pred_learned = []
        for i, r in enumerate(test_routes):
            route_name = ['demon', 'lofar', 'hht', 'mel'][r]
            route_counts_learned[route_name] += 1
            if r == 0:
                y_pred_learned.append(y_demon[i])
            elif r == 1:
                y_pred_learned.append(y_lofar[i])
            elif r == 2:
                y_pred_learned.append(y_hht[i])
            else:
                y_pred_learned.append(y_mel[i])

        y_pred_learned = np.array(y_pred_learned)
        results_gating_learned.append(compute_sp_metrics(y_test, y_pred_learned))

    # Resumo das Médias nos 10 Folds
    def aggregate(res_list):
        return {
            'SP (%)': f"{np.mean([r['sp'] for r in res_list]):.2f} ± {np.std([r['sp'] for r in res_list]):.2f}",
            'Bal Acc (%)': f"{np.mean([r['bal_acc'] for r in res_list]):.2f} ± {np.std([r['bal_acc'] for r in res_list]):.2f}",
            'Recall SMALL (%)': f"{np.mean([r['rec_small'] for r in res_list]):.1f}%",
            'Recall MEDIUM (%)': f"{np.mean([r['rec_medium'] for r in res_list]):.1f}%",
            'Recall LARGE (%)': f"{np.mean([r['rec_large'] for r in res_list]):.1f}%",
            'Recall BG (%)': f"{np.mean([r['rec_bg'] for r in res_list]):.1f}%",
        }

    summary_table = [
        {'Sistema / Estratégia': '1. Baseline Isolado (MEL)', **aggregate(results_baseline_mel)},
        {'Sistema / Estratégia': '2. Baseline Isolado (LOFAR)', **aggregate(results_baseline_lofar)},
        {'Sistema / Estratégia': '3. Gating Físico (Physics-Informed)', **aggregate(results_gating_physics)},
        {'Sistema / Estratégia': '4. Gating Aprendido (Data-Driven)', **aggregate(results_gating_learned)},
        {'Sistema / Estratégia': '5. Teto do Oráculo Teórico (100% Certo)', **aggregate(results_oracle)},
    ]

    df_summary = pd.DataFrame(summary_table)
    print("\n" + "=" * 90)
    print("QUADRO RESUMO — DESEMPENHO DO GATING FRONT-END (10 FOLDS 5x2cv)")
    print("=" * 90)
    print(df_summary.to_string(index=False))

    total_routes_p = sum(route_counts_physics.values())
    print("\n--- DISTRIBUIÇÃO DAS ROTAS ESCOLHIDAS PELO GATING FÍSICO ---")
    for r, c in route_counts_physics.most_common():
        print(f"  * Rota {r.upper():<6}: {c:4d} áudios ({c/total_routes_p*100:5.1f}%)")

    total_routes_l = sum(route_counts_learned.values())
    print("\n--- DISTRIBUIÇÃO DAS ROTAS ESCOLHIDAS PELO GATING APRENDIDO ---")
    for r, c in route_counts_learned.most_common():
        print(f"  * Rota {r.upper():<6}: {c:4d} áudios ({c/total_routes_l*100:5.1f}%)")

    out_pkl = 'data/iara_processed/gating_benchmark_results.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'df_summary': df_summary,
            'results_baseline_mel': results_baseline_mel,
            'results_gating_physics': results_gating_physics,
            'results_gating_learned': results_gating_learned,
            'results_oracle': results_oracle,
            'route_counts_physics': route_counts_physics,
            'route_counts_learned': route_counts_learned
        }, f)
    print(f"\nResultados salvos com sucesso em '{out_pkl}'.")

if __name__ == '__main__':
    main()
