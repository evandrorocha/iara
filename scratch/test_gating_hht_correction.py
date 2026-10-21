"""
Teste da Correção do Gating Front-End:
Substitui o foco em DEMON por HHT para a detecção de embarcações dinâmicas / SMALL.
Avalia nos 10 Folds Oficiais 5x2cv com Isolamento de Navios.
"""
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import scipy.stats as scipy_stats

def compute_metrics(y_true, y_pred, num_classes=4):
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
        'rec_bg': float(recalls[3] * 100),
        'cm': cm
    }

def main():
    print("=" * 90)
    print("EXPERIMENTO: GATING FRONT-END COM CORREÇÃO DE HHT PARA SMALL")
    print("Protocolo Oficial: 10 Folds 5x2cv com Isolamento de Navios")
    print("=" * 90)

    with open('data/iara_processed/cache_gating_metafeatures.pkl', 'rb') as f:
        meta = pickle.load(f)
    with open('data/iara_processed/rf_all_representations_predictions.pkl', 'rb') as f:
        pred_data = pickle.load(f)

    feature_cols = ['crest_factor', 'spectral_flatness', 'mod_index', 'flux_std', 'high_low_ratio', 'rms_db', 'zcr']

    # Avaliação Fold a Fold
    res_mel = []
    res_hht_iso = []
    res_gating_anterior = [] # focado em DEMON
    res_gating_hht_fisico = [] # focado em HHT para Small
    res_gating_hht_aprendido = [] # supervisionado com HHT prioritário
    res_oracle = []

    routes_ant = Counter()
    routes_hht = Counter()
    routes_apr = Counter()

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

        res_mel.append(compute_metrics(y_test, y_mel))
        res_hht_iso.append(compute_metrics(y_test, y_hht))

        # Oráculo
        y_o = []
        for i, yt in enumerate(y_test):
            cand = [y_mel[i], y_lofar[i], y_hht[i], y_demon[i]]
            y_o.append(yt if yt in cand else y_mel[i])
        res_oracle.append(compute_metrics(y_test, np.array(y_o)))

        # 1. Gating Anterior (DEMON prioritário)
        y_ant = []
        for i, fid in enumerate(fids_test):
            ft = meta.get(fid, None)
            if ft is None:
                r = 'mel'
            else:
                if ft['high_low_ratio'] > 0.8 and ft['mod_index'] > 1.15:
                    r = 'demon'
                elif ft['crest_factor'] > 5.0 and ft['high_low_ratio'] < 0.5:
                    r = 'lofar'
                elif ft['flux_std'] > 0.35:
                    r = 'hht'
                else:
                    r = 'mel'
            routes_ant[r] += 1
            mapping = {'demon': y_demon[i], 'lofar': y_lofar[i], 'hht': y_hht[i], 'mel': y_mel[i]}
            y_ant.append(mapping[r])
        res_gating_anterior.append(compute_metrics(y_test, np.array(y_ant)))

        # 2. Gating Corrigido (HHT prioritário para lanchas Small e sinais dinâmicos)
        # Regra Física Atualizada:
        # Se há fluxo temporal acentuado (flux_std > 0.28) e energia de alta frequência (high_low_ratio > 0.8) -> ROTA HHT!
        # Se há raias fortes diesel (< 250 Hz) e baixa cavitação -> ROTA LOFAR!
        # Se há cavitação muito evidente e sinal não-transiente -> ROTA DEMON
        # Caso contrário -> ROTA MEL (estável para banda larga e Medium)
        y_corrigido = []
        for i, fid in enumerate(fids_test):
            ft = meta.get(fid, None)
            if ft is None:
                r = 'mel'
            else:
                # Prioridade 1: Raias harmônicas diesel estreitas (LOFAR)
                if ft['crest_factor'] > 5.0 and ft['high_low_ratio'] < 0.45:
                    r = 'lofar'
                # Prioridade 2: Transientes / Dinâmica de Lancha rápida (HHT)
                elif ft['flux_std'] > 0.28 and ft['high_low_ratio'] > 0.80:
                    r = 'hht'
                # Prioridade 3: Cavitação periódica sustentada (DEMON)
                elif ft['mod_index'] > 1.25 and ft['high_low_ratio'] > 1.10:
                    r = 'demon'
                # Padrão: Ruído contínuo banda larga / Medium (MEL)
                else:
                    r = 'mel'
            routes_hht[r] += 1
            mapping = {'demon': y_demon[i], 'lofar': y_lofar[i], 'hht': y_hht[i], 'mel': y_mel[i]}
            y_corrigido.append(mapping[r])
        res_gating_hht_fisico.append(compute_metrics(y_test, np.array(y_corrigido)))

        # 3. Gating Aprendido com Prioridade no HHT para Small
        X_tr = []
        y_tr = []
        for trn_fold in range(10):
            if trn_fold == fold:
                continue
            trn_t = pred_data['targets'][trn_fold]
            trn_m = pred_data['mel'][trn_fold]
            trn_l = pred_data['lofar'][trn_fold]
            trn_h = pred_data['hht'][trn_fold]
            trn_d = pred_data['demon'][trn_fold]
            for fid, yt in trn_t.items():
                if fid not in meta:
                    continue
                fvec = [meta[fid][c] for c in feature_cols]
                h_m = (trn_m[fid] == yt)
                h_l = (trn_l[fid] == yt)
                h_h = (trn_h[fid] == yt)
                h_d = (trn_d[fid] == yt)

                # Priorização: se Small e HHT acertou -> rota HHT (1)
                # se Large e LOFAR acertou -> rota LOFAR (2)
                # se Medium e MEL acertou -> rota MEL (0)
                # se DEMON acertou exclusivamente -> rota DEMON (3)
                if yt == 0 and h_h:
                    r_opt = 1 # hht
                elif yt == 2 and h_l:
                    r_opt = 2 # lofar
                elif yt == 1 and h_m:
                    r_opt = 0 # mel
                elif h_m:
                    r_opt = 0 # mel
                elif h_h:
                    r_opt = 1 # hht
                elif h_l:
                    r_opt = 2 # lofar
                elif h_d:
                    r_opt = 3 # demon
                else:
                    r_opt = 0 # mel

                X_tr.append(fvec)
                y_tr.append(r_opt)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        clf_router = RandomForestClassifier(n_estimators=40, max_depth=5, class_weight='balanced', random_state=42)
        clf_router.fit(X_tr_s, y_tr)

        X_te = [[meta[fid][c] if fid in meta else 0.0 for c in feature_cols] for fid in fids_test]
        X_te_s = scaler.transform(X_te)
        test_r = clf_router.predict(X_te_s)

        y_apr = []
        for i, r in enumerate(test_r):
            r_name = ['mel', 'hht', 'lofar', 'demon'][r]
            routes_apr[r_name] += 1
            mapping = {0: y_mel[i], 1: y_hht[i], 2: y_lofar[i], 3: y_demon[i]}
            y_apr.append(mapping[r])
        res_gating_hht_aprendido.append(compute_metrics(y_test, np.array(y_apr)))

    def aggregate(res_list):
        return {
            'SP (%)': f"{np.mean([r['sp'] for r in res_list]):.2f} ± {np.std([r['sp'] for r in res_list]):.2f}",
            'Bal Acc (%)': f"{np.mean([r['bal_acc'] for r in res_list]):.2f} ± {np.std([r['bal_acc'] for r in res_list]):.2f}",
            'Recall SMALL (%)': f"{np.mean([r['rec_small'] for r in res_list]):.1f}%",
            'Recall MEDIUM (%)': f"{np.mean([r['rec_medium'] for r in res_list]):.1f}%",
            'Recall LARGE (%)': f"{np.mean([r['rec_large'] for r in res_list]):.1f}%",
            'Recall BG (%)': f"{np.mean([r['rec_bg'] for r in res_list]):.1f}%",
        }

    summary = [
        {'Configuração de Gating': '1. Baseline MEL Isolado', **aggregate(res_mel)},
        {'Configuração de Gating': '2. Baseline HHT Isolado', **aggregate(res_hht_iso)},
        {'Configuração de Gating': '3. Gating Anterior (Foco DEMON)', **aggregate(res_gating_anterior)},
        {'Configuração de Gating': '4. Gating Físico Corrigido (Foco HHT)', **aggregate(res_gating_hht_fisico)},
        {'Configuração de Gating': '5. Gating Aprendido (Prioridade HHT)', **aggregate(res_gating_hht_aprendido)},
        {'Configuração de Gating': '6. Teto do Oráculo Teórico', **aggregate(res_oracle)},
    ]

    df_sum = pd.DataFrame(summary)
    print("\n" + "=" * 95)
    print("RESULTADOS DO GATING COM CORREÇÃO PARA HHT (10 FOLDS 5x2cv)")
    print("=" * 95)
    print(df_sum.to_string(index=False))

    tot_h = sum(routes_hht.values())
    print("\n--- DISTRIBUIÇÃO DAS ROTAS NO GATING FÍSICO CORRIGIDO (HHT) ---")
    for r, c in routes_hht.most_common():
        print(f"  * Rota {r.upper():<6}: {c:4d} áudios ({c/tot_h*100:5.1f}%)")

    tot_a = sum(routes_apr.values())
    print("\n--- DISTRIBUIÇÃO DAS ROTAS NO GATING APRENDIDO (HHT) ---")
    for r, c in routes_apr.most_common():
        print(f"  * Rota {r.upper():<6}: {c:4d} áudios ({c/tot_a*100:5.1f}%)")

if __name__ == '__main__':
    main()
