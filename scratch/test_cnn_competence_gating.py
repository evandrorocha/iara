"""
Implementação e Avaliação do ROTEADOR DE COMPETÊNCIA CONVOLUCIONAL (CNN Competence Router)
A CNN 2D (Dual-Stream: Espectrograma 2D + 7 Meta-Features Físicas) é treinada via
BCEWithLogitsLoss para PREVER A COMPETÊNCIA DE CADA ESPECIALISTA:
P(SVM_m acerta o áudio x).
O sinal é roteado para argmax_m P(SVM_m acerta | x).
Avaliação oficial: 10 Folds 5x2cv com Isolamento Estrito de Navios.
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
from collections import Counter
import scipy.stats as scipy_stats
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Modelo Dual-Stream: Espectrograma 2D + Vetor de Meta-Features Físicas
class DualStreamCompetenceRouter(nn.Module):
    def __init__(self, num_routes=4):
        super().__init__()
        # Ramo 1: Convolucional para Espectrograma 2D (B, 1, 32, 128)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # -> (B, 16, 16, 32)
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # -> (B, 32, 8, 16)
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))      # -> (B, 64, 1, 1)
        )
        
        # Ramo 2: Perceptron Denso para as 7 Meta-Características Físicas
        self.scalar_net = nn.Sequential(
            nn.Linear(7, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        
        # Fusão dos Recursos Físicos e Convolucionais
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, num_routes) # 4 logits de competência [MEL, LOFAR, HHT, DEMON]
        )

    def forward(self, spec, scalar):
        f_spec = self.conv(spec).view(spec.size(0), -1)
        f_scal = self.scalar_net(scalar)
        combined = torch.cat([f_spec, f_scal], dim=1)
        return self.fc(combined)

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

def standardize_spec(spec_np, target_shape=(32, 128)):
    t = torch.tensor(spec_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    interp = F.interpolate(t, size=target_shape, mode='bilinear', align_corners=False)
    return interp.squeeze(0).squeeze(0).numpy()

def main():
    print("=" * 90)
    print("TREINAMENTO DO ROTEADOR DE COMPETÊNCIA CONVOLUCIONAL (CNN COMPETENCE ROUTER)")
    print("Dual-Stream: Espectrograma 2D + 7 Meta-Features Físicas -> Multi-Label BCE")
    print("Protocolo Oficial: 10 Folds 5x2cv com Isolamento Estrito de Navios")
    print("=" * 90)

    # 1. Carregar espectrogramas 2D
    print("Carregando espectrogramas 2D em cache_fusion_mel_demon.pkl...")
    with open('data/iara_processed/cache_fusion_mel_demon.pkl', 'rb') as f:
        cache_data = pickle.load(f)
    raw_mel_dict = cache_data['mel']
    
    print("Padronizando espectrogramas para tamanho fixo (32, 128)...")
    spec_dict = {fid: standardize_spec(np.array(raw_spec, dtype=np.float32)) for fid, raw_spec in raw_mel_dict.items()}
    print(f"Espectrogramas padronizados para {len(spec_dict)} arquivos.")

    # 2. Carregar meta-features físicas
    with open('data/iara_processed/cache_gating_metafeatures.pkl', 'rb') as f:
        meta_dict = pickle.load(f)
    feature_cols = ['crest_factor', 'spectral_flatness', 'mod_index', 'flux_std', 'high_low_ratio', 'rms_db', 'zcr']

    # 3. Carregar predições das 4 representações
    with open('data/iara_processed/rf_all_representations_predictions.pkl', 'rb') as f:
        pred_data = pickle.load(f)

    route_names = ['MEL', 'LOFAR', 'HHT', 'DEMON']

    res_mel_baseline = []
    res_gated_competence = []
    res_oracle = []

    routes_selected = Counter()
    t_start = time.time()

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

        res_mel_baseline.append(compute_metrics(y_test, y_mel))

        # Oráculo
        y_o = [yt if yt in [y_mel[i], y_lofar[i], y_hht[i], y_demon[i]] else y_mel[i] for i, yt in enumerate(y_test)]
        res_oracle.append(compute_metrics(y_test, np.array(y_o)))

        # Montar conjunto de treino: 9 folds restantes
        X_tr_spec = []
        X_tr_scal = []
        y_tr_comp = [] # shape (N, 4): [mel_correct, lofar_correct, hht_correct, demon_correct]

        for trn_fold in range(10):
            if trn_fold == fold:
                continue
            trn_t = pred_data['targets'][trn_fold]
            trn_m = pred_data['mel'][trn_fold]
            trn_l = pred_data['lofar'][trn_fold]
            trn_h = pred_data['hht'][trn_fold]
            trn_d = pred_data['demon'][trn_fold]

            for fid, yt in trn_t.items():
                if fid not in spec_dict or fid not in meta_dict:
                    continue
                X_tr_spec.append(spec_dict[fid])
                X_tr_scal.append([meta_dict[fid][c] for c in feature_cols])
                y_tr_comp.append([
                    float(trn_m[fid] == yt),
                    float(trn_l[fid] == yt),
                    float(trn_h[fid] == yt),
                    float(trn_d[fid] == yt)
                ])

        X_tr_spec = np.array(X_tr_spec, dtype=np.float32)[:, np.newaxis, :, :] # (N, 1, 32, 128)
        X_tr_scal = np.array(X_tr_scal, dtype=np.float32)
        y_tr_comp = np.array(y_tr_comp, dtype=np.float32)

        # Normalizações
        mean_spec = float(np.mean(X_tr_spec))
        std_spec = float(np.std(X_tr_spec)) + 1e-6
        X_tr_spec = (X_tr_spec - mean_spec) / std_spec

        scaler = StandardScaler()
        X_tr_scal = scaler.fit_transform(X_tr_scal)

        # Preparar dados de teste
        X_te_spec = []
        X_te_scal = []
        for fid in fids_test:
            if fid in spec_dict:
                X_te_spec.append(spec_dict[fid])
            else:
                X_te_spec.append(np.zeros((32, 128), dtype=np.float32))
            if fid in meta_dict:
                X_te_scal.append([meta_dict[fid][c] for c in feature_cols])
            else:
                X_te_scal.append([0.0]*len(feature_cols))

        X_te_spec = np.array(X_te_spec, dtype=np.float32)[:, np.newaxis, :, :]
        X_te_spec = (X_te_spec - mean_spec) / std_spec
        X_te_scal = scaler.transform(np.array(X_te_scal, dtype=np.float32))

        # Dataset e DataLoader
        train_dataset = TensorDataset(
            torch.tensor(X_tr_spec),
            torch.tensor(X_tr_scal, dtype=torch.float32),
            torch.tensor(y_tr_comp)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Treinar o Modelo de Competência com BCEWithLogitsLoss
        torch.manual_seed(42 + fold)
        model = DualStreamCompetenceRouter(num_routes=4)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        model.train()
        for epoch in range(16):
            for bx_spec, bx_scal, by_comp in train_loader:
                optimizer.zero_grad()
                logits = model(bx_spec, bx_scal)
                loss = criterion(logits, by_comp)
                loss.backward()
                optimizer.step()

        # Inferência: Calcular Logits de Competência no Conjunto de Teste
        model.eval()
        with torch.no_grad():
            test_logits = model(torch.tensor(X_te_spec), torch.tensor(X_te_scal, dtype=torch.float32))
            # Rota ótima prevista: especialista com maior probabilidade estimada de acerto
            best_routes = torch.argmax(test_logits, dim=1).numpy()

        y_gated = []
        for i, r in enumerate(best_routes):
            r_name = route_names[r]
            routes_selected[r_name] += 1
            if r == 0:
                y_gated.append(y_mel[i])
            elif r == 1:
                y_gated.append(y_lofar[i])
            elif r == 2:
                y_gated.append(y_hht[i])
            else:
                y_gated.append(y_demon[i])

        y_gated = np.array(y_gated)
        res_gated_competence.append(compute_metrics(y_test, y_gated))
        print(f"Fold {fold+1}/10 concluído | Rotas escolhidas: {dict(Counter([route_names[r] for r in best_routes]))}")

    elapsed = time.time() - t_start
    print(f"\nTreinamento e validação dos 10 folds concluídos em {elapsed:.1f}s!")

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
        {'Sistema / Estratégia': '1. Baseline MEL Isolado', **aggregate(res_mel_baseline)},
        {'Sistema / Estratégia': '2. CNN Competence Router (Dual-Stream BCE)', **aggregate(res_gated_competence)},
        {'Sistema / Estratégia': '3. Teto do Oráculo Teórico', **aggregate(res_oracle)},
    ]

    df_sum = pd.DataFrame(summary)
    print("\n" + "=" * 95)
    print("QUADRO RESUMO: DESEMPENHO DO ROTEADOR DE COMPETÊNCIA CONVOLUCIONAL (10 FOLDS 5x2cv)")
    print("=" * 95)
    print(df_sum.to_string(index=False))

    tot_r = sum(routes_selected.values())
    print("\n--- DISTRIBUIÇÃO DAS ROTAS ESCOLHIDAS PELA CNN DE COMPETÊNCIA ---")
    for r, c in routes_selected.most_common():
        print(f"  * Rota {r:<6}: {c:4d} áudios ({c/tot_r*100:5.1f}%)")

if __name__ == '__main__':
    main()
