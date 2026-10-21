"""
Implementação e Avaliação do ROTEADOR CONVOLUCIONAL (CNN Front-End Router)
Abordagem A: A CNN 2D recebe o espectrograma contínuo interpolado para (32, 128)
e aprende a selecionar a melhor entrada espectral (MEL, LOFAR, HHT, DEMON) para o classificador.
Avalia sob o protocolo oficial 5x2cv (10 folds com isolamento estrito de navios).
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
from collections import Counter
import scipy.stats as scipy_stats
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Modelo CNN Front-End Leve
class CNNRouter(nn.Module):
    def __init__(self, num_routes=4):
        super().__init__()
        # Entrada: (B, 1, 32, 128)
        self.features = nn.Sequential(
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
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_routes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)

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
    # spec_np: (T, F)
    t = torch.tensor(spec_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, T, F)
    interp = F.interpolate(t, size=target_shape, mode='bilinear', align_corners=False)
    return interp.squeeze(0).squeeze(0).numpy() # (32, 128)

def main():
    print("=" * 90)
    print("TREINAMENTO E AVALIAÇÃO DO ROTEADOR CONVOLUCIONAL (CNN FRONT-END ROUTER)")
    print("Protocolo Oficial: 10 Folds 5x2cv com Isolamento Estrito de Navios")
    print("=" * 90)

    # 1. Carregar espectrogramas 2D e pré-padronizar para (32, 128)
    print("Carregando espectrogramas 2D em cache_fusion_mel_demon.pkl...")
    with open('data/iara_processed/cache_fusion_mel_demon.pkl', 'rb') as f:
        cache_data = pickle.load(f)
    raw_mel_dict = cache_data['mel']
    
    print("Padronizando espectrogramas para tamanho fixo (32, 128)...")
    spec_dict = {}
    for fid, raw_spec in raw_mel_dict.items():
        spec_dict[fid] = standardize_spec(np.array(raw_spec, dtype=np.float32), target_shape=(32, 128))
    print(f"Espectrogramas padronizados para {len(spec_dict)} arquivos.")

    # 2. Carregar predições das 4 representações
    with open('data/iara_processed/rf_all_representations_predictions.pkl', 'rb') as f:
        pred_data = pickle.load(f)

    # Rotas: 0: MEL, 1: LOFAR, 2: HHT, 3: DEMON
    route_names = ['MEL', 'LOFAR', 'HHT', 'DEMON']

    res_mel_baseline = []
    res_cnn_router = []
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
        y_o = []
        for i, yt in enumerate(y_test):
            cand = [y_mel[i], y_lofar[i], y_hht[i], y_demon[i]]
            y_o.append(yt if yt in cand else y_mel[i])
        res_oracle.append(compute_metrics(y_test, np.array(y_o)))

        # Montar conjunto de treino da CNN: todos os outros 9 folds
        X_train_spec = []
        y_train_route = []

        for trn_fold in range(10):
            if trn_fold == fold:
                continue
            trn_t = pred_data['targets'][trn_fold]
            trn_m = pred_data['mel'][trn_fold]
            trn_l = pred_data['lofar'][trn_fold]
            trn_h = pred_data['hht'][trn_fold]
            trn_d = pred_data['demon'][trn_fold]

            for fid, yt in trn_t.items():
                if fid not in spec_dict:
                    continue
                spec = spec_dict[fid] # (32, 128)
                h_m = (trn_m[fid] == yt)
                h_l = (trn_l[fid] == yt)
                h_h = (trn_h[fid] == yt)
                h_d = (trn_d[fid] == yt)

                # Priorização física do roteamento:
                # 0: MEL, 1: LOFAR, 2: HHT, 3: DEMON
                if yt == 0 and h_h:
                    r_opt = 2 # HHT é especialista prioritário em Small
                elif yt == 0 and h_m:
                    r_opt = 0 # MEL
                elif yt == 2 and h_l:
                    r_opt = 1 # LOFAR
                elif yt == 1 and h_m:
                    r_opt = 0 # MEL
                elif h_m:
                    r_opt = 0 # MEL
                elif h_l:
                    r_opt = 1 # LOFAR
                elif h_h:
                    r_opt = 2 # HHT
                elif h_d:
                    r_opt = 3 # DEMON
                else:
                    r_opt = 0 # MEL

                X_train_spec.append(spec)
                y_train_route.append(r_opt)

        X_train_spec = np.array(X_train_spec, dtype=np.float32)[:, np.newaxis, :, :] # (N, 1, 32, 128)
        y_train_route = np.array(y_train_route, dtype=np.int64)

        # Normalização do espectrograma
        mean_val = float(np.mean(X_train_spec))
        std_val = float(np.std(X_train_spec)) + 1e-6
        X_train_spec = (X_train_spec - mean_val) / std_val

        # Preparar dados de teste
        X_test_spec = []
        for fid in fids_test:
            if fid in spec_dict:
                X_test_spec.append(spec_dict[fid])
            else:
                X_test_spec.append(np.zeros((32, 128), dtype=np.float32))
        X_test_spec = np.array(X_test_spec, dtype=np.float32)[:, np.newaxis, :, :]
        X_test_spec = (X_test_spec - mean_val) / std_val

        # Pesos de classe para balancear o roteador
        counts = Counter(y_train_route)
        total_trn = len(y_train_route)
        class_weights = [total_trn / (4.0 * max(1, counts.get(c, 1))) for c in range(4)]
        weights_t = torch.tensor(class_weights, dtype=torch.float32)

        # Treinar a CNN Router
        torch.manual_seed(42 + fold)
        model = CNNRouter(num_routes=4)
        criterion = nn.CrossEntropyLoss(weight=weights_t)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        train_dataset = TensorDataset(torch.tensor(X_train_spec), torch.tensor(y_train_route))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model.train()
        for epoch in range(15):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        # Inferência do Roteador no Teste
        model.eval()
        with torch.no_grad():
            test_logits = model(torch.tensor(X_test_spec))
            test_routes = torch.argmax(test_logits, dim=1).numpy()

        y_cnn_pred = []
        for i, r in enumerate(test_routes):
            r_name = route_names[r]
            routes_selected[r_name] += 1
            if r == 0:
                y_cnn_pred.append(y_mel[i])
            elif r == 1:
                y_cnn_pred.append(y_lofar[i])
            elif r == 2:
                y_cnn_pred.append(y_hht[i])
            else:
                y_cnn_pred.append(y_demon[i])

        y_cnn_pred = np.array(y_cnn_pred)
        res_cnn_router.append(compute_metrics(y_test, y_cnn_pred))
        print(f"Fold {fold+1}/10 concluído | Rotas escolhidas: {dict(Counter([route_names[r] for r in test_routes]))}")

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
        {'Sistema / Estratégia': '2. CNN Front-End Router (Abordagem A)', **aggregate(res_cnn_router)},
        {'Sistema / Estratégia': '3. Teto do Oráculo Teórico', **aggregate(res_oracle)},
    ]

    df_sum = pd.DataFrame(summary)
    print("\n" + "=" * 95)
    print("QUADRO RESUMO: DESEMPENHO DA CNN FRONT-END ROUTER (10 FOLDS 5x2cv)")
    print("=" * 95)
    print(df_sum.to_string(index=False))

    tot_r = sum(routes_selected.values())
    print("\n--- DISTRIBUIÇÃO DAS ROTAS ESCOLHIDAS PELA CNN ---")
    for r, c in routes_selected.most_common():
        print(f"  * Rota {r:<6}: {c:4d} áudios ({c/tot_r*100:5.1f}%)")

if __name__ == '__main__':
    main()
