"""
Gera e analisa as predições do modelo MLP (Multi-Layer Perceptron) nas 4 representações:
- MEL (do artigo Silva et al. 2025)
- LOFAR (do artigo Silva et al. 2025)
- HHT (PyTorch [32, 16])
- DEMON (PyTorch [32, 16])

Calcula a análise de consenso, teto do oráculo, resgates exclusivos e padrões de erro
para as 3 classes de embarcações:
- SMALL (Classe 0)
- MEDIUM (Classe 1)
- LARGE (Classe 2)
"""

import os
import time
import pickle
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from results_writer import ResultsWriter
import sklearn.metrics as sk_metrics
import scipy.stats as scipy_stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def mode(x):
    return Counter(x).most_common(1)[0][0]

def compute_sp_index(targets, preds, num_classes=4):
    cm = sk_metrics.confusion_matrix(targets, preds, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    acc = float((targets == preds).mean() * 100)
    return sp, acc, recalls * 100, cm

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

class PyTorchMLP(nn.Module):
    def __init__(self, in_features, num_classes=4, hidden=[32, 16], dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.BatchNorm1d(hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], num_classes)
        )

    def forward(self, x):
        return self.net(x)

def run_pytorch_mlp(feature_cache, folds, num_classes=4, epochs=30, batch_size=64, lr=1e-3, seed=42, writer=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando MLP PyTorch no dispositivo: {device}")

    preds_by_fold = {}
    for fold in folds:
        fold_idx = fold["fold_idx"]
        t0 = time.time()
        X_train, y_train, _ = build_fold_matrices(feature_cache, fold["train_files"])
        X_test, y_test, test_fids = build_fold_matrices(feature_cache, fold["test_files"])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        counts = np.bincount(y_train, minlength=num_classes)
        weights = len(y_train) / (num_classes * counts.astype(np.float32) + 1e-6)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

        trn_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)

        model = PyTorchMLP(in_features=X_train.shape[1], num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model.train()
        for epoch in range(epochs):
            for bx, by in trn_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            tst_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
            tst_logits = model(tst_tensor)
            win_preds = torch.argmax(tst_logits, dim=1).cpu().numpy()
            win_scores = torch.softmax(tst_logits, dim=1).cpu().numpy()
        if writer is not None:
            writer.save_fold(fold_idx, test_fids, y_test, win_preds,
                             train_files=fold["train_files"], scores=win_scores)

        file_preds = defaultdict(list)
        for fid, pred in zip(test_fids, win_preds):
            file_preds[fid].append(pred)

        preds_by_fold[fold_idx] = {fid: int(np.argmax(np.bincount(file_preds[fid], minlength=num_classes))) for fid in file_preds}
        print(f"  Fold {fold_idx} concluído em {time.time()-t0:.1f}s.")
    return preds_by_fold

def analyze_class(df_class, class_name, target_id):
    total = len(df_class)
    total_correct = (df_class["num_hits"] > 0).sum()

    print("\n" + "=" * 80)
    print(f"ANÁLISE MLP — CLASSE {class_name} ({total} ÁUDIOS DE TESTE NOS 10 FOLDS)")
    print("=" * 80)

    # 1. Desempenho Individual
    print("\n--- 1. DESEMPENHO INDIVIDUAL NO MLP ---")
    for name, col in [("MEL", "hit_mel"), ("LOFAR", "hit_lofar"), ("HHT", "hit_hht"), ("DEMON", "hit_demon")]:
        c = df_class[col].sum()
        pct = c / total * 100
        print(f"  Acertos MLP {name:<6}: {c:4d} / {total} ({pct:5.2f}%)")

    print(f"\n  >>> TETO DO ORÁCULO MLP : {total_correct:4d} / {total} ({total_correct/total*100:5.2f}%)")
    print(f"  >>> ERRO COMUM MLP      : {total - total_correct:4d} / {total} ({(total - total_correct)/total*100:5.2f}%)")

    # 2. Grau de Consenso
    print("\n--- 2. GRAU DE CONSENSO SOBRE O TOTAL ---")
    c4 = (df_class["num_hits"] == 4).sum()
    c3 = (df_class["num_hits"] == 3).sum()
    c2 = (df_class["num_hits"] == 2).sum()
    c1 = (df_class["num_hits"] == 1).sum()
    c0 = (df_class["num_hits"] == 0).sum()

    print(f"  * 4 Entradas (Unanimidade): {c4:4d} ({c4/total*100:5.2f}%)")
    print(f"  * 3 Entradas (Alto)       : {c3:4d} ({c3/total*100:5.2f}%)")
    print(f"  * 2 Entradas (Dividido)   : {c2:4d} ({c2/total*100:5.2f}%)")
    print(f"  * 1 Entrada  (Exclusivo)  : {c1:4d} ({c1/total*100:5.2f}%)")
    print(f"  * 0 Entradas (Erro Comum) : {c0:4d} ({c0/total*100:5.2f}%)")

    # 3. Resgates Exclusivos
    print("\n--- 3. RESGATES EXCLUSIVOS (Apenas 1 entrada salva o áudio) ---")
    only_mel = ((df_class['hit_mel']) & (~df_class['hit_lofar']) & (~df_class['hit_hht']) & (~df_class['hit_demon'])).sum()
    only_lofar = ((~df_class['hit_mel']) & (df_class['hit_lofar']) & (~df_class['hit_hht']) & (~df_class['hit_demon'])).sum()
    only_hht = ((~df_class['hit_mel']) & (~df_class['hit_lofar']) & (df_class['hit_hht']) & (~df_class['hit_demon'])).sum()
    only_demon = ((~df_class['hit_mel']) & (~df_class['hit_lofar']) & (~df_class['hit_hht']) & (df_class['hit_demon'])).sum()

    print(f"  Salvos EXCLUSIVAMENTE pelo MEL  : {only_mel:4d} ({only_mel/total*100:5.2f}%)")
    print(f"  Salvos EXCLUSIVAMENTE pelo LOFAR: {only_lofar:4d} ({only_lofar/total*100:5.2f}%)")
    print(f"  Salvos EXCLUSIVAMENTE pelo HHT  : {only_hht:4d} ({only_hht/total*100:5.2f}%)")
    print(f"  Salvos EXCLUSIVAMENTE pelo DEMON: {only_demon:4d} ({only_demon/total*100:5.2f}%)")

    # 4. Padrão de Erro
    print(f"\n--- 4. PADRÃO DE CONFUSÃO EM {class_name} ---")
    class_map = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "AMBIENTAL"}
    for mod_name, col in [("MEL", "pred_mel"), ("LOFAR", "pred_lofar"), ("HHT", "pred_hht"), ("DEMON", "pred_demon")]:
        errs = df_class[df_class[col] != target_id][col].map(class_map).value_counts()
        total_errs = (df_class[col] != target_id).sum()
        print(f"\n  Padrão de Erro no {mod_name} (Total Erros: {total_errs}):")
        for cls, cnt in errs.items():
            print(f"    -> Confundido com {cls:<10}: {cnt:4d} ({cnt/total_errs*100:5.2f}%)")

    # 5. Votação por Consenso
    print(f"\n--- 5. VOTAÇÃO POR MAIORIA / CONSENSO EM {class_name} ---")
    print(f"  * Unanimidade (4 de 4)    : {c4:4d} ({c4/total*100:5.2f}% de recall)")
    print(f"  * Maioria Estrita (>= 3/4): {c4+c3:4d} ({(c4+c3)/total*100:5.2f}% de recall)")
    print(f"  * Maioria Simples (>= 2/4): {c4+c3+c2:4d} ({(c4+c3+c2)/total*100:5.2f}% de recall)")
    print(f"  * Oráculo (>= 1/4)        : {total_correct:4d} ({total_correct/total*100:5.2f}% de recall)")

    return {
        "total": total,
        "total_correct": total_correct,
        "c4": c4, "c3": c3, "c2": c2, "c1": c1, "c0": c0,
        "only_mel": only_mel, "only_lofar": only_lofar, "only_hht": only_hht, "only_demon": only_demon,
    }

def main():
    cache_file = "data/iara_processed/mlp_all_representations_predictions.pkl"
    folds = get_official_folds()

    if os.path.exists(cache_file):
        print(f"Carregando predições MLP do cache '{cache_file}'...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        mel_preds = data["mel"]
        lofar_preds = data["lofar"]
        hht_preds = data["hht"]
        demon_preds = data["demon"]
        targets_by_fold = data["targets"]
    else:
        # 1. Carregar MLP MEL
        print("Carregando predições de MLP MEL...")
        p_mel = "results/trainings/tests/mlp/eval"
        mel_preds, targets_by_fold = {}, {}
        for i in range(10):
            df = pd.read_csv(f"{p_mel}/fold_{i}/mlp_multiclass_test.csv").groupby("File").agg({"Target": mode, "Prediction": mode}).reset_index()
            mel_preds[i] = dict(zip(df["File"], df["Prediction"]))
            targets_by_fold[i] = dict(zip(df["File"], df["Target"]))

        # 2. Carregar MLP LOFAR
        print("Carregando predições de MLP LOFAR...")
        p_lofar = "results/trainings/tests/mlp_lofar/eval"
        lofar_preds = {}
        for i in range(10):
            df = pd.read_csv(f"{p_lofar}/fold_{i}/mlp_lofar_multiclass_test.csv").groupby("File").agg({"Target": mode, "Prediction": mode}).reset_index()
            lofar_preds[i] = dict(zip(df["File"], df["Prediction"]))

        # 3. Rodar MLP HHT
        print("Treinando e avaliando MLP HHT (PyTorch [32, 16])...")
        with open("data/iara_processed/cache_hht.pkl", "rb") as f:
            cache_hht = pickle.load(f)
        hht_preds = run_pytorch_mlp(cache_hht, folds, writer=ResultsWriter(
            "mlp_hht_h32-16_e30_4c", script=__file__,
            params={"model": "PyTorchMLP", "representation": "HHT", "dim": 256, "hidden": [32, 16],
                    "dropout": 0.2, "epochs": 30, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-3,
                    "seed": 42, "early_stopping": False, "scaler": "StandardScaler(treino)", "num_classes": 4}))

        # 4. Rodar MLP DEMON
        print("Treinando e avaliando MLP DEMON (PyTorch [32, 16])...")
        with open("data/iara_processed/cache_fusion_mel_demon.pkl", "rb") as f:
            cache_fusion = pickle.load(f)
        cache_demon = cache_fusion["demon"]
        demon_preds = run_pytorch_mlp(cache_demon, folds, writer=ResultsWriter(
            "mlp_demon_h32-16_e30_4c", script=__file__,
            params={"model": "PyTorchMLP", "representation": "DEMON", "dim": 253, "hidden": [32, 16],
                    "dropout": 0.2, "epochs": 30, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-3,
                    "seed": 42, "early_stopping": False, "scaler": "StandardScaler(treino)", "num_classes": 4}))

        # Salvar cache
        with open(cache_file, "wb") as f:
            pickle.dump({
                "mel": mel_preds,
                "lofar": lofar_preds,
                "hht": hht_preds,
                "demon": demon_preds,
                "targets": targets_by_fold
            }, f)
        print(f"Cache de predições MLP salvo em '{cache_file}'.")

    # Montar dataframe com todos os 6.390 testes
    records = []
    for fold_idx in range(10):
        t_dict = targets_by_fold[fold_idx]
        for fid, tgt in t_dict.items():
            pm = mel_preds[fold_idx].get(fid, -1)
            pl = lofar_preds[fold_idx].get(fid, -1)
            ph = hht_preds[fold_idx].get(fid, -1)
            pd_ = demon_preds[fold_idx].get(fid, -1)

            hm = (pm == tgt)
            hl = (pl == tgt)
            hh = (ph == tgt)
            hd = (pd_ == tgt)

            records.append({
                "fold": fold_idx,
                "file_id": fid,
                "target": tgt,
                "pred_mel": pm,
                "pred_lofar": pl,
                "pred_hht": ph,
                "pred_demon": pd_,
                "hit_mel": hm,
                "hit_lofar": hl,
                "hit_hht": hh,
                "hit_demon": hd,
                "num_hits": hm + hl + hh + hd
            })

    df_all = pd.DataFrame(records)

    # 1. Analisar SMALL (Target 0)
    df_sml = df_all[df_all["target"] == 0].copy()
    res_sml = analyze_class(df_sml, "SMALL", 0)

    # 2. Analisar MEDIUM (Target 1)
    df_med = df_all[df_all["target"] == 1].copy()
    res_med = analyze_class(df_med, "MEDIUM", 1)

    # 3. Analisar LARGE (Target 2)
    df_lrg = df_all[df_all["target"] == 2].copy()
    res_lrg = analyze_class(df_lrg, "LARGE", 2)

    # Salvar resultados compilados para a geração dos notebooks e gráficos
    out_results = "data/iara_processed/mlp_complementarity_results.pkl"
    with open(out_results, "wb") as f:
        pickle.dump({
            "df_all": df_all,
            "res_sml": res_sml,
            "res_med": res_med,
            "res_lrg": res_lrg,
        }, f)
    print(f"\nResultados consolidados salvos com sucesso em '{out_results}'.")

if __name__ == "__main__":
    main()
