"""
Split por navio vs split por gravação para as redes neurais (complemento de eval_ship_split_5x2cv.py).

Mesmos folds de reports/split_navio/folds.csv (gerados por eval_ship_split_5x2cv.py):
  * split_id    -> StratifiedGroupKFold(2) com grupos = ID       (controle)
  * split_navio -> StratifiedGroupKFold(2) com grupos = Ship ID  (nenhum navio em treino e teste ao mesmo tempo)

Modelos (mesma arquitetura e hiperparâmetros dos scripts de origem, sem early stopping, treino = metade inteira):
  * MLP h32-16, 35 épocas, batch 64, Adam lr 1e-3 wd 1e-3, dropout 0,2 (eval_fusion_mel_demon_5x2cv.py)
    sobre MEL 5,12 s, LOFAR 5,12 s e HHT 5,12 s
  * BiGRU + atenção temporal, h128, 2 camadas, sequências 15/5, 15 épocas, AdamW 3e-4, cosseno
    (eval_bigru_lofar_multiclass_5x2cv.py) sobre LOFAR 5,12 s
Semente torch/numpy = índice do fold (as duas condições usam as mesmas sementes).

Saída: results/trainings/multirep/{split_id,split_navio}_{mlp,bigru}_{rep}_4c
Rodar na raiz do repositório, no container iara-dev (GPU):  python scratch/eval_ship_split_nn_5x2cv.py
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_writer import ResultsWriter  # noqa: E402
from eval_fusion_mel_demon_5x2cv import PyTorchFusionMLP  # noqa: E402
from eval_bigru_lofar_multiclass_5x2cv import BiGRUAttention, SequentialDataset, prepare_sequences  # noqa: E402
from eval_ship_split_5x2cv import CACHES, FOLDS_CSV, fold_files, matrices  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed(k):
    np.random.seed(k)
    torch.manual_seed(k)
    torch.cuda.manual_seed_all(k)


def run_mlp(cache, tr, te, k, num_classes=4, epochs=35, batch_size=64, lr=1e-3):
    seed(k)
    Xtr, ytr, _ = matrices(cache, tr)
    Xte, yte, fte = matrices(cache, te)
    sc = StandardScaler()
    Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
    counts = np.bincount(ytr, minlength=num_classes)
    w = len(ytr) / (num_classes * counts.astype(np.float32) + 1e-6)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    model = PyTorchFusionMLP(in_features=Xtr.shape[1], num_classes=num_classes, hidden=[32, 16], dropout=0.2).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:  # BatchNorm1d não aceita lote de 1
                continue
            opt.zero_grad()
            crit(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte, dtype=torch.float32, device=DEVICE))
    return fte, yte, logits.argmax(1).cpu().numpy(), torch.softmax(logits, 1).cpu().numpy()


def run_bigru(cache, tr, te, k, num_classes=4, epochs=15, batch_size=128, lr=3e-4, seq_len=15, stride=5):
    seed(k)
    trs, trt, trf, scaler = prepare_sequences(cache, tr, seq_len=seq_len, stride=stride, is_train=True)
    tes, tet, tef, _ = prepare_sequences(cache, te, seq_len=seq_len, stride=stride, scaler=scaler, is_train=False)
    trn_loader = DataLoader(SequentialDataset(trs, trt, trf), batch_size=batch_size, shuffle=True, drop_last=True)
    tst_loader = DataLoader(SequentialDataset(tes, tet, tef), batch_size=batch_size, shuffle=False)
    model = BiGRUAttention(input_dim=trs.shape[2], hidden_size=128, num_layers=2, num_classes=num_classes,
                           dropout=0.3).to(DEVICE)
    counts = np.bincount(trt, minlength=num_classes)
    w = 1.0 / (counts + 1e-6)
    w = w / w.sum() * num_classes
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for _ in range(epochs):
        for bx, by, _ in trn_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            crit(model(bx), by).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sch.step()
    model.eval()
    preds, scores = [], []
    with torch.no_grad():
        for bx, _, _ in tst_loader:
            logits = model(bx.to(DEVICE))
            preds.append(logits.argmax(1).cpu().numpy())
            scores.append(torch.softmax(logits, 1).cpu().numpy())
    return np.asarray(tef, dtype=np.int64), tet, np.concatenate(preds), np.concatenate(scores)


PARAMS = {
    "mlp": {"model": "PyTorchFusionMLP", "hidden": [32, 16], "dropout": 0.2, "epochs": 35, "batch_size": 64,
            "lr": 1e-3, "weight_decay": 1e-3, "early_stopping": False, "scaler": "StandardScaler(treino)",
            "origem": "scratch/eval_fusion_mel_demon_5x2cv.py"},
    "bigru": {"model": "BiGRUAttention (sem BatchNorm)", "hidden_size": 128, "num_layers": 2, "dropout": 0.3,
              "seq_len_frames": 15, "seq_stride_frames": 5, "epochs": 15, "batch_size": 128, "lr": 3e-4,
              "optimizer": "AdamW(wd=1e-3)", "scheduler": "CosineAnnealingLR", "early_stopping": False,
              "origem": "scratch/eval_bigru_lofar_multiclass_5x2cv.py"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="mlp:mel,mlp:lofar,mlp:hht,bigru:lofar")
    args = ap.parse_args()
    print(f"Dispositivo: {DEVICE}", flush=True)
    folds = pd.read_csv(FOLDS_CSV, dtype={"Ship ID": str})
    jobs = [j.split(":") for j in args.jobs.split(",")]

    for rep in dict.fromkeys(r for _, r in jobs):
        path, key = CACHES[rep]
        with open(path, "rb") as fh:
            cache = pickle.load(fh)
        cache = cache[key] if key else cache
        for model in [m for m, r in jobs if r == rep]:
            for cond in ("split_id", "split_navio"):
                name = f"{cond}_{model}_{rep}_4c"
                w = ResultsWriter(name, script=__file__, params={
                    **PARAMS[model], "representation": rep, "split": cond,
                    "grouping": "ID" if cond == "split_id" else "Ship ID (BACKGROUND: ID)",
                    "cv": "StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=rep), rep = 0..4",
                    "folds_csv": FOLDS_CSV, "seed": "torch/numpy = fold", "num_classes": 4})
                t0 = time.time()
                for k in range(10):
                    tr, te = fold_files(folds, cond, k, "trn"), fold_files(folds, cond, k, "test")
                    run = run_mlp if model == "mlp" else run_bigru
                    fte, yte, pred, scores = run(cache, tr, te, k)
                    w.save_fold(k, fte, yte, pred, train_files=tr, scores=scores)
                print(f"{name}: 10 folds em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
