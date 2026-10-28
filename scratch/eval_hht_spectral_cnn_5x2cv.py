"""
CNN espectral 1D (iara.ml.models.spectral_cnn.SpectralCNN) no HHT (256 bins, 1-1000 Hz) ou no DEMON (`--rep demon`), 5x2cv oficial.

Segunda CNN do projeto, específica para HHT: a CNN 2D de eval_hht_models_5x2cv.py (imagens 32x256,
~2 imagens por gravação, sem validação) ficou instável (SP 37,5 ± 13,0). Aqui:
  - 1 amostra = 1 janela HHT de 5,12 s (opcionalmente com `--context k` janelas vizinhas da mesma
    gravação como canais), convolução ao longo da frequência; voto majoritário por gravação.
  - pré-processamento: log(x + 1e-3) e z-score por bin, estatísticas só do treino do fold.
  - early stopping na perda de uma validação interna (15% das gravações de TREINO, estratificada
    por classe, separada por gravação); o conjunto de teste do fold só é usado na predição final.
  - semente fixa por fold, AdamW, entropia cruzada ponderada por classe, aumento leve
    (máscara de frequência e deslocamento de ±1 bin).

Protocolo: mesmos folds 5x2cv do pipeline (particionamento por gravação `ID`; o mesmo navio pode
estar em treino e teste). Predições por janela gravadas em results/trainings/multirep/<exp>/.

Uso (no container):
    python scratch/eval_hht_spectral_cnn_5x2cv.py --num_classes 0 --context 0
    python scratch/eval_hht_spectral_cnn_5x2cv.py --num_classes 0 --context 2
"""
import argparse
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import iara.ml.models.spectral_cnn as iara_scnn  # noqa: E402

from results_writer import ResultsWriter  # noqa: E402
from eval_hht_models_5x2cv import compute_sp_index, get_official_folds, load_all_hht_features  # noqa: E402
from eval_demon_baselines_5x2cv import load_all_demon_features  # noqa: E402

# Representação -> (carregador, rótulo, eps do log). O eps fica ~1/10 da mediana dos valores em cada
# representação (HHT: mediana 0,012; DEMON: mediana 0,0018), fixado antes de treinar.
REPS = {
    "hht": (load_all_hht_features, "HHT", 1e-3),
    "demon": (load_all_demon_features, "DEMON", 1e-4),
}
VAL_FRACTION = 0.15


def split_train_val(train_files, seed):
    """Separa VAL_FRACTION das gravações de treino (estratificado por classe) para early stopping."""
    rng = np.random.RandomState(seed)
    trn, val = {}, {}
    by_class = {}
    for fid, t in sorted(train_files.items()):
        by_class.setdefault(t, []).append(fid)
    for t, fids in by_class.items():
        fids = np.array(fids)
        rng.shuffle(fids)
        n_val = max(1, int(round(VAL_FRACTION * len(fids))))
        val.update({int(f): t for f in fids[:n_val]})
        trn.update({int(f): t for f in fids[n_val:]})
    return trn, val


def build_windows(cache, files, context):
    """Empilha as janelas das gravações e o índice das vizinhas (mesma gravação, borda replicada).

    Retorna X (N, 256), y (N,), fids (N,), neigh (N, 2*context+1) com índices em X.
    """
    X, y, fids, neigh = [], [], [], []
    offset = 0
    offsets = np.arange(-context, context + 1)
    for fid, t in files.items():
        if fid not in cache:
            continue
        w = cache[fid]
        n = len(w)
        X.append(w)
        y.append(np.full(n, t, dtype=np.int64))
        fids.append(np.full(n, fid, dtype=np.int64))
        idx = np.clip(np.arange(n)[:, None] + offsets[None, :], 0, n - 1)
        neigh.append(idx + offset)
        offset += n
    return np.vstack(X).astype(np.float32), np.concatenate(y), np.concatenate(fids), np.vstack(neigh)


def augment(xb, shift_p=0.5, mask_p=0.5, max_mask=16):
    """xb: (B, C, F). Deslocamento de ±1 bin e máscara de uma faixa de frequência (zero após z-score)."""
    B, _, F = xb.shape
    dev = xb.device
    shifts = torch.randint(-1, 2, (B,), device=dev) * (torch.rand(B, device=dev) < shift_p)
    if shifts.any():
        idx = (torch.arange(F, device=dev)[None, :] - shifts[:, None]).clamp(0, F - 1)
        xb = torch.gather(xb, 2, idx[:, None, :].expand_as(xb))
    width = torch.randint(1, max_mask + 1, (B,), device=dev)
    start = (torch.rand(B, device=dev) * (F - width)).long()
    pos = torch.arange(F, device=dev)[None, :]
    mask = (pos >= start[:, None]) & (pos < (start + width)[:, None]) & (torch.rand(B, 1, device=dev) < mask_p)
    return xb.masked_fill(mask[:, None, :], 0.0)


def predict(model, X, neigh, batch_size=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(neigh), batch_size):
            out.append(model(X[neigh[i:i + batch_size]]))
    return torch.cat(out)


def file_vote(fids, preds):
    votes = {}
    for f, p in zip(fids, preds):
        votes.setdefault(int(f), []).append(int(p))
    return {f: Counter(v).most_common(1)[0][0] for f, v in votes.items()}


def run(cache, folds, num_classes, context, args):
    _, rep_label, LOG_EPS = REPS[args.rep]
    exclude = 3 if num_classes == 3 else None
    exp = f"speccnn_{args.rep}_ctx{context}_e{args.epochs}p{args.patience}_{num_classes}c"
    params = {"model": "iara.ml.models.spectral_cnn.SpectralCNN", "representation": rep_label, "n_bins": int(next(iter(cache.values())).shape[1]),
              "context": context, "conv_channels": [32, 64, 64, 128], "kernel_size": 7, "dilations": [1, 1, 2, 4],
              "pooled_bins": 8, "classification_n_neurons": 64, "dropout": [0.1, 0.4],
              "preprocessing": f"log(x+{LOG_EPS}) + z-score por bin (treino)", "optimizer": "AdamW",
              "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
              "max_epochs": args.epochs, "patience": args.patience,
              "early_stopping": f"perda ponderada na validação interna ({VAL_FRACTION:.0%} das gravações de treino)",
              "augmentation": "deslocamento ±1 bin (p=0,5) + máscara de frequência até 16 bins (p=0,5)",
              "seed": f"{args.seed} + fold", "num_classes": num_classes}
    writer = ResultsWriter(exp, script=__file__, params=params)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== {exp} ({dev}) ===", flush=True)

    sps = []
    for fold in folds:
        k = fold["fold_idx"]
        t0 = time.time()
        seed = args.seed + k
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_all = {f: t for f, t in fold["train_files"].items() if t != exclude}
        test_files = {f: t for f, t in fold["test_files"].items() if t != exclude}
        trn_files, val_files = split_train_val(train_all, seed)

        Xtr, ytr, _, ntr = build_windows(cache, trn_files, context)
        Xva, yva, _, nva = build_windows(cache, val_files, context)
        Xte, yte, fte, nte = build_windows(cache, test_files, context)

        Ltr = np.log(Xtr + LOG_EPS)
        mu, sd = Ltr.mean(0), Ltr.std(0) + 1e-6
        to_t = lambda a: torch.tensor((np.log(a + LOG_EPS) - mu) / sd, dtype=torch.float32, device=dev)
        Xtr_t, Xva_t, Xte_t = to_t(Xtr), to_t(Xva), to_t(Xte)
        ntr_t, nva_t, nte_t = (torch.tensor(a, device=dev) for a in (ntr, nva, nte))
        ytr_t, yva_t = torch.tensor(ytr, device=dev), torch.tensor(yva, device=dev)

        counts = np.bincount(ytr, minlength=num_classes).astype(np.float32)
        w = torch.tensor(len(ytr) / (num_classes * counts), dtype=torch.float32, device=dev)
        criterion = nn.CrossEntropyLoss(weight=w)

        model = iara_scnn.SpectralCNN(n_bins=Xtr.shape[1], n_targets=num_classes, n_context=2 * context + 1).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, args.patience // 3))

        best_loss, best_state, best_epoch, bad = np.inf, None, 0, 0
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(ntr_t), device=dev)
            for i in range(0, len(perm), args.batch_size):
                b = perm[i:i + args.batch_size]
                xb = augment(Xtr_t[ntr_t[b]])
                opt.zero_grad()
                loss = criterion(model(xb), ytr_t[b])
                loss.backward()
                opt.step()
            val_loss = criterion(predict(model, Xva_t, nva_t), yva_t).item()
            sched.step(val_loss)
            if val_loss < best_loss - 1e-4:
                best_loss, best_epoch, bad = val_loss, epoch, 0
                best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        model.load_state_dict(best_state)

        logits = predict(model, Xte_t, nte_t)
        win_pred = logits.argmax(1).cpu().numpy()
        writer.save_fold(k, fte, yte, win_pred, train_files=train_all,
                         scores=torch.softmax(logits, 1).cpu().numpy())

        votes = file_vote(fte, win_pred)
        sp, rec, _ = compute_sp_index([test_files[f] for f in votes], list(votes.values()), num_classes)
        sps.append(sp)
        print(f"  fold {k}: SP={sp:.2f} | recalls={np.round(rec * 100, 1)} | melhor época {best_epoch + 1}"
              f" (val {best_loss:.3f}) | {time.time() - t0:.0f}s", flush=True)

    print(f">> {exp}: SP = {np.mean(sps):.2f} ± {np.std(sps):.2f}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--num_classes", type=int, default=0, choices=[0, 3, 4], help="0 = 4c e 3c")
    p.add_argument("--rep", default="hht", choices=list(REPS))
    p.add_argument("--context", type=int, default=0, help="janelas vizinhas de cada lado (0 = só a janela)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=str, default=None, help="ex.: 0,1 (teste rápido; padrão: os 10)")
    args = p.parse_args()

    cache = REPS[args.rep][0]()
    folds = get_official_folds()
    if args.folds:
        folds = [folds[int(i)] for i in args.folds.split(",")]
    for nc in ([4, 3] if args.num_classes == 0 else [args.num_classes]):
        run(cache, folds, nc, args.context, args)


if __name__ == "__main__":
    main()
