"""
CNN harmônica (iara.ml.models.harmonic_cnn.HarmonicCNN) no DEMON (253 bins, 0,39-25 Hz), 5x2cv oficial.

CNN específica para DEMON. No espectro DEMON a hélice aparece como a linha do eixo f0 e seus
harmônicos k*f0 (pás); as linhas são fracas (pico/mediana do espectro médio ~2,6 em SMALL).
  - TPSW (Two-Pass Split Window): cada espectro é dividido pela estimativa do fundo
    (janela dividida de 2n+1 bins sem os 2g+1 centrais; na 2ª passada, valores acima de alpha*fundo
    são trocados pelo fundo). n=16, g=3, alpha=2, fixados antes de treinar. Depois log e z-score
    por bin (estatísticas só do treino do fold).
  - empilhamento harmônico dentro do modelo: canais X(h*f0), h = 0,5, 1, 2, 3, 4, 5.
  - `--median k`: mediana de k janelas vizinhas da mesma gravação antes do TPSW (f0 varia pouco numa
    gravação; reduz o ruído). k=1 = sem mediana.
  - mesmo protocolo de treino da CNN espectral HHT (eval_hht_spectral_cnn_5x2cv.py): validação interna
    de 15% das gravações de treino, early stopping, semente fixa por fold, AdamW, entropia cruzada
    ponderada, aumento leve; voto majoritário por gravação.

Protocolo: mesmos folds 5x2cv do pipeline (particionamento por gravação `ID`; o mesmo navio pode
estar em treino e teste). Predições por janela gravadas em results/trainings/multirep/<exp>/.

Uso (no container):
    python scratch/eval_demon_harmonic_cnn_5x2cv.py --num_classes 0 --median 1
    python scratch/eval_demon_harmonic_cnn_5x2cv.py --num_classes 0 --median 5
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import iara.ml.models.harmonic_cnn as iara_hcnn  # noqa: E402

from results_writer import ResultsWriter  # noqa: E402
from eval_hht_models_5x2cv import compute_sp_index, get_official_folds  # noqa: E402
from eval_demon_baselines_5x2cv import load_all_demon_features  # noqa: E402
from eval_hht_spectral_cnn_5x2cv import VAL_FRACTION, augment, build_windows, file_vote, predict, split_train_val  # noqa: E402

DEMON_DIR = "data/iara_processed/demon_features"
TPSW_N, TPSW_G, TPSW_ALPHA = 16, 3, 2.0
LOG_EPS = 1e-3
HARMONICS = [0.5, 1, 2, 3, 4, 5]


def tpsw(X, n=TPSW_N, g=TPSW_G, alpha=TPSW_ALPHA):
    """Normaliza cada linha de X (janelas x bins) pelo fundo estimado com TPSW."""
    k = np.ones(2 * n + 1)
    k[n - g:n + g + 1] = 0
    k /= k.sum()
    bg1 = ndi.convolve1d(X, k, axis=1, mode="reflect")
    clipped = np.where(X > alpha * bg1, bg1, X)
    bg = ndi.convolve1d(clipped, k, axis=1, mode="reflect")
    return X / (bg + 1e-12)


def preprocess_cache(cache, median):
    """Mediana temporal (opcional) e TPSW por gravação; não usa rótulos nem estatísticas entre gravações."""
    out = {}
    for fid, w in cache.items():
        if median > 1:
            w = ndi.median_filter(w, size=(median, 1), mode="nearest")
        out[fid] = np.log(tpsw(w) + LOG_EPS).astype(np.float32)
    return out


def run(cache, freqs, folds, num_classes, args):
    exclude = 3 if num_classes == 3 else None
    exp = f"hcnn_demon_tpsw_med{args.median}_e{args.epochs}p{args.patience}_{num_classes}c"
    params = {"model": "iara.ml.models.harmonic_cnn.HarmonicCNN", "representation": "DEMON", "n_bins": len(freqs),
              "harmonics": HARMONICS, "conv_channels": [32, 64, 64], "kernel_size": 5, "dilations": [1, 2, 4],
              "pooled_bins": 8, "classification_n_neurons": 64, "dropout": [0.1, 0.4],
              "preprocessing": f"mediana temporal de {args.median} janela(s) + TPSW (n={TPSW_N}, g={TPSW_G}, alpha={TPSW_ALPHA}) "
                               f"+ log(x+{LOG_EPS}) + z-score por bin (treino)",
              "optimizer": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
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

        Xtr, ytr, _, ntr = build_windows(cache, trn_files, 0)
        Xva, yva, _, nva = build_windows(cache, val_files, 0)
        Xte, yte, fte, nte = build_windows(cache, test_files, 0)

        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        to_t = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32, device=dev)
        Xtr_t, Xva_t, Xte_t = to_t(Xtr), to_t(Xva), to_t(Xte)
        ntr_t, nva_t, nte_t = (torch.tensor(a, device=dev) for a in (ntr, nva, nte))
        ytr_t, yva_t = torch.tensor(ytr, device=dev), torch.tensor(yva, device=dev)

        counts = np.bincount(ytr, minlength=num_classes).astype(np.float32)
        w = torch.tensor(len(ytr) / (num_classes * counts), dtype=torch.float32, device=dev)
        criterion = nn.CrossEntropyLoss(weight=w)

        model = iara_hcnn.HarmonicCNN(frequencies=freqs, n_targets=num_classes, harmonics=HARMONICS).to(dev)
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
    p.add_argument("--median", type=int, default=1, help="janelas na mediana temporal (1 = sem mediana)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=str, default=None, help="ex.: 0,1 (teste rápido; padrão: os 10)")
    args = p.parse_args()

    with open(os.path.join(DEMON_DIR, "1.pkl"), "rb") as f:
        freqs = np.asarray(pickle.load(f)["frequencies"], dtype=np.float64)
    cache = preprocess_cache(load_all_demon_features(DEMON_DIR), args.median)
    folds = get_official_folds()
    if args.folds:
        folds = [folds[int(i)] for i in args.folds.split(",")]
    for nc in ([4, 3] if args.num_classes == 0 else [args.num_classes]):
        run(cache, freqs, folds, nc, args)


if __name__ == "__main__":
    main()
