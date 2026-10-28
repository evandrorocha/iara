"""
Treinamento e Avaliação da CNN 2D no espectrograma de Hilbert 2D (tempo x frequência por janela).
Modelo: iara.ml.models.hilbert_2d_cnn.Hilbert2DCNN
Feature: scratch/extract_all_hht_2d.py (32 passos de 80 ms x 128 bins de 7,9 Hz em cada janela de 2,56 s).

Diferença frente às duas CNNs HHT anteriores do projeto:
  1. PyTorchHHTCNN (eval_hht_models_5x2cv.py): imagens de 32 espectros marginais consecutivos (~164 s),
     ~2 imagens por gravação, sem validação e com z-score escalar global -> SP 37,5 ± 13,0 (colapsou;
     não há ablação que isole a causa).
  2. SpectralCNN (eval_hht_spectral_cnn_5x2cv.py): CNN 1D ao longo do espectro marginal de 256 bins.
  3. Hilbert2DCNN (este script): CNN 2D sobre o espectrograma de Hilbert dentro de cada janela,
     com AdaptiveAvgPool2d((2, 8)) para manter uma posição grosseira em tempo e frequência.

Pré-processamento: z-score por bin de frequência (estatísticas só do treino do fold).
Protocolo: 5x2cv oficial (particionamento por gravação `ID`; o mesmo navio pode estar em treino e teste).
Early stopping em 15% das gravações de treino (estratificado por classe). Voto majoritário por gravação.
"""

import argparse
import os
import sys
import time
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import iara.ml.models.hilbert_2d_cnn as iara_h2d  # noqa: E402

from results_writer import ResultsWriter  # noqa: E402
from eval_hht_models_5x2cv import compute_sp_index, get_official_folds  # noqa: E402

CACHE_2D_PATH = "data/iara_processed/cache_hht_2d.pkl"
FEATURES_2D_DIR = "data/iara_processed/hht_2d_features"
VAL_FRACTION = 0.15


def load_all_hht_2d_features(cache_path=CACHE_2D_PATH, features_dir=FEATURES_2D_DIR):
    """
    Carrega todos os espectrogramas Hilbert 2D (32x128) em memória.
    Retorna: dict mapping file_id (int) -> (numpy array of windows, shape (N, 32, 128))
    """
    if os.path.exists(cache_path):
        print(f"\nCarregando espectrogramas Hilbert 2D unificados de '{cache_path}'...", flush=True)
        t0 = time.time()
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"Sucesso: {len(cache)} arquivos carregados em {time.time()-t0:.2f}s.", flush=True)
        return cache

    print(f"\nCarregando espectrogramas 2D individuais de '{features_dir}'...", flush=True)
    t0 = time.time()
    cache = {}
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Diretório de features 2D não encontrado: {features_dir}. "
                                f"Execute primeiro 'python scratch/extract_all_hht_2d.py'.")

    file_list = [f for f in os.listdir(features_dir) if f.endswith(".pkl") and not f.endswith(".tmp")]
    for fname in file_list:
        file_id = int(os.path.splitext(fname)[0])
        fpath = os.path.join(features_dir, fname)
        try:
            with open(fpath, "rb") as f:
                d = pickle.load(f)
                cache[file_id] = d["spectrograms"].astype(np.float32)
        except Exception as e:
            print(f"Erro ao ler {fpath}: {e}")

    print(f"Sucesso: {len(cache)} arquivos lidos em {time.time()-t0:.2f}s.", flush=True)

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cache unificado salvo em '{cache_path}'.", flush=True)
    except Exception as e:
        print(f"Aviso ao salvar cache: {e}")

    return cache


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


def build_image_dataset(cache, files):
    """
    Retorna arrays com todas as janelas 2D das gravações selecionadas:
    - X: (Total_Windows, 32, 128) float32
    - y: (Total_Windows,) int64
    - fids: (Total_Windows,) int64
    """
    X, y, fids = [], [], []
    for fid, target in files.items():
        if fid not in cache:
            continue
        specs = cache[fid]  # (N, 32, 128)
        n_w = len(specs)
        X.append(specs)
        y.append(np.full(n_w, target, dtype=np.int64))
        fids.append(np.full(n_w, fid, dtype=np.int64))

    return np.vstack(X).astype(np.float32), np.concatenate(y), np.concatenate(fids)


def file_vote(fids, preds):
    """Voto majoritário por arquivo."""
    votes = {}
    for f, p in zip(fids, preds):
        votes.setdefault(int(f), []).append(int(p))
    return {f: Counter(v).most_common(1)[0][0] for f, v in votes.items()}


def predict(model, X_tensor, batch_size=512):
    """Predição em mini-batches."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            out.append(model(batch))
    return torch.cat(out)


def run_5x2cv(cache, folds, num_classes, args):
    exclude = 3 if num_classes == 3 else None
    exp = f"hht_2d_cnn_e{args.epochs}p{args.patience}_{num_classes}c"
    sample_spec = next(iter(cache.values()))
    n_time_bins, n_freq_bins = sample_spec.shape[1], sample_spec.shape[2]

    params = {
        "model": "iara.ml.models.hilbert_2d_cnn.Hilbert2DCNN",
        "representation": "espectrograma de Hilbert 2D (32 x 128 por janela de 2,56 s; 8 IMFs)",
        "normalization": "z-score por bin de frequência (treino)",
        "input_shape": [1, n_time_bins, n_freq_bins],
        "conv_channels": [32, 64, 128],
        "pooled_topology": [2, 8],
        "classification_n_neurons": 128,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "class_weighting": "balanced_loss",
        "early_stopping": f"val_loss ({VAL_FRACTION:.0%} train recordings)",
        "num_classes": num_classes,
        "seed": f"{args.seed} + fold"
    }

    writer = ResultsWriter(exp, script=__file__, params=params)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================", flush=True)
    print(f"AVALIAÇÃO OFICIAL 5x2cv: {exp} (Hardware: {dev})", flush=True)
    print(f"Formato da imagem 2D por janela: 1 canal x {n_time_bins} tempo x {n_freq_bins} freq", flush=True)
    print(f"=======================================================", flush=True)

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

        Xtr, ytr, _ = build_image_dataset(cache, trn_files)
        Xva, yva, _ = build_image_dataset(cache, val_files)
        Xte, yte, fte = build_image_dataset(cache, test_files)

        # Z-score por bin de frequência (média e desvio sobre janelas e passos de tempo do treino).
        # Um z-score escalar global deixa poucos bins de alta energia dominarem a escala, como na
        # CNN 2D antiga; por bin, cada faixa de frequência fica com média 0 e desvio 1.
        mu = Xtr.mean(axis=(0, 1), keepdims=True)
        sd = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
        Xtr_norm = (Xtr - mu) / sd
        Xva_norm = (Xva - mu) / sd
        Xte_norm = (Xte - mu) / sd

        # Tensor conversion: (N, 1, 32, 128)
        Xtr_t = torch.tensor(Xtr_norm[:, np.newaxis, :, :], dtype=torch.float32, device=dev)
        Xva_t = torch.tensor(Xva_norm[:, np.newaxis, :, :], dtype=torch.float32, device=dev)
        Xte_t = torch.tensor(Xte_norm[:, np.newaxis, :, :], dtype=torch.float32, device=dev)
        ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
        yva_t = torch.tensor(yva, dtype=torch.long, device=dev)

        # Class weights inversos da frequência
        counts = np.bincount(ytr, minlength=num_classes).astype(np.float32)
        w = torch.tensor(len(ytr) / (num_classes * counts), dtype=torch.float32, device=dev)
        criterion = nn.CrossEntropyLoss(weight=w)

        # Instanciação da Nova CNN 2D Hilbert
        model = iara_h2d.Hilbert2DCNN(
            n_targets=num_classes,
            in_channels=1,
            n_time_bins=n_time_bins,
            n_freq_bins=n_freq_bins,
            pooled_time=2,
            pooled_freq=8
        ).to(dev)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, args.patience // 3))

        best_loss, best_state, best_epoch, bad = np.inf, None, 0, 0
        n_samples = len(Xtr_t)

        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n_samples, device=dev)
            for i in range(0, n_samples, args.batch_size):
                b_idx = perm[i : i + args.batch_size]
                if len(b_idx) < 2:
                    continue  # BatchNorm1d da cabeça não treina com lote de 1 amostra
                xb = Xtr_t[b_idx]
                yb = ytr_t[b_idx]

                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()

            # Validação
            model.eval()
            val_logits = predict(model, Xva_t, batch_size=args.batch_size)
            val_loss = criterion(val_logits, yva_t).item()
            sched.step(val_loss)

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_epoch = epoch
                bad = 0
                best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Teste final no fold
        logits = predict(model, Xte_t, batch_size=args.batch_size)
        win_preds = logits.argmax(1).cpu().numpy()
        scores = torch.softmax(logits, 1).cpu().numpy()

        writer.save_fold(k, fte, yte, win_preds, train_files=train_all, scores=scores)

        # Voto majoritário por gravação
        votes = file_vote(fte, win_preds)
        y_true_audio = [test_files[f] for f in votes]
        y_pred_audio = list(votes.values())
        sp, recalls, _ = compute_sp_index(y_true_audio, y_pred_audio, num_classes)
        sps.append(sp)

        elapsed = time.time() - t0
        print(f"  Fold {k}: SP={sp:.2f}% | Recalls={np.round(recalls * 100, 1)}% | "
              f"Melhor época: {best_epoch + 1} (val_loss={best_loss:.3f}) | Tempo: {elapsed:.1f}s", flush=True)

    mean_sp = np.mean(sps)
    std_sp = np.std(sps)
    print("\n" + "=" * 80, flush=True)
    print(f">> RESULTADO FINAL {exp}: SP Oficial = {mean_sp:.2f}% ± {std_sp:.2f}%", flush=True)
    print("=" * 80 + "\n", flush=True)


def main():
    p = argparse.ArgumentParser(description="Avaliação 5x2cv da CNN 2D no Espectrograma de Hilbert")
    p.add_argument("--num_classes", type=int, default=0, choices=[0, 3, 4], help="0 = 4c e 3c")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=str, default=None, help="ex.: 0,1 (padrão: todos os 10)")
    args = p.parse_args()

    cache = load_all_hht_2d_features()
    folds = get_official_folds()
    if args.folds:
        folds = [folds[int(i)] for i in args.folds.split(",")]

    classes_to_run = [4, 3] if args.num_classes == 0 else [args.num_classes]
    for nc in classes_to_run:
        run_5x2cv(cache, folds, nc, args)


if __name__ == "__main__":
    main()
