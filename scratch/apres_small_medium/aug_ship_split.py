"""
Data augmentation e reamostragem por navio no MLP LOFAR, avaliado com SPLIT POR NAVIO (nenhum navio em treino e teste).

Base = split_navio_mlp_lofar_4c (scratch/eval_ship_split_nn_5x2cv.py): MLP h32-16, dropout 0,2, BatchNorm, 35 épocas,
batch 64, Adam lr 1e-3 wd 1e-3, StandardScaler(treino), CE com peso por classe, janelas LOFAR de 5,12 s (cache_lofar.pkl,
1024 faixas, versão L2 da CNN). Mesmos folds (reports/split_navio/folds.csv) e sementes (= fold) em todas as variantes,
então as diferenças entre variantes são pareadas fold a fold.

Variantes (--variants):
  base     : igual ao original (re-rodado no mesmo hardware das variantes)
  navio    : peso por amostra = 1/(nº de janelas do navio) dentro da classe -> cada navio pesa igual na sua classe
             (BACKGROUND: cada gravação é um "navio"); a soma por classe continua balanceada entre classes
  stretch  : esticamento aleatório do eixo de frequência, fator U(0,92; 1,08) por amostra (rotação do eixo/velocidade)
  mask     : SpecAugment em frequência: 2 máscaras de até 48 faixas (~375 Hz) zeradas
  mixnavio : mixup DENTRO da classe entre janelas de navios diferentes, lambda = max(u, 1-u), u ~ U(0,1) ("navio virtual")
  ruido    : ruído gaussiano no espaço padronizado, sigma 0,3
  bg       : mistura com BACKGROUND no áudio (cache_lofar_mixbg.pkl, scripts/extrair_lofar_mixbg.py, RSF 0-20 dB):
             a cada lote, 50% das amostras viram a mesma janela de uma cópia misturada; só cópias cujo fundo é
             BACKGROUND do treino do fold
  combinações com '+', ex.: navio+stretch+mask
As augmentations valem só no treino; o teste é o original.

Saída: results/trainings/multirep/augnavio_mlp_lofar_<variante>_4c
Rodar na raiz do repositório:  python scratch/apres_small_medium/aug_ship_split.py --variants base,navio,...
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from results_writer import ResultsWriter  # noqa: E402
from eval_fusion_mel_demon_5x2cv import PyTorchFusionMLP  # noqa: E402
from eval_ship_split_5x2cv import CACHES, FOLDS_CSV, fold_files, matrices, metadata  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NC = 4


def seed(k):
    np.random.seed(k)
    torch.manual_seed(k)
    torch.cuda.manual_seed_all(k)


def stretch(x, lo=0.92, hi=1.08):
    """Reamostra cada linha em f/s (s ~ U(lo, hi)): componentes em f vão para s·f."""
    b, d = x.shape
    s = torch.empty(b, 1, device=x.device).uniform_(lo, hi)
    pos = torch.arange(d, device=x.device, dtype=x.dtype).unsqueeze(0) / s
    i0 = pos.floor().clamp(0, d - 1).long()
    i1 = (i0 + 1).clamp(max=d - 1)
    w = (pos - i0.float()).clamp(0, 1)
    out = torch.gather(x, 1, i0) * (1 - w) + torch.gather(x, 1, i1) * w
    return out * (pos <= d - 1).float()


def freq_mask(x, n=2, width=48):
    b, d = x.shape
    x = x.clone()
    ar = torch.arange(d, device=x.device).unsqueeze(0)
    for _ in range(n):
        wdt = torch.randint(0, width + 1, (b, 1), device=x.device)
        f0 = (torch.rand(b, 1, device=x.device) * (d - wdt)).long()
        x = x.masked_fill((ar >= f0) & (ar < f0 + wdt), 0.0)
    return x


def mix_same_class_other_ship(x, y, ship, alpha=1.0):
    """Para cada amostra, sorteia um parceiro da MESMA classe e de OUTRO navio dentro do lote; mistura convexa."""
    b = x.shape[0]
    lam = torch.distributions.Beta(alpha, alpha).sample((b, 1)).to(x.device)
    lam = torch.maximum(lam, 1 - lam)  # a amostra original domina
    partner = torch.arange(b, device=x.device)
    perm = torch.randperm(b, device=x.device)
    # tenta várias permutações e fica com o 1º parceiro válido
    for _ in range(8):
        cand = perm[torch.randperm(b, device=x.device)]
        ok = (y[cand] == y) & (ship[cand] != ship) & (partner == torch.arange(b, device=x.device))
        partner = torch.where(ok, cand, partner)
    has = partner != torch.arange(b, device=x.device)
    lam = torch.where(has.unsqueeze(1), lam, torch.ones_like(lam))
    return lam * x + (1 - lam) * x[partner]


_MIX = None


def load_mix(path="data/iara_processed/cache_lofar_mixbg.pkl"):
    global _MIX
    if _MIX is None:
        with open(path, "rb") as fh:
            _MIX = pickle.load(fh)
    return _MIX


def run(cache, tr, te, k, variant, ship_of,epochs=35, batch_size=64, lr=1e-3):
    seed(k)
    parts = set(variant.split("+"))
    Xtr, ytr, ftr = matrices(cache, tr)
    Xte, yte, fte = matrices(cache, te)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8  # StandardScaler(treino); aplicado DEPOIS da augmentation
    ship = np.array([ship_of[f] for f in ftr])
    _, ship_code = np.unique(ship, return_inverse=True)

    counts = np.bincount(ytr, minlength=NC)
    cw = len(ytr) / (NC * counts.astype(np.float64) + 1e-6)
    if "navio" in parts:
        # peso da amostra = cw[c] * (janelas da classe / nº navios da classe) / (janelas do navio)
        sw = np.empty(len(ytr))
        for c in range(NC):
            mc = ytr == c
            s_c, inv, n_s = np.unique(ship_code[mc], return_inverse=True, return_counts=True)
            sw[mc] = cw[c] * (mc.sum() / len(s_c)) / n_s[inv]
    else:
        sw = cw[ytr]

    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=DEVICE)  # noqa: E731
    Xt, yt, wt, st = t(Xtr), t(ytr, torch.long), t(sw), t(ship_code, torch.long)
    if "bg" in parts:
        # cópias com BACKGROUND misturado (scripts/extrair_lofar_mixbg.py); só valem as cópias cujo fundo é uma
        # gravação de BACKGROUND do TREINO deste fold (nenhum áudio de teste entra no treino)
        mix = load_mix()
        jan, sor = mix["janelas"], mix["sorteios"]
        bg_trn = {f for f, c in tr.items() if c == 3}
        ncop = int(sor.copia.max()) + 1
        Xm = np.zeros((len(Xtr), ncop, Xtr.shape[1]), dtype=np.float32)
        Mm = np.zeros((len(Xtr), ncop), dtype=bool)
        bg_of = {(int(r.ID), int(r.copia)): int(r.bg_id) for r in sor.itertuples()}
        pos = 0
        for fid in tr:  # mesma ordem de matrices()
            n = len(cache[fid])
            for j in range(ncop):
                if (fid, j) in jan and bg_of[(fid, j)] in bg_trn:
                    assert len(jan[(fid, j)]) == n, (fid, j)
                    Xm[pos:pos + n, j] = jan[(fid, j)]
                    Mm[pos:pos + n, j] = True
            pos += n
        Xm_t, Mm_t = t(Xm), t(Mm, torch.bool)
        has_t = Mm_t.any(1)
        del Xm
    mu_t, sd_t = t(mu), t(sd)
    model = PyTorchFusionMLP(in_features=Xtr.shape[1], num_classes=NC, hidden=[32, 16], dropout=0.2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            xb, yb = Xt[idx], yt[idx]
            if "bg" in parts:  # metade das amostras com cópia válida vira a mesma janela com fundo misturado
                j = torch.multinomial(Mm_t[idx].float() + 1e-9, 1).squeeze(1)
                use = has_t[idx] & (torch.rand(len(idx), device=DEVICE) < 0.5)
                xb = torch.where(use.unsqueeze(1), Xm_t[idx, j], xb)
            if "mixnavio" in parts:
                xb = mix_same_class_other_ship(xb, yb, st[idx])
            if "stretch" in parts:
                xb = stretch(xb)
            if "mask" in parts:
                xb = freq_mask(xb)
            xb = (xb - mu_t) / sd_t
            if "ruido" in parts:
                xb = xb + 0.3 * torch.randn_like(xb)
            loss = (F.cross_entropy(model(xb), yb, reduction="none") * wt[idx]).sum() / wt[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model((t(Xte) - mu_t) / sd_t)
    return fte, yte, logits.argmax(1).cpu().numpy(), torch.softmax(logits, 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,navio,stretch,mask,mixnavio,ruido,navio+stretch+mask")
    ap.add_argument("--cond", default="split_navio")
    args = ap.parse_args()
    torch.set_num_threads(16)
    print(f"Dispositivo: {DEVICE}", flush=True)
    folds = pd.read_csv(FOLDS_CSV, dtype={"Ship ID": str})
    meta = metadata()
    ship_of = {f: (s if s != "-" else f"bg_{f}") for f, s in meta["Ship ID"].items()}
    path, key = CACHES["lofar"]
    with open(path, "rb") as fh:
        cache = pickle.load(fh)
    for v in args.variants.split(","):
        tag = v.replace("+", "-")
        prefix = "augnavio" if args.cond == "split_navio" else "augid"
        w = ResultsWriter(f"{prefix}_mlp_lofar_{tag}_4c", script=__file__, params={
            "base": "split_navio_mlp_lofar_4c", "variant": v, "split": args.cond, "folds_csv": FOLDS_CSV,
            "hidden": [32, 16], "dropout": 0.2, "epochs": 35, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-3,
            "device": str(DEVICE), "seed": "torch/numpy = fold"})
        t0 = time.time()
        for k in range(10):
            tr, te = fold_files(folds, args.cond, k, "trn"), fold_files(folds, args.cond, k, "test")
            fte, yte, pred, scores = run(cache, tr, te, k, v, ship_of)
            w.save_fold(k, fte, yte, pred, train_files=tr, scores=scores)
        print(f"{v}: 10 folds em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
