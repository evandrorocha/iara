"""
Extração DEMON_MIXBG_MAR: DEMON do pipeline sobre o áudio de navios LARGE MISTURADO com uma gravação de BACKGROUND
escolhida pelo ESTADO DO MAR. Data augmentation para o treino (experimentos DEMON + mar).

NÃO altera o DEMON usado pelos modelos (data/iara_processed/demon_features, cache_fusion_mel_demon.pkl): grava só
em pastas novas, e nenhum WAV é escrito.

Usa EXATAMENTE as mesmas misturas da extração LOFAR (scripts/extrair_lofar_mixbg_mar.py): para cada (ID, cópia) lê
bg_id, offset_s e rsf_db de data/iara_processed/lofar_mixbg_<grupo>/sorteios.csv, refaz a mistura
(navio + g·fundo, fundo repetido circularmente a partir do deslocamento, potência navio/fundo = 10^(RSF/10)) e calcula
o DEMON com iara.processing.demon.compute_demon e os parâmetros de scratch/extract_all_demon.py
(banda 1–2 kHz, envelope |x|, banda-base 50 Hz, STFT de 512 pontos com 50% de sobreposição, 4 bins DC descartados)
-> janelas de 10,24 s a cada 5,12 s, 253 frequências de modulação (0,4–25 Hz).
Com --check, recalcula sem mistura e compara com demon_features/<ID>.pkl.

Saídas:
  data/iara_processed/demon_mixbg_<grupo>/<ID>_<j>.npy   janelas (n × 253, float32)
  data/iara_processed/cache_demon_mixbg_<grupo>.pkl       {'janelas': {(ID, j): janelas}, 'sorteios': DataFrame}
Uso (raiz):  python scripts/extrair_demon_mixbg_mar.py --grupo agitado --workers 12
             python scripts/extrair_demon_mixbg_mar.py --check 4 8
"""
import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scratch"))
import extrair_lofar_mixbg as M  # noqa: E402
from iara.processing.demon import compute_demon  # noqa: E402
from eval_ship_split_5x2cv import metadata  # noqa: E402

DEMON_DIR = os.path.join(ROOT, "data", "iara_processed", "demon_features")


def demon_windows(x, fs):
    r = compute_demon(x, fs, n_fft=512, max_freq=50, overlap_ratio=0.5, passband=(1000.0, 2000.0),
                      method="abs", discard_dc_bins=4)
    return r["waterfall"].T.astype(np.float32)


def job(fid, j, bg, off_s, rsf, paths, out_dir):
    fs, x = M.read(fid, paths)
    fsb, b = M.read(bg, paths)
    assert fs == fsb
    x, b = x.astype(np.float64), b.astype(np.float64)
    off = int(round(off_s * fs))
    b = np.take(b, np.arange(off, off + len(x)), mode="wrap")
    xs, bs = x - x.mean(), b - b.mean()
    g = np.sqrt(np.mean(xs ** 2) / (np.mean(bs ** 2) * 10 ** (rsf / 10)))
    w = demon_windows(x + g * bs, fs)
    np.save(os.path.join(out_dir, f"{fid}_{j}.npy"), w)
    return fid, j, w.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grupo", choices=["agitado", "calmo"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--check", type=int, nargs="*")
    args = ap.parse_args()
    m = metadata()
    paths = {int(i): os.path.join(M.DATA, d, f"{d}-{int(i):04d}.wav") for i, d in zip(m.index, m.Dataset)}

    if args.check is not None:
        for fid in args.check or [1]:
            fs, x = M.read(fid, paths)
            w = demon_windows(x.astype(np.float64), fs)
            o = pd.read_pickle(os.path.join(DEMON_DIR, f"{fid}.pkl"))["df"].values
            print(f"ID {fid}: {w.shape} x demon_features {o.shape}; |dif| máx = {np.abs(w - o).max():.2e}")
        return

    sor = pd.read_csv(os.path.join(ROOT, "data", "iara_processed", f"lofar_mixbg_{args.grupo}", "sorteios.csv"))
    out_dir = os.path.join(ROOT, "data", "iara_processed", f"demon_mixbg_{args.grupo}")
    cache_path = os.path.join(ROOT, "data", "iara_processed", f"cache_demon_mixbg_{args.grupo}.pkl")
    os.makedirs(out_dir, exist_ok=True)
    todo = [r for r in sor.itertuples() if not os.path.exists(os.path.join(out_dir, f"{int(r.ID)}_{int(r.copia)}.npy"))]
    print(f"grupo {args.grupo}: {len(sor)} cópias (mesmas misturas do LOFAR); faltam {len(todo)}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(job, int(r.ID), int(r.copia), int(r.bg_id), float(r.offset_s), float(r.rsf_db), paths, out_dir)
                for r in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            fu.result()
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    cache = {(int(r.ID), int(r.copia)): np.load(os.path.join(out_dir, f"{int(r.ID)}_{int(r.copia)}.npy"))
             for r in sor.itertuples()}
    with open(cache_path, "wb") as fh:
        pickle.dump({"janelas": cache, "sorteios": sor}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{len(cache)} cópias em {time.time() - t0:.0f}s -> {cache_path}")


if __name__ == "__main__":
    main()
