"""
Extração LOFAR_MIXBG: LOFAR do pipeline calculado sobre o áudio do navio MISTURADO com uma gravação de BACKGROUND
(data augmentation para o treino). NÃO altera nem substitui o LOFAR usado pelos modelos (lofar_c5de…, cache_lofar.pkl):
grava só em pastas novas, e nenhum WAV é escrito.

Por gravação de navio (classes SMALL/MEDIUM/LARGE, subconjuntos A–D) e por cópia j = 0..N-1:
  1. sorteia (semente = ID·100 + j) uma gravação de BACKGROUND (subconjunto E), um deslocamento inicial e uma razão
     sinal/fundo-adicionado RSF ~ U(0, 20) dB; o fundo é repetido circularmente até cobrir a gravação do navio;
  2. escala o fundo para que potência(navio) / potência(fundo) = 10^(RSF/10) (potências médias da gravação inteira,
     após remover a média; os níveis do IARA não são calibrados, então a razão é a única escala possível);
  3. mistura = navio + fundo escalado (float64);
  4. LOFAR idêntico ao do pipeline (analysis.Analysis.LOFAR.apply com n_pts 1024, overlap 0, decimação 3,
     integração 0,512 s; Normalization.NORM_L2), depois média em janelas de 5,12 s nos mesmos instantes do
     cache_lofar.pkl (scratch/build_lofar_cache.py, tempos do demon_features).
Com RSF -> infinito o resultado reproduz o cache_lofar.pkl (conferido com --check).

Saídas:
  data/iara_processed/lofar_mixbg/<ID>_<j>.npy     janelas (n × 1024, float32)
  data/iara_processed/lofar_mixbg/sorteios.csv     ID, copia, bg_id, offset_s, rsf_db
  data/iara_processed/cache_lofar_mixbg.pkl         {(ID, j): janelas}, {'sorteios': DataFrame}
Uso (raiz do repositório):
  python scripts/extrair_lofar_mixbg.py --check 1 2       # confere a reprodução sem mistura
  python scripts/extrair_lofar_mixbg.py --copias 4 --workers 12
"""
import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from iara.processing.analysis import SpectralAnalysis, Normalization  # noqa: E402

DATA = os.path.join(ROOT, "data", "iara")
OUT_DIR = os.path.join(ROOT, "data", "iara_processed", "lofar_mixbg")
CACHE = os.path.join(ROOT, "data", "iara_processed", "cache_lofar_mixbg.pkl")
DEMON_DIR = os.path.join(ROOT, "data", "iara_processed", "demon_features")
CACHE_ORIG = os.path.join(ROOT, "data", "iara_processed", "cache_lofar.pkl")
META = os.path.join(ROOT, "src", "iara", "dataset_info", "iara.csv")
RSF_MIN, RSF_MAX = 0.0, 20.0


def lofar_windows(x, fs, fid):
    power, freqs, times = SpectralAnalysis.LOFAR.apply(data=x, fs=fs, n_pts=1024, n_overlap=0, n_mels=256,
                                               decimation_rate=3, integration_overlap=0,
                                               integration_interval=0.512)
    power = Normalization.NORM_L2(power)
    df = power.T.astype(np.float32)
    tm = np.asarray(times, dtype=np.float32)
    tdem = np.asarray(pd.read_pickle(os.path.join(DEMON_DIR, f"{fid}.pkl"))["times"], dtype=np.float32)
    out = []
    for t in tdem:  # igual a scratch/build_lofar_cache.py
        mask = (tm >= (t - 5.12)) & (tm <= t)
        out.append(df[mask].mean(axis=0) if np.any(mask) else df[np.argmin(np.abs(tm - t))])
    return np.array(out, dtype=np.float32)


def read(fid, paths):
    fs, x = wav.read(paths[fid])
    if x.ndim != 1:
        x = x[:, 0]
    return fs, x


def job(fid, j, bg_ids, paths):
    rng = np.random.default_rng(fid * 100 + j)
    bg = int(rng.choice(bg_ids))
    rsf = float(rng.uniform(RSF_MIN, RSF_MAX))
    fs, x = read(fid, paths)
    fsb, b = read(bg, paths)
    assert fs == fsb
    x = x.astype(np.float64)
    b = b.astype(np.float64)
    off = int(rng.integers(0, len(b)))
    b = np.take(b, np.arange(off, off + len(x)), mode="wrap")
    xs, bs = x - x.mean(), b - b.mean()
    g = np.sqrt(np.mean(xs ** 2) / (np.mean(bs ** 2) * 10 ** (rsf / 10)))
    w = lofar_windows(x + g * bs, fs, fid)
    np.save(os.path.join(OUT_DIR, f"{fid}_{j}.npy"), w)
    return {"ID": fid, "copia": j, "bg_id": bg, "offset_s": off / fs, "rsf_db": rsf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copias", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--check", type=int, nargs="*")
    args = ap.parse_args()
    m = pd.read_csv(META, dtype={"ID": str})
    m = m[m.Dataset.isin(list("ABCDE"))]
    paths = {int(i): os.path.join(DATA, d, f"{d}-{i}.wav") for i, d in zip(m.ID, m.Dataset)}
    ships = sorted(int(i) for i in m[m.Dataset != "E"].ID)
    bg_ids = sorted(int(i) for i in m[m.Dataset == "E"].ID)

    if args.check is not None:
        orig = pickle.load(open(CACHE_ORIG, "rb"))
        for fid in args.check or [1]:
            fs, x = read(fid, paths)
            w = lofar_windows(x.astype(np.float64), fs, fid)
            d = np.abs(w - orig[fid]).max()
            print(f"ID {fid}: {w.shape} x cache {orig[fid].shape}; |dif| máx = {d:.2e}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    todo = [(f, j) for f in ships for j in range(args.copias)
            if not os.path.exists(os.path.join(OUT_DIR, f"{f}_{j}.npy"))]
    print(f"{len(ships)} gravações de navio × {args.copias} cópias; faltam {len(todo)}", flush=True)
    rows, t0 = [], time.time()
    sp = os.path.join(OUT_DIR, "sorteios.csv")
    old = pd.read_csv(sp) if os.path.exists(sp) else pd.DataFrame()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(job, f, j, bg_ids, paths) for f, j in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
                pd.concat([old, pd.DataFrame(rows)]).to_csv(sp, index=False)
    sor = pd.concat([old, pd.DataFrame(rows)]).drop_duplicates(["ID", "copia"]).sort_values(["ID", "copia"])
    sor.to_csv(sp, index=False)
    cache = {(int(r.ID), int(r.copia)): np.load(os.path.join(OUT_DIR, f"{int(r.ID)}_{int(r.copia)}.npy"))
             for r in sor.itertuples()}
    with open(CACHE, "wb") as fh:
        pickle.dump({"janelas": cache, "sorteios": sor}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{len(cache)} cópias em {time.time() - t0:.0f}s -> {CACHE}")


if __name__ == "__main__":
    main()
