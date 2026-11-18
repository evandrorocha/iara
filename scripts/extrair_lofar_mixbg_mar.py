"""
Extração LOFAR_MIXBG_MAR: LOFAR do pipeline sobre o áudio de navios LARGE MISTURADO com uma gravação de BACKGROUND
escolhida pelo ESTADO DO MAR. Data augmentation para o treino do experimento mix_small_medium.

NÃO altera o LOFAR usado pelos modelos (cache_lofar.pkl) nem a extração lofar_mixbg: grava só em pastas novas,
e nenhum WAV é escrito. O processamento é o mesmo de scripts/extrair_lofar_mixbg.py (mesma função lofar_windows,
razão sinal/fundo-adicionado RSF ~ U(0, 20) dB, fundo repetido circularmente a partir de um deslocamento sorteado).

Grupos de fundo (subconjunto E, só gravações SEM chuva, para não confundir mar agitado com chuva):
  agitado : estado do mar >= 4   (76 gravações)
  calmo   : estado do mar <= 2   (50 gravações)   -> controle: mesmo procedimento com fundo de mar calmo
Semente = ID·100 + j + (50 se calmo).

Saídas:
  data/iara_processed/lofar_mixbg_<grupo>/<ID>_<j>.npy   janelas (n × 1024, float32)
  data/iara_processed/lofar_mixbg_<grupo>/sorteios.csv   ID, copia, bg_id, offset_s, rsf_db
  data/iara_processed/cache_lofar_mixbg_<grupo>.pkl       {'janelas': {(ID, j): janelas}, 'sorteios': DataFrame}
Uso (raiz):  python scripts/extrair_lofar_mixbg_mar.py --grupo agitado --copias 4 --workers 12
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
sys.path.insert(0, os.path.join(ROOT, "scratch"))
import extrair_lofar_mixbg as M  # noqa: E402
from eval_ship_split_5x2cv import metadata  # noqa: E402


def job(fid, j, bg_ids, paths, out_dir, extra_seed):
    rng = np.random.default_rng(fid * 100 + j + extra_seed)
    bg = int(rng.choice(bg_ids))
    rsf = float(rng.uniform(M.RSF_MIN, M.RSF_MAX))
    fs, x = M.read(fid, paths)
    fsb, b = M.read(bg, paths)
    assert fs == fsb
    x, b = x.astype(np.float64), b.astype(np.float64)
    off = int(rng.integers(0, len(b)))
    b = np.take(b, np.arange(off, off + len(x)), mode="wrap")
    xs, bs = x - x.mean(), b - b.mean()
    g = np.sqrt(np.mean(xs ** 2) / (np.mean(bs ** 2) * 10 ** (rsf / 10)))
    np.save(os.path.join(out_dir, f"{fid}_{j}.npy"), M.lofar_windows(x + g * bs, fs, fid))
    return {"ID": fid, "copia": j, "bg_id": bg, "offset_s": off / fs, "rsf_db": rsf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grupo", choices=["agitado", "calmo"], required=True)
    ap.add_argument("--copias", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "data", "iara_processed", f"lofar_mixbg_{args.grupo}")
    cache_path = os.path.join(ROOT, "data", "iara_processed", f"cache_lofar_mixbg_{args.grupo}.pkl")
    m = metadata()
    paths = {int(i): os.path.join(M.DATA, d, f"{d}-{int(i):04d}.wav") for i, d in zip(m.index, m.Dataset)}
    sea = pd.to_numeric(m["Sea state"], errors="coerce")
    bg = m[(m.Dataset == "E") & (m["Rain state"].str.strip() == "No rain")]
    sel = sea.loc[bg.index] >= 4 if args.grupo == "agitado" else sea.loc[bg.index] <= 2
    bg_ids = sorted(int(i) for i in bg.index[sel.values])
    large = sorted(int(i) for i in m.index[(m.Target == 2) & m.Dataset.isin(list("ABCD"))])
    os.makedirs(out_dir, exist_ok=True)
    todo = [(f, j) for f in large for j in range(args.copias) if not os.path.exists(os.path.join(out_dir, f"{f}_{j}.npy"))]
    print(f"grupo {args.grupo}: {len(bg_ids)} fundos; {len(large)} gravações LARGE × {args.copias} cópias; "
          f"faltam {len(todo)}", flush=True)
    rows, t0 = [], time.time()
    sp = os.path.join(out_dir, "sorteios.csv")
    old = pd.read_csv(sp) if os.path.exists(sp) else pd.DataFrame()
    extra = 50 if args.grupo == "calmo" else 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(job, f, j, bg_ids, paths, out_dir, extra) for f, j in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
                pd.concat([old, pd.DataFrame(rows)]).to_csv(sp, index=False)
    sor = pd.concat([old, pd.DataFrame(rows)]).drop_duplicates(["ID", "copia"]).sort_values(["ID", "copia"])
    sor.to_csv(sp, index=False)
    cache = {(int(r.ID), int(r.copia)): np.load(os.path.join(out_dir, f"{int(r.ID)}_{int(r.copia)}.npy"))
             for r in sor.itertuples()}
    with open(cache_path, "wb") as fh:
        pickle.dump({"janelas": cache, "sorteios": sor}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{len(cache)} cópias em {time.time() - t0:.0f}s -> {cache_path}")


if __name__ == "__main__":
    main()
