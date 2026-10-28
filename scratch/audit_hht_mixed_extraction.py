"""
Evidências do problema da extração HHT original (cache com dois limites de iterações de EMD) para o
notebook 28 (notebooks/src/generate_correcao_hht_notebook.py). Roda uma vez; o notebook só lê os CSVs.

Saídas em reports/correcao_hht/:
  arquivos.csv     uma linha por gravação: ID, horário de gravação do .pkl antigo, grupo (antes/depois da
                   última edição de scratch/extract_all_hht.py), e quanto a feature mudou na re-extração
                   (máximo e média de |novo - antigo|, fração de janelas com mudança > 1e-3)
  reproducao.csv   para uma amostra de gravações de cada grupo: em cada janela testada, a diferença máxima
                   entre o espectro antigo e o recalculado com MAX_ITERATION = 30, 100 e 1000 (padrão do PyEMD)
  contexto.json    horários de referência e configuração da re-extração

Uso (no container, na raiz): python scratch/audit_hht_mixed_extraction.py
"""
import concurrent.futures
import datetime
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import scipy.signal as scipy_sig
from PyEMD import EMD

OLD_DIR = "data/iara_processed/_backup_hht_iter_misto_20261015/hht_features"
NEW_DIR = "data/iara_processed/hht_features"
SCRIPT = "scratch/extract_all_hht.py"
OUT = "reports/correcao_hht"
ITERS = [30, 100, 1000]
N_SAMPLE_PER_GROUP = 8
WINDOWS = [0, 3, 10, 30]


def hht_window_spectrum(seg, fs, max_iter, n_bins=256, max_freq=1000.0, n_imfs=8):
    """Mesma cadeia de scratch/extract_all_hht.py para uma janela."""
    emd = EMD()
    emd.MAX_ITERATION = max_iter
    imfs = emd(seg, max_imf=n_imfs)
    df = (max_freq - 1.0) / (n_bins - 1)
    dt = 1.0 / fs
    w = np.zeros(n_bins, dtype=np.float32)
    for imf in imfs[:n_imfs]:
        a = scipy_sig.hilbert(imf)
        amp = np.abs(a)
        fr = np.gradient(np.unwrap(np.angle(a)), dt) / (2.0 * np.pi)
        m = (fr >= 1.0) & (fr <= max_freq)
        np.add.at(w, np.clip(np.floor((fr[m] - 1.0) / df).astype(int), 0, n_bins - 1), (amp[m] ** 2).astype(np.float32))
    return w / w.max() if w.max() > 0 else w


def reproduce(args):
    fid, wav, grupo = args
    fs, x = wavfile.read(wav)
    x = (x[:, 0] if x.ndim > 1 else x).astype(np.float32)
    d = int(round(fs / 2000))
    s = scipy_sig.decimate(x, d, ftype="fir", zero_phase=True)
    afs = fs / d
    s = (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)
    n, hop = int(2.56 * afs), int(5.12 * afs)
    with open(os.path.join(OLD_DIR, f"{fid}.pkl"), "rb") as f:
        old = pickle.load(f)["df"].values
    rows = []
    for w in WINDOWS:
        if w >= len(old) or w * hop + n > len(s):
            continue
        seg = s[w * hop:w * hop + n]
        row = {"ID": fid, "grupo": grupo, "janela": w}
        for it in ITERS:
            row[f"max_dif_iter{it}"] = float(np.abs(hht_window_spectrum(seg, afs, it) - old[w]).max())
        rows.append(row)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    ref = os.path.getmtime(SCRIPT)
    wavs = {int(os.path.basename(p)[:-4].split("-")[-1]): p
            for c in "ABCDE" for p in glob.glob(f"data/iara/{c}/*.wav")}

    rows = []
    for fname in sorted(os.listdir(OLD_DIR), key=lambda s: int(s[:-4])):
        fid = int(fname[:-4])
        t = os.path.getmtime(os.path.join(OLD_DIR, fname))
        with open(os.path.join(OLD_DIR, fname), "rb") as f:
            old = pickle.load(f)["df"].values
        with open(os.path.join(NEW_DIR, fname), "rb") as f:
            new = pickle.load(f)["df"].values
        dif = np.abs(new - old)
        rows.append({"ID": fid, "gravado_em": datetime.datetime.fromtimestamp(t).isoformat(timespec="seconds"),
                     "grupo": "antes" if t <= ref else "depois", "n_janelas": len(old),
                     "max_dif": float(dif.max()), "media_dif": float(dif.mean()),
                     "frac_janelas_mudou": float((dif.max(axis=1) > 1e-3).mean())})
    arquivos = pd.DataFrame(rows)
    arquivos.to_csv(os.path.join(OUT, "arquivos.csv"), index=False)

    rng = np.random.RandomState(0)
    tasks = []
    for g in ("antes", "depois"):
        ids = arquivos[arquivos.grupo == g].ID.values
        for fid in sorted(rng.choice(ids, size=min(N_SAMPLE_PER_GROUP, len(ids)), replace=False)):
            tasks.append((int(fid), wavs[int(fid)], g))
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
        repro = pd.DataFrame([r for rs in ex.map(reproduce, tasks) for r in rs])
    repro.to_csv(os.path.join(OUT, "reproducao.csv"), index=False)

    with open(os.path.join(OUT, "contexto.json"), "w", encoding="utf-8") as f:
        json.dump({"ultima_edicao_extract_all_hht": datetime.datetime.fromtimestamp(ref).isoformat(timespec="seconds"),
                   "primeiro_pkl_antigo": arquivos.gravado_em.min(), "ultimo_pkl_antigo": arquivos.gravado_em.max(),
                   "reextracao": {"n_imfs": 8, "MAX_ITERATION": 30, "script": SCRIPT,
                                  "data": datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(NEW_DIR, "1.pkl"))).isoformat(timespec="seconds")},
                   "amostra_reproducao": {"por_grupo": N_SAMPLE_PER_GROUP, "janelas": WINDOWS, "iteracoes": ITERS, "semente": 0}},
                  f, indent=2, ensure_ascii=False)
    print(arquivos.groupby("grupo")[["max_dif", "frac_janelas_mudou"]].describe().round(3).T)
    print(repro.groupby("grupo")[[f"max_dif_iter{it}" for it in ITERS]].agg(lambda v: (v < 1e-6).mean()).round(2))


if __name__ == "__main__":
    main()
