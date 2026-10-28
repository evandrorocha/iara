"""
Medidas acústicas simples por gravação (A–E do IARA) para o estudo do SMALL (Notebook 32).

Sem calibração absoluta dos hidrofones, por isso as medidas são relativas à própria gravação:
  * nível RMS em janelas de 1 s (dBFS, após remover a média da janela);
  * p5 / p50 / p95 do nível e faixa dinâmica (p95 - p5);
  * ganho no CPA: mediana do nível em ±30 s do CPA anotado menos a mediana dos 30 s iniciais e finais
    (aproximação de SNR da passagem; BACKGROUND usa o meio da gravação);
  * pico: instante do máximo do nível médio em 10 s e distância ao CPA anotado;
  * PSD de Welch (nperseg 4096): fração de energia em < 200 Hz, 200 Hz–1 kHz, 1–5 kHz, 5–24 kHz,
    centroide espectral e planura espectral (0–5 kHz).

Saída: reports/small/features_acusticas.csv
Rodar na raiz do repositório:  python scratch/small_acoustic_features.py
"""
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch

OUT = os.path.join("reports", "small", "features_acusticas.csv")
BANDAS = [(0, 200), (200, 1000), (1000, 5000), (5000, 24000)]


def medir(args):
    path, cpa = args
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    n = len(x) // fs
    fr = x[:n * fs].reshape(n, fs)
    fr = fr - fr.mean(1, keepdims=True)
    lv = 10 * np.log10(np.mean(fr ** 2, 1) + 1e-20)
    p5, p50, p95 = np.percentile(lv, [5, 50, 95])
    c = cpa if np.isfinite(cpa) and 0 <= cpa <= n else n / 2
    meio = lv[max(0, int(c) - 30):int(c) + 30]
    bordas = np.r_[lv[:30], lv[-30:]]
    lv10 = np.convolve(lv, np.ones(10) / 10, mode="valid")
    t_pico = int(np.argmax(lv10)) + 5
    f, P = welch(x - x.mean(), fs=fs, nperseg=4096)
    tot = P.sum()
    fr_b = {f"energia {a}-{b} Hz (%)": 100 * P[(f >= a) & (f < b)].sum() / tot for a, b in BANDAS}
    m5 = (f > 0) & (f <= 5000)
    return {"File": int(os.path.basename(path).split("-")[1].split(".")[0]),
            "duração (s)": n, "nível p5 (dBFS)": p5, "nível p50 (dBFS)": p50, "nível p95 (dBFS)": p95,
            "faixa dinâmica (dB)": p95 - p5, "ganho no CPA (dB)": np.median(meio) - np.median(bordas),
            "pico (s)": t_pico, "|pico − CPA| (s)": abs(t_pico - c),
            "centroide (Hz)": float((f * P).sum() / tot),
            "planura 0–5 kHz": float(np.exp(np.mean(np.log(P[m5] + 1e-30))) / np.mean(P[m5])), **fr_b}


def main():
    meta = pd.read_csv(os.path.join("src", "iara", "dataset_info", "iara.csv"), dtype={"ID": str})
    cpa = dict(zip(meta.ID.astype(int), pd.to_numeric(meta["CPA time"], errors="coerce")))
    files = sorted(sum((glob.glob(os.path.join("data", "iara", d, "*.wav")) for d in "ABCDE"), []))
    jobs = [(p, cpa.get(int(os.path.basename(p).split("-")[1].split(".")[0]), np.nan)) for p in files]
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(medir, jobs, chunksize=8))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).sort_values("File").to_csv(OUT, index=False)
    print(f"{len(rows)} gravações -> {OUT}")


if __name__ == "__main__":
    main()
