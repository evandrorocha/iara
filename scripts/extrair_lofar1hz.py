"""
Extração LOFAR1HZ: LOFAR de baixa frequência com faixas de ~1 Hz (0–1 kHz), para separar banda larga de raias
próximas (Tarefa 1, NB35). NÃO substitui o LOFAR do pipeline (lofar_bc15… / lofar_c5de…, faixas de 7,8 Hz até 8 kHz).

Diferenças em relação ao LOFAR do pipeline (src/iara/processing/analysis.py):
  - decimação 48 kHz -> 2 kHz (÷24) com resample_poly (filtro FIR polifásico; um Chebyshev de ordem 8 em 0,8/24 é
    numericamente instável na forma (b, a)). Espectro até 1 kHz; o filtro começa a atenuar perto de 1 kHz,
    então use 0–900 Hz;
  - FFT de 2048 pontos a 2 kHz: janela de 1,024 s, faixas de 0,977 Hz (1024 faixas de 0,98 Hz a 1 kHz, sem o DC);
  - sobreposição de 50%: uma linha a cada 0,512 s, o mesmo passo das linhas do LOFAR do pipeline. Sem integração;
  - TPSW com a mesma largura EM Hz do pipeline: ±168 faixas (±164 Hz), sem a faixa central e as 24 vizinhas de cada
    lado (±23 Hz). Com os parâmetros padrão (±21 faixas = ±21 Hz), uma banda de 55 Hz seria tratada como fundo;
  - o resto é igual: Hanning, magnitude * nfft/2, 20·log10, LOFAR = log − TPSW, valores < −0,2 viram 0, sem normalização.

Saídas:
  data/iara_processed/lofar1hz/<ID>.pkl   {'df': DataFrame float32 (linhas × 1024 faixas), 'times': list}
  data/iara_processed/lofar1hz/config.json
  data/iara_processed/cache_lofar1hz_resumo.pkl   {'freqs', 'lofar_medio', 'log_medio', 'fundo_medio', 'pers6'}
      (espectro médio do LOFAR, do log-espectro e do fundo TPSW, e persistência > 6 dB, por gravação)

Uso (raiz do repositório):
  python scripts/extrair_lofar1hz.py                 # todas as gravações (pula as já extraídas)
  python scripts/extrair_lofar1hz.py --ids 208 951   # só algumas (piloto)
  python scripts/extrair_lofar1hz.py --workers 6
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import scipy.signal as sci

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from iara.processing.analysis import tpsw  # noqa: E402

NOME = "lofar1hz"
OUT_DIR = os.path.join(ROOT, "data", "iara_processed", NOME)
RESUMO = os.path.join(ROOT, "data", "iara_processed", f"cache_{NOME}_resumo.pkl")
REF_DIR = os.path.join(ROOT, "data", "iara_processed", "lofar_bc1559516c522e2b993895bf5cb2ac94")

FS_IN, DEC = 48000, 24
FS = FS_IN / DEC                        # 2000 Hz
NFFT = 2048                             # 1,024 s, 0,977 Hz por faixa
HOP = NFFT // 2                         # 50% de sobreposição -> 0,512 s entre linhas
DF = FS / NFFT
TPSW_N = int(round(164.0625 / DF))      # 168 faixas = ±164 Hz (igual ao pipeline: 21 × 7,8125 Hz)
TPSW_P = 25                             # zeros: central ± 24 faixas = ±23,4 Hz (pipeline: central ± 3 × 7,8125 Hz)
FREQS = (np.arange(1, NFFT // 2 + 1)) * DF
CONFIG = {"nome": NOME, "descricao": "LOFAR de baixa frequência, faixas de ~1 Hz, 0–1 kHz (Tarefa 1)",
          "fs_original": FS_IN, "decimacao": DEC, "metodo_decimacao": "scipy.signal.resample_poly(x, 1, 24)",
          "fs": FS, "nfft": NFFT, "janela": "np.hanning(nfft)", "duracao_janela_s": NFFT / FS,
          "sobreposicao": NFFT - HOP, "passo_s": HOP / FS, "largura_faixa_hz": DF, "faixas": NFFT // 2,
          "faixa_util_hz": [DF, 900], "tpsw_n": TPSW_N, "tpsw_p": TPSW_P, "tpsw_a": 2.0,
          "corte": "valores < -0.2 -> 0", "normalizacao": "nenhuma", "dtype": "float32"}


def arquivos():
    return {int(os.path.basename(f)[-8:-4]): f for f in glob.glob(os.path.join(ROOT, "data", "iara", "*", "*.wav"))}


def extrai(i, caminho):
    destino = os.path.join(OUT_DIR, f"{i}.pkl")
    fs0, x = wav.read(caminho)
    assert fs0 == FS_IN, f"{i}: fs = {fs0}"
    x = (x[:, 0] if x.ndim > 1 else x).astype(np.float64)
    x = x - x.mean()
    xd = sci.resample_poly(x, 1, DEC)
    _, t, Z = sci.spectrogram(xd, fs=FS, nfft=NFFT, window=np.hanning(NFFT), noverlap=NFFT - HOP,
                              detrend=False, scaling="spectrum", mode="complex")
    P = (np.abs(Z) * NFFT / 2)[1:]
    P[P < 1e-9] = 1e-9
    LOG = 20 * np.log10(P)                                  # faixas × janelas
    BG = tpsw(LOG, n=TPSW_N, p=TPSW_P)
    L = LOG - BG
    L[L < -0.2] = 0
    df = pd.DataFrame(L.T.astype(np.float32), columns=[f"f {k}" for k in range(L.shape[0])])
    pd.to_pickle({"df": df, "times": list(t)}, destino)
    return i, L.mean(1), LOG.mean(1), BG.mean(1), (L > 6).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4, ensure_ascii=False)

    wavs = arquivos()
    ids = a.ids or sorted(int(f[:-4]) for f in os.listdir(REF_DIR) if f.endswith(".pkl"))
    faltam = [i for i in ids if i in wavs]
    sem_wav = sorted(set(ids) - set(wavs))
    if sem_wav:
        print(f"{len(sem_wav)} IDs sem .wav: {sem_wav[:10]}")

    res = pd.read_pickle(RESUMO) if os.path.exists(RESUMO) else {"freqs": FREQS}
    tabs = {k: dict(res[k].T.items()) if k in res else {} for k in ("lofar_medio", "log_medio", "fundo_medio", "pers6")}
    faltam = [i for i in faltam if not (os.path.exists(os.path.join(OUT_DIR, f"{i}.pkl")) and i in tabs["lofar_medio"])]
    print(f"{NOME}: {len(faltam)} gravações a extrair (Δf = {DF:.4f} Hz, TPSW n = {TPSW_N}, p = {TPSW_P})", flush=True)

    def salva():
        out = {"freqs": FREQS}
        for k, v in tabs.items():
            out[k] = pd.DataFrame(v).T.sort_index().astype(np.float32)
        pd.to_pickle(out, RESUMO)

    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        fut = [ex.submit(extrai, i, wavs[i]) for i in faltam]
        for n, f in enumerate(as_completed(fut), 1):
            i, lm, gm, bm, pm = f.result()
            for k, v in zip(tabs, (lm, gm, bm, pm)):
                tabs[k][i] = v
            if n % 50 == 0 or n == len(fut):
                salva()
                print(f"{n}/{len(fut)} ({time.time() - t0:.0f} s)", flush=True)
    salva()
    print("fim", flush=True)


if __name__ == "__main__":
    main()
