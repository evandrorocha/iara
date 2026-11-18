"""
Extração PSD24K: espectro de potência do áudio original de 48 kHz, de 0 a 24 kHz, sem decimação (Tarefa 1).

Motivação: LOFAR (até 8 kHz, com passa-baixas em 6,4 kHz), lofar1hz (até 1 kHz) e DEMON (envelope de 1–2 kHz) não
olham a faixa de 8 a 24 kHz. Esta extração cobre o espectro inteiro gravado, para comparar tipos de navio que
soam iguais nas outras representações (ex.: Offshore Supply Ships × rebocadores).

Método (por gravação):
  - remoção da média; Welch com janela de Hann de 8192 amostras (0,171 s), 50% de sobreposição,
    escala de densidade espectral (unidade do arquivo²/Hz) -> faixas de 5,86 Hz de 0 a 24 kHz (4097 faixas);
  - o Welch é calculado em blocos de 10 s, o que dá também a variação ao longo da passagem.
Guarda, por gravação:
  - 'psd_db': média dos blocos, em dB (10·log10), SEM normalização (nível do arquivo, depende de ganho e distância);
  - 'blocos_db': os blocos de 10 s (blocos × faixas, float32) para análises de dinâmica.
Resumo de todas as gravações em data/iara_processed/cache_psd24k_resumo.pkl: {'freqs', 'psd_db'} (gravações × faixas).

Saídas: data/iara_processed/psd24k/<ID>.pkl + config.json; data/iara_processed/cache_psd24k_resumo.pkl
Uso (raiz do repositório): python scripts/extrair_psd24k.py [--ids 208 951] [--workers 6]
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import scipy.signal as sci

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOME = "psd24k"
OUT_DIR = os.path.join(ROOT, "data", "iara_processed", NOME)
RESUMO = os.path.join(ROOT, "data", "iara_processed", f"cache_{NOME}_resumo.pkl")
REF_DIR = os.path.join(ROOT, "data", "iara_processed", "lofar_bc1559516c522e2b993895bf5cb2ac94")
FS = 48000
NPERSEG = 8192
BLOCO_S = 10
FREQS = np.fft.rfftfreq(NPERSEG, 1 / FS)
CONFIG = {"nome": NOME, "descricao": "PSD de Welch do áudio original, 0–24 kHz, sem decimação nem normalização",
          "fs": FS, "janela": "hann", "nperseg": NPERSEG, "sobreposicao": NPERSEG // 2, "largura_faixa_hz": FS / NPERSEG,
          "faixas": len(FREQS), "bloco_s": BLOCO_S, "escala": "10·log10(densidade espectral), unidade do arquivo int16",
          "normalizacao": "nenhuma"}


def arquivos():
    return {int(os.path.basename(f)[-8:-4]): f for f in glob.glob(os.path.join(ROOT, "data", "iara", "*", "*.wav"))}


def extrai(i, caminho):
    fs, x = wav.read(caminho)
    assert fs == FS, f"{i}: fs = {fs}"
    x = (x[:, 0] if x.ndim > 1 else x).astype(np.float64)
    x -= x.mean()
    n = BLOCO_S * FS
    blocos = [sci.welch(x[a:a + n], fs=FS, window="hann", nperseg=NPERSEG, noverlap=NPERSEG // 2, scaling="density")[1]
              for a in range(0, len(x) - NPERSEG, n) if len(x[a:a + n]) >= NPERSEG]
    B = np.array(blocos)
    psd_db = 10 * np.log10(B.mean(0) + 1e-20)
    pd.to_pickle({"psd_db": psd_db.astype(np.float32), "blocos_db": (10 * np.log10(B + 1e-20)).astype(np.float32),
                  "freqs": FREQS}, os.path.join(OUT_DIR, f"{i}.pkl"))
    return i, psd_db


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
    res = pd.read_pickle(RESUMO) if os.path.exists(RESUMO) else {"freqs": FREQS, "psd_db": pd.DataFrame()}
    feitos = dict(res["psd_db"].T.items()) if len(res["psd_db"]) else {}
    faltam = [i for i in ids if i in wavs and not (i in feitos and os.path.exists(os.path.join(OUT_DIR, f"{i}.pkl")))]
    print(f"{NOME}: {len(faltam)} gravações a extrair", flush=True)

    def salva():
        pd.to_pickle({"freqs": FREQS, "psd_db": pd.DataFrame(feitos).T.sort_index().astype(np.float32)}, RESUMO)

    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        fut = [ex.submit(extrai, i, wavs[i]) for i in faltam]
        for n, f in enumerate(as_completed(fut), 1):
            i, v = f.result(); feitos[i] = v
            if n % 100 == 0 or n == len(fut):
                salva(); print(f"{n}/{len(fut)} ({time.time() - t0:.0f} s)", flush=True)
    salva()
    print("fim", flush=True)


if __name__ == "__main__":
    main()
