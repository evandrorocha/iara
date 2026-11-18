"""
Extração DEMON_MULTIBANDA: DEMON calculado em cinco bandas de cavitação (Tarefa 1).

O DEMON do projeto (data/iara_processed/demon_features) usa só a banda de 1–2 kHz. Hélices de tamanho, passo e
regime diferentes podem cavitar com mais energia em outras bandas, e a modulação pode aparecer melhor nelas.
Aqui o DEMON é recalculado, com a MESMA função do pipeline (src/iara/processing/demon.py: compute_demon, mesmos
parâmetros do demon_features: n_fft = 512, fs final = 50 Hz, 50% de sobreposição, envelope |x|, descarta 4 faixas
de DC), em cinco bandas: 0,5–1, 1–2, 2–4, 4–8 e 8–16 kHz. A banda de 1–2 kHz serve de controle (deve reproduzir
o demon_features).

Guarda, por gravação e banda: o espectro médio linear ('avg') e, em dB acima do fundo TPSW (mesmo TPSW do HCNN
DEMON: ±16 faixas sem as ±3 centrais), o espectro médio ('rel_db') e a persistência > 6 dB após mediana de
5 janelas ('pers6').

Saídas: data/iara_processed/demon_multibanda/<ID>.pkl + config.json;
        data/iara_processed/cache_demon_multibanda_resumo.pkl: {'freqs', 'bandas', 'rel_db': {banda: DataFrame},
        'pers6': {banda: DataFrame}} (gravações × 253 faixas).
Uso (raiz do repositório): python scripts/extrair_demon_multibanda.py [--ids 208 951] [--workers 6]
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
import scipy.ndimage as ndi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from iara.processing.demon import compute_demon  # noqa: E402

NOME = "demon_multibanda"
OUT_DIR = os.path.join(ROOT, "data", "iara_processed", NOME)
RESUMO = os.path.join(ROOT, "data", "iara_processed", f"cache_{NOME}_resumo.pkl")
REF_DIR = os.path.join(ROOT, "data", "iara_processed", "lofar_bc1559516c522e2b993895bf5cb2ac94")
BANDAS = {"0,5–1 kHz": (500, 1000), "1–2 kHz": (1000, 2000), "2–4 kHz": (2000, 4000),
          "4–8 kHz": (4000, 8000), "8–16 kHz": (8000, 16000)}
PARAMS = dict(n_fft=512, max_freq=50, overlap_ratio=0.5, method="abs", discard_dc_bins=4)
CONFIG = {"nome": NOME, "descricao": "DEMON em cinco bandas de cavitação, com a função compute_demon do pipeline",
          "bandas_hz": BANDAS, **PARAMS, "tpsw": {"n": 16, "g": 3, "alpha": 2.0}, "persistencia": "> 6 dB, mediana de 5 janelas"}


def fundo_tpsw(x, n=16, g=3, alpha=2.0):
    k = np.ones(2 * n + 1); k[n - g:n + g + 1] = 0; k /= k.sum()
    b1 = ndi.convolve1d(x, k, axis=-1, mode="reflect")
    return ndi.convolve1d(np.where(x > alpha * b1, b1, x), k, axis=-1, mode="reflect")


def arquivos():
    return {int(os.path.basename(f)[-8:-4]): f for f in glob.glob(os.path.join(ROOT, "data", "iara", "*", "*.wav"))}


def extrai(i, caminho):
    fs, x = wav.read(caminho)
    x = x[:, 0] if x.ndim > 1 else x
    out, freqs = {}, None
    for nome, pb in BANDAS.items():
        r = compute_demon(x, fs=fs, passband=pb, **PARAMS)
        freqs = r["frequencies"]
        avg = r["avg_spectrum"]; W = r["waterfall"].T                      # janelas × faixas
        Wm = ndi.median_filter(W, size=(5, 1), mode="nearest")
        out[nome] = {"avg": avg.astype(np.float32),
                     "rel_db": (20 * np.log10(avg / (fundo_tpsw(avg) + 1e-12))).astype(np.float32),
                     "pers6": (20 * np.log10(Wm / (fundo_tpsw(Wm) + 1e-12)) > 6).mean(0).astype(np.float32)}
    pd.to_pickle({"freqs": freqs, "bandas": out}, os.path.join(OUT_DIR, f"{i}.pkl"))
    return i, freqs, out


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
    rel = {b: {} for b in BANDAS}; pers = {b: {} for b in BANDAS}; freqs = None
    if os.path.exists(RESUMO):
        r0 = pd.read_pickle(RESUMO); freqs = r0["freqs"]
        for b in BANDAS:
            rel[b] = dict(r0["rel_db"][b].T.items()); pers[b] = dict(r0["pers6"][b].T.items())
    faltam = [i for i in ids if i in wavs and not (i in rel["1–2 kHz"] and os.path.exists(os.path.join(OUT_DIR, f"{i}.pkl")))]
    print(f"{NOME}: {len(faltam)} gravações a extrair", flush=True)

    def salva():
        pd.to_pickle({"freqs": freqs, "bandas": list(BANDAS),
                      "rel_db": {b: pd.DataFrame(rel[b]).T.sort_index() for b in BANDAS},
                      "pers6": {b: pd.DataFrame(pers[b]).T.sort_index() for b in BANDAS}}, RESUMO)

    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        fut = [ex.submit(extrai, i, wavs[i]) for i in faltam]
        for n, f in enumerate(as_completed(fut), 1):
            i, fr, out = f.result(); freqs = fr
            for b in BANDAS:
                rel[b][i] = out[b]["rel_db"]; pers[b][i] = out[b]["pers6"]
            if n % 100 == 0 or n == len(fut):
                salva(); print(f"{n}/{len(fut)} ({time.time() - t0:.0f} s)", flush=True)
    salva()
    print("fim", flush=True)


if __name__ == "__main__":
    main()
