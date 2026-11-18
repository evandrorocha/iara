"""MEL e HHT na classe SMALL: existe alguma pista POSITIVA de SMALL (contra MEDIUM, LARGE e BACKGROUND)?

Usa os caches do NB37 (cache_mel_espectro_medio.pkl, cache_hht_resumo.pkl): espectro médio de cada gravação.
Representações (todas independentes do ganho):
  MEL L2            entrada dos modelos MEL (256 bandas até 8 kHz, L2 por janela)
  MEL forma         MEL em dB (ref = máximo da gravação) menos a média da gravação
  HHT 1D forma      log do espectro marginal de Hilbert (0–1 kHz) menos a média
  HHT 2D intermit.  coeficiente de variação da energia em passos de 80 ms (0–1 kHz); ~ energia relativa ao pico
Unidade: navio (média das gravações). Mann-Whitney faixa a faixa + FDR (BH), juntando A–D e dentro de A, C, D.
BACKGROUND: cada gravação é uma unidade, só no subconjunto E (confundido com equipamento/local).

Saídas: reports/small_medium/mel_hht_small_faixas.csv  e  presentation/20261116-classe-small/figuras/mel_hht_classes.png
Rodar na raiz, no container iara-dev:  python scratch/apres_small/mel_hht_small.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, false_discovery_control  # noqa: E402

sys.path.insert(0, "scratch")
from eval_ship_split_5x2cv import metadata  # noqa: E402

PROC = "data/iara_processed"
OUT_FIG = "presentation/20261116-classe-small/figuras/mel_hht_classes.png"
OUT_CSV = "reports/small_medium/mel_hht_small_faixas.csv"
CL = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}
COL = {"SMALL": "#2a78d6", "MEDIUM": "#eb6834", "LARGE": "#1baf7a", "BACKGROUND": "#8a8a8a"}

m = metadata(); m.index = m.index.map(int)
m["classe"] = m.Target.map(CL); m["ship"] = m["Ship ID"].astype(str)
m.loc[m.Target == 3, "ship"] = "bg" + m.index[m.Target == 3].astype(str)

mel = pd.read_pickle(os.path.join(PROC, "cache_mel_espectro_medio.pkl"))
hht = pd.read_pickle(os.path.join(PROC, "cache_hht_resumo.pkl"))
import librosa  # noqa: E402
FM = librosa.mel_frequencies(n_mels=258, fmin=0, fmax=8000)[1:-1]
F1 = np.linspace(0, 1000, 257)[:-1] + 1000 / 512
F2 = np.linspace(0, 1000, 129)[:-1] + 1000 / 256
_ms = mel["sem normalização"]
REPS = {"MEL L2 (entrada dos modelos)": (mel["L2 (modelos)"], FM),
        "MEL forma": (_ms.sub(_ms.mean(1), axis=0), FM),
        "HHT 1D forma": (hht["hht1d_log"].sub(hht["hht1d_log"].mean(1), axis=0), F1),
        "HHT 2D intermitência": (hht["hht2d_cv"], F2)}


def nav(X, ids):
    ids = [i for i in ids if i in X.index]
    return X.loc[ids].groupby(m.loc[ids, "ship"].values).mean()


def ids(classe, ds=None):
    c = m[m.classe == classe]
    return c.index if ds is None else c.index[c.Dataset == ds]


rows = []
for nome, (X, F) in REPS.items():
    for a, b in (("SMALL", "MEDIUM"), ("SMALL", "LARGE"), ("MEDIUM", "LARGE"), ("SMALL", "BACKGROUND"), ("LARGE", "BACKGROUND")):
        for ds in ((None,) if b == "BACKGROUND" else (None, "A", "C", "D")):
            A, B = nav(X, ids(a, ds)), nav(X, ids(b, None if b == "BACKGROUND" else ds))
            ok = (A.notna().all() & B.notna().all()).values
            U, p = mannwhitneyu(A.loc[:, ok].values, B.loc[:, ok].values, axis=0)
            q, auc = false_discovery_control(p), U / (len(A) * len(B))
            rows.append({"representação": nome, "comparação": f"{a} × {b}", "subconjunto": ds or "todos",
                         "navios": f"{len(A)} × {len(B)}", "faixas": int(ok.sum()),
                         "q<0,05": int((q < .05).sum()),
                         f"q<0,05 e {a} maior": int(((q < .05) & (auc > .5)).sum()),
                         "AUC mín": auc.min(), "AUC máx": auc.max(),
                         "_auc": auc, "_f": F[ok]})
T = pd.DataFrame(rows)
# "SMALL tem mais" (AUC > 0,5) contra LARGE e MEDIUM tem pista própria se não aparecer igual em MEDIUM × LARGE
sl = T[(T.comparação == "SMALL × LARGE") & (T.subconjunto == "todos")].set_index("representação")
ml = T[(T.comparação == "MEDIUM × LARGE") & (T.subconjunto == "todos")].set_index("representação")
dif, cor = {}, {}
for r in REPS:
    a_ = pd.Series(sl.loc[r, "_auc"], index=sl.loc[r, "_f"]); b_ = pd.Series(ml.loc[r, "_auc"], index=ml.loc[r, "_f"])
    c_ = a_.index.intersection(b_.index); a_, b_ = a_.loc[c_], b_.loc[c_]
    dif[r], cor[r] = float((a_ - b_).abs().max()), float(np.corrcoef(a_, b_)[0, 1])
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
cols = [c for c in T.columns if not c.startswith("_")]
T[cols].to_csv(OUT_CSV, index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print(T[cols].round(2).to_string())
print("\nAUC SMALL×LARGE vs AUC MEDIUM×LARGE, faixa a faixa (todos os subconjuntos):")
for r in REPS:
    print(f"  {r}: correlação {cor[r]:.2f}, maior diferença {dif[r]:.2f}")

# ---------------------------------------------------------------- figura
fig, axs = plt.subplots(2, 4, figsize=(13, 5.4), constrained_layout=True,
                        gridspec_kw={"height_ratios": [1.6, 1]})
TIT = {"MEL L2 (entrada dos modelos)": "MEL L2 (entrada dos modelos)", "MEL forma": "MEL: forma (dB − média)",
       "HHT 1D forma": "HHT 1D: forma (log − média)", "HHT 2D intermitência": "HHT 2D: intermitência (CV)"}
for j, (nome, (X, F)) in enumerate(REPS.items()):
    ax = axs[0, j]
    for c in ("BACKGROUND", "LARGE", "MEDIUM", "SMALL"):
        g = nav(X, ids(c))
        ok = g.notna().all().values & (F <= 830 if "intermit" in nome else True)
        md, q1, q3 = g.median().values[ok], g.quantile(.25).values[ok], g.quantile(.75).values[ok]
        ax.plot(F[ok], md, color=COL[c], lw=1.4 if c == "SMALL" else 1.1, label=c)
        if c in ("SMALL", "LARGE"):
            ax.fill_between(F[ok], q1, q3, color=COL[c], alpha=.12, lw=0)
    ax.set_title(TIT[nome], fontsize=9)
    if nome.startswith("MEL"):
        ax.set_xscale("symlog", linthresh=100); ax.set_xlim(20, 8000)
    else:
        ax.set_xlim(0, 830 if "intermit" in nome else 1000)
    ax.grid(alpha=.25)
    ax2 = axs[1, j]
    for comp, cc, ls in (("SMALL × LARGE", COL["SMALL"], "-"), ("MEDIUM × LARGE", COL["MEDIUM"], "--"),
                         ("SMALL × MEDIUM", "k", "-")):
        r = T[(T.representação == nome) & (T.comparação == comp) & (T.subconjunto == "todos")].iloc[0]
        ax2.plot(r["_f"], r["_auc"], color=cc, ls=ls, lw=1.1, label=comp)
    ax2.axhline(.5, color="gray", lw=.8); ax2.set_ylim(0, 1); ax2.set_xlabel("frequência (Hz)", fontsize=8)
    if nome.startswith("MEL"):
        ax2.set_xscale("symlog", linthresh=100); ax2.set_xlim(20, 8000)
    else:
        ax2.set_xlim(0, 830 if "intermit" in nome else 1000)
    ax2.grid(alpha=.25)
axs[0, 0].legend(fontsize=7, loc="lower left"); axs[1, 0].legend(fontsize=7, loc="lower left")
axs[1, 0].set_ylabel("AUC (1º grupo > 2º)", fontsize=8)
for a in axs.flat:
    a.tick_params(labelsize=7)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.savefig(OUT_FIG, dpi=170)
print("figura:", OUT_FIG)
