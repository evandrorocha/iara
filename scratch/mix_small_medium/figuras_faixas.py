"""Figuras da seção "Por que MIX_SMALL_MEDIUM e LARGE se separam" (mesmos cálculos do NB40, seção 6)."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "notebooks/src")
import generate_mix_small_medium_notebook as G
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "presentation/20261116-mix-small-medium/figuras/"; os.makedirs(OUT, exist_ok=True)
ns = {}
for src in (G.SETUP_CODE, G.FAIXA_CODE, G.PICO_CODE):
    ns["display"] = lambda *a, **k: None
    exec(src.replace("plt.show()", "plt.close('all')"), ns)
CF, TCF, PN, PASS = ns["CF"], ns["TCF"], ns["PICO_NAV"], ns["PASS"]
COR = {"SMALL": "#2a78d6", "MEDIUM": "#e8913a", "LARGE": "#1baf7a"}
xs = (CF.SL[:-1] + CF.SL[1:]) / 2

# 1. AUC ao longo da passagem, duas faixas
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
for ax, b, tit in zip(axs, ["37-58", "141-221"], ["37–58 Hz (graves)", "141–221 Hz"]):
    for k, cor, ls in [("SMALL × MEDIUM", "#8a8a8a", "-"), ("MEDIUM × LARGE", COR["LARGE"], "-"), ("SMALL × LARGE", COR["SMALL"], "--")]:
        y = TCF[k][TCF[k].banda == b].set_index("fatia").reindex(CF.SNAMES).AUC * 100
        ax.plot(xs, y, marker="o", ms=3.5, lw=1.8, color=cor, ls=ls, label=k)
    ax.axhline(50, color="k", lw=.8, ls=":"); ax.axvline(0, color="#999", lw=.8)
    ax.text(3, 14, "CPA", fontsize=8, color="#666"); ax.text(-148, 52, "empate", fontsize=8, color="#666")
    ax.set_title(tit, fontsize=10); ax.set_xlabel("tempo em relação ao CPA (s)"); ax.set_ylim(10, 95)
    ax.grid(alpha=.25, lw=.6)
axs[0].set_ylabel("% em que a 2ª classe\ntem mais energia")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT + "auc_passagem.png", dpi=200); plt.close(fig)

# 2. faixa de pico por classe (% dos navios)
fig, ax = plt.subplots(figsize=(7.0, 2.9)); x = np.arange(len(PN)); w = 0.27
for i, c in enumerate(["SMALL", "MEDIUM", "LARGE"]):
    ax.bar(x + (i - 1) * w, PN[c].values, w * 0.92, color=COR[c], label=c)
ax.set_xticks(x); ax.set_xticklabels([s.replace("-", "–") for s in PN.index], fontsize=8.5)
ax.set_xlabel("faixa de pico (Hz)"); ax.set_ylabel("% dos navios"); ax.legend(fontsize=8, frameon=False)
ax.grid(False); ax.grid(axis="y", alpha=.25, lw=.6)
fig.tight_layout(); fig.savefig(OUT + "pico_classes.png", dpi=200); plt.close(fig)

# 3. energia em 37-58 Hz ao longo da passagem
fig, ax = plt.subplots(figsize=(5.2, 2.9))
for c in ["SMALL", "MEDIUM", "LARGE"]:
    ax.plot(xs, PASS["37-58"].loc[c].values, marker="o", ms=3.5, lw=1.8, color=COR[c], label=c)
ax.axvline(0, color="#999", lw=.8); ax.text(3, 1.2, "CPA", fontsize=8, color="#666")
ax.set_xlabel("tempo em relação ao CPA (s)"); ax.set_ylabel("dB acima do fundo\n(mediana dos navios)")
ax.set_title("37–58 Hz, gravações com CPA (A e C)", fontsize=10); ax.set_ylim(0, 11)
ax.legend(fontsize=8, frameon=False, loc="center right"); ax.grid(alpha=.25, lw=.6)
fig.tight_layout(); fig.savefig(OUT + "energia_passagem.png", dpi=200); plt.close(fig)
print("ok")
