"""Figuras da apresentação do experimento mix_small_medium (mesmos cálculos do NB40)."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "notebooks/src")
import generate_mix_small_medium_notebook as G
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "presentation/20261116-mix-small-medium/figuras/"; os.makedirs(OUT, exist_ok=True)
ns = {}
for src in (G.SETUP_CODE, G.INV_CODE, G.RES_CODE, G.CONF_CODE, G.DET_CODE):
    ns["display"] = lambda *a, **k: None
    exec(src.replace("plt.show()", "plt.close('all')"), ns)
CM, NOVO_M, d = ns["CM"], ns["NOVO_M"], ns["d"]

# 1. recall por classe nos dois splits
fig, ax = plt.subplots(figsize=(6.8, 3.2)); x = np.arange(3); w = 0.38
for i, (c, cor, rot) in enumerate([("split_id", "#9fc3ea", "split por gravação (artigo)"), ("split_navio", "#2a78d6", "split por navio")]):
    v = [NOVO_M[c][f"R_{CM[j]}"] for j in range(3)]
    ax.bar(x + (i - 0.5) * w, v, w * 0.92, color=cor, label=rot)
    for xi, vi in zip(x + (i - 0.5) * w, v):
        ax.text(xi, vi + 1.5, f"{vi:.0f}", ha="center", fontsize=9)
ax.axhline(100 / 3, color="#999", ls="--", lw=1); ax.text(2.55, 35.5, "acaso (33%)", fontsize=8, color="#666", ha="right")
ax.set_xticks(x); ax.set_xticklabels(list(CM.values())); ax.set_ylabel("recall (%)"); ax.set_ylim(0, 100)
ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=2); ax.grid(axis="y", alpha=.25, lw=.6)
fig.tight_layout(); fig.savefig(OUT + "recall_splits.png", dpi=200); plt.close(fig)

# 2. % prevista LARGE por comprimento do navio (split por navio)
nav = d[d.Target < 2].groupby("Ship ID").agg(L=("Length", "first"), n=("File", "nunique"), y=("Target", "first"),
                                              prevL=("prev", lambda s: (s == "LARGE").mean() * 100))
fig, ax = plt.subplots(figsize=(6.8, 3.2))
for y, cor, rot in [(0, "#2a78d6", "MIX_SMALL_MEDIUM verdadeira"), (1, "#1baf7a", "LARGE verdadeira")]:
    s = nav[nav.y == y]
    ax.scatter(s.L, s.prevL, s=6 + 8 * np.sqrt(s.n), color=cor, alpha=.5, edgecolor="white", lw=.4, label=rot)
bins = [8, 25, 50, 75, 100, 150, 200, 250, 350]
g = nav.groupby(pd.cut(nav.L, bins, right=False), observed=True).prevL.mean()
ax.plot([np.sqrt(i.left * i.right) for i in g.index], g.values, color="k", lw=2, marker="o", ms=5, label="média por faixa")
ax.axvline(100, color="#555", ls="--", lw=1); ax.text(104, 3, "100 m", fontsize=8)
ax.set_xscale("log"); ax.set_xticks([10, 20, 50, 100, 200, 300]); ax.set_xticklabels(["10", "20", "50", "100", "200", "300"])
ax.set_xlabel("comprimento do navio (m, eixo log)"); ax.set_ylabel("% prevista LARGE")
ax.legend(fontsize=7.5, frameon=False, loc="upper left"); ax.grid(alpha=.25, lw=.6)
fig.tight_layout(); fig.savefig(OUT + "fronteira_100m.png", dpi=200); plt.close(fig)
print("ok")
