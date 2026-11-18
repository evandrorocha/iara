"""Figuras da apresentação de metadados ambientais (mesmos cálculos do NB41). --aug inclui o augmentation."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "notebooks/src")
import generate_metadados_ambientais_notebook as G
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "presentation/20261116-metadados-ambientais/figuras/"; os.makedirs(OUT, exist_ok=True)
AUG = "--aug" in sys.argv
ns = {}
for src in (G.SETUP_CODE, G.CLIMA_CODE, G.ERRO_CODE) + ((G.AUG_CODE,) if AUG else ()):
    ns["display"] = lambda *a, **k: None
    exec(src.replace("plt.show()", "plt.close('all')"), ns)
COR = {"SMALL": "#2a78d6", "MEDIUM": "#e8913a", "LARGE": "#1baf7a", "BACKGROUND": "#8a8a8a"}

# 1. estado do mar por classe
MD = ns["MAR_DIST"]
fig, ax = plt.subplots(figsize=(7.0, 2.9)); cols = [c for c in MD.columns if c <= 6]; x = np.arange(len(cols)); w = 0.2
for i, c in enumerate(MD.index):
    ax.bar(x + (i - 1.5) * w, MD.loc[c, cols].values, w * 0.92, color=COR[c], label=c)
ax.set_xticks(x); ax.set_xticklabels([str(c) for c in cols]); ax.set_xlabel("estado do mar (estimado pelo vento)")
ax.set_ylabel("% das gravações"); ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper right")
ax.grid(False); ax.grid(axis="y", alpha=.25, lw=.6)
fig.tight_layout(); fig.savefig(OUT + "mar_por_classe.png", dpi=200); plt.close(fig)

# 2. acerto de LARGE por CPA x mar (base, e com augmentation se houver)
if AUG:
    E = ns["EST_AUG"]
    cores = ["#8a8a8a", "#c9c9c9", "#9fc3ea", "#1baf7a"]
else:
    E = ns["EST_L"][["acerto"]].rename(columns={"acerto": "MLP mix_small_medium"}); cores = ["#1baf7a"]
fig, ax = plt.subplots(figsize=(7.0, 3.0)); lab = [f"{a}\n{b}" for a, b in E.index]; x = np.arange(len(lab))
w = 0.8 / E.shape[1]
for i, c in enumerate(E.columns):
    xs = x + (i - (E.shape[1] - 1) / 2) * w
    ax.bar(xs, E[c].values, w * 0.92, color=cores[i], label=c)
    for xi, v in zip(xs, E[c].values):
        ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.set_ylabel("acerto de LARGE (%)"); ax.set_ylim(0, 100)
ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right"); ax.grid(False); ax.grid(axis="y", alpha=.25, lw=.6)
fig.tight_layout(); fig.savefig(OUT + ("large_cpa_mar_aug.png" if AUG else "large_cpa_mar.png"), dpi=200); plt.close(fig)
print("ok")
