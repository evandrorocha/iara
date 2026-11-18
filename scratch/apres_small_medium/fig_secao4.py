"""Figuras das seções 4–7 da apresentação 20261116-eda-iara-dados (mesmos cálculos do NB39, seções 7, 9 e 10)."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "notebooks/src"); sys.path.insert(0, "scratch")
import iara_results as R
from eval_ship_split_5x2cv import metadata
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "presentation/20261116-eda-iara-dados/figuras/"
COL = {"SMALL": "#2a78d6", "MEDIUM": "#eb6834", "LARGE": "#1baf7a"}
m = metadata(); m["L"] = pd.to_numeric(m.Length, errors="coerce")

# 1. Δ ACC por modelo e variante
T = pd.read_csv("reports/small_medium/aug_modelos_resumo.csv")
VS = ["dup", "stretch", "mask", "mixnavio", "ruido", "bg"]; MODS = {"mlp": "MLP", "rf": "RF", "svm2000": "SVM", "bigru": "BiGRU"}
D = T[T.variante.isin(VS)].pivot(index="variante", columns="modelo", values="dACC").reindex(VS)[list(MODS)]
fig, ax = plt.subplots(figsize=(7.2, 3.3)); w = 0.2; x = np.arange(len(VS))
for i, (k, c) in enumerate(zip(MODS, ["#2a78d6", "#1baf7a", "#eb6834", "#8a5cc2"])):
    ax.bar(x + (i - 1.5) * w, D[k].fillna(0), w * 0.88, color=c, label=MODS[k])
ax.axhline(0, color="k", lw=1); ax.axhspan(1.5, 2.6, color="green", alpha=0.08)
ax.text(5.45, 1.75, "critério de ganho real (≥ +1,5)", fontsize=8, color="green", ha="right")
ax.annotate("RF + bg: −2,6 (p 0,047)\nBACKGROUND 72→62%", (5 - 0.1, -2.61), xytext=(3.3, -2.6), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#555"), va="center")
ax.set_xticks(x); ax.set_xticklabels(["dup\n(controle)"] + VS[1:]); ax.set_ylabel("Δ ACC vs. base (pontos)")
ax.set_ylim(-3.2, 2.6); ax.legend(ncol=4, fontsize=8, loc="lower left", frameon=False); ax.grid(axis="y", alpha=.25, lw=.6)
fig.tight_layout(); fig.savefig(OUT + "aug_modelos.png", dpi=200); plt.close(fig)

# 2. Espectro das lanchas
F = (np.arange(1024) + 1) * 16000 / 2048
S = pd.read_pickle("data/iara_processed/cache_lofar_espectro_medio.pkl")
p = pd.read_csv("reports/small_medium/lanchas_pratico.csv", index_col=0)
p["g"] = np.where(p.acerto_id >= 0.6, "acertada", np.where(p.pL_id > p.pM_id, "errada → LARGE", "errada → MEDIUM"))
sel = F <= 400
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.axvspan(31, 86, color="#888", alpha=.1); ax.text(58, 10.6, "31–86 Hz\n(banda LARGE)", ha="center", fontsize=7.5, color="#52514e", va="top")
for lab, idx, c, ls, lw in [("lancha acertada", p[p.g == "acertada"].index, COL["SMALL"], "-", 2.2),
                             ("lancha errada → MEDIUM", p[p.g == "errada → MEDIUM"].index, COL["MEDIUM"], "-", 2.2),
                             ("lancha errada → LARGE", p[p.g == "errada → LARGE"].index, COL["LARGE"], "-", 2.2),
                             ("MEDIUM (todas)", m[m.Target == 1].index, COL["MEDIUM"], ":", 1.5),
                             ("LARGE (todas)", m[m.Target == 2].index, COL["LARGE"], ":", 1.5)]:
    ax.plot(F[sel], S.loc[S.index.intersection(idx), sel].mean().values, color=c, ls=ls, lw=lw, label=lab)
ax.set_xlabel("frequência (Hz)"); ax.set_ylabel("LOFAR médio (dB acima do TPSW)"); ax.set_ylim(1.5, 11)
ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=.25, lw=.6)
fig.tight_layout(); fig.savefig(OUT + "lancha_espectro.png", dpi=200); plt.close(fig)

# 3. Fronteira de comprimento
N = pd.read_csv("reports/small_medium/fronteira_comprimento.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
for a, (c, col, nome, outra) in zip(axs, [(0, "M", "SMALL", "MEDIUM"), (1, "S", "MEDIUM", "SMALL")]):
    s = N[N.Target == c]
    a.scatter(s.L, s[col], s=10 + 4 * s.n, color=COL[nome], alpha=.55, edgecolor="white", lw=.5)
    b = pd.cut(s.L, [0, 12.5, 20, 30, 50] if c == 0 else [50, 65, 75, 85, 100], right=False)
    g = s.groupby(b, observed=True)[col].mean(); xm = [(i.left + i.right) / 2 for i in g.index]
    a.plot(xm, g.values, color="k", lw=2, marker="o", ms=5, label="média por faixa")
    a.set_title(f"{nome} verdadeiras: % prevista {outra}", fontsize=9.5); a.set_xlabel("comprimento do navio (m)")
    a.grid(alpha=.25, lw=.6)
axs[0].set_ylabel("% das gravações do navio"); axs[0].legend(fontsize=8, frameon=False, loc="upper left")
axs[0].set_xlim(0, 52); axs[1].set_xlim(48, 102); axs[0].axvline(50, color="#999", ls="--", lw=1); axs[1].axvline(50, color="#999", ls="--", lw=1)
fig.tight_layout(); fig.savefig(OUT + "fronteira_comprimento.png", dpi=200); plt.close(fig)
print("ok")
