"""Espectro de TODAS as gravações de lanchas de prático classificadas erradas (MLP LOFAR, split por gravação, NB39 seção 9),
uma linha por gravação, separadas pela classe prevista (MEDIUM, LARGE). Complementa lancha_espectro.png (slide 25).
Classe prevista = a mais votada entre MEDIUM, LARGE e BACKGROUND nos folds em que a gravação foi testada."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "scratch")
from eval_ship_split_5x2cv import metadata
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "presentation/20261116-eda-iara-dados/figuras/"
COL = {"SMALL": "#2a78d6", "MEDIUM": "#eb6834", "LARGE": "#1baf7a"}
m = metadata()
F = (np.arange(1024) + 1) * 16000 / 2048
S = pd.read_pickle("data/iara_processed/cache_lofar_espectro_medio.pkl")
p = pd.read_csv("reports/small_medium/lanchas_pratico.csv", index_col=0)
p["g"] = np.where(p.acerto_id >= 0.6, "acertada", p[["pM_id", "pL_id", "pB_id"]].idxmax(1).map(
    {"pM_id": "MEDIUM", "pL_id": "LARGE", "pB_id": "BACKGROUND"}))
sel = F <= 400
band = (F >= 31) & (F < 86)
ok = S.loc[p.index[p.g == "acertada"]]
ref = {"lancha acertada (média)": (ok, COL["SMALL"], "-"),
       "MEDIUM, todas (média)": (S.loc[S.index.intersection(m.index[m.Target == 1])], COL["MEDIUM"], ":"),
       "LARGE, todas (média)": (S.loc[S.index.intersection(m.index[m.Target == 2])], COL["LARGE"], ":")}
med_ok = ok.loc[:, band].mean(1).median()

fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
for ax, alvo in zip(axs, ["MEDIUM", "LARGE"]):
    idx = p.index[p.g == alvo]; X = S.loc[idx]
    ax.axvspan(31, 86, color="#888", alpha=.1)
    for _, r in X.iterrows():
        ax.plot(F[sel], r.values[sel], color=COL[alvo], lw=.6, alpha=.22)
    ax.plot(F[sel], X.loc[:, sel].mean().values, color=COL[alvo], lw=2.6, label=f"errada → {alvo} (média)")
    for lab, (Y, c, ls) in ref.items():
        ax.plot(F[sel], Y.loc[:, sel].mean().values, color=c, lw=1.6, ls=ls, label=lab)
    e = X.loc[:, band].mean(1)
    acima = (e > med_ok).mean() * 100
    ax.set_title(f"Lanchas previstas {alvo}: {len(idx)} gravações de {p.loc[idx, 'Ship ID'].nunique()} lanchas",
                 fontsize=9.5)
    ax.text(0.98, 0.97, f"31–86 Hz acima da lancha\nacertada típica: {acima:.0f}% das gravações",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#444")
    ax.set_xlabel("frequência (Hz)"); ax.grid(alpha=.25, lw=.6); ax.set_ylim(0, 22)
axs[0].set_ylabel("LOFAR médio (dB acima do TPSW)")
h, l = axs[1].get_legend_handles_labels(); h0, l0 = axs[0].get_legend_handles_labels()
fig.legend([h0[0], h[0]] + h[1:], [l0[0], l[0]] + l[1:], fontsize=7.5, frameon=False, loc="lower center", ncol=5)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT + "lanchas_erradas.png", dpi=200); plt.close(fig)
n = p.g.value_counts(); print(n.to_dict())
print("gravações por lancha (erradas → LARGE):", p[p.g == "LARGE"].groupby("Ship ID").size().sort_values(ascending=False).to_dict())
print("gravações por lancha (erradas → MEDIUM):", p[p.g == "MEDIUM"].groupby("Ship ID").size().sort_values(ascending=False).to_dict())
