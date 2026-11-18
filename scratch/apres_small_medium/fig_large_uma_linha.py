"""LARGE: uma linha por gravação (LOFAR médio, dB acima do TPSW, 0–400 Hz), acertadas × erradas pela CNN LOFAR do artigo
(mesmos grupos do slide "LARGE Acertadas Têm uma Corcova Grave": acertada = acerta em todos os folds de teste;
errada = erra em todos; oscila = o resto, fora da figura). Mesmo estilo de fig_lanchas_erradas.py.
Rodar na raiz, no container iara-dev:  python scratch/apres_small_medium/fig_large_uma_linha.py
Saída: presentation/20261116-eda-iara-dados/figuras/large_uma_linha.png"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
sys.path.insert(0, "notebooks/src"); sys.path.insert(0, "scratch")
import iara_results as R  # noqa: E402
from eval_ship_split_5x2cv import metadata  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "presentation/20261116-eda-iara-dados/figuras/"
COL = {"acertada": "#1baf7a", "errada": "#d62728", "SMALL": "#2a78d6"}
m = metadata(); m.index = m.index.map(int)
F = (np.arange(1024) + 1) * 16000 / 2048
S = pd.read_pickle("data/iara_processed/cache_lofar_espectro_medio.pkl"); S.index = S.index.map(int)

d = pd.concat(R.file_predictions("cnn_lofar").values()); d = d[d.Target == 2]
t = d.groupby("File").apply(lambda x: (x.Prediction == x.Target).mean()); t.index = t.index.astype(int)
g = pd.Series(np.where(t == 1, "acertada", np.where(t == 0, "errada", "oscila")), index=t.index)
print(g.value_counts().to_dict())
sel = F <= 400
band = (F >= 31) & (F < 86)
small = S.loc[S.index.intersection(m.index[m.Target == 0])]
ref_small = small.loc[:, band].mean(1).median()

fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.7), sharey=True)
for ax, grp in zip(axs, ["acertada", "errada"]):
    X = S.loc[g.index[g == grp].intersection(S.index)]
    ax.axvspan(31, 86, color="#888", alpha=.1)
    for _, r in X.iterrows():
        ax.plot(F[sel], r.values[sel], color=COL[grp], lw=.5, alpha=.12)
    ax.plot(F[sel], X.loc[:, sel].mean().values, color=COL[grp], lw=2.6, label=f"LARGE {grp} (média)")
    ax.plot(F[sel], small.loc[:, sel].mean().values, color=COL["SMALL"], lw=1.6, ls=":", label="SMALL, todas (média)")
    e = X.loc[:, band].mean(1)
    ax.set_title(f"LARGE {grp}s pela CNN LOFAR: {len(X)} gravações", fontsize=9.5)
    ax.text(0.98, 0.97, f"31–86 Hz acima do SMALL típico\n({ref_small:.1f} dB): ".replace(".", ",") + f"{100 * (e > ref_small).mean():.0f}% das gravações",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#444")
    ax.set_xlabel("frequência (Hz)"); ax.grid(alpha=.25, lw=.6); ax.set_ylim(0, 22)
    print(grp, len(X), "acima do SMALL típico: %.0f%%" % (100 * (e > ref_small).mean()), "mediana 31-86: %.1f" % e.median())
axs[0].set_ylabel("LOFAR médio (dB acima do TPSW)")
h0, l0 = axs[0].get_legend_handles_labels(); h1, l1 = axs[1].get_legend_handles_labels()
fig.legend([h0[0], h1[0], h0[1]], [l0[0], l1[0], l0[1]], fontsize=8, frameon=False, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT + "large_uma_linha.png", dpi=200); plt.close(fig)
