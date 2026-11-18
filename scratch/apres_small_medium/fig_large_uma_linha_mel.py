"""LARGE no MEL: uma linha por gravação (MEL sem normalização, dB relativo ao máximo da gravação, 20 Hz–6 kHz, eixo log),
acertadas × erradas pela CNN LOFAR do artigo (mesmos grupos do slide "LARGE: Uma Linha por Gravação" com LOFAR).
Medida-resumo: grave − agudo = média de 31–86 Hz menos média de 1–6 kHz (dB), comparada ao SMALL típico (mediana).
Rodar na raiz, no container iara-dev:  python scratch/apres_small_medium/fig_large_uma_linha_mel.py
Saída: presentation/20261116-eda-iara-dados/figuras/large_uma_linha_mel.png"""
import sys
import numpy as np, pandas as pd, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
import librosa  # noqa: E402
from scipy import stats as st  # noqa: E402
sys.path.insert(0, "notebooks/src"); sys.path.insert(0, "scratch")
import iara_results as R  # noqa: E402
from eval_ship_split_5x2cv import metadata  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "presentation/20261116-eda-iara-dados/figuras/"
COL = {"acertada": "#1baf7a", "errada": "#d62728", "SMALL": "#2a78d6"}
m = metadata(); m.index = m.index.map(int)
FM = librosa.mel_frequencies(n_mels=258, fmin=0, fmax=8000)[1:-1]
S = pd.read_pickle("data/iara_processed/cache_mel_espectro_medio.pkl")["sem normalização"]; S.index = S.index.map(int)

d = pd.concat(R.file_predictions("cnn_lofar").values()); d = d[d.Target == 2]
acc = d.assign(ok=d.Prediction == d.Target).groupby("File").ok.mean(); acc.index = acc.index.astype(int)
g = pd.Series(np.where(acc == 1, "acertada", np.where(acc == 0, "errada", "oscila")), index=acc.index)
print(g.value_counts().to_dict())

sel = (FM >= 20) & (FM <= 6000)
grave, agudo = (FM >= 31) & (FM < 86), (FM >= 1000) & (FM < 6000)
ga = S.loc[:, grave].mean(1) - S.loc[:, agudo].mean(1)
small = S.loc[S.index.intersection(m.index[m.Target == 0])]
ref = ga.loc[small.index].median()
A = g.index[g == "acertada"].intersection(S.index); E = g.index[g == "errada"].intersection(S.index)
u = st.mannwhitneyu(ga.loc[A], ga.loc[E])
print(f"SMALL típico grave−agudo {ref:.1f} dB | acertadas mediana {ga.loc[A].median():.1f}, acima {100 * (ga.loc[A] > ref).mean():.0f}% | "
      f"erradas mediana {ga.loc[E].median():.1f}, acima {100 * (ga.loc[E] > ref).mean():.0f}% | AUC acertada>errada {u.statistic / len(A) / len(E):.2f} p {u.pvalue:.1g}")
for nome, msk in (("31–86 Hz", grave), ("1–6 kHz", agudo)):
    a, e = S.loc[A, msk].mean(1), S.loc[E, msk].mean(1)
    uu = st.mannwhitneyu(a, e)
    print(f"  {nome}: acertadas {a.median():.1f} × erradas {e.median():.1f} dB, AUC acertada>errada {uu.statistic / len(a) / len(e):.2f}")

fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.7), sharey=True)
tk = [20, 50, 100, 200, 500, 1000, 2000, 5000]
for ax, grp, idx in zip(axs, ["acertada", "errada"], [A, E]):
    X = S.loc[idx]
    ax.axvspan(31, 86, color="#888", alpha=.1)
    for _, r in X.iterrows():
        ax.plot(FM[sel], r.values[sel], color=COL[grp], lw=.5, alpha=.12)
    ax.plot(FM[sel], X.mean().values[sel], color=COL[grp], lw=2.6, label=f"LARGE {grp} (média)")
    ax.plot(FM[sel], small.mean().values[sel], color=COL["SMALL"], lw=1.6, ls=":", label="SMALL, todas (média)")
    ax.set_title(f"LARGE {grp}s pela CNN LOFAR: {len(X)} gravações", fontsize=9.5)
    ax.text(0.98, 0.97, f"grave − agudo acima do SMALL típico\n({ref:.1f} dB): ".replace(".", ",") +
            f"{100 * (ga.loc[idx] > ref).mean():.0f}% das gravações", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="#444")
    ax.set_xscale("log"); ax.set_xlim(20, 6000); ax.set_xticks(tk)
    ax.set_xticklabels([f"{x:,}".replace(",", ".") for x in tk]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("frequência (Hz, escala logarítmica)"); ax.grid(alpha=.25, lw=.6); ax.set_ylim(-80, -5)
axs[0].set_ylabel("MEL médio (dB rel. ao máximo)")
h0, l0 = axs[0].get_legend_handles_labels(); h1, l1 = axs[1].get_legend_handles_labels()
fig.legend([h0[0], h1[0], h0[1]], [l0[0], l1[0], l0[1]], fontsize=8, frameon=False, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT + "large_uma_linha_mel.png", dpi=200); plt.close(fig)
