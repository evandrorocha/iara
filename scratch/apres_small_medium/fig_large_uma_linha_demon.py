"""LARGE no DEMON: uma linha por gravação (DEMON 1–2 kHz, dB acima do fundo TPSW ao longo da frequência de modulação,
0,4–25 Hz; mesmo cálculo de fig_lanchas_demon.py), acertadas × erradas pela CNN LOFAR do artigo (mesmos grupos dos slides
"LARGE: Uma Linha por Gravação" com LOFAR e MEL).
Medida-resumo: força da linha mais forte (máximo acima do TPSW entre 1 e 25 Hz), comparada ao SMALL típico (mediana).
Rodar na raiz, no container iara-dev:  python scratch/apres_small_medium/fig_large_uma_linha_demon.py
Saída: presentation/20261116-eda-iara-dados/figuras/large_uma_linha_demon.png"""
import sys
import numpy as np, pandas as pd, scipy.ndimage as ndi, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats as st  # noqa: E402
sys.path.insert(0, "notebooks/src"); sys.path.insert(0, "scratch")
import iara_results as R  # noqa: E402
from eval_ship_split_5x2cv import metadata  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "presentation/20261116-eda-iara-dados/figuras/"
COL = {"acertada": "#1baf7a", "errada": "#d62728", "SMALL": "#2a78d6"}
FD = (np.arange(253) + 4) * 50 / 512
PK = FD >= 1


def fundo_tpsw(x, n=16, g=3, alpha=2.0):
    k = np.ones(2 * n + 1); k[n - g:n + g + 1] = 0; k /= k.sum()
    b1 = ndi.convolve1d(x, k, axis=-1, mode="reflect")
    return ndi.convolve1d(np.where(x > alpha * b1, b1, x), k, axis=-1, mode="reflect")


m = metadata(); m.index = m.index.map(int)
cache = pd.read_pickle("data/iara_processed/cache_fusion_mel_demon.pkl")["demon"]
S = pd.DataFrame({int(f): 20 * np.log10(w.mean(0) / (fundo_tpsw(w.mean(0)) + 1e-12)) for f, w in cache.items()}).T

d = pd.concat(R.file_predictions("cnn_lofar").values()); d = d[d.Target == 2]
acc = d.assign(ok=d.Prediction == d.Target).groupby("File").ok.mean(); acc.index = acc.index.astype(int)
g = pd.Series(np.where(acc == 1, "acertada", np.where(acc == 0, "errada", "oscila")), index=acc.index)
A = g.index[g == "acertada"].intersection(S.index); E = g.index[g == "errada"].intersection(S.index)
small = S.loc[S.index.intersection(m.index[m.Target == 0])]
pico = S.loc[:, PK].max(1)
freq = pd.Series(FD[PK][S.loc[:, PK].values.argmax(1)], index=S.index)
ref = pico.loc[small.index].median()
u = st.mannwhitneyu(pico.loc[A], pico.loc[E]); uf = st.mannwhitneyu(freq.loc[A], freq.loc[E])
print(f"n acertadas {len(A)}, erradas {len(E)}, SMALL {len(small)}")
print(f"SMALL típico (força da linha) {ref:.1f} dB | acertadas mediana {pico.loc[A].median():.1f}, acima {100 * (pico.loc[A] > ref).mean():.0f}% | "
      f"erradas mediana {pico.loc[E].median():.1f}, acima {100 * (pico.loc[E] > ref).mean():.0f}% | AUC acertada>errada {u.statistic / len(A) / len(E):.2f} p {u.pvalue:.1g}")
print(f"freq da linha: acertadas mediana {freq.loc[A].median():.1f} Hz, erradas {freq.loc[E].median():.1f} Hz; AUC acertada>errada {uf.statistic / len(A) / len(E):.2f} p {uf.pvalue:.1g}")
print(f"LARGE todas força mediana {pico.loc[S.index.intersection(m.index[m.Target == 2])].median():.1f}; MEDIUM {pico.loc[S.index.intersection(m.index[m.Target == 1])].median():.1f}")

ESC = {"acertada": "#0b5e3f", "errada": "#7f1010"}   # média em tom escuro, para aparecer sobre as linhas finas
# mesmo padrão das figuras LOFAR e MEL (dois painéis de linhas) + um painel com a força da linha mais forte, onde a
# diferença do DEMON aparece (na média as linhas de cada navio se apagam)
fig = plt.figure(figsize=(12, 3.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.62])
axs = [fig.add_subplot(gs[0])]; axs.append(fig.add_subplot(gs[1], sharey=axs[0])); axb = fig.add_subplot(gs[2])
for ax, grp, idx in zip(axs, ["acertada", "errada"], [A, E]):
    X = S.loc[idx]
    for _, r in X.iterrows():
        ax.plot(FD, r.values, color=COL[grp], lw=.5, alpha=.10)
    ax.plot(FD, X.mean().values, color=ESC[grp], lw=2.6, label=f"LARGE {grp} (média)")
    ax.plot(FD, small.mean().values, color=COL["SMALL"], lw=1.6, ls=":", label="SMALL, todas (média)")
    ax.set_title(f"LARGE {grp}s pela CNN LOFAR: {len(X)} gravações", fontsize=9.5)
    ax.text(0.98, 0.97, f"linha mais forte acima do SMALL típico\n({ref:.1f} dB): ".replace(".", ",") +
            f"{100 * (pico.loc[idx] > ref).mean():.0f}% das gravações", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="#444")
    ax.set_xlabel("frequência de modulação (Hz)"); ax.grid(alpha=.25, lw=.6); ax.set_ylim(-4, 22); ax.set_xlim(0, 25)
axs[0].set_ylabel("DEMON (dB acima do fundo)")
vals = [pico.loc[A].values, pico.loc[E].values, pico.loc[small.index].values]
bp = axb.boxplot(vals, widths=.55, showfliers=False, patch_artist=True, medianprops=dict(color="k"))
for b, c in zip(bp["boxes"], [COL["acertada"], COL["errada"], COL["SMALL"]]):
    b.set_facecolor(c); b.set_alpha(.55)
axb.axhline(ref, color=COL["SMALL"], ls=":", lw=1.2)
axb.set_xticks([1, 2, 3]); axb.set_xticklabels(["LARGE\nacertadas", "LARGE\nerradas", "SMALL\n(todas)"], fontsize=8)
axb.set_ylabel("dB acima do fundo"); axb.set_ylim(0, 22); axb.grid(axis="y", alpha=.25, lw=.6)
axb.set_title("Força da linha mais forte (1–25 Hz)\npor gravação", fontsize=9.5)
axb.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
plt.setp(axs[1].get_yticklabels(), visible=False)
h0, l0 = axs[0].get_legend_handles_labels(); h1, l1 = axs[1].get_legend_handles_labels()
fig.legend([h0[0], h1[0], h0[1]], [l0[0], l1[0], l0[1]], fontsize=8, frameon=False, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT + "large_uma_linha_demon.png", dpi=200); plt.close(fig)
