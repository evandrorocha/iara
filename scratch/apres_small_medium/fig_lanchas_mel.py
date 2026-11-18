"""MEL das lanchas de prático agrupadas pelos ACERTOS/ERROS da MLP LOFAR (split por gravação; os mesmos grupos dos
slides de lanchas da apresentação 20261116-eda-iara-dados), análogo a fig_lanchas_erradas.py (LOFAR) e fig_lanchas_demon.py.

MEL: espectro médio da gravação no log-melgram do pipeline SEM normalização (256 bandas mel de 0 a 8 kHz, dB relativo
ao máximo da própria gravação; cache_mel_espectro_medio.pkl do NB37). Sem TPSW: o MEL mostra também elevações largas,
que o TPSW do LOFAR transforma em "fundo". Eixo logarítmico, 20 Hz–6 kHz (acima de ~6,4 kHz o filtro da decimação corta), para ver grave e agudo.

Figuras (presentation/20261116-eda-iara-dados/figuras/):
  mel_lancha_espectro.png  médias por grupo (como lancha_espectro.png)
  mel_lanchas_erradas.png  uma linha por gravação errada, por classe prevista (como lanchas_erradas.png)
Rodar na raiz, no container iara-dev:  python scratch/apres_small_medium/fig_lanchas_mel.py"""
import sys
import numpy as np, pandas as pd, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
import librosa  # noqa: E402
from scipy import stats as st  # noqa: E402
sys.path.insert(0, "scratch")
from eval_ship_split_5x2cv import metadata  # noqa: E402
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "presentation/20261116-eda-iara-dados/figuras/"
COL = {"SMALL": "#2a78d6", "MEDIUM": "#eb6834", "LARGE": "#1baf7a"}
FM = librosa.mel_frequencies(n_mels=258, fmin=0, fmax=8000)[1:-1]

m = metadata(); m.index = m.index.map(int)
S = pd.read_pickle("data/iara_processed/cache_mel_espectro_medio.pkl")["sem normalização"]; S.index = S.index.map(int)
p = pd.read_csv("reports/small_medium/lanchas_pratico.csv", index_col=0); p.index = p.index.map(int)
p["g25"] = np.where(p.acerto_id >= 0.6, "acertada", np.where(p.pL_id > p.pM_id, "errada → LARGE", "errada → MEDIUM"))
p["g26"] = np.where(p.acerto_id >= 0.6, "acertada", p[["pM_id", "pL_id", "pB_id"]].idxmax(1).map(
    {"pM_id": "MEDIUM", "pL_id": "LARGE", "pB_id": "BACKGROUND"}))
MED = S.loc[S.index.intersection(m.index[m.Target == 1])]
LAR = S.loc[S.index.intersection(m.index[m.Target == 2])]
sel = (FM >= 20) & (FM <= 6000)
grave, agudo = (FM >= 31) & (FM < 86), (FM >= 1000) & (FM < 6000)

# números por grupo: grave 31–86 Hz, agudo 1–6 kHz e grave − agudo (dB relativo ao máximo)
def resumo(X):
    return pd.Series({"31–86 Hz": X.loc[:, grave].mean(1).median(), "1–6 kHz": X.loc[:, agudo].mean(1).median(),
                      "grave − agudo": (X.loc[:, grave].mean(1) - X.loc[:, agudo].mean(1)).median(), "n": len(X)})
tab = pd.DataFrame({g: resumo(S.loc[p.index[p.g25 == g]]) for g in ["acertada", "errada → MEDIUM", "errada → LARGE"]}
                   | {"MEDIUM (todas)": resumo(MED), "LARGE (todas)": resumo(LAR)}).T
print(tab.round(1).to_string())
ok = S.loc[p.index[p.g25 == "acertada"]]
for g in ["errada → LARGE", "errada → MEDIUM"]:
    e = S.loc[p.index[p.g25 == g]]
    for nome, msk in (("31–86 Hz", grave), ("1–6 kHz", agudo)):
        u = st.mannwhitneyu(e.loc[:, msk].mean(1), ok.loc[:, msk].mean(1))
        print(f"  {g} × acertada, {nome}: AUC {u.statistic / (len(e) * len(ok)):.2f}, p {u.pvalue:.2g}")


def eixo(ax):
    ax.axvspan(31, 86, color="#888", alpha=.1)
    ax.set_xscale("log"); ax.set_xlim(20, 6000); ax.set_xlabel("frequência (Hz, escala logarítmica)"); ax.grid(alpha=.25, lw=.6)
    tk = [20, 50, 100, 200, 500, 1000, 2000, 5000]   # números comuns em vez de 10^2, 10^3
    ax.set_xticks(tk); ax.set_xticklabels([f"{t:,}".replace(",", ".") for t in tk])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


# 1. médias por grupo
fig, ax = plt.subplots(figsize=(7.2, 3.7))
eixo(ax)
for lab, g, c in [("lancha acertada", "acertada", COL["SMALL"]), ("lancha errada → MEDIUM", "errada → MEDIUM", COL["MEDIUM"]),
                  ("lancha errada → LARGE", "errada → LARGE", COL["LARGE"])]:
    idx = p.index[p.g25 == g]
    ax.plot(FM[sel], S.loc[idx].mean().values[sel], color=c, lw=2.2, label=f"{lab} ({len(idx)})")
ax.plot(FM[sel], MED.mean().values[sel], color=COL["MEDIUM"], ls=":", lw=1.5, label="MEDIUM (todas)")
ax.plot(FM[sel], LAR.mean().values[sel], color=COL["LARGE"], ls=":", lw=1.5, label="LARGE (todas)")
ax.text(52, 0.04, "31–86 Hz\n(banda LARGE)", ha="center", va="bottom", fontsize=7, color="#555",
        transform=ax.get_xaxis_transform())
ax.set_ylabel("MEL médio (dB rel. ao máximo)")
fig.legend(*ax.get_legend_handles_labels(), fontsize=7.5, frameon=False, ncol=3, loc="lower center")
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(OUT + "mel_lancha_espectro.png", dpi=200); plt.close(fig)

# 2. uma linha por gravação errada
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
for ax, alvo in zip(axs, ["MEDIUM", "LARGE"]):
    idx = p.index[p.g26 == alvo]; X = S.loc[idx]
    eixo(ax)
    for _, r in X.iterrows():
        ax.plot(FM[sel], r.values[sel], color=COL[alvo], lw=.6, alpha=.22)
    ax.plot(FM[sel], X.mean().values[sel], color=COL[alvo], lw=2.6, label=f"errada → {alvo} (média)")
    ax.plot(FM[sel], ok.mean().values[sel], color=COL["SMALL"], lw=1.6, label="lancha acertada (média)")
    ax.plot(FM[sel], MED.mean().values[sel], color=COL["MEDIUM"], ls=":", lw=1.6, label="MEDIUM, todas (média)")
    ax.plot(FM[sel], LAR.mean().values[sel], color=COL["LARGE"], ls=":", lw=1.6, label="LARGE, todas (média)")
    ax.set_title(f"Lanchas previstas {alvo}: {len(idx)} gravações de {p.loc[idx, 'Ship ID'].nunique()} lanchas", fontsize=9.5)
    ga = (X.loc[:, grave].mean(1) - X.loc[:, agudo].mean(1))
    ref = (ok.loc[:, grave].mean(1) - ok.loc[:, agudo].mean(1)).median()
    ax.text(0.98, 0.97, f"grave − agudo acima da lancha\nacertada típica: {100 * (ga > ref).mean():.0f}% das gravações",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#444")
axs[0].set_ylabel("MEL médio (dB rel. ao máximo)")
h, l = axs[1].get_legend_handles_labels(); h0, l0 = axs[0].get_legend_handles_labels()
fig.legend([h0[0], h[0]] + h[1:], [l0[0], l[0]] + l[1:], fontsize=7.5, frameon=False, loc="lower center", ncol=5)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT + "mel_lanchas_erradas.png", dpi=200); plt.close(fig)
print(p.g26.value_counts().to_dict())
