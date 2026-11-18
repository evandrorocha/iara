"""Figuras LOFAR × DEMON dentro de cada lancha com os grupos (acertada / errada → MEDIUM / errada → LARGE) definidos por
OUTROS modelos (lanchas_grupos_outros_modelos.py), para mostrar que o resultado não depende de a MLP LOFAR definir os grupos.

  lanchas_lofar_demon_rfmel.png   igual a lanchas_lofar_demon.png, mas com os grupos da RF MEL (maior recall de SMALL)
  lanchas_modelos_pilot14.png     a mesma lancha (PILOT 14, 86 gravações) colorida pelos 6 modelos com recall SMALL > 50%
Rodar na raiz, no container iara-dev:  python scratch/apres_small_medium/fig_lanchas_lofar_demon_modelos.py"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "presentation/20261116-eda-iara-dados/figuras/"
COL = {"SMALL": "#2a78d6", "MEDIUM": "#eb6834", "LARGE": "#1baf7a"}
cg = {"acertada": COL["SMALL"], "errada → MEDIUM": COL["MEDIUM"], "errada → LARGE": COL["LARGE"]}
G = list(cg)
D = pd.read_csv("reports/small_medium/lanchas_lofar_demon.csv", index_col=0); D.index = D.index.map(int)
GR = pd.read_csv("reports/small_medium/lanchas_grupos_outros_modelos_grupos.csv", index_col=0); GR.index = GR.index.map(int)
T = pd.read_csv("reports/small_medium/lanchas_grupos_outros_modelos.csv").drop_duplicates("modelo").set_index("modelo")
D["ship"] = D.ship.astype(str)


def pontos(ax, x, grupo):
    for g in G:
        y = x[grupo.loc[x.index] == g]
        ax.scatter(y.lofar_31_86, y.demon_freq_Hz, s=14 + 1.5 * y.demon_pico_dB.clip(0) ** 1.3, color=cg[g], alpha=.7,
                   edgecolor="white", lw=.4, label=g)
    ax.grid(alpha=.25, lw=.6)


# 1. slide 32 com os grupos da RF MEL
gr = GR["RF MEL"]
dd = D[gr.loc[D.index].isin(G)]
nun = dd.groupby("ship").apply(lambda t: gr.loc[t.index].nunique())
sh = [s for s in dd.ship.value_counts().index if nun[s] >= 2][:8]
fig, axs = plt.subplots(2, 4, figsize=(11, 5.8), sharex=True, sharey=True)
for ax, s in zip(axs.flat, sh):
    x = dd[dd.ship == s]
    pontos(ax, x, gr)
    ax.set_title(f"{x.nome.iloc[0]} ({len(x)} grav.)", fontsize=8.5)
for ax in axs.flat[len(sh):]:
    ax.axis("off")
for ax in axs.flat[:len(sh)]:
    ax.tick_params(labelbottom=True)
    ax.set_xlabel("LOFAR: grave 31–86 Hz (dB acima do TPSW)", fontsize=7.5)
for ax in axs[:, 0]:
    ax.set_ylabel("DEMON: frequência da\nlinha mais forte (Hz)", fontsize=8)
h, l = axs.flat[0].get_legend_handles_labels()
fig.legend(h, [f"{t} (RF MEL)" for t in l], loc="lower center", ncol=3, frameon=False, fontsize=8.5)
fig.text(0.99, 0.01, "tamanho do ponto = força da linha DEMON", ha="right", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT + "lanchas_lofar_demon_rfmel.png", dpi=200); plt.close(fig)
print("RF MEL, lanchas:", sh)

# 2. PILOT 14 segundo os 6 modelos com recall SMALL > 50%
MOD = [m for m in ["RF MEL", "BiGRU LOFAR", "SVM MEL", "RF HHT", "MLP LOFAR", "RF LOFAR"]]
x = D[D.nome == "PILOT 14"]
fig, axs = plt.subplots(2, 3, figsize=(11.5, 5.8), sharex=True, sharey=True)
for ax, mo in zip(axs.flat, MOD):
    g = GR[mo]
    xx = x[g.loc[x.index].isin(G)]
    pontos(ax, xx, g)
    gl = g.loc[xx.index]
    medL = xx.lofar_31_86[gl == "errada → LARGE"].median(); medA = xx.lofar_31_86[gl == "acertada"].median()
    for v, c in ((medA, cg["acertada"]), (medL, cg["errada → LARGE"])):
        ax.axvline(v, color=c, ls="--", lw=1.1, alpha=.9)
    ax.set_title(f"{mo} (recall SMALL {T.loc[mo, 'recall SMALL (todas)']:.0f}%)\n"
                 f"grave típico: {medA:.1f} (acertadas) × {medL:.1f} dB (→ LARGE)", fontsize=8.5)
    ax.tick_params(labelbottom=True)
    ax.set_xlabel("LOFAR: grave 31–86 Hz (dB acima do TPSW)", fontsize=7.5)
for ax in axs[:, 0]:
    ax.set_ylabel("DEMON: frequência da\nlinha mais forte (Hz)", fontsize=8)
h, l = axs.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
fig.text(0.99, 0.01, "tracejado = grave típico do grupo", ha="right", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT + "lanchas_modelos_pilot14.png", dpi=200); plt.close(fig)
print("ok")
