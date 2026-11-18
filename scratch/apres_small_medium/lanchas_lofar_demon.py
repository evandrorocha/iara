"""LOFAR × DEMON dentro da MESMA lancha de prático: as gravações que a MLP LOFAR (split por gravação) acerta, manda para
MEDIUM ou manda para LARGE diferem no LOFAR (grave 31–86 Hz) e/ou no DEMON (linha de modulação mais forte: força e
frequência)? Comparação centrada em cada lancha (cada gravação menos a mediana da sua lancha) e figura com um painel por
lancha. Grupos = os dos slides 25–29 (g26: classe mais votada entre MEDIUM, LARGE e BACKGROUND nas 5 avaliações).

Saídas: presentation/20261116-eda-iara-dados/figuras/lanchas_lofar_demon.png; reports/small_medium/lanchas_lofar_demon.csv"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats as st
sys.path.insert(0, "scratch/apres_small_medium")
exec(open("scratch/apres_small_medium/fig_lanchas_demon.py", encoding="utf-8").read().split("# 1. médias")[0]
     .replace("print(", "(lambda *a, **k: None)("))  # S (DEMON dB acima do TPSW), p (lanchas), FD, COL

F = (np.arange(1024) + 1) * 16000 / 2048
L = pd.read_pickle("data/iara_processed/cache_lofar_espectro_medio.pkl")
PK = FD >= 1
D = pd.DataFrame({"lofar_31_86": L.loc[p.index, (F >= 31) & (F < 86)].mean(1),
                  "demon_pico_dB": S.loc[p.index, PK].max(1),
                  "demon_freq_Hz": FD[PK][S.loc[p.index, PK].values.argmax(1)],
                  "grupo": p.g26.replace({"MEDIUM": "errada → MEDIUM", "LARGE": "errada → LARGE", "BACKGROUND": "errada → BACKGROUND"}),
                  "ship": p["Ship ID"].astype(str), "nome": p.Name}, index=p.index)
D.to_csv("reports/small_medium/lanchas_lofar_demon.csv")
VARS = {"lofar_31_86": "LOFAR 31–86 Hz (dB)", "demon_pico_dB": "DEMON: força da linha (dB)", "demon_freq_Hz": "DEMON: freq. da linha (Hz)"}
G = ["acertada", "errada → MEDIUM", "errada → LARGE"]


def dentro(a, b):
    """AUC (b > a) com cada gravação centrada na mediana da sua lancha; só lanchas com os dois grupos."""
    x = D[D.grupo.isin([a, b])]
    ok = x.groupby("ship").grupo.nunique() == 2
    x = x[x.ship.isin(ok[ok].index)]
    out = {"lanchas": int(ok.sum()), "gravações": len(x)}
    for v in VARS:
        z = x[v] - x.groupby("ship")[v].transform("median")
        u = st.mannwhitneyu(z[x.grupo == b], z[x.grupo == a])
        out[v] = (u.statistic / ((x.grupo == b).sum() * (x.grupo == a).sum()), u.pvalue)
    return out


rows = []
for a, b in [("acertada", "errada → LARGE"), ("acertada", "errada → MEDIUM"), ("errada → MEDIUM", "errada → LARGE")]:
    r = dentro(a, b)
    rows.append({"comparação": f"{b} × {a}", "lanchas": r["lanchas"], "gravações": r["gravações"],
                 **{f"AUC {VARS[v]}": round(r[v][0], 2) for v in VARS}, **{f"p {VARS[v]}": r[v][1] for v in VARS}})
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("Dentro da mesma lancha (AUC: 2º grupo > 1º grupo):"); print(T.to_string(index=False))
print("\nMediana por lancha e grupo (só lanchas com >= 2 grupos):")
M = D[D.grupo.isin(G)].groupby(["ship", "grupo"]).agg(n=("lofar_31_86", "size"), **{v: (v, "median") for v in VARS})
print(M.round(1).to_string())

# figura: um painel por lancha com gravações em pelo menos 2 grupos; x = LOFAR 31–86, y = freq. da linha DEMON
cg = {"acertada": COL["SMALL"], "errada → MEDIUM": COL["MEDIUM"], "errada → LARGE": COL["LARGE"]}
sh = D[D.grupo.isin(G)].groupby("ship").grupo.nunique()
sh = [s for s in D[D.grupo.isin(G)].ship.value_counts().index if sh[s] >= 2][:8]
fig, axs = plt.subplots(2, 4, figsize=(11, 5.8), sharex=True, sharey=True)
for ax, s in zip(axs.flat, sh):
    x = D[(D.ship == s) & D.grupo.isin(G)]
    for g in G:
        y = x[x.grupo == g]
        ax.scatter(y.lofar_31_86, y.demon_freq_Hz, s=14 + 1.5 * y.demon_pico_dB.clip(0) ** 1.3, color=cg[g], alpha=.7,
                   edgecolor="white", lw=.4, label=g)
    ax.set_title(f"{x.nome.iloc[0]} ({len(x)} grav.)", fontsize=8.5); ax.grid(alpha=.25, lw=.6)
for ax in axs.flat[len(sh):]:
    ax.axis("off")
for ax in axs.flat[:len(sh)]:   # rótulo do eixo x em todos os painéis (sharex esconderia os da linha de cima)
    ax.tick_params(labelbottom=True)
    ax.set_xlabel("LOFAR: grave 31–86 Hz (dB acima do TPSW)", fontsize=7.5)
for ax in axs[:, 0]:
    ax.set_ylabel("DEMON: frequência da\nlinha mais forte (Hz)", fontsize=8)
h, l = axs.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
fig.text(0.99, 0.01, "tamanho do ponto = força da linha DEMON", ha="right", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT + "lanchas_lofar_demon.png", dpi=200); plt.close(fig)
print("ok", sh)
