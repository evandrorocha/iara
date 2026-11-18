"""Espectro de TODAS as gravações de lanchas de prático ACERTADAS (MLP LOFAR 4 classes, split por gravação,
split_id_mlp_lofar_4c, 10 folds = 5 repetições x 2; cada gravação é testada 5 vezes), uma linha por gravação.
Painéis: acertada nos 5 testes x acertada em 3–4 dos 5. Complementa lanchas_erradas.png (mesmo estilo e mesma escala)."""
import sys, numpy as np, pandas as pd, matplotlib
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
med_ok = ok.loc[:, band].mean(1).median()
ref = {"errada → LARGE (média)": (S.loc[p.index[p.g == "LARGE"]], COL["LARGE"], "-"),
       "errada → MEDIUM (média)": (S.loc[p.index[p.g == "MEDIUM"]], COL["MEDIUM"], "-"),
       "MEDIUM, todas (média)": (S.loc[S.index.intersection(m.index[m.Target == 1])], COL["MEDIUM"], ":"),
       "LARGE, todas (média)": (S.loc[S.index.intersection(m.index[m.Target == 2])], COL["LARGE"], ":")}
grupos = [("acertadas em 5 de 5 testes", p.index[p.acerto_id == 1]),
          ("acertadas em 3 ou 4 de 5 testes", p.index[(p.acerto_id >= 0.6) & (p.acerto_id < 1)])]

fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
for ax, (tit, idx) in zip(axs, grupos):
    X = S.loc[idx]
    ax.axvspan(31, 86, color="#888", alpha=.1)
    for _, r in X.iterrows():
        ax.plot(F[sel], r.values[sel], color=COL["SMALL"], lw=.6, alpha=.22)
    ax.plot(F[sel], X.loc[:, sel].mean().values, color=COL["SMALL"], lw=2.6, label="lancha acertada (média do painel)")
    for lab, (Y, c, ls) in ref.items():
        ax.plot(F[sel], Y.loc[:, sel].mean().values, color=c, lw=1.4 if ls == "-" else 1.6, ls=ls,
                alpha=.75 if ls == "-" else 1, label=lab)
    acima = (X.loc[:, band].mean(1) > med_ok).mean() * 100
    pico = F[sel][np.argmax(np.where(F[sel] >= 15, X.loc[:, sel].mean().values, -np.inf))]
    ax.set_title(f"{tit.capitalize()}: {len(idx)} gravações, {p.loc[idx, 'Ship ID'].nunique()} lanchas", fontsize=9.5)
    ax.text(0.98, 0.97, f"31–86 Hz acima da lancha\nacertada típica: {acima:.0f}% das gravações",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#444")
    ax.set_xlabel("frequência (Hz)"); ax.grid(alpha=.25, lw=.6); ax.set_ylim(0, 22)
    print(f"{tit}: n={len(idx)}, lanchas={p.loc[idx, 'Ship ID'].nunique()}, 31-86 acima={acima:.0f}%, "
          f"pico médio≈{pico:.0f} Hz, 31-86 médio={X.loc[:, band].mean(1).mean():.2f} dB, "
          f"CPA={p.loc[idx, 'CPA'].mean() * 100:.0f}%")
axs[0].set_ylabel("LOFAR médio (dB acima do TPSW)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7.5, frameon=False, loc="lower center", ncol=5)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT + "lanchas_acertadas.png", dpi=200); plt.close(fig)
for g in ["acertada", "MEDIUM", "LARGE"]:
    X = S.loc[p.index[p.g == g]]
    print(f"{g}: 31-86 Hz médio = {X.loc[:, band].mean(1).mean():.2f} dB; 90-141 Hz = "
          f"{X.loc[:, (F >= 90) & (F < 141)].mean(1).mean():.2f} dB")
