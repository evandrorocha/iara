"""DEMON das lanchas de prático agrupadas pelos ACERTOS/ERROS da MLP LOFAR (split por gravação, split_id_mlp_lofar_4c,
10 folds; cada gravação testada 5 vezes) — os mesmos grupos dos slides 25–27 da apresentação 20261116-eda-iara-dados.

DEMON "acima do fundo": espectro DEMON médio da gravação (demon_features: banda 1–2 kHz, 0,4–25 Hz) em dB acima do
fundo TPSW ao longo das frequências de modulação (mesmo TPSW do HCNN DEMON e do demon_multibanda: ±16 bins sem os
±3 centrais, alpha 2). Análogo ao LOFAR (dB acima do TPSW), que tira a escala arbitrária do DEMON do pipeline
(envelope dividido pelo próprio pico da gravação).

Figuras (presentation/20261116-eda-iara-dados/figuras/):
  demon_lancha_espectro.png  médias por grupo (como lancha_espectro.png)
  demon_lanchas_erradas.png  uma linha por gravação errada, por classe prevista (como lanchas_erradas.png)
  demon_lanchas_acertadas.png uma linha por gravação acertada, 5/5 e 3–4/5 (como lanchas_acertadas.png)
Tabela: reports/small_medium/demon_lanchas.csv (por gravação: pico, frequência do pico, média 0,4–25 Hz)"""
import os, sys
import numpy as np, pandas as pd, scipy.ndimage as ndi, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats as st
sys.path.insert(0, "scratch")
from eval_ship_split_5x2cv import metadata
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "presentation/20261116-eda-iara-dados/figuras/"
COL = {"SMALL": "#2a78d6", "MEDIUM": "#eb6834", "LARGE": "#1baf7a"}
FD = (np.arange(253) + 4) * 50 / 512


def fundo_tpsw(x, n=16, g=3, alpha=2.0):
    k = np.ones(2 * n + 1); k[n - g:n + g + 1] = 0; k /= k.sum()
    b1 = ndi.convolve1d(x, k, axis=-1, mode="reflect")
    return ndi.convolve1d(np.where(x > alpha * b1, b1, x), k, axis=-1, mode="reflect")


m = metadata()
cache = pd.read_pickle("data/iara_processed/cache_fusion_mel_demon.pkl")["demon"]
S = pd.DataFrame({f: 20 * np.log10(w.mean(0) / (fundo_tpsw(w.mean(0)) + 1e-12)) for f, w in cache.items()}).T
p = pd.read_csv("reports/small_medium/lanchas_pratico.csv", index_col=0)
p["g25"] = np.where(p.acerto_id >= 0.6, "acertada", np.where(p.pL_id > p.pM_id, "errada → LARGE", "errada → MEDIUM"))
p["g26"] = np.where(p.acerto_id >= 0.6, "acertada", p[["pM_id", "pL_id", "pB_id"]].idxmax(1).map(
    {"pM_id": "MEDIUM", "pL_id": "LARGE", "pB_id": "BACKGROUND"}))
MED = S.loc[S.index.intersection(m.index[m.Target == 1])]
LAR = S.loc[S.index.intersection(m.index[m.Target == 2])]

# tabela por gravação
T = pd.DataFrame({"pico_dB": S.loc[p.index].max(1), "freq_pico_Hz": FD[S.loc[p.index].values.argmax(1)],
                  "media_dB": S.loc[p.index].mean(1)}).join(p[["Ship ID", "Name", "acerto_id", "g25", "g26"]])
T.to_csv("reports/small_medium/demon_lanchas.csv")
ref = pd.DataFrame({"pico_dB": [MED.max(1).median(), LAR.max(1).median()],
                    "media_dB": [MED.mean(1).median(), LAR.mean(1).median()]}, index=["MEDIUM (todas)", "LARGE (todas)"])
print("Mediana por grupo (gravações de lancha) e referências:")
print(pd.concat([T.groupby("g25")[["pico_dB", "media_dB", "freq_pico_Hz"]].median(), ref]).round(2).to_string())
print(T.g25.value_counts().to_dict())
ok = T[T.g25 == "acertada"]
for g in ["errada → LARGE", "errada → MEDIUM"]:
    e = T[T.g25 == g]
    for c in ["pico_dB", "media_dB"]:
        u = st.mannwhitneyu(e[c], ok[c])
        print(f"  {g} × acertada, {c}: AUC {u.statistic / (len(e) * len(ok)):.2f}, p {u.pvalue:.2g}")
# dentro da mesma lancha (centrado na mediana da lancha)
d = T[T.g25 != "oscila"].copy(); d["err"] = d.g25 != "acertada"
amb = d.groupby("Ship ID").err.agg(lambda s: s.any() and (~s).any()); d = d[d["Ship ID"].isin(amb[amb].index)]
for c in ["pico_dB", "media_dB"]:
    z = d[c] - d.groupby("Ship ID")[c].transform("median")
    u = st.mannwhitneyu(z[d.err], z[~d.err])
    print(f"  dentro da lancha ({d['Ship ID'].nunique()} lanchas), {c}: AUC errada>acertada {u.statistic / (d.err.sum() * (~d.err).sum()):.2f}, p {u.pvalue:.2g}")

# 1. médias por grupo (como o slide 25)
fig, ax = plt.subplots(figsize=(7.2, 3.3))
for lab, idx, c, ls, lw in [("lancha acertada", p.index[p.g25 == "acertada"], COL["SMALL"], "-", 2.2),
                             ("lancha errada → MEDIUM", p.index[p.g25 == "errada → MEDIUM"], COL["MEDIUM"], "-", 2.2),
                             ("lancha errada → LARGE", p.index[p.g25 == "errada → LARGE"], COL["LARGE"], "-", 2.2)]:
    ax.plot(FD, S.loc[idx].mean().values, color=c, ls=ls, lw=lw, label=f"{lab} ({len(idx)})")
ax.plot(FD, MED.mean().values, color=COL["MEDIUM"], ls=":", lw=1.5, label="MEDIUM (todas)")
ax.plot(FD, LAR.mean().values, color=COL["LARGE"], ls=":", lw=1.5, label="LARGE (todas)")
ax.set_xlabel("frequência de modulação (Hz)"); ax.set_ylabel("DEMON médio (dB acima do TPSW)")
ax.legend(fontsize=7.5, frameon=False, ncol=2); ax.grid(alpha=.25, lw=.6)
fig.tight_layout(); fig.savefig(OUT + "demon_lancha_espectro.png", dpi=200); plt.close(fig)


# 2 e 3. uma linha por gravação
def painel(grupos, cor, nome, ylim):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (tit, idx, c) in zip(axs, grupos):
        X = S.loc[idx]
        for _, r in X.iterrows():
            ax.plot(FD, r.values, color=c, lw=.6, alpha=.22)
        ax.plot(FD, X.mean().values, color=c, lw=2.6, label="média do painel")
        ax.plot(FD, S.loc[p.index[p.g25 == "acertada"]].mean().values, color=COL["SMALL"], lw=1.4, alpha=.8,
                label="lancha acertada (média)")
        ax.plot(FD, MED.mean().values, color=COL["MEDIUM"], ls=":", lw=1.6, label="MEDIUM, todas (média)")
        ax.plot(FD, LAR.mean().values, color=COL["LARGE"], ls=":", lw=1.6, label="LARGE, todas (média)")
        ax.set_title(f"{tit}: {len(idx)} gravações, {p.loc[idx, 'Ship ID'].nunique()} lanchas", fontsize=9.5)
        pk = X.max(1).median()
        ax.text(0.98, 0.97, f"pico mediano: {pk:.1f} dB", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#444")
        ax.set_xlabel("frequência de modulação (Hz)"); ax.grid(alpha=.25, lw=.6); ax.set_ylim(*ylim)
    axs[0].set_ylabel("DEMON (dB acima do TPSW)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, frameon=False, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT + nome, dpi=200); plt.close(fig)


lo, hi = np.percentile(S.loc[p.index].values, [0.5, 99.8])
ylim = (min(-3, lo), hi + 1)
painel([("Previstas MEDIUM", p.index[p.g26 == "MEDIUM"], COL["MEDIUM"]),
        ("Previstas LARGE", p.index[p.g26 == "LARGE"], COL["LARGE"])], None, "demon_lanchas_erradas.png", ylim)
painel([("Acertadas em 5 de 5 testes", p.index[p.acerto_id == 1], COL["SMALL"]),
        ("Acertadas em 3 ou 4 de 5 testes", p.index[(p.acerto_id >= 0.6) & (p.acerto_id < 1)], COL["SMALL"])],
       None, "demon_lanchas_acertadas.png", ylim)
print("ok")


# 4. força da linha mais forte (pico, >= 1 Hz: abaixo disso é resíduo da componente contínua) por grupo e onde ela está
PK = FD >= 1
grp = {"acertada 5/5": p.index[p.acerto_id == 1], "acertada 3–4/5": p.index[(p.acerto_id >= 0.6) & (p.acerto_id < 1)],
       "errada → MEDIUM": p.index[p.g26 == "MEDIUM"], "errada → LARGE": p.index[p.g26 == "LARGE"]}
cg = {"acertada 5/5": COL["SMALL"], "acertada 3–4/5": "#7fb0ea", "errada → MEDIUM": COL["MEDIUM"], "errada → LARGE": COL["LARGE"]}
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axs[0]
vals = [S.loc[i, PK].max(1).values for i in grp.values()] + [MED.loc[:, PK].max(1).values, LAR.loc[:, PK].max(1).values]
labs = list(grp) + ["MEDIUM\n(todas)", "LARGE\n(todas)"]
cols = list(cg.values()) + [COL["MEDIUM"], COL["LARGE"]]
bp = ax.boxplot(vals, widths=.55, showfliers=False, patch_artist=True, medianprops=dict(color="k"))
for b, c in zip(bp["boxes"], cols):
    b.set_facecolor(c); b.set_alpha(.55)
rng = np.random.default_rng(0)
for i, v in enumerate(vals[:4]):
    ax.scatter(i + 1 + rng.uniform(-.18, .18, len(v)), v, s=6, color=cols[i], alpha=.6, lw=0)
ax.set_xticks(range(1, 7)); ax.set_xticklabels([l.replace(" → ", "\n→ ").replace("acertada ", "acertada\n") for l in labs], fontsize=7)
ax.set_ylabel("dB acima do TPSW"); ax.set_ylim(0, 25); ax.grid(axis="y", alpha=.25, lw=.6)
ax.set_title("Força da linha de modulação mais forte", fontsize=9.5)
ax = axs[1]
for g, idx in grp.items():
    X = S.loc[idx, PK]
    ax.scatter(FD[PK][X.values.argmax(1)], X.max(1), s=12, color=cg[g], alpha=.7, lw=0, label=f"{g} ({len(idx)})")
ax.set_xlabel("frequência da linha mais forte (Hz)"); ax.set_ylabel("dB acima do TPSW"); ax.set_ylim(0, 25)
ax.set_title("Onde está a linha mais forte", fontsize=9.5); ax.legend(fontsize=7, frameon=False); ax.grid(alpha=.25, lw=.6)
fig.tight_layout(); fig.savefig(OUT + "demon_lanchas_pico.png", dpi=200); plt.close(fig)
for g, idx in grp.items():
    X = S.loc[idx, PK]; fq = FD[PK][X.values.argmax(1)]
    print(f"{g}: n={len(idx)}, pico mediano {X.max(1).median():.1f} dB, pico>10 dB {100 * (X.max(1) > 10).mean():.0f}%, "
          f"freq do pico < 3 Hz {100 * (fq < 3).mean():.0f}%")
print(f"MEDIUM todas: pico mediano {MED.loc[:, PK].max(1).median():.1f}, >10 dB {100 * (MED.loc[:, PK].max(1) > 10).mean():.0f}%; "
      f"LARGE: {LAR.loc[:, PK].max(1).median():.1f}, >10 dB {100 * (LAR.loc[:, PK].max(1) > 10).mean():.0f}%")
