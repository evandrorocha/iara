"""Figuras da apresentação focada na classe SMALL (presentation/20261116-classe-small/figuras/).

  exemplos_extracoes.png  uma lancha de prático e um LARGE em LOFAR, DEMON, MEL e HHT 1D
  lofar_classes.png       LOFAR médio (dB acima do TPSW) por classe e por tipo de SMALL (mediana dos navios)
  lofar_destinos.png      LOFAR médio das SMALL por destino na MLP LOFAR 4 classes (split por navio)
  demon_classes.png       DEMON (dB acima do fundo TPSW) por classe e por tipo de SMALL
  psd24k_classes.png      psd24k: forma por oitava e variação na passagem, por classe e por tipo de SMALL
Rodar na raiz, no container iara-dev:  python scratch/apres_small/figuras.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "scratch"); sys.path.insert(0, "notebooks/src")
from eval_ship_split_5x2cv import metadata  # noqa: E402
import iara_results as R  # noqa: E402

OUT = "presentation/20261116-classe-small/figuras/"
os.makedirs(OUT, exist_ok=True)
COL = {"SMALL": "#2a78d6", "MEDIUM": "#eb6834", "LARGE": "#1baf7a", "BACKGROUND": "#8a8a8a"}
TCOL = {"Pilot Vessel": "#2a78d6", "Tug": "#7b4fc9", "Fishing Vessel": "#d4a017"}
TNOME = {"Pilot Vessel": "SMALL: lancha de prático", "Tug": "SMALL: rebocador", "Fishing Vessel": "SMALL: pesca"}
CL = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25})

m = metadata(); m.index = m.index.map(int)
m["classe"] = m.Target.map(CL)
m["ship"] = m["Ship ID"].astype(str)
m.loc[m.Target == 3, "ship"] = "bg" + m.index[m.Target == 3].astype(str)  # cada BG é a sua própria unidade


def por_navio(X, idx):
    """mediana por navio das linhas de X (índice = ID), depois mediana e quartis entre navios"""
    g = X.loc[idx].groupby(m.loc[idx, "ship"].values).median()
    return g.median(), g.quantile(.25), g.quantile(.75), len(g)


def fundo_tpsw(x, n=16, g=3, alpha=2.0):
    k = np.r_[np.ones(n - g), np.zeros(2 * g + 1), np.ones(n - g)]; k /= k.sum()
    pad = np.pad(x, n, mode="reflect"); b = np.convolve(pad, k, "same")[n:-n]
    y = np.minimum(x, alpha * b); pad = np.pad(y, n, mode="reflect")
    return np.convolve(pad, k, "same")[n:-n]


# ---------------------------------------------------------------- 1. exemplos
import librosa  # noqa: E402
FMEL = librosa.mel_frequencies(n_mels=258, fmin=0, fmax=8000)[1:-1]

def exemplos():
    def escolhe(cond):
        c = m[cond & m.Dataset.isin(["A", "C"])]
        return (c["Recording time"].astype(float) - 300).abs().idxmin() if len(c) else None
    ids = {"SMALL (lancha de prático)": escolhe((m.Target == 0) & (m["DETAILED TYPE"] == "Pilot Vessel") & (m.ship == "456")),
           "LARGE (navio de carga)": escolhe((m.Target == 2) & m["DETAILED TYPE"].str.contains("Cargo", na=False))}
    fig, axs = plt.subplots(2, 4, figsize=(14, 5.4), constrained_layout=True)
    for r, (nome, i) in enumerate(ids.items()):
        lo = pd.read_pickle(f"data/iara_processed/lofar_bc1559516c522e2b993895bf5cb2ac94/{i}.pkl")
        F = (np.arange(1024) + 1) * 16000 / 2048; t = np.array(lo["times"]); k = F <= 1000
        ax = axs[r, 0]
        im = ax.pcolormesh(F[k], t, lo["df"].values[:, k] if hasattr(lo["df"], "values") else lo["df"][:, k],
                           shading="auto", cmap="magma", vmin=0, vmax=12, rasterized=True)
        ax.set_title(f"LOFAR – {nome}", fontsize=8.5); ax.set_xlabel("frequência (Hz)"); ax.set_ylabel("tempo (s)")
        ax.grid(False); fig.colorbar(im, ax=ax, label="dB acima do TPSW", shrink=.85)
        de = pd.read_pickle(f"data/iara_processed/demon_features/{i}.pkl")
        W = np.asarray(de["df"]); fr = np.asarray(de["frequencies"])
        Wd = np.array([20 * np.log10(np.maximum(w, 1e-12) / np.maximum(fundo_tpsw(w), 1e-12)) for w in W])
        ax = axs[r, 1]
        im = ax.pcolormesh(fr, de["times"], Wd, shading="auto", cmap="magma", vmin=0, vmax=12, rasterized=True)
        ax.set_title(f"DEMON 1–2 kHz – {nome}", fontsize=8.5); ax.set_xlabel("frequência de modulação (Hz)")
        ax.grid(False); fig.colorbar(im, ax=ax, label="dB acima do fundo", shrink=.85)
        me = pd.read_pickle(f"data/iara_processed/log_melgram_d50517371dbc579e9074fcd15aa81e4f/{i}.pkl")
        Mm = np.asarray(me["df"]); kk = FMEL >= 30
        ax = axs[r, 2]
        im = ax.pcolormesh(FMEL[kk], np.asarray(me["times"], float), Mm[:, kk], shading="auto", cmap="magma",
                           vmin=-80, vmax=0, rasterized=True)
        ax.set_xscale("log"); ax.set_title(f"MEL – {nome}", fontsize=8.5); ax.set_xlabel("frequência (Hz)")
        ax.grid(False); fig.colorbar(im, ax=ax, label="dB relativo ao máximo", shrink=.85)
        hh = pd.read_pickle(f"data/iara_processed/hht_features/{i}.pkl")
        H = 10 * np.log10(np.maximum(np.asarray(hh["df"], float), 1e-6))
        ax = axs[r, 3]
        im = ax.pcolormesh(np.asarray(hh["frequencies"]), np.asarray(hh["times"], float), H, shading="auto",
                           cmap="magma", vmin=-40, vmax=0, rasterized=True)
        ax.set_title(f"HHT 1D – {nome}", fontsize=8.5); ax.set_xlabel("frequência (Hz)")
        ax.grid(False); fig.colorbar(im, ax=ax, label="dB relativo ao pico", shrink=.85)
        print("exemplo", nome, i, m.loc[i, "Name"], m.loc[i, "Dataset"], m.loc[i, "Length"])
    fig.savefig(OUT + "exemplos_extracoes.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- 2. LOFAR por classe e tipo
L = pd.read_pickle("data/iara_processed/cache_lofar_espectro_medio.pkl"); L.index = L.index.map(int)
FL = (np.arange(1024) + 1) * 16000 / 2048
ids_all = [i for i in L.index if i in m.index]


def faixa(ax):
    ax.axvspan(31, 86, color="#999", alpha=.15, lw=0); ax.text(58, ax.get_ylim()[1] * .97, "31–86 Hz", ha="center",
                                                               va="top", fontsize=7.5, color="#555")


def lofar_classes():
    k = FL <= 1000
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True, constrained_layout=True)
    for c in ["SMALL", "MEDIUM", "LARGE", "BACKGROUND"]:
        med, q1, q3, n = por_navio(L.loc[:, :], [i for i in ids_all if m.loc[i, "classe"] == c])
        axs[0].plot(FL[k], med.values[k], color=COL[c], lw=1.8, label=f"{c} ({n} {'grav.' if c == 'BACKGROUND' else 'navios'})")
        axs[0].fill_between(FL[k], q1.values[k], q3.values[k], color=COL[c], alpha=.10, lw=0)
    for t in TCOL:
        med, q1, q3, n = por_navio(L, [i for i in ids_all if m.loc[i, "Target"] == 0 and m.loc[i, "DETAILED TYPE"] == t])
        axs[1].plot(FL[k], med.values[k], color=TCOL[t], lw=1.8, label=f"{TNOME[t]} ({n} navios)")
    med, *_ = por_navio(L, [i for i in ids_all if m.loc[i, "classe"] == "LARGE"])
    axs[1].plot(FL[k], med.values[k], color=COL["LARGE"], lw=1.4, ls=":", label="LARGE (todos)")
    med, *_ = por_navio(L, [i for i in ids_all if m.loc[i, "classe"] == "MEDIUM"])
    axs[1].plot(FL[k], med.values[k], color=COL["MEDIUM"], lw=1.4, ls=":", label="MEDIUM (todos)")
    for ax, tt in zip(axs, ["por classe (faixa = 25–75% dos navios)", "SMALL por tipo de embarcação"]):
        ax.set_xscale("log"); ax.set_xlim(15, 1000); ax.set_xlabel("frequência (Hz)"); ax.set_title(tt, fontsize=9)
        ax.set_ylim(0, 11); faixa(ax); ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    axs[0].set_ylabel("LOFAR médio\n(dB acima do TPSW)")
    fig.savefig(OUT + "lofar_classes.png", dpi=200); plt.close(fig)


def destinos():
    P = R.file_predictions("split_navio_mlp_lofar_4c"); d = pd.concat(P.values())
    d["File"] = d.File.astype(int); d = d[d.Target == 0]
    dest = d.groupby("File").Prediction.agg(lambda s: s.value_counts().index[0]).map(CL)
    k = FL <= 1000
    fig, ax = plt.subplots(figsize=(8.6, 3.6), constrained_layout=True)
    cor = {"SMALL": COL["SMALL"], "MEDIUM": COL["MEDIUM"], "LARGE": COL["LARGE"], "BACKGROUND": COL["BACKGROUND"]}
    nomes = {"SMALL": "acertada (→ SMALL)", "MEDIUM": "errada → MEDIUM", "LARGE": "errada → LARGE",
             "BACKGROUND": "errada → BACKGROUND"}
    res = {}
    for g in ["SMALL", "MEDIUM", "LARGE", "BACKGROUND"]:
        idx = [i for i in dest.index[dest == g] if i in L.index]
        med, q1, q3, n = por_navio(L, idx)
        ax.plot(FL[k], med.values[k], color=cor[g], lw=1.9, label=f"{nomes[g]}: {len(idx)} grav., {n} navios")
        res[g] = (len(idx), n)
    ax.set_xscale("log"); ax.set_xlim(15, 1000); ax.set_ylim(0, 11); faixa(ax)
    ax.set_xlabel("frequência (Hz)"); ax.set_ylabel("LOFAR médio\n(dB acima do TPSW)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.savefig(OUT + "lofar_destinos.png", dpi=200); plt.close(fig)
    print("destinos", res)


# ---------------------------------------------------------------- 3. DEMON
def demon():
    D = pd.read_pickle("data/iara_processed/cache_demon_relativo.pkl"); X = D["medio"]; X.index = X.index.map(int)
    fd = D["freqs"]; ids = [i for i in X.index if i in m.index]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True, constrained_layout=True)
    for c in ["SMALL", "MEDIUM", "LARGE", "BACKGROUND"]:
        med, q1, q3, n = por_navio(X, [i for i in ids if m.loc[i, "classe"] == c])
        axs[0].plot(fd, med.values, color=COL[c], lw=1.6, label=f"{c}")
    for t in TCOL:
        med, *_ = por_navio(X, [i for i in ids if m.loc[i, "Target"] == 0 and m.loc[i, "DETAILED TYPE"] == t])
        axs[1].plot(fd, med.values, color=TCOL[t], lw=1.6, label=TNOME[t])
    med, *_ = por_navio(X, [i for i in ids if m.loc[i, "classe"] == "LARGE"])
    axs[1].plot(fd, med.values, color=COL["LARGE"], lw=1.3, ls=":", label="LARGE (todos)")
    for ax, tt in zip(axs, ["por classe (mediana dos navios)", "SMALL por tipo de embarcação"]):
        ax.set_xlabel("frequência de modulação (Hz)"); ax.set_title(tt, fontsize=9); ax.set_xlim(0, 25)
        ax.legend(fontsize=7.5, frameon=False)
    axs[0].set_ylabel("DEMON médio\n(dB acima do fundo TPSW)")
    fig.savefig(OUT + "demon_classes.png", dpi=200); plt.close(fig)
    # pico dominante de cada gravação (>= 1 Hz): frequência e força, mediana por navio
    pk = fd >= 1
    P = pd.DataFrame({"freq": fd[pk][X.loc[ids, pk].values.argmax(1)], "forca": X.loc[ids, pk].values.max(1)}, index=ids)
    bins = [1, 2, 3, 4, 6, 8, 12, 25]; lab = ["1–2", "2–3", "3–4", "4–6", "6–8", "8–12", "12–25"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
    grupos = [(c, COL[c], [i for i in ids if m.loc[i, "classe"] == c]) for c in ["SMALL", "MEDIUM", "LARGE"]]
    w = .26
    for j, (c, cor, idx) in enumerate(grupos):
        nv = P.loc[idx].groupby(m.loc[idx, "ship"].values).median()
        h = pd.cut(nv.freq, bins, labels=lab).value_counts(normalize=True).reindex(lab) * 100
        axs[0].bar(np.arange(len(lab)) + (j - 1) * w, h.values, w, color=cor, label=f"{c} (mediana {nv.freq.median():.1f} Hz)")
        axs[1].hist(nv.forca, bins=np.arange(0, 26, 1.5), histtype="step", lw=1.8, color=cor, density=True,
                    label=f"{c} (mediana {nv.forca.median():.1f} dB)")
        print("demon pico", c, round(nv.freq.median(), 2), round(nv.forca.median(), 2), len(nv))
    axs[0].set_xticks(range(len(lab))); axs[0].set_xticklabels(lab); axs[0].set_xlabel("frequência do pico dominante (Hz)")
    axs[0].set_ylabel("% dos navios"); axs[0].legend(fontsize=7.5, frameon=False)
    axs[0].set_title("onde fica a linha DEMON mais forte", fontsize=9)
    axs[1].set_xlabel("força da linha mais forte (dB acima do fundo)"); axs[1].set_ylabel("densidade")
    axs[1].set_title("quão forte ela é", fontsize=9); axs[1].legend(fontsize=7.5, frameon=False)
    fig.savefig(OUT + "demon_pico.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- 4. psd24k
def psd24k():
    X = pd.read_csv("reports/small_medium/psd24k_small_medidas.csv", index_col=0)
    X.index = X.index.map(int)
    fo = [c for c in X.columns if c.startswith("forma")]; di = [c for c in X.columns if c.startswith("dinâmica")]
    cent = np.sqrt([31.25 * 62.5 * 2 ** (2 * k) for k in range(10)]); cent[-1] = np.sqrt(16000 * 24000)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for c in ["SMALL", "MEDIUM", "LARGE", "BACKGROUND"]:
        idx = X.index[X.classe == c]
        for ax, cols in zip(axs[:2], [fo, di]):
            med, q1, q3, n = por_navio(X[cols], idx)
            ax.plot(cent, med.values, "-o", ms=3, color=COL[c], lw=1.7, label=c)
            ax.fill_between(cent, q1.values, q3.values, color=COL[c], alpha=.10, lw=0)
    for t in TCOL:
        idx = X.index[(X.classe == "SMALL") & (X["DETAILED TYPE"] == t)]
        med, *_ = por_navio(X[fo], idx)
        axs[2].plot(cent, med.values, "-o", ms=3, color=TCOL[t], lw=1.7, label=TNOME[t])
    med, *_ = por_navio(X[fo], X.index[X.classe == "LARGE"])
    axs[2].plot(cent, med.values, ":", color=COL["LARGE"], lw=1.4, label="LARGE (todos)")
    tit = ["forma: dB da oitava − média 125 Hz–16 kHz", "variação na passagem: p90 − p10 dos blocos de 10 s",
           "forma: SMALL por tipo"]
    yl = ["dB (relativo à própria gravação)", "dB", "dB (relativo à própria gravação)"]
    for ax, tt, y in zip(axs, tit, yl):
        ax.set_xscale("log"); ax.set_xlabel("frequência (Hz, centro da oitava)"); ax.set_title(tt, fontsize=8.5)
        ax.set_ylabel(y); ax.legend(fontsize=7, frameon=False)
    fig.savefig(OUT + "psd24k_classes.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    exemplos(); lofar_classes(); destinos(); demon(); psd24k()
    print("ok")
