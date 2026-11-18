"""Slide didático "O que muda dentro de cada lancha": para cada faixa do LOFAR (dB acima do TPSW) e para vento/duração,
AUC das gravações erradas (→ LARGE e → MEDIUM, separadas) contra as acertadas, DENTRO de cada lancha (valor menos a
mediana da lancha). Grupos da RF MEL (split por gravação), como nos slides LOFAR × DEMON.
Rodar na raiz, no container iara-dev:  python scratch/apres_small_medium/fig_lanchas_o_que_muda.py
Saídas: presentation/20261116-eda-iara-dados/figuras/lanchas_o_que_muda.png; reports/small_medium/lanchas_o_que_muda.csv"""
import sys
import numpy as np, pandas as pd, matplotlib
import matplotlib.ticker  # noqa: E401
matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats as st  # noqa: E402
sys.path.insert(0, "scratch")
from eval_ship_split_5x2cv import metadata  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "presentation/20261116-eda-iara-dados/figuras/"
COL = {"errada → LARGE": "#1baf7a", "errada → MEDIUM": "#eb6834"}
m = metadata(); m.index = m.index.map(int)
GR = pd.read_csv("reports/small_medium/lanchas_grupos_outros_modelos_grupos.csv", index_col=0); GR.index = GR.index.map(int)
L = pd.read_pickle("data/iara_processed/cache_lofar_espectro_medio.pkl"); L.index = L.index.map(int)
F = (np.arange(1024) + 1) * 16000 / 2048
ids = GR.index
X = pd.DataFrame(index=ids)
BANDAS = [("grave muito baixo\n20–31 Hz", 20, 31), ("grave de LARGE\n31–86 Hz", 31, 86), ("grave-médio\n86–250 Hz", 86, 250),
          ("médio\n250 Hz–1 kHz", 250, 1000), ("agudo\n1–3 kHz", 1000, 3000), ("agudo\n3–8 kHz", 3000, 8000)]
for nome, a, b in BANDAS:
    X[nome] = L.loc[ids, (F >= a) & (F < b)].mean(1)
X["vento\n(metadado)"] = pd.to_numeric(m.loc[ids, "Wind"], errors="coerce")
X["duração da\ngravação"] = pd.to_numeric(m.loc[ids, "Recording time"], errors="coerce")
X["ship"] = m.loc[ids, "Ship ID"].astype(str); X["g"] = GR["RF MEL"]

rows = []
for err in COL:
    x = X[X.g.isin(["acertada", err])]
    ok = x.groupby("ship").g.nunique() == 2
    x = x[x.ship.isin(ok[ok].index)]
    for v in X.columns[:-2]:
        z = x[v] - x.groupby("ship")[v].transform("median")
        k = z.notna()
        u = st.mannwhitneyu(z[k & (x.g == err)], z[k & (x.g == "acertada")])
        rows.append({"erro": err, "variável": v.replace("\n", " "), "v": v, "AUC": u.statistic / ((k & (x.g == err)).sum() * (k & (x.g == "acertada")).sum()),
                     "p": u.pvalue, "lanchas": int(ok.sum()), "gravações": len(x)})
T = pd.DataFrame(rows)
print(T.drop(columns="v").round(3).to_string(index=False))
T.drop(columns="v").to_csv("reports/small_medium/lanchas_o_que_muda.csv", index=False)

vars_ = list(X.columns[:-2])
fig, ax = plt.subplots(figsize=(8.6, 4.3))
y = np.arange(len(vars_))[::-1]
h = 0.38
for j, (err, c) in enumerate(COL.items()):
    t = T[T.erro == err].set_index("v").loc[vars_]
    yy = y + (h / 2 if j == 0 else -h / 2)
    ax.barh(yy, t.AUC - 0.5, left=0.5, height=h, color=c, alpha=.85,
            label=f"{err} ({t.lanchas.iloc[0]} lanchas, {t['gravações'].iloc[0]} grav.)")
    for yi, a, p in zip(yy, t.AUC, t.p):
        ax.text(a + (0.01 if a >= .5 else -0.01), yi, f"{a:.2f}".replace(".", ",") + ("*" if p < .05 else ""), va="center",
                ha="left" if a >= .5 else "right", fontsize=8)
ax.axvline(.5, color="k", lw=1)
ax.set_yticks(y); ax.set_yticklabels(vars_, fontsize=8.5)
ax.set_xlim(.15, .85); ax.set_xlabel("AUC (errada > acertada, dentro da lancha)")
ax.text(.17, len(vars_) - .35, "← as erradas têm MENOS", fontsize=8.5, color="#444")
ax.text(.83, len(vars_) - .35, "as erradas têm MAIS →", fontsize=8.5, color="#444", ha="right")
ax.axhline(1.5, color="#bbb", lw=.8, ls=":")
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.grid(axis="x", alpha=.25)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
fig.tight_layout(); fig.savefig(OUT + "lanchas_o_que_muda.png", dpi=200); plt.close(fig)
