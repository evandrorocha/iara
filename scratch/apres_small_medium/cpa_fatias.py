"""
SMALL x MEDIUM diferem em algum MOMENTO da passagem? LOFAR alinhado pelo CPA.

Só gravações com CPA (subconjuntos A e C). Para cada gravação, o LOFAR por quadro (lofar_bc15..., dB acima do fundo
TPSW, o mesmo do NB35) é cortado em fatias de 30 s relativas ao instante do CPA (coluna "CPA time"), de -150 a +150 s,
e resumido em 14 bandas log de 15 Hz a 8 kHz. Comparação SMALL x MEDIUM com UM PONTO POR NAVIO (média das gravações do
navio), AUC + Mann-Whitney, FDR (Benjamini-Hochberg) sobre todas as fatias x bandas. Controle positivo: MEDIUM x LARGE.
Achados com q < 0,05 são verificados em metades de navios (100 sorteios).

Cache: data/iara_processed/cache_lofar_cpa_fatias.pkl
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from eval_ship_split_5x2cv import metadata

LOFAR_DIR = "data/iara_processed/lofar_bc1559516c522e2b993895bf5cb2ac94"
CACHE = "data/iara_processed/cache_lofar_cpa_fatias.pkl"
F = (np.arange(1024) + 1) * 16000 / 2048
EDGES = np.geomspace(15, 8000, 15)  # cada banda com pelo menos 1 faixa do LOFAR (7,8 Hz)
BANDS = [(F >= a) & (F < b) for a, b in zip(EDGES[:-1], EDGES[1:])]
BNAMES = [f"{a:.0f}-{b:.0f}" for a, b in zip(EDGES[:-1], EDGES[1:])]
SL = np.arange(-150, 151, 30)  # bordas das fatias
SNAMES = [f"{a:+d}..{b:+d}" for a, b in zip(SL[:-1], SL[1:])]


def build(m):
    if os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    rows = []
    sel = m[m.Dataset.isin(["A", "C"]) & (m.Target < 3)]
    for fid, r in sel.iterrows():
        p = os.path.join(LOFAR_DIR, f"{fid}.pkl")
        if not os.path.exists(p):
            continue
        d = pd.read_pickle(p)
        X = d["df"].values; t = np.asarray(d["times"], dtype=float) - float(r["CPA time"])
        for i, (a, b) in enumerate(zip(SL[:-1], SL[1:])):
            mk = (t >= a) & (t < b)
            if mk.sum() < 20:  # pelo menos ~10 s de quadros na fatia
                continue
            s = X[mk].mean(0)
            rows.append({"File": fid, "fatia": SNAMES[i], **{bn: s[bm].mean() for bn, bm in zip(BNAMES, BANDS)}})
    D = pd.DataFrame(rows)
    D.to_pickle(CACHE)
    return D


def compara(D, m, a, b, ships=None):
    d = D.join(m[["Target", "Ship ID"]], on="File")
    d = d[d.Target.isin([a, b])]
    if ships is not None:
        d = d[d["Ship ID"].isin(ships)]
    nav = d.groupby(["fatia", "Ship ID"]).agg({**{bn: "mean" for bn in BNAMES}, "Target": "first"}).reset_index()
    rows = []
    for s in SNAMES:
        x = nav[nav.fatia == s]
        for bn in BNAMES:
            u, v = x[x.Target == b][bn], x[x.Target == a][bn]
            if len(u) < 5 or len(v) < 5:
                continue
            mw = st.mannwhitneyu(u, v)
            rows.append({"fatia": s, "banda": bn, "n_a": len(v), "n_b": len(u), "AUC": mw.statistic / (len(u) * len(v)),
                         "p": mw.pvalue, "delta_dB": u.median() - v.median()})
    T = pd.DataFrame(rows)
    T["q"] = st.false_discovery_control(T.p)
    return T


def main():
    m = metadata()
    D = build(m)
    print(f"{D.File.nunique()} gravações com CPA (A/C); quadros por fatia de 30 s; {len(BNAMES)} bandas")
    n = D.join(m[["Target", "Ship ID"]], on="File").groupby(["fatia", "Target"])["Ship ID"].nunique().unstack()
    print("navios por fatia (0=SMALL,1=MEDIUM,2=LARGE):"); print(n.reindex(SNAMES).to_string())
    for a, b, nome in [(0, 1, "SMALL x MEDIUM"), (1, 2, "MEDIUM x LARGE (controle)")]:
        T = compara(D, m, a, b)
        print(f"\n=== {nome}: {len(T)} testes; q<0,05: {(T.q < 0.05).sum()}; p<0,05: {(T.p < 0.05).sum()} "
              f"(esperado por acaso ~{0.05 * len(T):.0f})")
        A = T.pivot(index="banda", columns="fatia", values="AUC").reindex(index=BNAMES, columns=SNAMES)
        print("AUC (2ª classe > 1ª) por banda x fatia:"); print(A.round(2).to_string())
        print("menores p:"); print(T.sort_values("p").head(8).round(4).to_string(index=False))
        T.to_csv(f"reports/small_medium/cpa_fatias_{a}{b}.csv", index=False)
    # validação em metades para SMALL x MEDIUM: os 5 menores p da análise toda
    T = compara(D, m, 0, 1)
    top = T.sort_values("p").head(5)[["fatia", "banda", "AUC"]]
    ships = D.join(m[["Target", "Ship ID"]], on="File").query("Target < 2").groupby("Ship ID").Target.first()
    rng = np.random.default_rng(0); res = {(r.fatia, r.banda): [] for r in top.itertuples()}
    for _ in range(100):
        half = set()
        for c in [0, 1]:
            s = ships[ships == c].index.values
            half |= set(rng.choice(s, len(s) // 2, replace=False))
        other = set(ships.index) - half
        Td = compara(D, m, 0, 1, ships=half).set_index(["fatia", "banda"])
        Tv = compara(D, m, 0, 1, ships=other).set_index(["fatia", "banda"])
        for k in res:
            if k in Td.index and k in Tv.index:
                res[k].append((np.sign(Td.loc[k, "AUC"] - .5) == np.sign(Tv.loc[k, "AUC"] - .5), Tv.loc[k, "p"] < 0.05))
    print("\nValidação em metades de navios (100 sorteios), 5 menores p de SMALL x MEDIUM:")
    for k, v in res.items():
        v = np.array(v)
        print(f"  {k}: mesmo sentido {v[:, 0].mean() * 100:.0f}%, p<0,05 na outra metade {v[:, 1].mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
