"""
Existe uma pista POSITIVA de SMALL no espectro cru (psd24k, sem TPSW)? SMALL × LARGE e SMALL × BACKGROUND.

O nível absoluto do IARA não é calibrado (ganho, hidrofone, distância), então só usamos medidas que não dependem do ganho:
  - forma: média em dB por oitava (31 Hz–24 kHz) menos a média da gravação em 125 Hz–16 kHz;
  - inclinação (dB/oitava) entre 125 Hz e 8 kHz e entre 1 e 16 kHz;
  - faixa dinâmica por oitava: p90 − p10 dos blocos de 10 s (quanto a faixa varia ao longo da passagem).
Navios: cada medida é centrada na mediana do seu subconjunto (A–D; remove resposta do hidrofone/local) e resumida
pela mediana por navio (Ship ID); testes por navio (Mann-Whitney, AUC = P(SMALL > outro), FDR de Benjamini-Hochberg).
BACKGROUND está todo no subconjunto E: a comparação com BG usa valores NÃO centrados e mistura classe com equipamento.

Saídas: reports/small_medium/psd24k_small_{medidas,testes,por_tipo,por_subconjunto}.csv
Rodar na raiz, no container iara-dev:  python scratch/small_assinatura/psd24k_small.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, "scratch")
from eval_ship_split_5x2cv import metadata  # noqa: E402

OUT = "reports/small_medium/psd24k_small"
D = "data/iara_processed/psd24k"
CL = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}
OIT = [31.25 * 2 ** k for k in range(10)] + [24000]  # 31–62, 62–125, ..., 8k–16k, 16k–24k
NOMES = [f"{int(a) if a >= 100 else a:g}–{int(b)}" for a, b in zip(OIT[:-1], OIT[1:])]


def medidas(i):
    d = pd.read_pickle(f"{D}/{i}.pkl")
    f, p, B = d["freqs"], d["psd_db"], d["blocos_db"]
    sel = [(f >= a) & (f < b) for a, b in zip(OIT[:-1], OIT[1:])]
    ref = p[(f >= 125) & (f < 16000)].mean()
    r = {f"forma {n} Hz": p[s].mean() - ref for n, s in zip(NOMES, sel)}
    bands = np.array([p[s].mean() for s in sel])
    x = np.arange(len(bands))
    r["inclinação 125 Hz–8 kHz (dB/oit)"] = np.polyfit(x[2:8], bands[2:8], 1)[0]
    r["inclinação 1–16 kHz (dB/oit)"] = np.polyfit(x[5:9], bands[5:9], 1)[0]
    for n, s in zip(NOMES, sel):
        b = B[:, s].mean(1)
        r[f"dinâmica {n} Hz"] = np.percentile(b, 90) - np.percentile(b, 10) if len(b) >= 3 else np.nan
    return r


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out


def auc(a, b):
    a, b = np.asarray(a)[~np.isnan(a)], np.asarray(b)[~np.isnan(b)]
    u = st.mannwhitneyu(a, b); return u.statistic / (len(a) * len(b)), u.pvalue


def main():
    m = metadata(); m.index = m.index.map(int)
    ids = [i for i in m.index if os.path.exists(f"{D}/{i}.pkl")]
    X = pd.DataFrame({i: medidas(i) for i in ids}).T
    V = list(X.columns)
    X = X.join(m[["Dataset", "Target", "Ship ID", "DETAILED TYPE"]])
    X["classe"] = X.Target.map(CL)
    X["tipo"] = X["DETAILED TYPE"].where(X["DETAILED TYPE"].isin(["Pilot Vessel", "Tug", "Fishing Vessel"]), "outros")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    X.to_csv(f"{OUT}_medidas.csv")

    nav = X[X.Target < 3].copy()
    nav[V] = nav[V] - nav.groupby("Dataset")[V].transform("median")  # centrado no subconjunto
    S = nav.groupby("Ship ID").agg({**{v: "median" for v in V}, "classe": "first", "tipo": "first"})
    bg = X[X.Target == 3]
    raw = X[X.Target < 3].groupby("Ship ID").agg({**{v: "median" for v in V}, "classe": "first"})

    rows = []
    for v in V:
        s, l, md = (S.loc[S.classe == c, v] for c in ["SMALL", "LARGE", "MEDIUM"])
        a1, p1 = auc(s, l); a3, p3 = auc(md, l)
        a2, p2 = auc(raw.loc[raw.classe == "SMALL", v], bg[v]); a4, p4 = auc(raw.loc[raw.classe == "LARGE", v], bg[v])
        rows.append({"medida": v, "SMALL med": s.median(), "MEDIUM med": md.median(), "LARGE med": l.median(),
                     "AUC SMALL×LARGE": a1, "p S×L": p1, "AUC MEDIUM×LARGE": a3, "p M×L": p3,
                     "AUC SMALL×BG (bruto)": a2, "p S×BG": p2, "AUC LARGE×BG (bruto)": a4, "p L×BG": p4})
    T = pd.DataFrame(rows)
    for c in ["p S×L", "p M×L", "p S×BG", "p L×BG"]:
        T["q" + c[1:]] = bh(T[c])
    T.to_csv(f"{OUT}_testes.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"navios: SMALL {sum(S.classe=='SMALL')}, MEDIUM {sum(S.classe=='MEDIUM')}, LARGE {sum(S.classe=='LARGE')}; BG {len(bg)} gravações")
    print(T[["medida", "SMALL med", "MEDIUM med", "LARGE med", "AUC SMALL×LARGE", "q S×L", "AUC MEDIUM×LARGE",
             "AUC SMALL×BG (bruto)", "q S×BG", "AUC LARGE×BG (bruto)"]].round(3).to_string(index=False))

    # por tipo de SMALL contra LARGE (navio)
    L = S[S.classe == "LARGE"]
    pt = []
    for t in ["Pilot Vessel", "Tug", "Fishing Vessel"]:
        x = S[(S.classe == "SMALL") & (S.tipo == t)]
        for v in V:
            a, p = auc(x[v], L[v]); pt.append({"tipo": t, "navios": len(x), "medida": v, "AUC vs LARGE": a, "p": p})
    PT = pd.DataFrame(pt); PT.to_csv(f"{OUT}_por_tipo.csv", index=False)
    print("\nAUC (tipo de SMALL > LARGE), por navio:")
    print(PT.pivot(index="medida", columns="tipo", values="AUC vs LARGE").round(2).loc[V].to_string())

    # consistência por subconjunto (navio dentro do subconjunto)
    ps = []
    for ds, g in nav.groupby("Dataset"):
        G = g.groupby("Ship ID").agg({**{v: "median" for v in V}, "classe": "first"})
        for v in V:
            a, p = auc(G.loc[G.classe == "SMALL", v], G.loc[G.classe == "LARGE", v])
            ps.append({"subconjunto": ds, "medida": v, "AUC SMALL×LARGE": a, "p": p})
    PS = pd.DataFrame(ps); PS.to_csv(f"{OUT}_por_subconjunto.csv", index=False)
    print("\nAUC SMALL×LARGE por subconjunto:")
    print(PS.pivot(index="medida", columns="subconjunto", values="AUC SMALL×LARGE").round(2).loc[V].to_string())


if __name__ == "__main__":
    main()
