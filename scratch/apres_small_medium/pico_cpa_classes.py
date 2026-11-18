"""Pico em 37-90 Hz depende do CPA também em SMALL e MEDIUM? (1) % de gravações com pico 37-90 Hz por subconjunto;
(2) energia 37-58 e 58-90 Hz ao longo da passagem (fatias alinhadas pelo CPA, A/C), mediana por navio."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "scratch"); sys.path.insert(0, "scratch/apres_small_medium")
from eval_ship_split_5x2cv import metadata
import cpa_fatias as C
m = metadata(); S = pd.read_pickle("data/iara_processed/cache_lofar_espectro_medio.pkl")
F = (np.arange(1024) + 1) * 16000 / 2048; E = np.geomspace(15, 8000, 15)
NB = [f"{a:.0f}-{b:.0f}" for a, b in zip(E[:-1], E[1:])]
B = pd.DataFrame({n: S.loc[:, (F >= a) & (F < b)].mean(1) for n, a, b in zip(NB, E[:-1], E[1:])})
g = B.idxmax(1).rename("pico").to_frame().join(m[["Target", "Ship ID", "Dataset"]]); g = g[g.Target < 3]
g["p3790"] = g.pico.isin(["37-58", "58-90"])
HB = ["237", "238"]
CL = {0: "SMALL", 1: "MEDIUM", 2: "LARGE"}
print("1) % de gravações com pico em 37-90 Hz, por subconjunto (n entre parênteses)")
rows = {}
for t in [0, 1, 2]:
    x = g[g.Target == t]; rows[CL[t]] = {d: f"{100*x[x.Dataset==d].p3790.mean():.0f}% ({(x.Dataset==d).sum()})" for d in "ABCD"}
x = g[(g.Target == 1) & ~g["Ship ID"].astype(str).isin(HB)]
rows["MEDIUM sem HANG BO"] = {d: f"{100*x[x.Dataset==d].p3790.mean():.0f}% ({(x.Dataset==d).sum()})" for d in "ABCD"}
x = g[(g.Target == 1) & g["Ship ID"].astype(str).isin(HB)]
rows["só HANG BO"] = {d: f"{100*x[x.Dataset==d].p3790.mean():.0f}% ({(x.Dataset==d).sum()})" for d in "ABCD"}
print(pd.DataFrame(rows).T.to_string())
# por navio: navios com gravações com e sem CPA -> % pico 37-90 com CPA vs sem
print("\n   pareado por navio (navios com gravações COM e SEM CPA): % pico 37-90 com CPA vs sem")
g["cpa"] = g.Dataset.isin(["A", "C"])
for t in [0, 1, 2]:
    x = g[g.Target == t].groupby(["Ship ID", "cpa"]).p3790.mean().unstack().dropna()
    print(f"   {CL[t]}: {len(x)} navios; com CPA {100*x[True].mean():.0f}%, sem CPA {100*x[False].mean():.0f}%; "
          f"navios em que sobe com CPA {(x[True]>x[False]).sum()}, cai {(x[True]<x[False]).sum()}, igual {(x[True]==x[False]).sum()}")
print("\n2) energia (dB acima do fundo) ao longo da passagem, A/C, mediana dos navios")
D = C.build(m).join(m[["Target", "Ship ID"]], on="File")
D["HB"] = D["Ship ID"].astype(str).isin(HB)
for band in ["37-58", "58-90", "90-141"]:
    nav = D.groupby(["Target", "HB", "fatia", "Ship ID"])[band].mean().reset_index()
    T = {}
    for t in [0, 1, 2]:
        T[CL[t]] = nav[nav.Target == t].groupby("fatia")[band].median()
    T["MEDIUM sem HANG BO"] = nav[(nav.Target == 1) & ~nav.HB].groupby("fatia")[band].median()
    T = pd.DataFrame(T).reindex(C.SNAMES).T
    print(f"\n   banda {band} Hz:"); print(T.round(1).to_string())
