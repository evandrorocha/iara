"""Faixa de pico (14 bandas log 15 Hz-8 kHz, LOFAR médio, dB acima do fundo) por navio e por gravação, para SMALL/MEDIUM/LARGE."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "scratch"); from eval_ship_split_5x2cv import metadata
m = metadata(); S = pd.read_pickle("data/iara_processed/cache_lofar_espectro_medio.pkl")
F = (np.arange(1024) + 1) * 16000 / 2048; E = np.geomspace(15, 8000, 15)
NB = [f"{a:.0f}-{b:.0f} Hz" for a, b in zip(E[:-1], E[1:])]
B = pd.DataFrame({n: S.loc[:, (F >= a) & (F < b)].mean(1) for n, a, b in zip(NB, E[:-1], E[1:])})
meta = m.loc[B.index, ["Target", "Ship ID", "Dataset"]]; meta = meta[meta.Target < 3]; B = B.loc[meta.index]
CL = {0: "SMALL", 1: "MEDIUM", 2: "LARGE"}
nav = S.loc[meta.index].groupby([meta.Target, meta["Ship ID"]]).mean()
Bn = pd.DataFrame({n: nav.loc[:, (F >= a) & (F < b)].mean(1) for n, a, b in zip(NB, E[:-1], E[1:])})
pn = Bn.idxmax(1).rename("pico").reset_index()
pg = B.idxmax(1).rename("pico").to_frame().join(meta)
out = []
for nome, df in [("navios", pn), ("gravacoes", pg)]:
    t = pd.crosstab(df.pico, df.Target).reindex(index=NB, columns=[0, 1, 2]).fillna(0).astype(int).rename(columns=CL)
    p = (100 * t / t.sum()).round(0).astype(int)
    print(f"\n=== por {nome} (total: {t.sum().to_dict()}) ==="); print(pd.concat({"n": t, "%": p}, axis=1).to_string())
    out.append(pd.concat({"n": t, "%": p}, axis=1).add_prefix(nome + "_" if False else ""))
pd.concat({"navios": out[0], "gravacoes": out[1]}, axis=1).to_csv("reports/small_medium/faixa_pico_classes.csv")
# LARGE por subconjunto (gravações): com CPA (A/C) x sem (B/D)
L = pg[pg.Target == 2]; print("\nLARGE por subconjunto (gravações, %):")
print((100 * pd.crosstab(L.pico, L.Dataset, normalize="columns")).reindex(NB).fillna(0).round(0).astype(int).to_string())
