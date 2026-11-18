"""MEDIUM x LARGE (A/C, alinhado pelo CPA): o resultado pode ser viciado?
1) permutação de rótulos por navio; 2) metades de navios; 3) A e C separados; 4) distribuição de A/C por classe."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "scratch/apres_small_medium"); sys.path.insert(0, "scratch")
import cpa_fatias as C
m = C.metadata(); D = C.build(m)
d = D.join(m[["Target", "Ship ID", "Dataset"]], on="File")
d = d[d.Target.isin([1, 2])]
print("gravações por classe e subconjunto:"); print(d.drop_duplicates("File").groupby(["Target", "Dataset"]).size().unstack().to_string())
CELLS = [("-30..+0", "37-58"), ("+90..+120", "37-58"), ("-30..+0", "141-221"), ("-30..+0", "5109-8000")]
nav = d.groupby(["fatia", "Ship ID"]).agg({**{b: "mean" for b in C.BNAMES}, "Target": "first"}).reset_index()
def auc(x, y, col):
    u, v = x[y == 2][col].values, x[y == 1][col].values
    r = pd.concat([pd.Series(u), pd.Series(v)]).rank().values
    return (r[:len(u)].sum() - len(u) * (len(u) + 1) / 2) / (len(u) * len(v))
rng = np.random.default_rng(0)
print("\n1) permutação (rótulo MEDIUM/LARGE embaralhado entre navios, 2000x):")
for s, b in CELLS:
    x = nav[nav.fatia == s]; y = x.Target.values; obs = auc(x, y, b)
    null = np.array([auc(x, rng.permutation(y), b) for _ in range(2000)])
    print(f"  {s:>10} {b:>9}: AUC real {obs:.2f}; embaralhado {null.mean():.2f} (95% entre {np.percentile(null,2.5):.2f} e {np.percentile(null,97.5):.2f}); "
          f"vezes que o acaso igualou o real: {(np.abs(null-.5) >= abs(obs-.5)).sum()}/2000")
print("\n2) metades de navios (100 sorteios): descobre numa metade, confere na outra")
ships = d.groupby("Ship ID").Target.first()
res = {c: [] for c in CELLS}
for _ in range(100):
    h = set()
    for c in [1, 2]:
        s = ships[ships == c].index.values; h |= set(rng.choice(s, len(s) // 2, replace=False))
    o = set(ships.index) - h
    Ta = C.compara(D, m, 1, 2, ships=h).set_index(["fatia", "banda"]); Tb = C.compara(D, m, 1, 2, ships=o).set_index(["fatia", "banda"])
    for k in CELLS:
        res[k].append((np.sign(Ta.loc[k, "AUC"] - .5) == np.sign(Tb.loc[k, "AUC"] - .5), Tb.loc[k, "p"] < 0.05, Tb.loc[k, "AUC"]))
for k, v in res.items():
    v = np.array(v, dtype=float)
    print(f"  {k}: mesmo sentido {v[:,0].mean()*100:.0f}%, p<0,05 na outra metade {v[:,1].mean()*100:.0f}%, AUC na outra metade {v[:,2].min():.2f}-{v[:,2].max():.2f}")
print("\n3) A (CPA ≤ 250 m) e C (250–500 m) separados:")
for ds in ["A", "C"]:
    Dx = D[D.File.isin(m.index[m.Dataset == ds])]
    T = C.compara(Dx, m, 1, 2).set_index(["fatia", "banda"])
    n = T[["n_a", "n_b"]].iloc[0].tolist()
    print(f"  {ds} (navios MEDIUM {n[0]}, LARGE {n[1]}): q<0,05 em {(T.q<.05).sum()}/{len(T)}; " +
          "; ".join(f"{b}@{s} AUC {T.loc[(s,b),'AUC']:.2f}" for s, b in CELLS))
