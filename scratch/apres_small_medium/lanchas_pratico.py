"""Lanchas de prático (Pilot Vessel, SMALL): o que diferencia gravações acertadas x erradas da MESMA lancha?
Rótulo por gravação: fração de acerto do MLP LOFAR no split por gravação (split_id_mlp_lofar_4c; cada gravação é
testada 5 vezes) e no split por navio. Comparações dentro do navio (z-score / ranks dentro de cada lancha)."""
import os, sys, numpy as np, pandas as pd
from scipy import stats as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, "notebooks/src")
import iara_results as R
from eval_ship_split_5x2cv import metadata
m = metadata()
F = (np.arange(1024) + 1) * 16000 / 2048
S = pd.read_pickle("data/iara_processed/cache_lofar_espectro_medio.pkl")
NIV = pd.read_pickle("data/iara_processed/cache_psd24k_niveis.pkl"); NIV.index = NIV.index.astype(int)
AMB = pd.read_pickle("data/iara_processed/cache_amb_min_psd24k.pkl")

def per_rec(exp):
    P = pd.concat(R.file_predictions(exp).values()); P["File"] = P.File.astype(int)
    g = P.groupby("File")
    out = pd.DataFrame({"acerto": g.apply(lambda d: (d.Prediction == d.Target).mean()),
                        "agree": g.agreement.mean()})
    for c, n in enumerate(["S", "M", "L", "B"]):
        out[f"p{n}"] = g.Prediction.apply(lambda s: (s == c).mean())
    return out

ID = per_rec("split_id_mlp_lofar_4c"); NV = per_rec("split_navio_mlp_lofar_4c")
pil = m[(m["DETAILED TYPE"] == "Pilot Vessel")].copy()
pil = pil.join(ID.add_suffix("_id")).join(NV.add_suffix("_nv"))
pil["grupo"] = np.where(pil.acerto_id >= 0.6, "acertada", np.where(pil.acerto_id <= 0.4, "errada", "oscila"))
pil["CPA"] = pil.Dataset.isin(["A", "C"]); pil["perto"] = pil.Dataset.isin(["A", "B"])
print(f"{len(pil)} gravações de lancha de prático, {pil['Ship ID'].nunique()} lanchas")
print(pil.grupo.value_counts().to_string())
print(f"correlação acerto split_id x split_navio por gravação: {st.spearmanr(pil.acerto_id, pil.acerto_nv)[0]:.2f}")
print("\nPara onde vão os erros (split_id, média por gravação errada): ",
      pil[pil.grupo == "errada"][["pS_id", "pM_id", "pL_id", "pB_id"]].mean().round(2).to_dict())
print("concordância entre janelas (agreement): acertadas %.2f  erradas %.2f" %
      tuple(pil.groupby("grupo").agree_id.mean()[["acertada", "errada"]]))

# ---- 1. metadados
print("\n=== 1. Metadados: % de erradas por categoria (acertada/errada, sem 'oscila')")
d = pil[pil.grupo != "oscila"].copy(); d["err"] = d.grupo == "errada"
for col in ["Dataset", "Rain state", "Sea state"]:
    t = d.groupby(col).err.agg(["mean", "size"]); t["mean"] *= 100
    print(f"  {col}: " + ", ".join(f"{i}: {r['mean']:.0f}% (n={int(r['size'])})" for i, r in t.iterrows()))
# por lancha
t = d.groupby("Ship ID").agg(nome=("Name", "first"), n=("err", "size"), pct_err=("err", "mean"))
t.pct_err *= 100; print(t[t.n >= 5].round(0).to_string())

# ---- 2. variáveis contínuas, dentro do navio
d = d.join(NIV).join(AMB.mean(1).rename("amb_8_23k"))
d["wind"] = pd.to_numeric(d.Wind, errors="coerce"); d["dur"] = pd.to_numeric(d["Recording time"], errors="coerce")
d["snr_proxy"] = d["nível médio"] - d["amb_8_23k"]
# bandas do LOFAR (dB acima do TPSW)
bands = {"20-31": (20, 31), "31-86": (31, 86), "86-250": (86, 250), "250-1k": (250, 1000), "1k-3k": (1000, 3000), "3k-8k": (3000, 8000)}
for b, (a, z) in bands.items():
    d[f"L{b}"] = S.loc[d.index, (F >= a) & (F < z)].mean(1).values
feats = ["dur", "wind", "nível médio", "nível máximo", "faixa dinâmica (p90 − p10)", "amb_8_23k", "snr_proxy"] + [f"L{b}" for b in bands]
print("\n=== 2. Dentro da mesma lancha (lanchas com acertadas E erradas): AUC errada>acertada e Δ mediano (errada − acertada)")
ok_ships = d.groupby("Ship ID").err.agg(lambda s: s.any() and (~s).any())
dd = d[d["Ship ID"].isin(ok_ships[ok_ships].index)].copy()
print(f"  {dd['Ship ID'].nunique()} lanchas, {len(dd)} gravações ({dd.err.sum()} erradas)")
rows = []
for f in feats:
    x = dd[f].astype(float)
    z = x - dd.groupby("Ship ID")[f].transform("median")  # centrado no navio
    ok = z.notna()
    auc = st.mannwhitneyu(z[ok & dd.err], z[ok & ~dd.err]).statistic / ((ok & dd.err).sum() * (ok & ~dd.err).sum())
    p = st.mannwhitneyu(z[ok & dd.err], z[ok & ~dd.err]).pvalue
    rows.append({"variável": f, "AUC": round(auc, 2), "p": p, "Δ": round(z[ok & dd.err].median() - z[ok & ~dd.err].median(), 2)})
T = pd.DataFrame(rows); T["p"] = T.p.map(lambda v: f"{v:.1e}"); print(T.to_string(index=False))

# ---- 3. espectro: com qual classe a lancha errada se parece?
print("\n=== 3. Espectro médio LOFAR (dB acima do TPSW) por banda: lanchas acertadas/erradas x classes")
lab = m.Target.map({0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BG"})
nonpil = m[(m.Target == 0) & (m["DETAILED TYPE"] != "Pilot Vessel")].index
groups = {"lancha acertada": d[~d.err].index, "lancha errada→MEDIUM": d[d.err & (d.pM_id >= d.pL_id)].index,
          "lancha errada→LARGE": d[d.err & (d.pL_id > d.pM_id)].index, "SMALL não-lancha": nonpil,
          "MEDIUM": m[m.Target == 1].index, "LARGE": m[m.Target == 2].index, "BACKGROUND": m[m.Target == 3].index}
rows = []
for g, idx in groups.items():
    idx = [i for i in idx if i in S.index]
    r = {"grupo": g, "n": len(idx)}
    for b, (a, z) in bands.items():
        r[b] = round(S.loc[idx, (F >= a) & (F < z)].mean(1).mean(), 2)
    rows.append(r)
G = pd.DataFrame(rows); print(G.to_string(index=False))
# distância do perfil de 0-2 kHz
prof = lambda idx: S.loc[[i for i in idx if i in S.index], F < 2000].mean(0).values
print("\nRMS (dB) entre perfis 0–2 kHz:")
names = list(groups)
M = pd.DataFrame([[np.sqrt(np.mean((prof(groups[a]) - prof(groups[b])) ** 2)) for b in names] for a in names], index=names, columns=names)
print(M.round(2).iloc[:3].to_string())
pil.to_csv("reports/small_medium/lanchas_pratico.csv")
