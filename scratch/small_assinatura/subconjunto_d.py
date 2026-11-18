"""Protótipo: por que, no subconjunto D, o MEL não separa SMALL de LARGE (mas separa MEDIUM de LARGE)?
Rodar na raiz, no container:  python scratch/small_assinatura/subconjunto_d.py
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, false_discovery_control

sys.path.insert(0, "scratch")
from eval_ship_split_5x2cv import metadata  # noqa: E402

P = "data/iara_processed/"
m = metadata(); m.index = m.index.map(int)
m = m[m.Target < 3].copy()
m["classe"] = m.Target.map({0: "SMALL", 1: "MEDIUM", 2: "LARGE"}); m["ship"] = m["Ship ID"].astype(str)

lofar = pd.read_pickle(P + "cache_lofar_espectro_medio.pkl"); lofar.index = lofar.index.map(int)
FL = np.arange(1024) * 16000 / 2048
mel = pd.read_pickle(P + "cache_mel_espectro_medio.pkl")["L2 (modelos)"]
import librosa  # noqa: E402
FM = librosa.mel_frequencies(n_mels=258, fmin=0, fmax=8000)[1:-1]
ps = pd.read_pickle(P + "cache_psd24k_resumo.pkl")
FP = ps["freqs"]; psd = pd.DataFrame(ps["psd_db"]); print("psd index", psd.index[:3], type(ps.get("ids")), list(ps.keys()))
psd = ps["psd_db"] if isinstance(ps["psd_db"], pd.DataFrame) else psd
psd.index = psd.index.map(int)
kp = (FP >= 20) & (FP <= 8000)
PF = psd.loc[:, kp]; PF = PF.sub(PF.mean(1), axis=0); PF.columns = FP[kp]   # forma do psd24k (dB − média), 20 Hz–8 kHz

def banda(X, F, lo, hi):
    return X.loc[:, (F >= lo) & (F < hi)].mean(1)

MED = pd.DataFrame({
    "LOFAR 31–86 Hz (dB acima do fundo)": banda(lofar, FL, 31, 86),
    "MEL L2: grave − agudo (<1 kHz menos 1–6 kHz)": banda(mel, FM, 0, 1000) - banda(mel, FM, 1000, 6000),
    "psd24k: grave relativo (31–125 Hz menos 1–4 kHz, dB)": banda(PF, FP[kp], 31, 125) - banda(PF, FP[kp], 1000, 4000),
})
REPS = {"LOFAR": (lofar.loc[:, (FL >= 20) & (FL <= 6000)], None), "MEL L2": (mel, None), "psd24k forma": (PF, None)}

def nav(X, ids):
    ids = [i for i in ids if i in X.index]
    return X.loc[ids].groupby(m.loc[ids, "ship"].values).mean()

def sel(c, ds):
    return m.index[(m.classe == c) & m.Dataset.isin(ds)]

# passo 0: o efeito existe nas três representações?
rows = []
for nome, (X, _) in REPS.items():
    for ds in "ABCD":
        for a, b in (("SMALL", "LARGE"), ("MEDIUM", "LARGE"), ("SMALL", "MEDIUM")):
            A, B = nav(X, sel(a, [ds])), nav(X, sel(b, [ds]))
            if min(len(A), len(B)) < 5: continue
            U, p = mannwhitneyu(A.values, B.values, axis=0); q = false_discovery_control(p)
            rows.append({"rep": nome, "ds": ds, "comp": f"{a}×{b}", "navios": f"{len(A)}×{len(B)}", "faixas q<0,05 (%)": round(100 * (q < .05).mean())})
T0 = pd.DataFrame(rows).pivot_table(index=["rep", "comp"], columns="ds", values="faixas q<0,05 (%)")
print(T0)
nav_n = pd.DataFrame(rows).drop_duplicates(["ds", "comp"]).pivot(index="comp", columns="ds", values="navios"); print(nav_n)

# passo 1: quem mudou? cada classe, D contra A+C, por navio (medidas-resumo)
rows = []
for c in ("SMALL", "MEDIUM", "LARGE"):
    for med in MED:
        x, y = nav(MED[[med]], sel(c, ["D"]))[med], nav(MED[[med]], sel(c, ["A", "C"]))[med]
        bb = nav(MED[[med]], sel(c, ["B"]))[med]
        U, p = mannwhitneyu(x, y)
        rows.append({"classe": c, "medida": med, "A+C": y.median(), "B": bb.median(), "D": x.median(), "AUC D>A+C": U / len(x) / len(y), "p": p})
T1 = pd.DataFrame(rows); print(T1.round(3).to_string())
for ds in "ABCD":
    print(ds, {c: round(nav(MED, sel(c, [ds])).median()[MED.columns[0]], 2) for c in ("SMALL", "MEDIUM", "LARGE")})

# passo 2: mesmo navio, D contra A/C (pareado)
rows = []
for c in ("SMALL", "MEDIUM", "LARGE"):
    d, ac = nav(MED, sel(c, ["D"])), nav(MED, sel(c, ["A", "C"]))
    com = d.index.intersection(ac.index)
    for med in MED:
        dif = d.loc[com, med] - ac.loc[com, med]
        rows.append({"classe": c, "medida": med, "navios em D e A/C": len(com), "Δ mediano (D − A/C)": dif.median(),
                     "% navios com D > A/C": 100 * (dif > 0).mean(), "p Wilcoxon": wilcoxon(dif).pvalue if len(com) >= 5 else np.nan})
T2 = pd.DataFrame(rows); print(T2.round(3).to_string())

# passo 3: distribuição dos navios SMALL no D
med0 = MED.columns[0]
d = nav(MED, sel("SMALL", ["D"])); ac = nav(MED, sel("SMALL", ["A", "C"])); L = nav(MED, sel("LARGE", ["D"]))
info = m[m.classe == "SMALL"].groupby("ship").agg(tipo=("DETAILED TYPE", "first"), comp=("Length", "first"))
nD = m[(m.classe == "SMALL") & (m.Dataset == "D")].groupby("ship").size()
d = d.join(info).assign(gravações_D=nD, em_AC=lambda t: t.index.isin(ac.index))
for med in MED:
    lm = L[med].median()
    print(med, "LARGE D mediana", round(lm, 2), "| SMALL D acima:", round(100 * (d[med] > lm).mean()), "% | SMALL A+C acima:",
          round(100 * (ac[med] > nav(MED, sel('LARGE', ['A', 'C']))[med].median()).mean()), "%")
print(d.groupby("tipo")[list(MED)].median().round(2))
print(d.groupby("em_AC")[list(MED)].median().round(2), d.em_AC.value_counts())
print(d.sort_values(med0, ascending=False).head(12).round(2).to_string())
