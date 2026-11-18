import sys, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, false_discovery_control
sys.path.insert(0, "scratch")
from eval_ship_split_5x2cv import metadata
m = metadata(); m.index = m.index.map(int)
m["classe"] = m.Target.map({0:"SMALL",1:"MEDIUM",2:"LARGE",3:"BG"}); m["ship"] = m["Ship ID"].astype(str)
m.loc[m.Target==3,"ship"] = "bg"+m.index[m.Target==3].astype(str)
L = pd.read_pickle("data/iara_processed/cache_lofar_espectro_medio.pkl"); L.index = L.index.map(int)
F = np.arange(1024)*16000/2048
# bandas de 1/3 de oitava de 31 Hz a 6,3 kHz
cent = 1000*2**(np.arange(-15,9)/3)
ids = [i for i in L.index if i in m.index]
B = pd.DataFrame({f"{c:.0f}": L.loc[ids, (F>=c/2**(1/6))&(F<c*2**(1/6))].mean(1) for c in cent})
nav = B.groupby([m.loc[ids,"classe"].values, m.loc[ids,"ship"].values]).mean()
med = nav.groupby(level=0).median().T[["SMALL","MEDIUM","LARGE","BG"]]
print("Mediana entre navios, dB acima do TPSW, por 1/3 de oitava"); print(med.round(2).to_string())
# AUC por banda dentro dos subconjuntos A/C/D, por navio, juntando via centragem no subconjunto
rows=[]
for ds in "ACD":
    s = [i for i in ids if m.loc[i,"Dataset"]==ds]
    nb = B.loc[s].groupby([m.loc[s,"classe"].values, m.loc[s,"ship"].values]).mean()
    for a,b in (("SMALL","LARGE"),("MEDIUM","LARGE"),("SMALL","MEDIUM")):
        if a not in nb.index.get_level_values(0) or b not in nb.index.get_level_values(0): continue
        A, Bb = nb.loc[a], nb.loc[b]
        U,p = mannwhitneyu(A.values,Bb.values,axis=0); q=false_discovery_control(p)
        for j,c in enumerate(B.columns): rows.append((ds,f"{a}x{b}",c,U[j]/len(A)/len(Bb),q[j]))
R = pd.DataFrame(rows, columns=["ds","comp","banda","AUC","q"])
T = R.pivot_table(index="banda", columns=["comp","ds"], values="AUC").reindex(B.columns)
print("\nAUC (1º > 2º) por subconjunto; * q<0,05"); 
S = R.assign(t=R.AUC.round(2).astype(str)+np.where(R.q<.05,"*"," ")).pivot_table(index="banda",columns=["comp","ds"],values="t",aggfunc="first").reindex(B.columns)
print(S.to_string())
