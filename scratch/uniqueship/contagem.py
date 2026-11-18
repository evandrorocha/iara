"""UniqueShip: quantos navios por classe e faixa de comprimento (limiares do IARA: <50, 50-100, >100 m),
continuidade das passagens (trechos de 5 s consecutivos) e cobertura nos folds main5_balanced.
Saída: reports/uniqueship/contagem_*.csv"""
import os, glob
import numpy as np, pandas as pd
R = "data/uniqueship/ONC_segments_5s"; OUT = "reports/uniqueship"; os.makedirs(OUT, exist_ok=True)
cols = ["label", "path", "duration_sec", "MMSI", "length", "beam", "hydrophone", "mean_distance", "mean_sog", "date"]
d = pd.read_csv(f"{R}/metadata_5s.csv", usecols=cols).drop_duplicates("path")
print("trechos:", len(d), "| horas:", round(d.duration_sec.sum() / 3600))
v = d[d.MMSI > 0].copy()
v["tam"] = pd.cut(v.length, [0, 50, 100, 1e4], right=False, labels=["SMALL<50", "MEDIUM50-100", "LARGE>100"])
v["tam"] = v.tam.cat.add_categories("sem_length").fillna("sem_length")
# um tamanho por navio (mediana do length)
nav = v.groupby("MMSI").agg(label=("label", lambda s: s.mode()[0]), length=("length", "median"),
                            trechos=("path", "size"), hidros=("hydrophone", "nunique"))
nav["tam"] = pd.cut(nav.length, [0, 50, 100, 1e4], right=False, labels=["SMALL<50", "MEDIUM50-100", "LARGE>100"])
nav["tam"] = nav.tam.cat.add_categories("sem_length").fillna("sem_length")
nav["horas"] = nav.trechos * 5 / 3600
T = pd.crosstab(nav.label, nav.tam, margins=True); T.to_csv(f"{OUT}/contagem_navios_classe_tamanho.csv")
print("\nNAVIOS por classe x tamanho:\n", T)
H = nav.pivot_table(index="label", columns="tam", values="horas", aggfunc="sum", observed=False).round(1)
H.to_csv(f"{OUT}/contagem_horas_classe_tamanho.csv"); print("\nHORAS:\n", H)
# variação do length dentro do navio (consistência do AIS)
sd = v.groupby("MMSI").length.agg(lambda s: s.max() - s.min()); print("\nnavios com length variando >5 m:", (sd > 5).sum())
# comprimento por classe
print("\nlength por classe (quantis 10/50/90):\n", nav.groupby("label").length.quantile([.1, .5, .9]).unstack().round(0))
# passagens: trechos consecutivos do mesmo MMSI (início = fim do anterior)
t = v.path.str.extract(r"/(\d{8}T\d{6}\.\d{3}Z)_(\d{8}T\d{6}\.\d{3}Z)")
v["t0"] = pd.to_datetime(t[0], format="%Y%m%dT%H%M%S.%fZ"); v["t1"] = pd.to_datetime(t[1], format="%Y%m%dT%H%M%S.%fZ")
v = v.sort_values(["MMSI", "t0"])
novo = (v.MMSI != v.MMSI.shift()) | ((v.t0 - v.t1.shift()).dt.total_seconds().abs() > 0.5)
v["passagem"] = novo.cumsum()
P = v.groupby("passagem").agg(MMSI=("MMSI", "first"), label=("label", "first"), tam=("tam", "first"), n=("path", "size"),
                              dist_min=("mean_distance", "min"))
P["dur_s"] = P.n * 5; P.to_csv(f"{OUT}/contagem_passagens.csv")
print("\npassagens:", len(P), "| duração (s) quantis:", P.dur_s.quantile([.1, .25, .5, .75, .9]).to_dict())
print("passagens >= 60 s:", (P.dur_s >= 60).sum(), "| >= 120 s:", (P.dur_s >= 120).sum())
print("\npassagens por navio (mediana):", P.groupby("MMSI").size().median())
print("\nduração mediana da passagem por classe x tamanho (s):\n",
      P.pivot_table(index="label", columns="tam", values="dur_s", aggfunc="median", observed=False))
# folds main5_balanced
for sp in ["main5_balanced", "other7"]:
    f = pd.concat([pd.read_csv(p, usecols=["MMSI", "path"]).assign(fold=p.split(os.sep)[-2] if os.sep in p else p.split("/")[-2],
                   part=os.path.basename(p)) for p in glob.glob(f"{R}/splits/{sp}/fold_*/metadata_5s_*.csv")])
    m = nav.reset_index()[["MMSI", "label", "tam"]]
    x = f[f.MMSI > 0].drop_duplicates("MMSI").merge(m, on="MMSI")
    print(f"\n{sp}: navios únicos por classe x tamanho:\n", pd.crosstab(x.label, x.tam))
