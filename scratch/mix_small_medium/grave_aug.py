"""O augmentation com fundo de mar apaga o grave dos LARGE?
Compara a energia LOFAR (dB acima do TPSW) em 31–86 Hz (banda LARGE) e 90–221 Hz (banda dos < 100 m) entre o LOFAR
original de cada gravação LARGE e as suas cópias misturadas com fundo de mar agitado/calmo
(cache_lofar_mixbg_{agitado,calmo}.pkl), em função da razão sinal/fundo sorteada (rsf_db).
Referência: mediana por gravação de SMALL+MEDIUM (< 100 m) e de LARGE no LOFAR original.
Saída: reports/small_medium/grave_aug.csv (uma linha por cópia)"""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "scratch")
from eval_ship_split_5x2cv import metadata

F = (np.arange(1024) + 1) * 16000 / 2048
B1 = (F >= 31) & (F < 86)
B2 = (F >= 90) & (F < 221)
m = metadata()
cache = pickle.load(open("data/iara_processed/cache_lofar.pkl", "rb"))


def bandas(x):
    return x[:, B1].mean(), x[:, B2].mean()


orig = pd.DataFrame({f: bandas(x) for f, x in cache.items()}, index=["b1", "b2"]).T
orig = orig.join(m.Target)
ref_mix = orig[orig.Target.isin([0, 1])].b1.median()
ref_large = orig[orig.Target == 2].b1.median()
print(f"Mediana 31–86 Hz no LOFAR original: LARGE {ref_large:.2f} dB, SMALL+MEDIUM {ref_mix:.2f} dB")

rows = []
for g in ["agitado", "calmo"]:
    mix = pickle.load(open(f"data/iara_processed/cache_lofar_mixbg_{g}.pkl", "rb"))
    for _, s in mix["sorteios"].iterrows():
        y = mix["janelas"][(s.ID, s.copia)]
        b1, b2 = bandas(y)
        rows.append({"grupo": g, "ID": s.ID, "copia": s.copia, "rsf_db": s.rsf_db, "b1": b1, "b2": b2,
                     "b1_orig": orig.b1[s.ID], "b2_orig": orig.b2[s.ID]})
D = pd.DataFrame(rows)
D["d1"] = D.b1 - D.b1_orig
D["d2"] = D.b2 - D.b2_orig
D["abaixo_mix"] = D.b1 < ref_mix
D.to_csv("reports/small_medium/grave_aug.csv", index=False)

pd.set_option("display.width", 200)
D["faixa_rsf"] = pd.cut(D.rsf_db, [0, 5, 10, 15, 20], include_lowest=True)
orig_abaixo = (orig[orig.Target == 2].b1 < ref_mix).mean() * 100
print(f"LARGE originais com 31–86 Hz abaixo da mediana SMALL+MEDIUM: {orig_abaixo:.0f}%\n")
for g, d in D.groupby("grupo"):
    print(f"== fundo {g}: {len(d)} cópias de {d.ID.nunique()} LARGE")
    print(f"   31–86 Hz: original {d.b1_orig.mean():.2f} -> cópia {d.b1.mean():.2f} dB (Δ {d.d1.mean():+.2f}; "
          f"queda em {(d.d1 < 0).mean() * 100:.0f}% das cópias)")
    print(f"   90–221 Hz: original {d.b2_orig.mean():.2f} -> cópia {d.b2.mean():.2f} dB (Δ {d.d2.mean():+.2f})")
    print(f"   cópias com 31–86 Hz abaixo da mediana SMALL+MEDIUM: {d.abaixo_mix.mean() * 100:.0f}%")
    t = d.groupby("faixa_rsf", observed=True).agg(n=("d1", "size"), d1=("d1", "mean"), d2=("d2", "mean"),
                                                  abaixo_mix=("abaixo_mix", "mean"))
    t["abaixo_mix"] *= 100
    print(t.round(2).to_string(), "\n")
