"""Erros SMALL<->MEDIUM se concentram nos navios perto da fronteira de comprimento (SMALL <50 m, MEDIUM 50–100 m)?
Split por navio (navio nunca visto), média de 4 modelos LOFAR. Uma linha por navio."""
import os, sys, numpy as np, pandas as pd
from scipy import stats as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, "notebooks/src")
import iara_results as R
from eval_ship_split_5x2cv import metadata
m = metadata(); m["L"] = pd.to_numeric(m.Length, errors="coerce")
EXPS = ["augnavio_mlp_lofar_base_4c", "augnavio_rf_lofar_base_4c", "augnavio_svm2000_lofar_base_4c", "augnavio_bigru_lofar_base_4c"]
P = []
for e in EXPS:
    p = pd.concat(R.file_predictions(e).values()); p["File"] = p.File.astype(int); p["exp"] = e; P.append(p)
P = pd.concat(P); P = P[P.Target < 3]
P = P.join(m[["Ship ID", "L", "DETAILED TYPE"]], on="File")
for c, n in enumerate(["S", "M"]): P[n] = (P.Prediction == c).astype(float)
P["LG"] = (P.Prediction == 2).astype(float)
# por navio: média por gravação dentro do navio (cada gravação pesa igual), depois média dos modelos
ship = P.groupby(["Target", "Ship ID"]).agg(L=("L", "first"), tipo=("DETAILED TYPE", "first"), n=("File", "nunique"),
                                           S=("S", "mean"), M=("M", "mean"), Lpred=("LG", "mean")).reset_index()
ship[["S", "M", "Lpred"]] *= 100
out = []
for c, nome, outra, col, bins in [(0, "SMALL", "MEDIUM", "M", [0, 12.5, 20, 30, 50]),
                                   (1, "MEDIUM", "SMALL", "S", [50, 65, 75, 85, 100])]:
    s = ship[ship.Target == c]
    rho, p = st.spearmanr(s.L, s[col])
    print(f"\n=== {nome} verdadeiras ({len(s)} navios): % prevista {outra} x comprimento: Spearman {rho:+.2f} (p {p:.3f})")
    s = s.assign(faixa=pd.cut(s.L, bins, right=False))
    t = s.groupby("faixa", observed=True).agg(navios=("Ship ID", "size"), grav=("n", "sum"), acerto=("S" if c == 0 else "M", "mean"),
                                              prev_outra=(col, "mean"), prev_LARGE=("Lpred", "mean"))
    print(t.round(1).to_string())
    if c == 0:
        s2 = s[s.tipo != "Pilot Vessel"]; rho, p = st.spearmanr(s2.L, s2[col])
        print(f"  sem lanchas de prático ({len(s2)} navios): Spearman {rho:+.2f} (p {p:.3f})")
    print("  por tipo:"); print(s.groupby("tipo").agg(navios=("Ship ID", "size"), L_med=("L", "median"),
          acerto=("S" if c == 0 else "M", "mean"), prev_outra=(col, "mean")).query("navios>=3").round(1).to_string())
ship.to_csv("reports/small_medium/fronteira_comprimento.csv", index=False)
