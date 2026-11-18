exec(open("scratch/small_assinatura/subconjunto_d.py", encoding="utf-8").read().split("# passo 0")[0])
X = mel
LD = nav(X, sel("LARGE", ["D"]))
sm = m[(m.classe == "SMALL") & (m.Dataset == "D")]
acs = set(m[(m.classe == "SMALL") & m.Dataset.isin(["A", "C"])].ship)
grupos = {"todos": sm.index, "lanchas": sm.index[sm["DETAILED TYPE"] == "Pilot Vessel"],
          "rebocadores": sm.index[sm["DETAILED TYPE"] == "Tug"], "pesca": sm.index[sm["DETAILED TYPE"] == "Fishing Vessel"],
          "navios também em A/C": sm.index[sm.ship.isin(acs)], "navios só no D": sm.index[~sm.ship.isin(acs)],
          "MEDIUM (referência)": sel("MEDIUM", ["D"])}
rng = np.random.default_rng(0)
for g, ix in grupos.items():
    A = nav(X, ix)
    U, p = mannwhitneyu(A.values, LD.values, axis=0); q = false_discovery_control(p)
    # controle de tamanho: subamostra de LARGE não muda n; mostra também AUC mediana nas faixas < 1 kHz
    auc = U / len(A) / len(LD)
    print(f"{g:25s} navios {len(A):3d}  faixas q<0,05 {100*(q<.05).mean():4.0f}%  AUC mediana <1 kHz {np.median(auc[FM<1000]):.2f}  >1 kHz {np.median(auc[FM>1000]):.2f}")
# mesmo com A e C: SMALL x LARGE no A+C, por grupo
print("--- A+C")
LAC = nav(X, sel("LARGE", ["A", "C"]))
s2 = m[(m.classe == "SMALL") & m.Dataset.isin(["A", "C"])]
for g, ix in {"todos": s2.index, "lanchas": s2.index[s2["DETAILED TYPE"] == "Pilot Vessel"], "rebocadores": s2.index[s2["DETAILED TYPE"] == "Tug"], "pesca": s2.index[s2["DETAILED TYPE"] == "Fishing Vessel"]}.items():
    A = nav(X, ix); U, p = mannwhitneyu(A.values, LAC.values, axis=0); q = false_discovery_control(p); auc = U / len(A) / len(LAC)
    print(f"{g:25s} navios {len(A):3d}  faixas q<0,05 {100*(q<.05).mean():4.0f}%  AUC mediana <1 kHz {np.median(auc[FM<1000]):.2f}")
# dispersão entre navios (IQR da medida MEL grave−agudo)
med = MED.columns[1]
for c in ("SMALL", "MEDIUM", "LARGE"):
    print(c, {ds: round(float(nav(MED[[med]], sel(c, list(ds)))[med].quantile(.75) - nav(MED[[med]], sel(c, list(ds)))[med].quantile(.25)), 4) for ds in ("AC", "D")})
