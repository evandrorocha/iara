"""Se o modelo aprendesse a identificar LANCHAS DE PRÁTICO, o recall seria bom? (RF LOFAR, split por navio, NB29)
  A) binário: lancha de prático x todo o resto (inclui BACKGROUND)
  B) 5 classes: SMALL dividida em PRÁTICO e SMALL-outros; depois PRÁTICO+SMALL-outros juntados de volta em SMALL
     para comparar com o RF 4 classes original (augnavio_rf_lofar_base_4c)
Avaliação por gravação (voto majoritário das janelas), mesma regra do pipeline."""
import os, pickle, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, "notebooks/src")
from eval_ship_split_5x2cv import CACHES, FOLDS_CSV, fold_files, matrices, metadata, fit_predict
import iara_results as R

m = metadata(); pil = set(m.index[m["DETAILED TYPE"] == "Pilot Vessel"])
folds = pd.read_csv(FOLDS_CSV, dtype={"Ship ID": str})
cache = pickle.load(open(CACHES["lofar"][0], "rb"))

def vote(fte, pred):
    d = pd.DataFrame({"f": fte, "p": pred})
    return d.groupby("f").p.agg(lambda s: s.value_counts().index[0])

rows = []
for k in range(10):
    tr, te = fold_files(folds, "split_navio", k, "trn"), fold_files(folds, "split_navio", k, "test")
    # A) binário
    trA = {f: int(f in pil) for f in tr}; teA = {f: int(f in pil) for f in te}
    Xtr, ytr, _ = matrices(cache, trA); Xte, yte, fte = matrices(cache, teA)
    pA = vote(fte, fit_predict("rf", Xtr, ytr, Xte)[0])
    # B) 5 classes: 4 = prático
    trB = {f: (4 if f in pil else c) for f, c in tr.items()}; teB = {f: (4 if f in pil else c) for f, c in te.items()}
    Xtr, ytr, _ = matrices(cache, trB); Xte, yte, fte = matrices(cache, teB)
    pB = vote(fte, fit_predict("rf", Xtr, ytr, Xte)[0])
    for f in pA.index:
        rows.append({"fold": k, "File": f, "y4": te[f], "pratico": f in pil, "predA": pA[f], "predB": pB[f]})
    print(k, flush=True)
D = pd.DataFrame(rows)
D.to_csv("reports/small_medium/teste_pratico.csv", index=False)

print("\n=== A) binário prático x resto (split por navio, RF LOFAR)")
pr = D[D.pratico]; ot = D[~D.pratico]
print(f"recall prático: {(pr.predA == 1).mean() * 100:.1f}%  |  especificidade (resto não chamado de prático): {(ot.predA == 0).mean() * 100:.1f}%")
print(f"precisão (das chamadas prático, quantas são): {pr.predA.sum() / D.predA.sum() * 100:.1f}%")
print("falsos 'prático' por classe (%):", (ot.groupby("y4").predA.mean() * 100).round(1).rename({0: "SMALL-outros", 1: "MEDIUM", 2: "LARGE", 3: "BG"}).to_dict())

print("\n=== B) 5 classes (4 = prático)")
lab = {0: "SMALL-outros", 1: "MEDIUM", 2: "LARGE", 3: "BG", 4: "PRÁTICO"}
D["yB"] = np.where(D.pratico, 4, D.y4)
print((pd.crosstab(D.yB.map(lab), D.predB.map(lab), normalize="index") * 100).round(1).to_string())
# juntando prático + SMALL-outros = SMALL, comparar com o RF 4c
D["predB4"] = D.predB.replace(4, 0)
rec = lambda y, p: [np.mean(p[y == c] == c) * 100 for c in range(4)]
r5 = rec(D.y4.values, D.predB4.values)
base = pd.concat(R.file_predictions("augnavio_rf_lofar_base_4c").values()); base["File"] = base.File.astype(int)
base = base.set_index(["File"]).Prediction
D["pbase"] = [base.loc[f].iloc[0] if isinstance(base.loc[f], pd.Series) else base.loc[f] for f in D.File]  # mesmo fold
r4 = rec(D.y4.values, D.pbase.values)
print("\nrecall 4 classes (%):       SMALL  MEDIUM  LARGE  BG   ACC")
print("  RF 4 classes (base)   ", "  ".join(f"{x:5.1f}" for x in r4), f"{np.mean(r4):5.1f}")
print("  RF 5 classes juntado  ", "  ".join(f"{x:5.1f}" for x in r5), f"{np.mean(r5):5.1f}")
for nome, msk in [("prático", D.pratico), ("SMALL-outros", (D.y4 == 0) & ~D.pratico)]:
    print(f"  SMALL acertada, só {nome}: base {np.mean(D.pbase[msk] == 0) * 100:.1f}%  ×  5 classes {np.mean(D.predB4[msk] == 0) * 100:.1f}%")
