"""Resumo das augmentations em RF/SVM/BiGRU (+ MLP): ΔACC vs base e vs dup, F 5x2cv, folds melhores, SMALL/MEDIUM."""
import sys, pandas as pd
sys.path.insert(0, "notebooks/src"); import iara_results as R
V = ["base", "dup", "stretch", "mask", "mixnavio", "ruido", "bg"]
rows = []
for m in ["mlp", "rf", "svm2000", "bigru"]:
    base = f"augnavio_{m}_lofar_base_4c"; fb = R.fold_metrics(base)
    for v in V:
        e = f"augnavio_{m}_lofar_{v}_4c"
        try: fm = R.fold_metrics(e)
        except FileNotFoundError: continue
        d = fm.ACC - fb.ACC
        r = {"modelo": m, "variante": v, "ACC": round(fm.ACC.mean(), 1), "SP": round(fm.SP.mean(), 1),
             "SMALL": round(fm.R_SMALL.mean(), 1), "MEDIUM": round(fm.R_MEDIUM.mean(), 1),
             "LARGE": round(fm.R_LARGE.mean(), 1), "BG": round(fm.R_BACKGROUND.mean(), 1),
             "dACC": round(d.mean(), 2), "p_F5x2": None if v == "base" else round(R.f_test_5x2(e, base, metric="ACC")[1], 3),
             "melhores": int((d > 0).sum())}
        if v not in ("base", "dup"):
            try:
                fd = R.fold_metrics(f"augnavio_{m}_lofar_dup_4c")
                r["dACC_vs_dup"] = round((fm.ACC - fd.ACC).mean(), 2)
            except FileNotFoundError: pass
        rows.append(r)
T = pd.DataFrame(rows)
T.to_csv("reports/small_medium/aug_modelos_resumo.csv", index=False)
print(T.to_string(index=False))
