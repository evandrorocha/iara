"""Figuras da apresentação presentation/20261116-demon-mar/ a partir dos mesmos blocos de código do NB42
(notebooks/src/generate_demon_mar_notebook.py). Rodar na raiz, no container iara-dev."""
import os, sys
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, "notebooks/src")
import generate_demon_mar_notebook as G

OUT = "presentation/20261116-demon-mar/figuras"
os.makedirs(OUT, exist_ok=True)
ns = {"display": lambda *a, **k: None}
for blk, fig in [(G.SETUP_CODE, None), (G.ESPEC_CODE, "demon_classes.png"), (G.REF_CODE, None),
                 (G.AUG_CODE, "large_cpa_mar_demon.png"), (G.CONTR_CODE, "contraste_rsf.png")]:
    src = blk if fig is None else blk.replace("plt.show()", f"plt.savefig('{OUT}/{fig}', dpi=200); plt.close('all')")
    exec(src, ns)
print("ok:", os.listdir(OUT))
