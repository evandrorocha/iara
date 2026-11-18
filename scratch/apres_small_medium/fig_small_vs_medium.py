"""Figura small_vs_medium_auc.png da apresentação 20261116-eda-iara-dados a partir do bloco SXM_CODE do NB39."""
import sys, matplotlib
matplotlib.use("Agg")
sys.path.insert(0, "notebooks/src")
import generate_small_medium_augmentation_notebook as G
ns = {"display": lambda *a, **k: None}
exec(G.SETUP_CODE, ns)
exec(G.SXM_CODE.replace("plt.show()", "plt.savefig('presentation/20261116-eda-iara-dados/figuras/small_vs_medium_auc.png', dpi=200); plt.close('all')"), ns)
print(ns["SXM"].to_string())
