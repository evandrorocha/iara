import csv, math, os
from collections import defaultdict

def eval_exp(exp_dir, remap=None):
    """remap: dict to convert prediction labels back (e.g. {1: 2} for SL specialist)"""
    cm_total = defaultdict(int)
    folds_done = 0
    for fold in range(10):
        fold_dir = os.path.join(exp_dir, "eval", "fold_{}".format(fold))
        if not os.path.exists(fold_dir):
            continue
        csvs = [f for f in os.listdir(fold_dir) if "test" in f and f.endswith(".csv")]
        if not csvs:
            continue
        ft, fp = {}, defaultdict(list)
        with open(os.path.join(fold_dir, csvs[0])) as f:
            for row in csv.DictReader(f):
                fid = int(row["File"])
                t = int(row["Target"])
                p = int(row["Prediction"])
                if remap:
                    t = remap.get(t, t)
                    p = remap.get(p, p)
                ft[fid] = t
                fp[fid].append(p)
        for fid, preds in fp.items():
            pred = max(set(preds), key=preds.count)
            cm_total[(ft[fid], pred)] += 1
        folds_done += 1

    names = {0: 'SMALL', 2: 'LARGE'}
    print("Folds: {}".format(folds_done))
    classes = sorted(set(k[0] for k in cm_total) | set(k[1] for k in cm_total))
    for r in classes:
        row_total = sum(cm_total[(r, c)] for c in classes)
        if row_total == 0:
            continue
        recall = cm_total[(r, r)] / row_total * 100
        row_str = "  {:<8}:".format(names.get(r, str(r)))
        for c in classes:
            row_str += "  {}={:3d}".format(names.get(c, str(c)), cm_total[(r, c)])
        row_str += "   Recall={:.1f}%".format(recall)
        print(row_str)


base = "results/trainings/tests"

print("=== Especialista SL (SMALL vs LARGE) — avaliação standalone ===")
print("  (labels internos: SMALL=0, LARGE=1 remapeado para 2)")
print()
sl_dir = os.path.join(base, "svm_nystroem_6000_hybrid_binary_small_large_pca8_elasticnet_l1r0.15")
# SL specialist: interno usa 0=SMALL, 1=LARGE; remapear 1->2 para comparar com classes reais
eval_exp(sl_dir, remap={1: 2})
