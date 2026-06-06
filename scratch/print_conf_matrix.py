import csv, math, os
from collections import defaultdict

NAMES = {0: 'SMALL', 1: 'MEDIUM', 2: 'LARGE', 3: 'BG'}
N = 4

def build_conf_matrix_by_audio(exp_dir):
    cm = defaultdict(int)
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
                ft[fid] = int(row["Target"])
                fp[fid].append(int(row["Prediction"]))
        for fid, preds in fp.items():
            pred = max(set(preds), key=preds.count)
            cm[(ft[fid], pred)] += 1
    return cm

def print_conf_matrix(label, cm):
    totals_true = {c: sum(cm[(c, p)] for p in range(N)) for c in range(N)}
    totals_pred = {p: sum(cm[(t, p)] for t in range(N)) for p in range(N)}
    col_w = 9

    print("=" * 62)
    print("Modelo: {}".format(label))
    print("by_audio (votacao majoritaria), 10 folds acumulados")
    print()

    hdr = "  {:<8}".format("Real\\Pred")
    for p in range(N):
        hdr += "  {:>{}}".format(NAMES[p], col_w)
    hdr += "  {:>{}}".format("Recall", col_w)
    print(hdr)
    print("  " + "-" * (10 + (col_w + 2) * (N + 1)))

    recalls = {}
    for t in range(N):
        row = "  {:<8}".format(NAMES[t])
        total = totals_true[t]
        for p in range(N):
            n = cm[(t, p)]
            pct = n / total * 100 if total > 0 else 0
            row += "  {:>{}.1f}%".format(pct, col_w - 1)
        rec = cm[(t, t)] / total * 100 if total > 0 else 0
        recalls[t] = rec
        row += "  {:>{}.1f}%".format(rec, col_w - 1)
        print(row)

    print("  " + "-" * (10 + (col_w + 2) * (N + 1)))
    prec_row = "  {:<8}".format("Prec.")
    for p in range(N):
        total = totals_pred[p]
        prec = cm[(p, p)] / total * 100 if total > 0 else 0
        prec_row += "  {:>{}.1f}%".format(prec, col_w - 1)
    print(prec_row)
    print()

    recs = [recalls[c] / 100 for c in range(N)]
    mean_r = sum(recs) / N
    gmean_r = math.exp(sum(math.log(max(r, 1e-9)) for r in recs) / N)
    sp = math.sqrt(mean_r * gmean_r) * 100
    print("  ACC (balanced): {:.2f}%".format(mean_r * 100))
    print("  SP:             {:.2f}%".format(sp))
    print()


base = "results/trainings/tests"

exps = [
    ("MLP MEL",   "mlp"),
    ("MLP LOFAR", "mlp_lofar"),
    ("SVM Cascata HYBRID-MC+SM+SL (melhor)",
     "svm_nystroem_4000_hybrid_cascade_pretrained_svm_nystroem_6000_hybrid_binary_small_medium_pca64_elasticnet_l1r0.15_smallalso_fbmc_large_svm_nystroem_6000_hybrid_binary_small_large_pca8_elasticnet_l1r0.15_hybmc4000C2.0"),
]

for label, exp in exps:
    path = os.path.join(base, exp)
    if not os.path.exists(path):
        print("NAO ENCONTRADO: {}".format(path))
        continue
    cm = build_conf_matrix_by_audio(path)
    print_conf_matrix(label, cm)
