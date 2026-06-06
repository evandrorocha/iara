import csv, math, os
from collections import defaultdict

def eval_exp(exp_dir):
    all_rec = {c: [] for c in range(4)}
    all_acc = []; all_sp = []
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
        cm = defaultdict(int); correct = total = 0
        for fid, preds in fp.items():
            pred = max(set(preds), key=preds.count)
            cm[(ft[fid], pred)] += 1
            if pred == ft[fid]:
                correct += 1
            total += 1
        rec = {c: cm[(c, c)] / max(sum(cm[(c, p)] for p in range(4)), 1) for c in range(4)}
        for c in range(4):
            all_rec[c].append(rec[c] * 100)
        all_acc.append(correct / total * 100)
        recs = [rec[c] for c in range(4)]
        gm = math.exp(sum(math.log(max(r, 1e-9)) for r in recs) / 4)
        all_sp.append(math.sqrt(sum(recs) / 4 * gm) * 100)
    return all_rec, all_acc, all_sp

def st(v):
    mu = sum(v) / len(v)
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / len(v))
    return mu, sd

base = "results/trainings/tests"
exps = [
    ("MEL m=4000 C=2 (base)", "svm_nystroem_4000_mel_elasticnet_l1r0.15_C2.0"),
    ("HYBRID m=4000 C=0.5",   "svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5"),
    ("HYBRID m=4000 C=2.0",   "svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C2.0"),
]

for label, exp in exps:
    path = os.path.join(base, exp)
    if not os.path.exists(path):
        print("{} -> DIR NAO ENCONTRADO".format(label))
        continue
    rec, acc, sp = eval_exp(path)
    n = len(acc)
    if not acc:
        print("{} -> SEM FOLDS".format(label))
        continue
    s = st(sp); a = st(acc)
    print("{} [n={}]".format(label, n))
    print("  SMALL:  {:.1f} +- {:.1f}%".format(*st(rec[0])))
    print("  MEDIUM: {:.1f} +- {:.1f}%".format(*st(rec[1])))
    print("  LARGE:  {:.1f} +- {:.1f}%".format(*st(rec[2])))
    print("  BG:     {:.1f} +- {:.1f}%".format(*st(rec[3])))
    print("  ACC:    {:.2f} +- {:.2f}%".format(*a))
    print("  SP:     {:.2f} +- {:.2f}%".format(*s))
    print()
