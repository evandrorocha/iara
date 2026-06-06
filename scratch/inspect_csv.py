import csv, os
from collections import defaultdict, Counter

names = {0: 'SMALL', 1: 'MEDIUM', 2: 'LARGE', 3: 'BG'}

def analyze_runner_up(exp_dir, focus_pred=2):
    """For ships whose majority vote = focus_pred (LARGE=2),
    show distribution of runner-up class by true class."""

    runner_up = defaultdict(Counter)  # [true_class][runner_up_class]
    missed = Counter()               # true LARGE predicted as X

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
            target = ft[fid]
            counts = Counter(preds)
            ordered = counts.most_common()
            majority = ordered[0][0]
            runner = ordered[1][0] if len(ordered) > 1 else majority

            if majority == focus_pred:
                runner_up[target][runner] += 1
            elif target == focus_pred:
                missed[majority] += 1

    print("=== Navios com votacao majoritaria = {} ===".format(names[focus_pred]))
    print("    (segunda classe mais votada por classe real)\n")

    all_true = sorted(runner_up.keys())
    all_runners = sorted(set(r for c in runner_up.values() for r in c.keys()))

    header = "  {:<10}  {:>6}".format("Classe real", "Total")
    for r in all_runners:
        header += "  {:>9}".format("2a=" + names.get(r, str(r)))
    print(header)
    print("  " + "-" * (18 + 11 * len(all_runners)))

    for t in all_true:
        total = sum(runner_up[t].values())
        row = "  {:<10}  {:>6}".format(names.get(t, str(t)), total)
        for r in all_runners:
            n = runner_up[t].get(r, 0)
            pct = n / total * 100 if total > 0 else 0
            row += "  {:>5} {:>3.0f}%".format(n, pct)
        print(row)

    print()
    print("=== Verdadeiros {} NAO votados como {} (perdidos para outros) ===".format(
        names[focus_pred], names[focus_pred]))
    total_missed = sum(missed.values())
    for cls, cnt in missed.most_common():
        print("  Predito como {:<8}: {:>4}  ({:.1f}%)".format(
            names.get(cls, str(cls)), cnt, cnt / total_missed * 100 if total_missed else 0))
    print("  Total perdidos: {}".format(total_missed))


exp = "results/trainings/tests/svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C2.0"
print("Experimento: HYBRID multiclasse (MEL+LOFAR, m=4000, C=2.0)\n")
analyze_runner_up(exp, focus_pred=2)
