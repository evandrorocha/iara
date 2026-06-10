"""
Overfitting check for SVM LOFAR m=6000 (elasticnet).

Compares train vs test SP index per fold.
If train SP >> test SP -> overfitting.
If train SP ~= test SP -> healthy generalization.

Evaluation is done at audio level (majority vote across windows).
SP index = sqrt(arithmetic_mean(recalls) * geometric_mean(recalls))
"""
import csv, math, os, collections

EXP_DIR = "results/trainings/tests/svm_nystroem_6000_lofar_elasticnet_l1r0.15"
EXP_ID  = "svm_nystroem_6000_lofar_elasticnet_l1r0.15"
N_CLASSES = 4
CLASS_NAMES = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BG"}


def compute_sp(csv_path):
    """Read a window-level CSV and return (sp, acc, recalls) at audio level."""
    ft = {}   # file_id -> target
    fp = collections.defaultdict(list)  # file_id -> [predictions]

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            fid = int(row["File"])
            ft[fid] = int(row["Target"])
            fp[fid].append(int(row["Prediction"]))

    # Majority vote per audio file
    cm = collections.defaultdict(int)
    correct = total = 0
    for fid, preds in fp.items():
        pred = max(set(preds), key=preds.count)
        cm[(ft[fid], pred)] += 1
        if pred == ft[fid]:
            correct += 1
        total += 1

    acc = correct / total * 100 if total > 0 else 0.0

    recalls = {}
    for c in range(N_CLASSES):
        tp = cm[(c, c)]
        total_c = sum(cm[(c, p)] for p in range(N_CLASSES))
        recalls[c] = tp / total_c if total_c > 0 else 0.0

    recs = [recalls[c] for c in range(N_CLASSES)]
    am = sum(recs) / N_CLASSES
    gm = math.exp(sum(math.log(max(r, 1e-9)) for r in recs) / N_CLASSES)
    sp = math.sqrt(am * gm) * 100

    return sp, acc, recalls


print("=" * 80)
print(f"OVERFITTING CHECK: {EXP_ID}")
print("=" * 80)
print(f"{'Fold':<6} | {'Train SP':>10} | {'Test SP':>10} | {'Delta':>8} | {'Train ACC':>10} | {'Test ACC':>10}")
print("-" * 80)

train_sps, test_sps, deltas = [], [], []

for fold in range(10):
    fold_dir = os.path.join(EXP_DIR, "eval", f"fold_{fold}")
    trn_csv  = os.path.join(fold_dir, f"{EXP_ID}_multiclass_trn.csv")
    test_csv = os.path.join(fold_dir, f"{EXP_ID}_multiclass_test.csv")

    if not os.path.exists(trn_csv) or not os.path.exists(test_csv):
        print(f"Fold {fold:<2} | MISSING")
        continue

    sp_trn,  acc_trn,  _ = compute_sp(trn_csv)
    sp_test, acc_test, _ = compute_sp(test_csv)
    delta = sp_trn - sp_test

    train_sps.append(sp_trn)
    test_sps.append(sp_test)
    deltas.append(delta)

    print(f"Fold {fold:<2} | {sp_trn:>10.2f}% | {sp_test:>10.2f}% | {delta:>+7.2f}pp | {acc_trn:>10.2f}% | {acc_test:>10.2f}%")

if train_sps:
    print("-" * 80)
    mu_trn  = sum(train_sps) / len(train_sps)
    mu_test = sum(test_sps)  / len(test_sps)
    mu_d    = sum(deltas)    / len(deltas)
    sd_trn  = math.sqrt(sum((x - mu_trn) ** 2  for x in train_sps)  / len(train_sps))
    sd_test = math.sqrt(sum((x - mu_test) ** 2 for x in test_sps)   / len(test_sps))
    sd_d    = math.sqrt(sum((x - mu_d) ** 2    for x in deltas)     / len(deltas))
    print(f"{'Mean':<6} | {mu_trn:>10.2f}% | {mu_test:>10.2f}% | {mu_d:>+7.2f}pp |")
    print(f"{'Std':<6} | {sd_trn:>10.2f}pp | {sd_test:>10.2f}pp | {sd_d:>7.2f}pp |")
    print("=" * 80)
    print()
    if mu_d > 10:
        print(">> POSSÍVEL OVERFITTING: delta médio > 10 pp entre treino e teste.")
    elif mu_d > 5:
        print(">> ATENÇÃO: pequena diferença treino/teste (~{:.1f} pp). Monitorar.".format(mu_d))
    else:
        print(">> GENERALIZAÇÃO SAUDÁVEL: delta médio {:.1f} pp — sem indícios de overfitting.".format(mu_d))

print()
print("Detalhe por fold — recalls no conjunto de TESTE:")
print(f"{'Fold':<6} | {'SMALL':>8} | {'MEDIUM':>8} | {'LARGE':>8} | {'BG':>8}")
print("-" * 50)
for fold in range(10):
    fold_dir = os.path.join(EXP_DIR, "eval", f"fold_{fold}")
    test_csv = os.path.join(fold_dir, f"{EXP_ID}_multiclass_test.csv")
    if not os.path.exists(test_csv):
        continue
    _, _, recalls = compute_sp(test_csv)
    print(f"Fold {fold:<2} | {recalls[0]*100:>7.1f}% | {recalls[1]*100:>7.1f}% | {recalls[2]*100:>7.1f}% | {recalls[3]*100:>7.1f}%")
