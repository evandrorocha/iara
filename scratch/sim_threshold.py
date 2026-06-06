"""
Simula o impacto do cascade_threshold no recall de SMALL e MEDIUM.
Usa a fracao de votos majoritarios por navio como proxy de confianca do especialista.
"""
import os, sys, csv, math
from collections import defaultdict

spec_dir = 'results/trainings/tests/svm_nystroem_6000_hybrid_binary_small_medium_pca64_elasticnet_l1r0.15'
thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

fold_results = {t: {'small_rec': [], 'medium_rec': [], 'fallbacks': []} for t in thresholds}

for fold in range(10):
    eval_dir = os.path.join(spec_dir, 'eval', f'fold_{fold}')
    if not os.path.exists(eval_dir):
        continue
    csvs = [f for f in os.listdir(eval_dir) if 'test' in f and f.endswith('.csv')]
    if not csvs:
        continue

    ft, fp = {}, defaultdict(list)
    with open(os.path.join(eval_dir, csvs[0])) as f:
        for row in csv.DictReader(f):
            fid = int(row['File'])
            ft[fid] = int(row['Target'])
            fp[fid].append(int(row['Prediction']))

    # Proxy de confianca: fracao de janelas no voto majoritario
    ship_conf = {}
    for fid, preds in fp.items():
        n = len(preds)
        majority = max({0: preds.count(0), 1: preds.count(1)}, key=lambda k: preds.count(k))
        ship_conf[fid] = (majority, preds.count(majority) / n)

    for t in thresholds:
        sm_correct = sm_total = med_correct = med_total = fallbacks = total_sm_med = 0
        for fid, preds in fp.items():
            target = ft[fid]
            if target not in [0, 1]:
                continue
            majority, conf = ship_conf[fid]
            total_sm_med += 1

            if conf >= t:
                final = majority  # especialista decide
            else:
                # fallback: assume multiclasse acertou (upper bound otimista)
                # na pratica, o multiclasse tem ~55-62% de acerto para estas classes
                final = target
                fallbacks += 1

            if target == 0:
                sm_total += 1
                if final == 0:
                    sm_correct += 1
            else:
                med_total += 1
                if final == 1:
                    med_correct += 1

        if sm_total:
            fold_results[t]['small_rec'].append(sm_correct / sm_total * 100)
        if med_total:
            fold_results[t]['medium_rec'].append(med_correct / med_total * 100)
        if total_sm_med:
            fold_results[t]['fallbacks'].append(fallbacks / total_sm_med * 100)

def st(v):
    if not v:
        return 0, 0
    mu = sum(v) / len(v)
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / len(v))
    return mu, sd

print(f"{'Threshold':>10}  {'Rec SMALL':>14}  {'Rec MEDIUM':>14}  {'Fallback%':>10}")
print("-" * 58)
for t in thresholds:
    sr = fold_results[t]['small_rec']
    mr = fold_results[t]['medium_rec']
    fb = fold_results[t]['fallbacks']
    mu_s, sd_s = st(sr)
    mu_m, sd_m = st(mr)
    mu_fb, _ = st(fb)
    print(f"{t:>10.2f}  {mu_s:>6.1f}+-{sd_s:.1f}%  {mu_m:>6.1f}+-{sd_m:.1f}%  {mu_fb:>8.1f}%")

print()
print("Nota: fallback assume multiclasse correto (upper bound otimista).")
print("Na cascata real, o fallback retorna a predicao original do multiclasse,")
print("que tem ~62% de acerto para SMALL e ~55% para MEDIUM neste subconjunto.")
