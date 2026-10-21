import pickle
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def compute_sp(y_true, y_pred, num_classes=4):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    recalls = cm.diagonal() / (cm.sum(axis=1) + 1e-12)
    recalls = np.nan_to_num(recalls, nan=0.0)
    gmean = 0.0 if np.any(recalls == 0) else float(scipy_stats.gmean(recalls))
    sp = float(np.sqrt(np.mean(recalls * gmean)) * 100)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100)
    acc = float((y_true == y_pred).mean() * 100)
    return sp, bal_acc, acc, recalls * 100

files = {
    'RF': 'data/iara_processed/rf_all_representations_predictions.pkl',
    'MLP': 'data/iara_processed/mlp_all_representations_predictions.pkl',
}
data = {k: pickle.load(open(v, 'rb')) for k, v in files.items()}
with open('data/iara_processed/cache_gating_metafeatures.pkl', 'rb') as f:
    meta_dict = pickle.load(f)

feature_cols = ['crest_factor', 'spectral_flatness', 'mod_index', 'flux_std', 'high_low_ratio', 'rms_db', 'zcr']
fixed_expert_map = {'mel': 'RF', 'lofar': 'MLP', 'hht': 'RF', 'demon': 'MLP'}

demon_audios = []
res_4branch = []
res_3branch = []

reps_4 = ['mel', 'lofar', 'hht', 'demon']
reps_3 = ['mel', 'lofar', 'hht']

for fold in range(10):
    tst_targets = data['RF']['targets'][fold]
    fids_test = sorted(list(tst_targets.keys()))
    y_test = np.array([tst_targets[fid] for fid in fids_test])

    train_fids = []
    for trn_fold in range(10):
        if trn_fold == fold: continue
        for fid in data['RF']['targets'][trn_fold].keys():
            if fid in meta_dict:
                train_fids.append((trn_fold, fid))

    X_train, y_comp_4, y_comp_3 = [], [], []
    for trn_fold, fid in train_fids:
        yt = data['RF']['targets'][trn_fold][fid]
        X_train.append([meta_dict[fid][c] for c in feature_cols])
        y_comp_4.append([int(data[fixed_expert_map[r]][r][trn_fold][fid] == yt) for r in reps_4])
        y_comp_3.append([int(data[fixed_expert_map[r]][r][trn_fold][fid] == yt) for r in reps_3])

    X_train = np.array(X_train)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    clfs_4 = [RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42).fit(X_train_s, np.array(y_comp_4)[:, m]) for m in range(4)]
    clfs_3 = [RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42).fit(X_train_s, np.array(y_comp_3)[:, m]) for m in range(3)]

    X_test = np.array([[meta_dict[fid][c] if fid in meta_dict else 0.0 for c in feature_cols] for fid in fids_test])
    X_test_s = scaler.transform(X_test)

    probs_4 = np.column_stack([clfs_4[m].predict_proba(X_test_s)[:, 1] for m in range(4)])
    routes_4 = np.argmax(probs_4, axis=1)

    probs_3 = np.column_stack([clfs_3[m].predict_proba(X_test_s)[:, 1] for m in range(3)])
    routes_3 = np.argmax(probs_3, axis=1)

    y_pred_4, y_pred_3 = [], []
    for i, fid in enumerate(fids_test):
        yt = y_test[i]
        r4 = reps_4[routes_4[i]]
        pred4 = data[fixed_expert_map[r4]][r4][fold][fid]
        y_pred_4.append(pred4)

        r3 = reps_3[routes_3[i]]
        pred3 = data[fixed_expert_map[r3]][r3][fold][fid]
        y_pred_3.append(pred3)

        if r4 == 'demon':
            p_mel = data['RF']['mel'][fold][fid]
            p_lofar = data['MLP']['lofar'][fold][fid]
            p_hht = data['RF']['hht'][fold][fid]
            demon_audios.append({
                'fold': fold,
                'fid': fid,
                'y_true': yt,
                'class_name': ['Small', 'Medium', 'Large', 'Background'][yt],
                'demon_pred': pred4,
                'mel_pred': p_mel,
                'lofar_pred': p_lofar,
                'hht_pred': p_hht,
                'demon_correct': int(pred4 == yt),
                'mel_correct': int(p_mel == yt),
                'lofar_correct': int(p_lofar == yt),
                'hht_correct': int(p_hht == yt),
            })

    res_4branch.append(compute_sp(y_test, np.array(y_pred_4)))
    res_3branch.append(compute_sp(y_test, np.array(y_pred_3)))

df_demon = pd.DataFrame(demon_audios)
print("=" * 80)
print(f"ANÁLISE DETALHADA DOS {len(df_demon)} ÁUDIOS ROTEADOS PARA O DEMON")
print("=" * 80)
print("Distribuição das Classes Reais dos áudios enviados ao DEMON:")
print(df_demon['class_name'].value_counts())

n_hits_demon = df_demon['demon_correct'].sum()
print(f"\n1. Quantos áudios o DEMON:MLP ACERTOU?")
print(f"   Acertos do DEMON: {n_hits_demon} de {len(df_demon)} ({n_hits_demon/len(df_demon)*100:.1f}%)")

n_hits_mel = df_demon['mel_correct'].sum()
n_hits_lofar = df_demon['lofar_correct'].sum()
n_hits_hht = df_demon['hht_correct'].sum()

print(f"\n2. E se esses {len(df_demon)} áudios fossem para os outros especialistas?")
print(f"   Se fossem para MEL (RF):   {n_hits_mel} acertos ({n_hits_mel/len(df_demon)*100:.1f}%)")
print(f"   Se fossem para LOFAR (MLP): {n_hits_lofar} acertos ({n_hits_lofar/len(df_demon)*100:.1f}%)")
print(f"   Se fossem para HHT (RF):    {n_hits_hht} acertos ({n_hits_hht/len(df_demon)*100:.1f}%)")

exclusive = df_demon[(df_demon['demon_correct'] == 1) & (df_demon['mel_correct'] == 0) & (df_demon['lofar_correct'] == 0) & (df_demon['hht_correct'] == 0)]
print(f"\n3. Acertos EXCLUSIVOS do DEMON (onde todos os outros três erraram):")
print(f"   Total de resgates exclusivos pelo DEMON: {len(exclusive)} áudios")
if len(exclusive) > 0:
    print(exclusive[['fold', 'class_name', 'demon_pred', 'mel_pred', 'lofar_pred', 'hht_pred']])

sp4 = [r[0] for r in res_4branch]
sp3 = [r[0] for r in res_3branch]
rec4 = np.mean([r[3] for r in res_4branch], axis=0)
rec3 = np.mean([r[3] for r in res_3branch], axis=0)

print("\n" + "=" * 80)
print("O DEMON FAZ DIFERENÇA NO SISTEMA COMO UM TODO?")
print("Comparando 4 Ramos (com DEMON) vs. 3 Ramos (sem DEMON: apenas MEL, LOFAR, HHT)")
print("=" * 80)
print(f"4 Ramos (MEL, LOFAR, HHT, DEMON): SP = {np.mean(sp4):.2f}% ± {np.std(sp4):.2f}% | Small: {rec4[0]:.1f}%, Med: {rec4[1]:.1f}%, Lrg: {rec4[2]:.1f}%, BG: {rec4[3]:.1f}%")
print(f"3 Ramos (MEL, LOFAR, HHT):        SP = {np.mean(sp3):.2f}% ± {np.std(sp3):.2f}% | Small: {rec3[0]:.1f}%, Med: {rec3[1]:.1f}%, Lrg: {rec3[2]:.1f}%, BG: {rec3[3]:.1f}%")
diff = np.mean(sp4) - np.mean(sp3)
print(f"Impacto líquido do DEMON no SP Index: {diff:+.2f}%")
