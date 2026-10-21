import pickle
import numpy as np
import pandas as pd
from collections import Counter

files = {
    'RF': 'data/iara_processed/rf_all_representations_predictions.pkl',
    'SVM': 'data/iara_processed/svm_all_representations_predictions.pkl',
    'MLP': 'data/iara_processed/mlp_all_representations_predictions.pkl',
    'CNN': 'data/iara_processed/cnn_all_representations_predictions.pkl',
}
data = {k: pickle.load(open(v, 'rb')) for k, v in files.items()}

with open('data/iara_processed/cache_gating_metafeatures.pkl', 'rb') as f:
    meta_dict = pickle.load(f)

df_info = pd.read_csv('src/iara/dataset_info/iara.csv')
df_info['ID'] = df_info['ID'].astype(int)
info_dict = df_info.set_index('ID').to_dict(orient='index')

reps = ['mel', 'lofar', 'hht', 'demon']
models = ['RF', 'SVM', 'MLP', 'CNN']
class_names = ['Small', 'Medium', 'Large', 'Background']

failures = []

for fold in range(10):
    tst_t = data['RF']['targets'][fold]
    fids = sorted(list(tst_t.keys()))

    for fid in fids:
        yt = tst_t[fid]
        
        preds_16 = {}
        hits = 0
        for m in models:
            for r in reps:
                p = data[m][r][fold][fid]
                preds_16[f"{m}_{r}"] = p
                if p == yt:
                    hits += 1

        if hits == 0:
            pred_counts = Counter(preds_16.values())
            dominant_pred = pred_counts.most_common(1)[0][0]
            
            meta_f = meta_dict.get(fid, {})
            info_f = info_dict.get(fid, {})
            
            failures.append({
                'fold': fold,
                'fid': fid,
                'y_true_id': yt,
                'y_true': class_names[yt],
                'dominant_pred': class_names[dominant_pred],
                'dominant_votes': pred_counts[dominant_pred],
                'name': info_f.get('Name', 'Unknown'),
                'ship_id': info_f.get('Ship ID', np.nan),
                'shiptype': info_f.get('SHIPTYPE', 'Unknown'),
                'detailed_type': info_f.get('DETAILED TYPE', 'Unknown'),
                'length': info_f.get('Length', np.nan),
                'sea_state': info_f.get('Sea state', np.nan),
                'wind': info_f.get('Wind', np.nan),
                'rain': info_f.get('Rain state', 'Unknown'),
                'crest_factor': meta_f.get('crest_factor', np.nan),
                'high_low_ratio': meta_f.get('high_low_ratio', np.nan),
                'rms_db': meta_f.get('rms_db', np.nan),
                'spectral_flatness': meta_f.get('spectral_flatness', np.nan),
            })

df_fail = pd.DataFrame(failures)
print("=" * 100)
print(f"ANÁLISE FORENSE DOS {len(df_fail)} ÁUDIOS ONDE TODOS OS 16 ESPECIALISTAS ERRARAM")
print("=" * 100)

print("\n1. Distribuição de Classes Reais:")
print(df_fail['y_true'].value_counts())

print("\n2. Matriz de Confusão do Consenso (Real -> Predição Dominante dos 16 Modelos):")
ct = pd.crosstab(df_fail['y_true'], df_fail['dominant_pred'], margins=True)
print(ct)

print("\n3. Principais Tipos de Embarcações Nesses 164 Áudios:")
print(df_fail['shiptype'].value_counts().head(8))

print("\n4. Principais Navios Específicos que Geraram Falha Unânime (Top Ships):")
print(df_fail.groupby(['name', 'y_true', 'dominant_pred', 'length', 'shiptype']).size().reset_index(name='n_audios').sort_values('n_audios', ascending=False).head(15).to_string(index=False))

print("\n5. Análise de Comprimento (Length) nas Falhas:")
for c in ['Small', 'Medium', 'Large']:
    lens = df_fail[df_fail['y_true'] == c]['length'].dropna()
    print(f"   Comprimento real das falhas em {c}: Média = {lens.mean():.1f}m, Min = {lens.min():.1f}m, Max = {lens.max():.1f}m")

print("\n6. Características Ambientais (Mar, Vento, Ruído):")
print(f"   Sea state médio nas falhas: {df_fail['sea_state'].mean():.2f} (vs dataset geral: {df_info['Sea state'].mean():.2f})")
print(f"   RMS Power médio (dB): {df_fail['rms_db'].mean():.2f} dB")
print(f"   Chuva nas falhas: {df_fail['rain'].value_counts().to_dict()}")
