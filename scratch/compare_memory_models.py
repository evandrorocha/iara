import os
import pandas as pd

base = 'results/trainings/tests'
dirs = [d for d in os.listdir(base) if 'gru' in d.lower() or 'crnn' in d.lower()]
rows = []
for d in sorted(dirs):
    p = os.path.join(base, d, 'metrics_summary.csv')
    if os.path.exists(p):
        df = pd.read_csv(p)
        m = dict(zip(df['metric'], df['mean']))
        s = dict(zip(df['metric'], df['std']))
        
        is_multiclass = 'multiclass' in d
        rows.append({
            'Tipo': 'Multiclasse (4C)' if is_multiclass else 'Especialista Binário (S vs L)',
            'Modelo': d,
            'SP (%)': f"{m.get('sp', 0):.2f} ± {s.get('sp', 0):.2f}",
            'Bal Acc (%)': f"{m.get('bal_acc', 0):.2f} ± {s.get('bal_acc', 0):.2f}",
            'Acc (%)': f"{m.get('acc', 0):.2f} ± {s.get('acc', 0):.2f}",
            'Recall SMALL (%)': f"{m.get('rec_small', 0):.2f}" if 'rec_small' in m and not pd.isna(m.get('rec_small')) else '-',
            'Recall MEDIUM (%)': f"{m.get('rec_medium', 0):.2f}" if 'rec_medium' in m and not pd.isna(m.get('rec_medium')) else '-',
            'Recall LARGE (%)': f"{m.get('rec_large', 0):.2f}" if 'rec_large' in m and not pd.isna(m.get('rec_large')) else '-',
        })

df_out = pd.DataFrame(rows)
df_multiclass = df_out[df_out['Tipo'] == 'Multiclasse (4C)'].sort_values('SP (%)', ascending=False)
df_spec = df_out[df_out['Tipo'] == 'Especialista Binário (S vs L)'].sort_values('SP (%)', ascending=False)

print("=== MODELOS MULTICLASSE (4 CLASSES) ===")
print(df_multiclass.to_string(index=False))
print("\n=== MODELOS ESPECIALISTAS BINÁRIOS (SMALL vs LARGE) ===")
print(df_spec.to_string(index=False))
