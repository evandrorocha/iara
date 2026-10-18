import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load both results
rows_mel = pickle.load(open("data/iara_processed/bigru_multiclass_results.pkl", "rb"))
rows_lofar = pickle.load(open("data/iara_processed/bigru_lofar_multiclass_results.pkl", "rb"))

# Merge 4C and 3C
rows_4c = rows_mel[:5] + rows_lofar[:3]
rows_3c = rows_mel[5:] + rows_lofar[3:]

labels = [
    "DEMON", "HHT", "MEL", "MEL+DEM", "MEL+HHT",
    "LOFAR", "LOF+DEM", "LOF+HHT"
]
colors = [
    '#95a5a6', '#e67e22', '#3498db', '#9b59b6', '#2ecc71',
    '#34495e', '#e74c3c', '#1abc9c'
]

def parse_sp(s):
    parts = s.split('±')
    return float(parts[0].strip()), float(parts[1].strip())

def parse_val(s):
    if '±' in s:
        return float(s.split('±')[0].strip())
    return float(s.replace('%', '').strip())

sp_4c_mean, sp_4c_std = zip(*[parse_sp(r['SP (%)']) for r in rows_4c])
ba_4c = [parse_val(r['Bal Acc (%)']) for r in rows_4c]
small_4c = [parse_val(r['Recall SMALL (%)']) for r in rows_4c]
med_4c = [parse_val(r['Recall MEDIUM (%)']) for r in rows_4c]
lrg_4c = [parse_val(r['Recall LARGE (%)']) for r in rows_4c]

sp_3c_mean, sp_3c_std = zip(*[parse_sp(r['SP (%)']) for r in rows_3c])
ba_3c = [parse_val(r['Bal Acc (%)']) for r in rows_3c]
small_3c = [parse_val(r['Recall SMALL (%)']) for r in rows_3c]
med_3c = [parse_val(r['Recall MEDIUM (%)']) for r in rows_3c]
lrg_3c = [parse_val(r['Recall LARGE (%)']) for r in rows_3c]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))

# Panel 1: 4 Classes SP
bars1 = ax1.bar(np.arange(len(labels)), sp_4c_mean, yerr=sp_4c_std, capsize=4, color=colors, alpha=0.85, edgecolor='black', lw=1.2)
ax1.axhline(63.80, color='blue', linestyle='--', label='SVM MEL (63.8%)', lw=1.5)
ax1.axhline(66.51, color='red', linestyle='--', label='MLP LOFAR (66.5%)', lw=1.5)
ax1.set_title("BiGRU com Atenção — 4 Classes (SP Global)", fontsize=12, fontweight='bold', pad=10)
ax1.set_ylabel("Standard Product (SP) %", fontsize=10, fontweight='bold')
ax1.set_xticks(np.arange(len(labels)))
ax1.set_xticklabels(labels, fontsize=9.5, fontweight='bold', rotation=20)
ax1.set_ylim(45, 70)
ax1.legend(loc='lower left', frameon=True)
for bar, sp, ba in zip(bars1, sp_4c_mean, ba_4c):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, f"{sp:.1f}%\n({ba:.1f}%)", 
             ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# Panel 2: 3 Classes SP
bars2 = ax2.bar(np.arange(len(labels)), sp_3c_mean, yerr=sp_3c_std, capsize=4, color=colors, alpha=0.85, edgecolor='black', lw=1.2)
ax2.set_title("BiGRU com Atenção — 3 Classes (SP Global)", fontsize=12, fontweight='bold', pad=10)
ax2.set_ylabel("Standard Product (SP) %", fontsize=10, fontweight='bold')
ax2.set_xticks(np.arange(len(labels)))
ax2.set_xticklabels(labels, fontsize=9.5, fontweight='bold', rotation=20)
ax2.set_ylim(45, 68)
for bar, sp, ba in zip(bars2, sp_3c_mean, ba_3c):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, f"{sp:.1f}%\n({ba:.1f}%)", 
             ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# Panel 3: 4 Classes Recalls
w = 0.25
x = np.arange(len(labels))
ax3.bar(x - w, small_4c, width=w, label='SMALL', color='#3498db', edgecolor='black', alpha=0.85)
ax3.bar(x, med_4c, width=w, label='MEDIUM', color='#f39c12', edgecolor='black', alpha=0.85)
ax3.bar(x + w, lrg_4c, width=w, label='LARGE', color='#2ecc71', edgecolor='black', alpha=0.85)
ax3.set_title("BiGRU — Recalls por Classe em 4 Classes (Destaque para SMALL em LOFAR)", fontsize=12, fontweight='bold', pad=10)
ax3.set_ylabel("Recall (%)", fontsize=10, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=9.5, fontweight='bold', rotation=20)
ax3.set_ylim(20, 80)
ax3.legend(loc='upper left', frameon=True)
for i in range(len(labels)):
    ax3.text(x[i] - w, small_4c[i] + 1, f"{small_4c[i]:.0f}%", ha='center', fontsize=7.5, fontweight='bold')

# Panel 4: 3 Classes Recalls
ax4.bar(x - w, small_3c, width=w, label='SMALL', color='#3498db', edgecolor='black', alpha=0.85)
ax4.bar(x, med_3c, width=w, label='MEDIUM', color='#f39c12', edgecolor='black', alpha=0.85)
ax4.bar(x + w, lrg_3c, width=w, label='LARGE', color='#2ecc71', edgecolor='black', alpha=0.85)
ax4.set_title("BiGRU — Recalls por Classe em 3 Classes", fontsize=12, fontweight='bold', pad=10)
ax4.set_ylabel("Recall (%)", fontsize=10, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=9.5, fontweight='bold', rotation=20)
ax4.set_ylim(30, 80)
ax4.legend(loc='upper left', frameon=True)
for i in range(len(labels)):
    ax4.text(x[i] - w, small_3c[i] + 1, f"{small_3c[i]:.0f}%", ha='center', fontsize=7.5, fontweight='bold')

plt.tight_layout()
out_path = "reports/comparacao_completa_bigru_multiclasse.png"
plt.savefig(out_path, dpi=300)
print(f"Gráfico completo salvo em {out_path}")
