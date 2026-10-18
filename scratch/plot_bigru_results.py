import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load results list
with open("data/iara_processed/bigru_multiclass_results.pkl", "rb") as f:
    rows = pickle.load(f)

# Rows 0-4 are 4 Classes, rows 5-9 are 3 Classes
labels = ["DEMON", "HHT", "MEL", "MEL+DEMON", "MEL+HHT"]
colors = ['#e74c3c', '#e67e22', '#3498db', '#9b59b6', '#2ecc71']

# Parse values
def parse_sp(s):
    # '52.15 ± 1.13'
    parts = s.split('±')
    return float(parts[0].strip()), float(parts[1].strip())

def parse_val(s):
    # '53.15 ± 1.04' or '33.0%'
    if '±' in s:
        return float(s.split('±')[0].strip())
    return float(s.replace('%', '').strip())

# 4 Classes
sp_4c_mean, sp_4c_std = zip(*[parse_sp(r['SP (%)']) for r in rows[:5]])
ba_4c = [parse_val(r['Bal Acc (%)']) for r in rows[:5]]
rec_small_4c = [parse_val(r['Recall SMALL (%)']) for r in rows[:5]]
rec_med_4c = [parse_val(r['Recall MEDIUM (%)']) for r in rows[:5]]
rec_large_4c = [parse_val(r['Recall LARGE (%)']) for r in rows[:5]]

# 3 Classes
sp_3c_mean, sp_3c_std = zip(*[parse_sp(r['SP (%)']) for r in rows[5:]])
ba_3c = [parse_val(r['Bal Acc (%)']) for r in rows[5:]]
rec_small_3c = [parse_val(r['Recall SMALL (%)']) for r in rows[5:]]
rec_med_3c = [parse_val(r['Recall MEDIUM (%)']) for r in rows[5:]]
rec_large_3c = [parse_val(r['Recall LARGE (%)']) for r in rows[5:]]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. 4 Classes SP
bars1 = ax1.bar(np.arange(len(labels)), sp_4c_mean, yerr=sp_4c_std, capsize=5, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
ax1.set_title("BiGRU com Atenção — 4 Classes (SP Global)", fontsize=12, fontweight='bold', pad=10)
ax1.set_ylabel("Standard Product (SP) %", fontsize=10, fontweight='bold')
ax1.set_xticks(np.arange(len(labels)))
ax1.set_xticklabels(labels, fontsize=10, fontweight='bold')
ax1.set_ylim(45, 68)
for bar, sp, ba in zip(bars1, sp_4c_mean, ba_4c):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f"{sp:.1f}%\n(Bal: {ba:.1f}%)", 
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. 3 Classes SP
bars2 = ax2.bar(np.arange(len(labels)), sp_3c_mean, yerr=sp_3c_std, capsize=5, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
ax2.set_title("BiGRU com Atenção — 3 Classes (SP Global)", fontsize=12, fontweight='bold', pad=10)
ax2.set_ylabel("Standard Product (SP) %", fontsize=10, fontweight='bold')
ax2.set_xticks(np.arange(len(labels)))
ax2.set_xticklabels(labels, fontsize=10, fontweight='bold')
ax2.set_ylim(45, 66)
for bar, sp, ba in zip(bars2, sp_3c_mean, ba_3c):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f"{sp:.1f}%\n(Bal: {ba:.1f}%)", 
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. 4 Classes Recalls por classe
w = 0.25
x = np.arange(len(labels))
ax3.bar(x - w, rec_small_4c, width=w, label='SMALL', color='#3498db', edgecolor='black', alpha=0.85)
ax3.bar(x, rec_med_4c, width=w, label='MEDIUM', color='#f39c12', edgecolor='black', alpha=0.85)
ax3.bar(x + w, rec_large_4c, width=w, label='LARGE', color='#2ecc71', edgecolor='black', alpha=0.85)
ax3.set_title("BiGRU — Recalls por Classe em 4 Classes (Efeito da Fusão)", fontsize=12, fontweight='bold', pad=10)
ax3.set_ylabel("Recall (%)", fontsize=10, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=10, fontweight='bold')
ax3.set_ylim(20, 80)
ax3.legend(loc='upper left', frameon=True)
for i in range(len(labels)):
    ax3.text(x[i] - w, rec_small_4c[i] + 1, f"{rec_small_4c[i]:.0f}%", ha='center', fontsize=8, fontweight='bold')
    ax3.text(x[i], rec_med_4c[i] + 1, f"{rec_med_4c[i]:.0f}%", ha='center', fontsize=8, fontweight='bold')
    ax3.text(x[i] + w, rec_large_4c[i] + 1, f"{rec_large_4c[i]:.0f}%", ha='center', fontsize=8, fontweight='bold')

# 4. 3 Classes Recalls por classe
ax4.bar(x - w, rec_small_3c, width=w, label='SMALL', color='#3498db', edgecolor='black', alpha=0.85)
ax4.bar(x, rec_med_3c, width=w, label='MEDIUM', color='#f39c12', edgecolor='black', alpha=0.85)
ax4.bar(x + w, rec_large_3c, width=w, label='LARGE', color='#2ecc71', edgecolor='black', alpha=0.85)
ax4.set_title("BiGRU — Recalls por Classe em 3 Classes (Efeito da Fusão)", fontsize=12, fontweight='bold', pad=10)
ax4.set_ylabel("Recall (%)", fontsize=10, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=10, fontweight='bold')
ax4.set_ylim(30, 80)
ax4.legend(loc='upper left', frameon=True)
for i in range(len(labels)):
    ax4.text(x[i] - w, rec_small_3c[i] + 1, f"{rec_small_3c[i]:.0f}%", ha='center', fontsize=8, fontweight='bold')
    ax4.text(x[i], rec_med_3c[i] + 1, f"{rec_med_3c[i]:.0f}%", ha='center', fontsize=8, fontweight='bold')
    ax4.text(x[i] + w, rec_large_3c[i] + 1, f"{rec_large_3c[i]:.0f}%", ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
out_path = "reports/comparacao_bigru_multiclasse.png"
plt.savefig(out_path, dpi=300)
print(f"Gráfico salvo com sucesso em {out_path}")
