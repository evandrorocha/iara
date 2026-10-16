"""
Gera gráfico publication-ready da análise de complementaridade e efeito oráculo na classe SMALL.
Salva em reports/analise_complementaridade_small_oraculo.png e copia para artifacts.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150
})

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))

# -------------------------------------------------------------
# Painel 1: Recall Individual vs. Oráculo (União Teórica)
# -------------------------------------------------------------
models = ['DEMON\n(253-D)', 'LOFAR\n(1024-D)', 'HHT Puro\n(256-D)', 'MEL\n(256-D)', 'ORÁCULO\n(União 4 Modos)']
recalls = [35.96, 38.15, 48.43, 50.73, 83.99]
colors = ['#95a5a6', '#34495e', '#9b59b6', '#2980b9', '#27ae60']

bars1 = ax1.bar(models, recalls, color=colors, width=0.55, edgecolor='black', linewidth=0.8, alpha=0.9)
bars1[-1].set_hatch('//')

ax1.set_title('(A) Recall de SMALL: Modelos Isolados vs. Teto do Oráculo', fontweight='bold', pad=10)
ax1.set_ylabel('Recall na Classe SMALL (%)')
ax1.set_ylim(0, 100)
ax1.axhline(83.99, color='#27ae60', linestyle='--', lw=1.5, alpha=0.7)

for bar in bars1:
    h = bar.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                 fontsize=10.5, fontweight='bold', color='#111')

# -------------------------------------------------------------
# Painel 2: Distribuição de Concordância (Quantos modelos acertam)
# -------------------------------------------------------------
labels_agree = ['Nenhum\n(Erro Comum)', 'Apenas 1\n(Exclusivo)', '2 Modelos', '3 Modelos', 'Todos os 4\n(Consenso)']
counts_agree = [285, 513, 496, 365, 121]
pcts_agree = [16.01, 28.82, 27.87, 20.51, 6.80]
colors_agree = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71']

bars2 = ax2.bar(labels_agree, pcts_agree, color=colors_agree, width=0.55, edgecolor='black', linewidth=0.8, alpha=0.9)
ax2.set_title('(B) Grau de Concordância sobre os 1.780 Áudios de SMALL', fontweight='bold', pad=10)
ax2.set_ylabel('Percentual de Áudios de SMALL (%)')
ax2.set_ylim(0, 35)

for bar, cnt in zip(bars2, counts_agree):
    h = bar.get_height()
    ax2.annotate(f'{h:.1f}%\n({cnt} áudios)', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#111')

# -------------------------------------------------------------
# Painel 3: Resgates Exclusivos (Onde só UMA entrada acerta)
# -------------------------------------------------------------
exclusive_models = ['LOFAR\nExclusivo', 'MEL\nExclusivo', 'DEMON\nExclusivo', 'HHT\nExclusivo']
exclusive_counts = [37, 131, 138, 207]
exclusive_pcts = [2.08, 7.36, 7.75, 11.63]
colors_excl = ['#34495e', '#2980b9', '#95a5a6', '#9b59b6']

bars3 = ax3.barh(exclusive_models, exclusive_counts, color=colors_excl, height=0.5, edgecolor='black', linewidth=0.8, alpha=0.9)
ax3.set_title('(C) Resgates Exclusivos (Total: 513 Áudios Únicos)', fontweight='bold', pad=10)
ax3.set_xlabel('Número de Áudios Salvos Exclusivamente')
ax3.set_xlim(0, 270)

for idx, (bar, pct) in enumerate(zip(bars3, exclusive_pcts)):
    w = bar.get_width()
    tag = "  <-- LÍDER EM RESGATES" if idx == 3 else ""
    ax3.annotate(f'{w} áudios ({pct:.1f}%){tag}', xy=(w, bar.get_y() + bar.get_height()/2),
                 xytext=(5, 0), textcoords="offset points", ha='left', va='center',
                 fontsize=9.5, fontweight='bold', color='#111' if idx != 3 else '#8e44ad')

# -------------------------------------------------------------
# Painel 4: Padrão de Confusão Oposto (Para onde o SMALL vai?)
# -------------------------------------------------------------
representations = ['MEL (877 err)', 'LOFAR (1101 err)', 'HHT (918 err)', 'DEMON (1140 err)']
y_pos = np.arange(len(representations))

err_medium = [49.26, 36.78, 28.21, 49.91]
err_large  = [35.92, 46.41, 52.61, 35.79]
err_amb    = [14.82, 16.80, 19.17, 14.30]

w_bar = 0.55
b1 = ax4.bar(representations, err_medium, w_bar, label='Confundido com MEDIUM', color='#f39c12', alpha=0.9, edgecolor='black', linewidth=0.5)
b2 = ax4.bar(representations, err_large, w_bar, bottom=err_medium, label='Confundido com LARGE', color='#e74c3c', alpha=0.9, edgecolor='black', linewidth=0.5)
b3 = ax4.bar(representations, err_amb, w_bar, bottom=np.array(err_medium) + np.array(err_large), label='Confundido com AMBIENTAL', color='#3498db', alpha=0.9, edgecolor='black', linewidth=0.5)

ax4.set_title('(D) Destino do Erro na Classe SMALL por Extrator (%)', fontweight='bold', pad=10)
ax4.set_ylabel('Distribuição das Confusões (%)')
ax4.set_ylim(0, 110)
ax4.legend(loc='upper right', frameon=True, fontsize=9.5)

for i in range(len(representations)):
    ax4.text(i, err_medium[i]/2, f'{err_medium[i]:.1f}%', ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')
    ax4.text(i, err_medium[i] + err_large[i]/2, f'{err_large[i]:.1f}%', ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')
    ax4.text(i, err_medium[i] + err_large[i] + err_amb[i]/2, f'{err_amb[i]:.1f}%', ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')

plt.suptitle('Descoberta Científica: Ortogonalidade e Complementaridade na Classe SMALL (Dataset IARA)', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = 'reports/analise_complementaridade_small_oraculo.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Figura salva em {out_path}')

artifact_dir = r'C:\Users\ev_ro\.gemini\antigravity-ide\brain\ffb9500e-86ad-47f3-9cf6-d5e7cd24de4f'
artifact_path = os.path.join(artifact_dir, 'analise_complementaridade_small_oraculo.png')
shutil.copy(out_path, artifact_path)
print(f'Figura copiada para artifact: {artifact_path}')
