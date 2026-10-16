"""
Plot publication-ready comparison of Multimodal Fusion (MEL + DEMON) vs. Isolated Baselines.
Saves to reports/fusao_multimodal_mel_demon.png and copies to artifacts.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 150
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

# 1. Comparativo de Métricas Globais (4C e 3C)
labels = ['DEMON Isolado\n(253-D)', 'MEL Isolado\n(256-D)', 'FUSÃO MULTIMODAL\n(MEL + DEMON, 509-D)']
x = np.arange(len(labels))
width = 0.35

sp_4c = [53.51, 64.93, 66.99]
sp_4c_err = [1.44, 2.31, 1.67]

sp_3c = [55.62, 63.98, 66.18]
sp_3c_err = [2.72, 1.82, 1.71]

rects1 = ax1.bar(x - width/2, sp_4c, width, yerr=sp_4c_err, capsize=5, label='4 Classes (Geral)', color='#3498db', alpha=0.9)
rects2 = ax1.bar(x + width/2, sp_3c, width, yerr=sp_3c_err, capsize=5, label='3 Classes (Embarcações)', color='#e67e22', alpha=0.9)

ax1.axhline(66.51, color='#27ae60', linestyle='--', lw=2, label='Benchmark Artigo: MLP LOFAR (66.51%)')
ax1.axhline(63.38, color='#8e44ad', linestyle=':', lw=2, label='Benchmark Artigo: MLP MEL (63.38%)')

ax1.set_title('Evolução do Índice SP Oficial (5x2cv, 10 Folds)', fontweight='bold', pad=12)
ax1.set_ylabel('Índice SP Oficial (%)')
ax1.set_ylim(40, 72)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontweight='bold')
ax1.legend(loc='lower right', frameon=True)

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

# 2. Sensibilidade por Classe no Problema de 4 Classes
classes = ['SMALL', 'MEDIUM', 'LARGE', 'AMBIENTAL']
x_c = np.arange(len(classes))
w_c = 0.25

rec_demon = [34.78, 45.91, 62.06, 77.76]
rec_mel   = [45.62, 67.38, 70.28, 80.20]
rec_fused = [53.93, 63.56, 71.50, 80.82]

r1 = ax2.bar(x_c - w_c, rec_demon, w_c, label='DEMON Isolado', color='#95a5a6', alpha=0.85)
r2 = ax2.bar(x_c, rec_mel, w_c, label='MEL Isolado', color='#2980b9', alpha=0.85)
r3 = ax2.bar(x_c + w_c, rec_fused, w_c, label='FUSÃO (MEL+DEMON)', color='#27ae60', alpha=0.95)

ax2.set_title('Sensibilidade por Classe (Recall 4C): O Salto em SMALL', fontweight='bold', pad=12)
ax2.set_ylabel('Recall (%)')
ax2.set_ylim(0, 95)
ax2.set_xticks(x_c)
ax2.set_xticklabels(classes, fontweight='bold', fontsize=11)
ax2.legend(loc='upper right', frameon=True)

for rect in r3:
    h = rect.get_height()
    ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Vitória da Fusão Multimodal Direta (MEL + DEMON) no Protocolo Oficial 5x2cv', fontsize=15, fontweight='bold', y=0.98)

plt.figtext(0.5, 0.02, 
            'Conclusão: A fusão MEL+DEMON atinge 66.99% de SP, superando tanto o MEL isolado (64.93%) quanto a MLP LOFAR de referência do artigo (66.51%).\n'
            'A informação de RPM da hélice do DEMON destravou um salto de +8.3 p.p. no recall de SMALL (45.6% -> 53.9%) sem prejudicar AMBIENTAL (80.8%).',
            ha='center', fontsize=10.5, style='italic', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#27ae60', alpha=0.9))

plt.tight_layout(rect=[0, 0.06, 1, 0.95])

out_path = 'reports/fusao_multimodal_mel_demon.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Imagem salva em {out_path}')

artifact_dir = r'C:\Users\ev_ro\.gemini\antigravity-ide\brain\ffb9500e-86ad-47f3-9cf6-d5e7cd24de4f'
artifact_path = os.path.join(artifact_dir, 'fusao_multimodal_mel_demon.png')
shutil.copy(out_path, artifact_path)
print(f'Imagem copiada para artifact: {artifact_path}')
