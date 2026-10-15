"""
Plot publication-ready comprehensive comparison of Multimodal Fusion (MEL + DEMON)
across MLP, SVM (m=6000) and CNN 2D PyTorch under the official 5x2cv protocol.
Saves to reports/comparacao_fusao_modelos.png and copies to artifacts.
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# -------------------------------------------------------------
# 1. Comparativo de Índice SP Oficial por Família de Modelo
# -------------------------------------------------------------
models = ['MLP (PyTorch)', 'SVM Nyström (m=6000)', 'CNN 2D (PyTorch GPU)']
x = np.arange(len(models))
w = 0.25

# SP 4C: [DEMON Isolado, MEL Isolado, FUSÃO MEL+DEMON]
sp_demon = [53.51, 46.57, 35.72]
sp_demon_err = [1.44, 4.45, 12.63]

sp_mel = [64.93, 63.78, 63.52]
sp_mel_err = [2.31, 1.14, 2.26]

sp_fused = [66.99, 64.85, 23.96]
sp_fused_err = [1.67, 1.90, 17.32]

r1 = ax1.bar(x - w, sp_demon, w, yerr=sp_demon_err, capsize=4, label='DEMON Isolado (253-D)', color='#95a5a6', alpha=0.9)
r2 = ax1.bar(x, sp_mel, w, yerr=sp_mel_err, capsize=4, label='MEL Isolado (256-D)', color='#2980b9', alpha=0.9)
r3 = ax1.bar(x + w, sp_fused, w, yerr=sp_fused_err, capsize=4, label='FUSÃO MEL + DEMON (509-D)', color='#27ae60', alpha=0.95)

# Linhas de referência do artigo
ax1.axhline(66.51, color='#e67e22', linestyle='--', lw=2, label='Benchmark Artigo: MLP LOFAR (66.51%)')
ax1.axhline(66.05, color='#8e44ad', linestyle=':', lw=2, label='Benchmark Artigo: CNN LOFAR (66.05%)')

ax1.set_title('Impacto da Fusão no Índice SP Oficial (4 Classes, 5x2cv)', fontweight='bold', pad=12)
ax1.set_ylabel('Índice SP Oficial (%)')
ax1.set_ylim(0, 78)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontweight='bold', fontsize=11)
ax1.legend(loc='lower left', frameon=True, fontsize=9.5)

for rect in r1:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

for rect in r2:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

for rect in r3:
    h = rect.get_height()
    c = '#c0392b' if h < 30 else '#27ae60'
    ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c)

# -------------------------------------------------------------
# 2. Sensibilidade na Classe Crítica: SMALL (Lanchas Rápidas)
# -------------------------------------------------------------
# Recall SMALL: [DEMON Isolado, MEL Isolado, FUSÃO MEL+DEMON]
rec_demon = [34.78, 15.51, 19.33]
rec_mel   = [45.62, 45.60, 46.20]
rec_fused = [53.93, 47.08, 28.09]

b1 = ax2.bar(x - w, rec_demon, w, label='DEMON Isolado', color='#95a5a6', alpha=0.9)
b2 = ax2.bar(x, rec_mel, w, label='MEL Isolado', color='#2980b9', alpha=0.9)
b3 = ax2.bar(x + w, rec_fused, w, label='FUSÃO MEL + DEMON', color='#27ae60', alpha=0.95)

ax2.set_title('Sensibilidade na Classe Crítica (Recall SMALL - 4 Classes)', fontweight='bold', pad=12)
ax2.set_ylabel('Recall SMALL (%)')
ax2.set_ylim(0, 68)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontweight='bold', fontsize=11)
ax2.legend(loc='upper right', frameon=True, fontsize=9.5)

for rect in b1:
    h = rect.get_height()
    ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

for rect in b2:
    h = rect.get_height()
    ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

for rect in b3:
    h = rect.get_height()
    c = '#c0392b' if h < 30 else '#27ae60'
    ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c)

plt.suptitle('Comportamento dos Modelos na Fusão Multimodal (MEL + DEMON) — Protocolo 5x2cv', fontsize=15, fontweight='bold', y=0.98)

plt.figtext(0.5, 0.015,
            'Conclusão Central: Modelos vetoriais densos (MLP e SVM) se beneficiam fortemente da Fusão MEL+DEMON (MLP atinge 66.99% de SP; SVM sobe de 46.57% para 64.85%).\n'
            'A CNN 2D colapsa (23.96% SP) pela quebra de contiguidade espacial 2D: convoluções através da fronteira artificial entre Mel e DEMON borram os harmônicos finos.',
            ha='center', fontsize=10.5, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#2980b9', alpha=0.9))

plt.tight_layout(rect=[0, 0.06, 1, 0.95])

out_path = 'reports/comparacao_fusao_modelos.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Imagem salva em {out_path}')

artifact_dir = r'C:\Users\ev_ro\.gemini\antigravity-ide\brain\ffb9500e-86ad-47f3-9cf6-d5e7cd24de4f'
artifact_path = os.path.join(artifact_dir, 'comparacao_fusao_modelos.png')
shutil.copy(out_path, artifact_path)
print(f'Imagem copiada para artifact: {artifact_path}')
