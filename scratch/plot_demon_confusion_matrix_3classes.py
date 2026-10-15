"""
Plot publication-ready confusion matrix for DEMON 3-class evaluation (Vessels only) on 5x2cv.
Saves to reports/matriz_confusao_demon_3classes.png and artifacts directory.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

cm_svm_3c = np.array([
    [57.9, 15.2, 26.9],
    [40.5, 41.4, 18.1],
    [20.3, 10.1, 69.6]
])

cm_rf_3c = np.array([
    [60.5,  7.8, 31.7],
    [47.6, 34.9, 17.5],
    [27.8,  3.6, 68.6]
])

cm_mlp_3c = np.array([
    [57.4, 20.4, 22.2],
    [38.1, 50.5, 11.4],
    [27.5, 11.0, 61.4]
])

cm_cnn_3c = np.array([
    [21.3, 55.0, 23.7],
    [22.5, 61.3, 16.2],
    [14.0, 34.8, 51.2]
])

cms_3c = [cm_svm_3c, cm_rf_3c, cm_mlp_3c, cm_cnn_3c]
titles_3c = [
    'SVM Nyström (m=6000) — SP: 54.83%',
    'Random Forest (100t) — SP: 53.47%',
    'MLP PyTorch [32, 16] — SP: 56.00% (Melhor)',
    'CNN 2D PyTorch (GPU) — SP: 41.67%'
]
classes_3c = ['SMALL', 'MEDIUM', 'LARGE']

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (cm, title) in enumerate(zip(cms_3c, titles_3c)):
    ax = axes[idx]
    im = ax.imshow(cm, cmap='YlGnBu', vmin=0, vmax=100)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(classes_3c, fontsize=11, fontweight='bold')
    ax.set_yticklabels(classes_3c, fontsize=11, fontweight='bold')
    ax.set_xlabel('Classe Predita', fontsize=11)
    ax.set_ylabel('Classe Real', fontsize=11)
    
    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            color = 'white' if val > 45 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=12, fontweight=weight)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Taxa (%)', fontsize=10)

plt.suptitle('Matrizes de Confusão Normalizadas — DEMON Isolado [3 Classes: Apenas Embarcações]', 
             fontsize=15, fontweight='bold', y=0.98)

# Explicação de rodapé
plt.figtext(0.5, 0.02, 
            'Destaques: O recall de SMALL salta para 60.5% no RF e 57.4% no MLP quando o ruído AMBIENTAL é removido.\n'
            'Contudo, a confusão SMALL -> LARGE persiste (22.2% a 31.9%) devido à oscilação das ondas marítimas (0.5-2 Hz) que imita giros lentos de eixo.',
            ha='center', fontsize=10.5, style='italic', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.9))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

out_path = 'reports/matriz_confusao_demon_3classes.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Imagem salva em {out_path}')

# Copy to artifacts directory
artifact_dir = r'C:\Users\ev_ro\.gemini\antigravity-ide\brain\ffb9500e-86ad-47f3-9cf6-d5e7cd24de4f'
artifact_path = os.path.join(artifact_dir, 'matriz_confusao_demon_3classes.png')
shutil.copy(out_path, artifact_path)
print(f'Imagem copiada para artifact: {artifact_path}')
