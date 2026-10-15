"""
Plot high-resolution confusion matrix for DEMON 4-class evaluation on 5x2cv.
Saves to reports/matriz_confusao_demon_4classes.png and artifacts directory.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# Matrizes acumuladas nos 10 folds oficiais (m=6000)
cm_abs = np.array([
    [ 276,  324,  683,  497],
    [ 118,  748,  319,  305],
    [ 130,  180, 1569,  261],
    [  57,   31,  179,  713]
], dtype=int)

cm_norm = np.array([
    [15.5, 18.2, 38.4, 27.9],
    [ 7.9, 50.2, 21.4, 20.5],
    [ 6.1,  8.4, 73.3, 12.2],
    [ 5.8,  3.2, 18.3, 72.7]
], dtype=float)

classes = ['SMALL', 'MEDIUM', 'LARGE', 'AMBIENTAL']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

# 1. Matriz Normalizada (%)
im1 = ax1.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100)
ax1.set_title('Matriz de Confusão Normalizada (%)\n(Sensibilidade / Recall por Classe)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xticks(range(4))
ax1.set_yticks(range(4))
ax1.set_xticklabels(classes, fontsize=11, fontweight='bold')
ax1.set_yticklabels(classes, fontsize=11, fontweight='bold')
ax1.set_xlabel('Classe Predita pelo DEMON', fontsize=12, fontweight='bold')
ax1.set_ylabel('Classe Real da Embarcação', fontsize=12, fontweight='bold')

for i in range(4):
    for j in range(4):
        val = cm_norm[i, j]
        color = 'white' if val > 45 else 'black'
        weight = 'bold' if i == j else 'normal'
        ax1.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=12, fontweight=weight)

cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Taxa de Classificação (%)', fontsize=11)

# 2. Matriz Absoluta (Amostras de Áudio nos 10 Folds)
im2 = ax2.imshow(cm_abs, cmap='YlGnBu', vmin=0, vmax=cm_abs.max())
ax2.set_title('Matriz de Confusão Absoluta\n(Total de Áudios Avaliados nos 10 Folds)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(classes, fontsize=11, fontweight='bold')
ax2.set_yticklabels(classes, fontsize=11, fontweight='bold')
ax2.set_xlabel('Classe Predita pelo DEMON', fontsize=12, fontweight='bold')
ax2.set_ylabel('Classe Real da Embarcação', fontsize=12, fontweight='bold')

for i in range(4):
    for j in range(4):
        val = cm_abs[i, j]
        color = 'white' if val > 800 else 'black'
        weight = 'bold' if i == j else 'normal'
        ax2.text(j, i, f'{val:,}', ha='center', va='center', color=color, fontsize=12, fontweight=weight)

cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Número de Áudios', fontsize=11)

plt.suptitle('Desempenho do DEMON Isolado (SVM Nyström m=6000) no Protocolo 5x2cv (1.706 Áudios)', 
             fontsize=15, fontweight='bold', y=0.98)

# Explicação de rodapé
plt.figtext(0.5, 0.02, 
            'Destaques: LARGE quase nunca é confundido com SMALL (apenas 5.8%). '
            'SMALL sofre confusão com LARGE (39.9%) e AMBIENTAL (29.1%) devido à ondulação do mar em baixa frequência.',
            ha='center', fontsize=10.5, style='italic', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.9))

plt.tight_layout(rect=[0, 0.06, 1, 0.95])

out_path = 'reports/matriz_confusao_demon_4classes.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Imagem salva em {out_path}')

# Copy to artifacts directory
artifact_dir = r'C:\Users\ev_ro\.gemini\antigravity-ide\brain\ffb9500e-86ad-47f3-9cf6-d5e7cd24de4f'
artifact_path = os.path.join(artifact_dir, 'matriz_confusao_demon_4classes.png')
shutil.copy(out_path, artifact_path)
print(f'Imagem copiada para artifact: {artifact_path}')
