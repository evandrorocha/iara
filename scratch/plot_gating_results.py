"""
Gera o gráfico comparativo do Sistema de Gating Front-End (Roteador Dinâmico).
Salva em 'reports/comparacao_gating_frontend.png'.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Gráfico de Desempenho Global (SP Index e Balanced Accuracy)
    models = ['MEL Isolado\n(256-D)', 'LOFAR Isolado\n(1024-D)', 'Gating Físico\n(Sem Cascata)', 'Gating Aprendido\n(Front-End)', 'Teto do Oráculo\n(Teórico)']
    sp_vals = [61.45, 63.12, 61.03, 61.70, 85.64]
    bal_vals = [61.55, 63.81, 61.31, 62.65, 85.73]

    x = np.arange(len(models))
    w = 0.35

    b1 = axes[0].bar(x - w/2, sp_vals, w, label='Índice SP (%)', color='#1d3557', alpha=0.9, edgecolor='black', linewidth=0.8)
    b2 = axes[0].bar(x + w/2, bal_vals, w, label='Acurácia Balanceada (%)', color='#e63946', alpha=0.85, edgecolor='black', linewidth=0.8)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontweight='bold')
    axes[0].set_ylabel('Desempenho (%)', fontweight='bold')
    axes[0].set_title('(A) Desempenho Global sob Protocolo Oficial 5x2cv', fontweight='bold', pad=12)
    axes[0].set_ylim(45, 96)
    axes[0].legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left')

    for bar in b1:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1d3557')
    for bar in b2:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#e63946')

    # 2. Distribuição das Rotas Selecionadas no Front-End
    routes = ['DEMON\n(Cavitação)', 'LOFAR\n(Diesel)', 'HHT\n(Manobra)', 'MEL\n(Banda Larga)']
    routes_physics = [4.1, 32.4, 12.6, 51.0]
    routes_learned = [38.1, 32.5, 15.4, 14.0]

    x2 = np.arange(len(routes))
    b3 = axes[1].bar(x2 - w/2, routes_physics, w, label='Gating Físico (Regras)', color='#2a9d8f', alpha=0.9, edgecolor='black', linewidth=0.8)
    b4 = axes[1].bar(x2 + w/2, routes_learned, w, label='Gating Aprendido (Treinado)', color='#f4a261', alpha=0.9, edgecolor='black', linewidth=0.8)

    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(routes, fontweight='bold')
    axes[1].set_ylabel('Percentual de Áudios Roteados (%)', fontweight='bold')
    axes[1].set_title('(B) Distribuição de Roteamento de Entradas no Front-End', fontweight='bold', pad=12)
    axes[1].set_ylim(0, 62)
    axes[1].legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper right')

    for bar in b3:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 1.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2a9d8f')
    for bar in b4:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 1.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#f4a261')

    plt.tight_layout()
    out_img = 'reports/comparacao_gating_frontend.png'
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo com sucesso em '{out_img}'!")

if __name__ == '__main__':
    main()
