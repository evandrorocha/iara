"""
Gera o gráfico oficial de 4 quadrantes para a análise de complementaridade da classe MEDIUM.
Salva em reports/analise_complementaridade_medium_oraculo.png e copia para os artefatos.
"""

import os
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt

def generate_medium_plot():
    pkl_path = "data/iara_processed/medium_complementarity_analysis.pkl"
    if not os.path.exists(pkl_path):
        print(f"Erro: {pkl_path} não encontrado.")
        return

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

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

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))

    # =========================================================================
    # Quadrante 1 (Top-Left): Desempenho Isolado vs Teto Teórico do Oráculo
    # =========================================================================
    ax1 = axes[0, 0]
    reps = ['HHT Puro\n(256-D)', 'DEMON\n(253-D)', 'LOFAR\n(1024-D)', 'MEL\n(256-D)', 'ORÁCULO\n(União 4 Modos)']
    recalls = [44.03, 56.04, 60.40, 60.40, 83.69]
    colors1 = ['#9b59b6', '#95a5a6', '#e67e22', '#2980b9', '#27ae60']

    b1 = ax1.bar(reps, recalls, color=colors1, width=0.55, edgecolor='black', linewidth=0.8)
    b1[-1].set_hatch('//')
    ax1.axhline(83.69, color='#27ae60', linestyle='--', lw=1.5, alpha=0.7, label='Teto do Oráculo (83.69%)')
    ax1.axhline(60.40, color='#2980b9', linestyle=':', lw=1.5, alpha=0.7, label='Melhor Isolado MEL/LOFAR (60.40%)')

    ax1.set_title('A) Recall em MEDIUM: Modelos Isolados vs. Teto do Oráculo', fontweight='bold', pad=12)
    ax1.set_ylabel('Recall na Classe MEDIUM (%)')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper left', frameon=True, fontsize=9.5)

    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                     fontsize=10.5, fontweight='bold', color='#111')

    # =========================================================================
    # Quadrante 2 (Top-Right): Pirâmide de Consenso dos 1.490 Áudios
    # =========================================================================
    ax2 = axes[0, 1]
    labels_cons = ['4 Entradas\n(Unânime)', '3 Entradas\n(Alto Consenso)', '2 Entradas\n(Dividido)', '1 Entrada\n(Exclusivo/Frágil)', '0 Entradas\n(Erro Comum)']
    counts_cons = [403, 275, 285, 284, 243]
    pcts_cons = [27.05, 18.46, 19.13, 19.06, 16.31]
    colors_cons = ['#27ae60', '#2980b9', '#f39c12', '#e74c3c', '#7f8c8d']

    b2 = ax2.bar(labels_cons, counts_cons, color=colors_cons, edgecolor='black', linewidth=0.8, width=0.55)
    ax2.set_title('B) Grau de Consenso em MEDIUM (Total: 1.490 Áudios no 5x2cv)', fontweight='bold', pad=12)
    ax2.set_ylabel('Quantidade de Áudios de MEDIUM')
    ax2.set_ylim(0, 480)

    for bar, pct in zip(b2, pcts_cons):
        h = bar.get_height()
        ax2.annotate(f'{h} ({pct:.1f}%)', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                     fontsize=9.5, fontweight='bold')

    # =========================================================================
    # Quadrante 3 (Bottom-Left): Resgates Exclusivos por Representação
    # =========================================================================
    ax3 = axes[1, 0]
    excl_reps = ['HHT Exclusivo', 'MEL Exclusivo', 'LOFAR Exclusivo', 'DEMON Exclusivo']
    excl_counts = [43, 57, 77, 107]
    excl_pcts = [2.89, 3.83, 5.17, 7.18]
    colors_excl = ['#9b59b6', '#2980b9', '#e67e22', '#16a085']

    b3 = ax3.barh(excl_reps, excl_counts, color=colors_excl, height=0.5, edgecolor='black', linewidth=0.8)
    ax3.set_title('C) Resgates Exclusivos em MEDIUM (Total: 284 Áudios Únicos)', fontweight='bold', pad=12)
    ax3.set_xlabel('Áudios Salvos Exclusivamente por Apenas UMA Entrada')
    ax3.set_xlim(0, 160)

    for idx, (bar, pct) in enumerate(zip(b3, excl_pcts)):
        w = bar.get_width()
        tag = "  <-- CAMPEÃO DE RESGATES EM MEDIUM" if idx == 3 else ""
        ax3.annotate(f'{w} áudios ({pct:.1f}%){tag}', xy=(w, bar.get_y() + bar.get_height()/2),
                     xytext=(5, 0), textcoords="offset points", ha='left', va='center',
                     fontsize=9.5, fontweight='bold', color='#111' if idx != 3 else '#117864')

    # =========================================================================
    # Quadrante 4 (Bottom-Right): Padrão de Erro em MEDIUM
    # =========================================================================
    ax4 = axes[1, 1]
    err_reps = ['MEL (590 erros)', 'DEMON (655 erros)', 'HHT (834 erros)', 'LOFAR (590 erros)']
    sml_err = [65.76, 59.69, 63.91, 44.41]
    lrg_err = [31.53, 27.33, 25.06, 43.56]
    amb_err = [2.71,  12.98, 11.03, 12.03]
    w = 0.52

    p1 = ax4.bar(err_reps, sml_err, w, label='Confundido com SMALL (Lanchas)', color='#3498db', alpha=0.9, edgecolor='black', linewidth=0.6)
    p2 = ax4.bar(err_reps, lrg_err, w, bottom=sml_err, label='Confundido com LARGE (Navios Grandes)', color='#e74c3c', alpha=0.9, edgecolor='black', linewidth=0.6)
    p3 = ax4.bar(err_reps, amb_err, w, bottom=np.array(sml_err)+np.array(lrg_err), label='Confundido com AMBIENTAL (Ruído)', color='#95a5a6', alpha=0.9, edgecolor='black', linewidth=0.6)

    ax4.set_title('D) Para onde o MEDIUM vai quando erra? (Distribuição de Erros %)', fontweight='bold', pad=12)
    ax4.set_ylabel('Distribuição Percentual dos Erros (%)')
    ax4.set_ylim(0, 118)
    ax4.legend(loc='upper right', frameon=True, fontsize=9.5)

    for i in range(len(err_reps)):
        ax4.text(i, sml_err[i]/2, f'{sml_err[i]:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        ax4.text(i, sml_err[i] + lrg_err[i]/2, f'{lrg_err[i]:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        if amb_err[i] > 4:
            ax4.text(i, sml_err[i] + lrg_err[i] + amb_err[i]/2, f'{amb_err[i]:.1f}%', ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')

    plt.suptitle('Análise Científica de Complementaridade e Teto Teórico da Classe MEDIUM (Dataset IARA, 5x2cv)',
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_png = "reports/analise_complementaridade_medium_oraculo.png"
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo com sucesso em: {out_png}")

    artifact_dir = r"C:\Users\ev_ro\.gemini\antigravity-ide\brain\ffb9500e-86ad-47f3-9cf6-d5e7cd24de4f"
    art_path = os.path.join(artifact_dir, "analise_complementaridade_medium_oraculo.png")
    shutil.copy2(out_png, art_path)
    print(f"Copiado para artefatos em: {art_path}")

if __name__ == "__main__":
    generate_medium_plot()
