"""
Gera os gráficos oficiais de 4 quadrantes para a análise de complementaridade do MLP:
1. reports/analise_complementaridade_mlp_small_oraculo.png
2. reports/analise_complementaridade_mlp_medium_oraculo.png
3. reports/analise_complementaridade_mlp_large_oraculo.png
Copia para os artefatos.
"""

import os
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_class(class_name, reps, recalls, consensus_labels, consensus_counts, consensus_pcts, excl_reps, excl_counts, excl_pcts, err_reps, err_bars, out_png, artifact_dir):
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

    # Quadrante 1: Recall Individual vs Oráculo
    ax1 = axes[0, 0]
    colors1 = ['#9b59b6', '#95a5a6', '#2980b9', '#e67e22', '#27ae60']
    b1 = ax1.bar(reps, recalls, color=colors1, width=0.55, edgecolor='black', linewidth=0.8)
    b1[-1].set_hatch('//')
    ax1.axhline(recalls[-1], color='#27ae60', linestyle='--', lw=1.5, alpha=0.7, label=f'Teto do Oráculo ({recalls[-1]:.1f}%)')
    ax1.set_title(f'A) Recall MLP em {class_name}: Modelos Isolados vs. Teto do Oráculo', fontweight='bold', pad=12)
    ax1.set_ylabel(f'Recall na Classe {class_name} (%)')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper left', frameon=True, fontsize=9.5)
    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                     fontsize=10.5, fontweight='bold', color='#111')

    # Quadrante 2: Grau de Consenso
    ax2 = axes[0, 1]
    colors_cons = ['#27ae60', '#2980b9', '#f39c12', '#e74c3c', '#7f8c8d']
    b2 = ax2.bar(consensus_labels, consensus_counts, color=colors_cons, edgecolor='black', linewidth=0.8, width=0.55)
    ax2.set_title(f'B) Grau de Consenso MLP em {class_name} (Total: {sum(consensus_counts)} Áudios)', fontweight='bold', pad=12)
    ax2.set_ylabel('Quantidade de Áudios')
    ax2.set_ylim(0, max(consensus_counts) * 1.25)
    for bar, pct in zip(b2, consensus_pcts):
        h = bar.get_height()
        ax2.annotate(f'{h} ({pct:.1f}%)', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                     fontsize=9.5, fontweight='bold')

    # Quadrante 3: Resgates Exclusivos
    ax3 = axes[1, 0]
    colors_excl = ['#9b59b6', '#2980b9', '#e67e22', '#16a085']
    b3 = ax3.barh(excl_reps, excl_counts, color=colors_excl, height=0.5, edgecolor='black', linewidth=0.8)
    ax3.set_title(f'C) Resgates Exclusivos MLP em {class_name} (Total: {sum(excl_counts)} Áudios)', fontweight='bold', pad=12)
    ax3.set_xlabel('Áudios Salvos Exclusivamente por Apenas UMA Entrada')
    ax3.set_xlim(0, max(excl_counts) * 1.35)
    for idx, (bar, pct) in enumerate(zip(b3, excl_pcts)):
        w = bar.get_width()
        tag = "  <-- CAMPEÃO DE RESGATES" if idx == 3 else ""
        ax3.annotate(f'{w} áudios ({pct:.1f}%){tag}', xy=(w, bar.get_y() + bar.get_height()/2),
                     xytext=(5, 0), textcoords="offset points", ha='left', va='center',
                     fontsize=9.5, fontweight='bold', color='#111' if idx != 3 else '#117864')

    # Quadrante 4: Padrão de Erro
    ax4 = axes[1, 1]
    w = 0.52
    c1, c2, c3 = err_bars['c1'], err_bars['c2'], err_bars['c3']
    p1 = ax4.bar(err_reps, c1['pct'], w, label=c1['name'], color=c1['color'], alpha=0.9, edgecolor='black', linewidth=0.6)
    p2 = ax4.bar(err_reps, c2['pct'], w, bottom=c1['pct'], label=c2['name'], color=c2['color'], alpha=0.9, edgecolor='black', linewidth=0.6)
    p3 = ax4.bar(err_reps, c3['pct'], w, bottom=np.array(c1['pct'])+np.array(c2['pct']), label=c3['name'], color=c3['color'], alpha=0.9, edgecolor='black', linewidth=0.6)

    ax4.set_title(f'D) Padrão de Erro MLP em {class_name} (Para onde vai quando erra? %)', fontweight='bold', pad=12)
    ax4.set_ylabel('Distribuição Percentual dos Erros (%)')
    ax4.set_ylim(0, 118)
    ax4.legend(loc='upper right', frameon=True, fontsize=9.5)

    for i in range(len(err_reps)):
        ax4.text(i, c1['pct'][i]/2, f'{c1["pct"][i]:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        ax4.text(i, c1['pct'][i] + c2['pct'][i]/2, f'{c2["pct"][i]:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        if c3['pct'][i] > 5:
            ax4.text(i, c1['pct'][i] + c2['pct'][i] + c3['pct'][i]/2, f'{c3["pct"][i]:.1f}%', ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')

    plt.suptitle(f'Análise de Complementaridade e Teto Teórico do MLP PyTorch na Classe {class_name} (Dataset IARA, 5x2cv)',
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()

    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {out_png}")

    art_path = os.path.join(artifact_dir, os.path.basename(out_png))
    shutil.copy2(out_png, art_path)
    print(f"Copiado para artefatos: {art_path}")

def main():
    artifact_dir = r"C:\Users\ev_ro\git\IARA\reports"

    # =========================================================================
    # 1. SMALL
    # =========================================================================
    plot_class(
        class_name="SMALL",
        reps=['HHT Puro\n(256-D)', 'DEMON\n(253-D)', 'MEL\n(256-D)', 'LOFAR\n(1024-D)', 'ORÁCULO\n(União 4 Modos)'],
        recalls=[30.73, 34.44, 44.78, 46.35, 76.85],
        consensus_labels=['4 Entradas\n(Unânime)', '3 Entradas\n(Alto Consenso)', '2 Entradas\n(Dividido)', '1 Entrada\n(Exclusivo/Frágil)', '0 Entradas\n(Erro Comum)'],
        consensus_counts=[95, 336, 457, 480, 412],
        consensus_pcts=[5.34, 18.88, 25.67, 26.97, 23.15],
        excl_reps=['HHT Exclusivo', 'MEL Exclusivo', 'LOFAR Exclusivo', 'DEMON Exclusivo'],
        excl_counts=[72, 125, 128, 155],
        excl_pcts=[4.04, 7.02, 7.19, 8.71],
        err_reps=['MEL (983 erros)', 'DEMON (1167 erros)', 'HHT (1233 erros)', 'LOFAR (955 erros)'],
        err_bars={
            'c1': {'name': 'Confundido com MEDIUM', 'color': '#f39c12', 'pct': [46.19, 25.88, 31.63, 39.58]},
            'c2': {'name': 'Confundido com LARGE', 'color': '#e74c3c', 'pct': [32.45, 40.70, 41.44, 47.33]},
            'c3': {'name': 'Confundido com AMBIENTAL', 'color': '#3498db', 'pct': [21.36, 33.42, 26.93, 13.09]}
        },
        out_png="reports/analise_complementaridade_mlp_small_oraculo.png",
        artifact_dir=r"C:\Users\ev_ro\.gemini\antigravity-ide\brain\ffb9500e-86ad-47f3-9cf6-d5e7cd24de4f"
    )

    # =========================================================================
    # 2. MEDIUM
    # =========================================================================
    plot_class(
        class_name="MEDIUM",
        reps=['DEMON\n(253-D)', 'HHT Puro\n(256-D)', 'LOFAR\n(1024-D)', 'MEL\n(256-D)', 'ORÁCULO\n(União 4 Modos)'],
        recalls=[46.04, 57.92, 61.48, 64.09, 84.90],
        consensus_labels=['4 Entradas\n(Unânime)', '3 Entradas\n(Alto Consenso)', '2 Entradas\n(Dividido)', '1 Entrada\n(Exclusivo/Frágil)', '0 Entradas\n(Erro Comum)'],
        consensus_counts=[446, 282, 253, 284, 225],
        consensus_pcts=[29.93, 18.93, 16.98, 19.06, 15.10],
        excl_reps=['LOFAR Exclusivo', 'DEMON Exclusivo', 'HHT Exclusivo', 'MEL Exclusivo'],
        excl_counts=[51, 67, 78, 88],
        excl_pcts=[3.42, 4.50, 5.23, 5.91],
        err_reps=['MEL (535 erros)', 'LOFAR (574 erros)', 'HHT (627 erros)', 'DEMON (804 erros)'],
        err_bars={
            'c1': {'name': 'Confundido com SMALL', 'color': '#3498db', 'pct': [56.45, 58.01, 38.28, 42.04]},
            'c2': {'name': 'Confundido com LARGE', 'color': '#e74c3c', 'pct': [36.64, 35.89, 31.74, 28.11]},
            'c3': {'name': 'Confundido com AMBIENTAL', 'color': '#95a5a6', 'pct': [6.92, 6.10, 29.98, 29.85]}
        },
        out_png="reports/analise_complementaridade_mlp_medium_oraculo.png",
        artifact_dir=r"C:\Users\ev_ro\.gemini\antigravity-ide\brain\ffb9500e-86ad-47f3-9cf6-d5e7cd24de4f"
    )

    # =========================================================================
    # 3. LARGE
    # =========================================================================
    plot_class(
        class_name="LARGE",
        reps=['DEMON\n(253-D)', 'HHT Puro\n(256-D)', 'MEL\n(256-D)', 'LOFAR\n(1024-D)', 'ORÁCULO\n(União 4 Modos)'],
        recalls=[64.16, 67.76, 68.83, 74.72, 88.83],
        consensus_labels=['4 Entradas\n(Unânime)', '3 Entradas\n(Alto Consenso)', '2 Entradas\n(Dividido)', '1 Entrada\n(Exclusivo/Frágil)', '0 Entradas\n(Erro Comum)'],
        consensus_counts=[920, 483, 268, 230, 239],
        consensus_pcts=[42.99, 22.57, 12.52, 10.75, 11.17],
        excl_reps=['MEL Exclusivo', 'HHT Exclusivo', 'LOFAR Exclusivo', 'DEMON Exclusivo'],
        excl_counts=[30, 37, 46, 117],
        excl_pcts=[1.40, 1.73, 2.15, 5.47],
        err_reps=['MEL (667 erros)', 'LOFAR (541 erros)', 'HHT (690 erros)', 'DEMON (767 erros)'],
        err_bars={
            'c1': {'name': 'Confundido com MEDIUM', 'color': '#f39c12', 'pct': [47.68, 39.37, 37.68, 27.77]},
            'c2': {'name': 'Confundido com SMALL', 'color': '#3498db', 'pct': [40.78, 50.28, 41.59, 50.85]},
            'c3': {'name': 'Confundido com AMBIENTAL', 'color': '#95a5a6', 'pct': [11.54, 10.35, 20.72, 21.38]}
        },
        out_png="reports/analise_complementaridade_mlp_large_oraculo.png",
        artifact_dir=r"C:\Users\ev_ro\.gemini\antigravity-ide\brain\ffb9500e-86ad-47f3-9cf6-d5e7cd24de4f"
    )

if __name__ == "__main__":
    main()
