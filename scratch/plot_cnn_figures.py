"""
Gera os 3 gráficos oficiais de 4 quadrantes para a CNN 2D a partir de
data/iara_processed/cnn_complementarity_results.pkl e copia para os artefatos.
"""

import os
import shutil
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_4panel_figure(res, class_name, out_filename, artifact_dir):
    total = res["total"]
    total_corr = res["total_correct"]
    oracle_pct = total_corr / total * 100

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

    # Quadrante 1: Desempenho Isolado vs Oráculo
    ax1 = axes[0, 0]
    reps = ['DEMON\n(253-D)', 'HHT Puro\n(256-D)', 'LOFAR\n(1024-D)', 'MEL\n(256-D)', 'ORÁCULO CNN\n(União 4 Modos)']
    rec_vals = [
        res["ind_recalls"]["DEMON"],
        res["ind_recalls"]["HHT"],
        res["ind_recalls"]["LOFAR"],
        res["ind_recalls"]["MEL"],
        oracle_pct
    ]
    colors1 = ['#95a5a6', '#9b59b6', '#e67e22', '#2980b9', '#27ae60']
    b1 = ax1.bar(reps, rec_vals, color=colors1, width=0.55, edgecolor='black', linewidth=0.8)
    b1[-1].set_hatch('//')
    best_ind = max(rec_vals[:-1])
    best_rep = ["DEMON", "HHT", "LOFAR", "MEL"][np.argmax(rec_vals[:-1])]

    ax1.axhline(oracle_pct, color='#27ae60', linestyle='--', lw=1.5, alpha=0.7, label=f'Teto do Oráculo ({oracle_pct:.1f}%)')
    ax1.axhline(best_ind, color='#e67e22', linestyle=':', lw=1.5, alpha=0.7, label=f'Melhor Isolado {best_rep} ({best_ind:.1f}%)')

    ax1.set_title(f'A) Recall CNN 2D em {class_name}: Modelos Isolados vs. Teto do Oráculo', fontweight='bold', pad=12)
    ax1.set_ylabel(f'Recall na Classe {class_name} (%)')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper left', frameon=True, fontsize=9.5)

    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                     fontsize=10.5, fontweight='bold', color='#111')

    # Quadrante 2: Pirâmide de Consenso
    ax2 = axes[0, 1]
    labels_cons = ['4 Entradas\n(Unânime)', '3 Entradas\n(Alto Consenso)', '2 Entradas\n(Dividido)', '1 Entrada\n(Exclusivo/Frágil)', '0 Entradas\n(Erro Comum)']
    counts_cons = [res["c4"], res["c3"], res["c2"], res["c1"], res["c0"]]
    pcts_cons = [c / total * 100 for c in counts_cons]
    colors_cons = ['#27ae60', '#2980b9', '#f39c12', '#e74c3c', '#7f8c8d']

    b2 = ax2.bar(labels_cons, counts_cons, color=colors_cons, edgecolor='black', linewidth=0.8, width=0.55)
    ax2.set_title(f'B) Grau de Consenso CNN 2D em {class_name} (Total: {total} Áudios)', fontweight='bold', pad=12)
    ax2.set_ylabel(f'Quantidade de Áudios de {class_name}')
    ax2.set_ylim(0, max(counts_cons) * 1.18 if max(counts_cons) > 0 else 100)

    for bar, pct in zip(b2, pcts_cons):
        h = bar.get_height()
        ax2.annotate(f'{h} ({pct:.1f}%)', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                     fontsize=9.5, fontweight='bold')

    # Quadrante 3: Resgates Exclusivos
    ax3 = axes[1, 0]
    excl_reps = ['HHT Exclusivo', 'MEL Exclusivo', 'LOFAR Exclusivo', 'DEMON Exclusivo']
    excl_counts = [res["only_hht"], res["only_mel"], res["only_lofar"], res["only_demon"]]
    excl_pcts = [c / total * 100 for c in excl_counts]
    colors_excl = ['#9b59b6', '#2980b9', '#e67e22', '#16a085']

    b3 = ax3.barh(excl_reps, excl_counts, color=colors_excl, height=0.5, edgecolor='black', linewidth=0.8)
    ax3.set_title(f'C) Resgates Exclusivos CNN 2D em {class_name} (Total: {sum(excl_counts)} Áudios Únicos)', fontweight='bold', pad=12)
    ax3.set_xlabel('Áudios Salvos Exclusivamente por Apenas UMA Entrada')
    ax3.set_xlim(0, max(excl_counts) * 1.35 if max(excl_counts) > 0 else 50)

    champ_idx = np.argmax(excl_counts)
    for idx, (bar, pct) in enumerate(zip(b3, excl_pcts)):
        w = bar.get_width()
        tag = f"  <-- CAMPEÃO DE RESGATES EM {class_name}" if idx == champ_idx else ""
        ax3.annotate(f'{w} áudios ({pct:.1f}%){tag}', xy=(w, bar.get_y() + bar.get_height()/2),
                     xytext=(5, 0), textcoords="offset points", ha='left', va='center',
                     fontsize=9.5, fontweight='bold', color='#111' if idx != champ_idx else '#117864')

    # Quadrante 4: Padrão de Erro
    ax4 = axes[1, 1]
    reps_err = ['LOFAR', 'MEL', 'HHT', 'DEMON']
    other_classes = [c for c in ["SMALL", "MEDIUM", "LARGE", "AMBIENTAL"] if c != class_name]

    c_colors = {'SMALL': '#3498db', 'MEDIUM': '#f39c12', 'LARGE': '#2ecc71', 'AMBIENTAL': '#95a5a6'}
    bottom = np.zeros(len(reps_err))
    w = 0.52

    for other_c in other_classes:
        vals = [res["err_patterns"][r].get(other_c, 0.0) for r in reps_err]
        ax4.bar(reps_err, vals, w, bottom=bottom, label=f'Confundido com {other_c}', color=c_colors.get(other_c, '#888'), alpha=0.9, edgecolor='black', linewidth=0.6)
        for i, val in enumerate(vals):
            if val > 6:
                ax4.text(i, bottom[i] + val/2, f'{val:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        bottom += np.array(vals)

    ax4.set_title(f'D) Padrão de Erro CNN 2D em {class_name}: Para onde vai a confusão? (%)', fontweight='bold', pad=12)
    ax4.set_ylabel('Distribuição Percentual dos Erros (%)')
    ax4.set_ylim(0, 118)
    ax4.legend(loc='upper right', frameon=True, fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    plt.savefig(out_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo com sucesso em '{out_filename}'.")

    artifact_copy = os.path.join(artifact_dir, os.path.basename(out_filename))
    shutil.copy2(out_filename, artifact_copy)
    print(f"Copiado para os artefatos: '{artifact_copy}'.")

def main():
    pkl_file = "data/iara_processed/cnn_complementarity_results.pkl"
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    res_sml = data["res_sml"]
    res_med = data["res_med"]
    res_lrg = data["res_lrg"]

    artifact_dir = r"C:\Users\ev_ro\.gemini\antigravity-ide\brain\ffb9500e-86ad-47f3-9cf6-d5e7cd24de4f"
    plot_4panel_figure(res_sml, "SMALL", "reports/analise_complementaridade_cnn_small_oraculo.png", artifact_dir)
    plot_4panel_figure(res_med, "MEDIUM", "reports/analise_complementaridade_cnn_medium_oraculo.png", artifact_dir)
    plot_4panel_figure(res_lrg, "LARGE", "reports/analise_complementaridade_cnn_large_oraculo.png", artifact_dir)
    print("Todas as 3 figuras foram geradas e salvas com sucesso!")

if __name__ == "__main__":
    main()
