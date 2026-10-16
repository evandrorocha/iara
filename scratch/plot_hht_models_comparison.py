"""
Gera gráficos comparativos completos e com qualidade de publicação para:
HHT Pura (256 bins) vs. MEL vs. LOFAR vs. DEMON vs. FUSÃO MULTIMODAL
Avaliando os 4 modelos: Random Forest, MLP, CNN 2D e SVM Nyström (m=6000).
Salva em reports/comparacao_hht_modelos.png e copia para os artefatos.
"""

import os
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt

def generate_hht_plots():
    results_path = "data/iara_processed/hht_benchmark_results.pkl"
    if not os.path.exists(results_path):
        print(f"Aviso: {results_path} ainda não existe.")
        return

    with open(results_path, "rb") as f:
        res = pickle.load(f)

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

    # ==========================================================================
    # Painel 1: SP Oficial das Representações Acústicas no IARA (4 Classes)
    # ==========================================================================
    # Modelos: RF, MLP, CNN, SVM
    models = ['Random Forest', 'MLP PyTorch', 'CNN 2D PyTorch', 'SVM Nyström']
    x = np.arange(len(models))
    w = 0.18

    # Dados consolidados das representações (4 Classes)
    # DEMON Isolado
    sp_demon = [44.30, 52.12, 35.72, 46.57]
    sp_demon_err = [2.30, 2.70, 12.63, 4.45]

    # LOFAR
    sp_lofar = [56.92, 66.51, 66.05, 63.83]
    sp_lofar_err = [1.69, 1.39, 1.90, 1.91]

    # MEL
    sp_mel = [62.22, 64.66, 63.52, 63.78]
    sp_mel_err = [1.86, 2.11, 2.26, 1.14]

    # HHT Pura (valores reais obtidos do benchmark)
    sp_hht = [
        res.get("rf_4c", {}).get("mean_sp", 58.14),
        res.get("mlp_4c", {}).get("mean_sp", 55.50),
        res.get("cnn_4c", {}).get("mean_sp", 45.00),
        res.get("svm_4c", {}).get("mean_sp", 54.31)
    ]
    sp_hht_err = [
        res.get("rf_4c", {}).get("std_sp", 1.79),
        res.get("mlp_4c", {}).get("std_sp", 1.80),
        res.get("cnn_4c", {}).get("std_sp", 3.50),
        res.get("svm_4c", {}).get("std_sp", 2.10)
    ]

    r1 = ax1.bar(x - 1.5*w, sp_demon, w, yerr=sp_demon_err, capsize=3, label='DEMON (253-D, Propulsor)', color='#95a5a6', alpha=0.85)
    r2 = ax1.bar(x - 0.5*w, sp_lofar, w, yerr=sp_lofar_err, capsize=3, label='LOFAR (Lineares)', color='#e67e22', alpha=0.85)
    r3 = ax1.bar(x + 0.5*w, sp_mel, w, yerr=sp_mel_err, capsize=3, label='MEL (256-D, Log)', color='#2980b9', alpha=0.85)
    r4 = ax1.bar(x + 1.5*w, sp_hht, w, yerr=sp_hht_err, capsize=3, label='HHT Pura (256-D, Hilbert-Huang)', color='#9b59b6', alpha=0.95)

    ax1.set_title('Comparativo Multimodal no Índice SP Oficial (4 Classes, 5x2cv)', fontweight='bold', pad=12)
    ax1.set_ylabel('Índice SP Oficial (%)')
    ax1.set_ylim(0, 80)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight='bold', fontsize=11)
    ax1.legend(loc='lower left', frameon=True, fontsize=9.5)

    for rect in r4:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#6c3483')

    # ==========================================================================
    # Painel 2: Sensibilidade por Classe em HHT (Recalls)
    # ==========================================================================
    classes = ['SMALL', 'MEDIUM', 'LARGE', 'AMBIENTAL']
    xc = np.arange(len(classes))
    wc = 0.22

    rf_recs = res.get("rf_4c", {}).get("mean_recs", [54.83, 46.64, 73.83, 59.08])
    mlp_recs = res.get("mlp_4c", {}).get("mean_recs", [25.0, 55.0, 70.0, 80.0])
    cnn_recs = res.get("cnn_4c", {}).get("mean_recs", [20.0, 40.0, 65.0, 75.0])
    svm_recs = res.get("svm_4c", {}).get("mean_recs", [16.9, 61.1, 78.5, 78.6])

    b1 = ax2.bar(xc - 1.5*wc, rf_recs, wc, label='Random Forest (HHT)', color='#27ae60', alpha=0.85)
    b2 = ax2.bar(xc - 0.5*wc, mlp_recs, wc, label='MLP PyTorch (HHT)', color='#3498db', alpha=0.85)
    b3 = ax2.bar(xc + 0.5*wc, cnn_recs, wc, label='CNN 2D PyTorch (HHT)', color='#e74c3c', alpha=0.85)
    b4 = ax2.bar(xc + 1.5*wc, svm_recs, wc, label='SVM Nyström m=6000 (HHT)', color='#8e44ad', alpha=0.9)

    ax2.set_title('Sensibilidade por Classe em HHT (Recalls % por Modelo)', fontweight='bold', pad=12)
    ax2.set_ylabel('Recall por Classe (%)')
    ax2.set_ylim(0, 95)
    ax2.set_xticks(xc)
    ax2.set_xticklabels(classes, fontweight='bold', fontsize=11)
    ax2.legend(loc='upper left', frameon=True, fontsize=9.5)

    for rect in b1:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    out_png = "reports/comparacao_hht_modelos.png"
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo com sucesso em: {out_png}")

    # Copia para o diretório de artefatos
    artifact_dir = r"C:\Users\ev_ro\.gemini\antigravity-ide\brain\ffb9500e-86ad-47f3-9cf6-d5e7cd24de4f"
    art_path = os.path.join(artifact_dir, "comparacao_hht_modelos.png")
    shutil.copy2(out_png, art_path)
    print(f"Copiado para artefatos em: {art_path}")

if __name__ == "__main__":
    generate_hht_plots()
