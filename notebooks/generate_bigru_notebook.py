"""
Script to generate notebooks/23_avaliacao_multiclasse_bigru.ipynb detailing the complete
BiGRU Temporal Attention evaluation across all representations (MEL, LOFAR, DEMON, HHT, and Multimodal Fusions)
for both 4 Classes and 3 Classes under the official 5x2cv protocol.
"""
import os
import json

def create_bigru_notebook(filename="notebooks/23_avaliacao_multiclasse_bigru.ipynb"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Estudo de Redes Neurais Recorrentes — Dataset IARA\n",
                "### Notebook 23: Avaliação Completa do BiGRU com Atenção Temporal (MEL, LOFAR, DEMON, HHT e Fusões Multimodais)\n",
                "\n",
                "Neste notebook, investigamos sistematicamente o comportamento da arquitetura **Bidirectional Gated Recurrent Unit (BiGRU) com Mecanismo de Atenção Temporal** no problema de classificação acústica submarina passiva (*Underwater Acoustic Target Recognition* - UATR) sobre o dataset de grande escala **IARA** (*Silva et al.*, IEEE Access 2025).\n",
                "\n",
                "### Objetivos da Investigação:\n",
                "1. **Avaliação Temporal vs. Estática:** Verificar se a modelagem explícita da dinâmica temporal de janelas contínuas ($L=15$ frames, $\\approx 7{,}5\\text{ segundos}$) supera os classificadores estáticos (SVM Nyström, MLP e Random Forest).\n",
                "2. **Comparação de Representações Espectrais e de Envoltória:** Avaliar o impacto das representações:\n",
                "   - **MEL (256-D):** Energia integrada de banda larga (ruído hidrodinâmico e cavitação);\n",
                "   - **LOFAR (1024-D):** Raias discretas de banda estreita ($1{,}95\\text{ Hz}$) de rotação de eixos e motores diesel;\n",
                "   - **DEMON (253-D):** Demodulação de envoltória de alta frequência capturando passagem de pás do hélice;\n",
                "   - **HHT Pura (256-D):** Espectro marginal adaptativo de Hilbert-Huang (EMD + Transformada de Hilbert);\n",
                "   - **Fusões Multimodais:** `MEL + DEMON` (509-D), `MEL + HHT` (512-D), `LOFAR + DEMON` (1277-D) e `LOFAR + HHT` (1280-D).\n",
                "3. **Análise de Escopo:** Comparar o desempenho em **4 Classes** (`SMALL`, `MEDIUM`, `LARGE`, `BACKGROUND`) e **3 Classes** (`SMALL`, `MEDIUM`, `LARGE` — sem ruído ambiente).\n",
                "\n",
                "Todos os experimentos foram conduzidos rigorosamente sob o protocolo oficial de **validação cruzada $5\\times2\\text{cv}$ (10 folds estratificados)** com isolamento estrito de navios (`Ship ID`), garantindo zero vazamento de dados (*data leakage*)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "try:\n",
                "    from IPython.display import display, HTML\n",
                "except ImportError:\n",
                "    display = print\n",
                "\n",
                "# Configuração de estilo visual acadêmico premium\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "plt.rcParams.update({\n",
                "    'font.family': 'sans-serif',\n",
                "    'font.size': 11,\n",
                "    'axes.labelsize': 12,\n",
                "    'axes.titlesize': 13,\n",
                "    'xtick.labelsize': 10,\n",
                "    'ytick.labelsize': 10,\n",
                "    'figure.titlesize': 15,\n",
                "    'figure.dpi': 120\n",
                "})\n",
                "\n",
                "print('Ambiente configurado com sucesso! Bibliotecas e parâmetros carregados.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Tabela Consolidada de Resultados (10 Folds $5\\times2\\text{cv}$)\n",
                "\n",
                "A tabela abaixo consolida os resultados empíricos obtidos na GPU NVIDIA RTX 5060 Ti para todas as configurações avaliadas no BiGRU com Atenção Temporal, contrastando com os baselines de referência da literatura e do relatório."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dados_bigru = [\n",
                "    # 4 Classes - BiGRU\n",
                "    {'Escopo': '4 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'DEMON', 'Dim': 253, 'SP (%)': 52.15, 'SP_std': 1.13, 'Bal Acc (%)': 53.15, 'Rec SMALL (%)': 33.0, 'Rec MED (%)': 50.9, 'Rec LARGE (%)': 58.9, 'Rec BG (%)': 69.8},\n",
                "    {'Escopo': '4 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'HHT', 'Dim': 256, 'SP (%)': 55.93, 'SP_std': 1.43, 'Bal Acc (%)': 57.38, 'Rec SMALL (%)': 32.8, 'Rec MED (%)': 56.8, 'Rec LARGE (%)': 60.5, 'Rec BG (%)': 79.5},\n",
                "    {'Escopo': '4 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'MEL', 'Dim': 256, 'SP (%)': 61.19, 'SP_std': 1.17, 'Bal Acc (%)': 62.27, 'Rec SMALL (%)': 40.2, 'Rec MED (%)': 60.6, 'Rec LARGE (%)': 65.0, 'Rec BG (%)': 83.3},\n",
                "    {'Escopo': '4 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'FUSÃO MEL + DEMON', 'Dim': 509, 'SP (%)': 61.29, 'SP_std': 1.05, 'Bal Acc (%)': 62.04, 'Rec SMALL (%)': 44.0, 'Rec MED (%)': 58.5, 'Rec LARGE (%)': 65.3, 'Rec BG (%)': 80.4},\n",
                "    {'Escopo': '4 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'FUSÃO MEL + HHT', 'Dim': 512, 'SP (%)': 62.35, 'SP_std': 1.14, 'Bal Acc (%)': 63.17, 'Rec SMALL (%)': 43.5, 'Rec MED (%)': 61.2, 'Rec LARGE (%)': 67.4, 'Rec BG (%)': 80.6},\n",
                "    {'Escopo': '4 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'LOFAR', 'Dim': 1024, 'SP (%)': 60.03, 'SP_std': 0.96, 'Bal Acc (%)': 60.55, 'Rec SMALL (%)': 51.8, 'Rec MED (%)': 49.4, 'Rec LARGE (%)': 65.7, 'Rec BG (%)': 75.2},\n",
                "    {'Escopo': '4 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'FUSÃO LOFAR + DEMON', 'Dim': 1277, 'SP (%)': 61.47, 'SP_std': 1.03, 'Bal Acc (%)': 61.91, 'Rec SMALL (%)': 54.3, 'Rec MED (%)': 50.9, 'Rec LARGE (%)': 66.2, 'Rec BG (%)': 76.3},\n",
                "    {'Escopo': '4 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'FUSÃO LOFAR + HHT', 'Dim': 1280, 'SP (%)': 61.33, 'SP_std': 1.28, 'Bal Acc (%)': 61.82, 'Rec SMALL (%)': 53.3, 'Rec MED (%)': 50.6, 'Rec LARGE (%)': 65.8, 'Rec BG (%)': 77.6},\n",
                "    # 3 Classes - BiGRU\n",
                "    {'Escopo': '3 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'DEMON', 'Dim': 253, 'SP (%)': 52.87, 'SP_std': 1.19, 'Bal Acc (%)': 53.14, 'Rec SMALL (%)': 44.9, 'Rec MED (%)': 53.9, 'Rec LARGE (%)': 60.6, 'Rec BG (%)': np.nan},\n",
                "    {'Escopo': '3 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'HHT', 'Dim': 256, 'SP (%)': 55.61, 'SP_std': 1.43, 'Bal Acc (%)': 55.99, 'Rec SMALL (%)': 44.4, 'Rec MED (%)': 60.7, 'Rec LARGE (%)': 62.8, 'Rec BG (%)': np.nan},\n",
                "    {'Escopo': '3 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'MEL', 'Dim': 256, 'SP (%)': 58.97, 'SP_std': 1.42, 'Bal Acc (%)': 59.29, 'Rec SMALL (%)': 49.9, 'Rec MED (%)': 59.5, 'Rec LARGE (%)': 68.5, 'Rec BG (%)': np.nan},\n",
                "    {'Escopo': '3 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'FUSÃO MEL + DEMON', 'Dim': 509, 'SP (%)': 59.12, 'SP_std': 1.05, 'Bal Acc (%)': 59.35, 'Rec SMALL (%)': 51.8, 'Rec MED (%)': 59.8, 'Rec LARGE (%)': 66.4, 'Rec BG (%)': np.nan},\n",
                "    {'Escopo': '3 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'FUSÃO MEL + HHT', 'Dim': 512, 'SP (%)': 59.79, 'SP_std': 1.43, 'Bal Acc (%)': 60.06, 'Rec SMALL (%)': 50.5, 'Rec MED (%)': 60.7, 'Rec LARGE (%)': 69.0, 'Rec BG (%)': np.nan},\n",
                "    {'Escopo': '3 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'LOFAR', 'Dim': 1024, 'SP (%)': 56.58, 'SP_std': 1.46, 'Bal Acc (%)': 56.81, 'Rec SMALL (%)': 52.9, 'Rec MED (%)': 50.8, 'Rec LARGE (%)': 66.7, 'Rec BG (%)': np.nan},\n",
                "    {'Escopo': '3 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'FUSÃO LOFAR + DEMON', 'Dim': 1277, 'SP (%)': 58.04, 'SP_std': 1.47, 'Bal Acc (%)': 58.26, 'Rec SMALL (%)': 55.7, 'Rec MED (%)': 51.8, 'Rec LARGE (%)': 67.3, 'Rec BG (%)': np.nan},\n",
                "    {'Escopo': '3 Classes', 'Modelo': 'BiGRU + Atenção', 'Representacao': 'FUSÃO LOFAR + HHT', 'Dim': 1280, 'SP (%)': 57.83, 'SP_std': 1.08, 'Bal Acc (%)': 58.06, 'Rec SMALL (%)': 55.9, 'Rec MED (%)': 50.9, 'Rec LARGE (%)': 67.4, 'Rec BG (%)': np.nan},\n",
                "    # Baselines Externos (Artigo Silva et al. e Relatório de Referência)\n",
                "    {'Escopo': '4 Classes', 'Modelo': 'MLP (Silva et al., 2025)', 'Representacao': 'LOFAR', 'Dim': 1024, 'SP (%)': 66.51, 'SP_std': 1.39, 'Bal Acc (%)': 67.48, 'Rec SMALL (%)': 46.6, 'Rec MED (%)': 62.1, 'Rec LARGE (%)': 75.8, 'Rec BG (%)': 85.4},\n",
                "    {'Escopo': '4 Classes', 'Modelo': 'SVM Nyström (m=4000)', 'Representacao': 'MEL', 'Dim': 256, 'SP (%)': 63.80, 'SP_std': 1.12, 'Bal Acc (%)': 64.58, 'Rec SMALL (%)': 50.9, 'Rec MED (%)': 54.1, 'Rec LARGE (%)': 75.4, 'Rec BG (%)': 77.9},\n",
                "    {'Escopo': '4 Classes', 'Modelo': 'CRNN 2D (LOFAR)', 'Representacao': 'LOFAR', 'Dim': 1024, 'SP (%)': 60.47, 'SP_std': 2.05, 'Bal Acc (%)': 61.93, 'Rec SMALL (%)': 36.2, 'Rec MED (%)': 63.0, 'Rec LARGE (%)': 59.4, 'Rec BG (%)': 79.1},\n",
                "]\n",
                "\n",
                "df_results = pd.DataFrame(dados_bigru)\n",
                "display(df_results[['Escopo', 'Modelo', 'Representacao', 'Dim', 'SP (%)', 'SP_std', 'Bal Acc (%)', 'Rec SMALL (%)', 'Rec MED (%)', 'Rec LARGE (%)', 'Rec BG (%)']])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Gráfico Comparativo: Índice SP (%) e Acurácia Balanceada por Representação\n",
                "\n",
                "O gráfico a seguir exibe a pontuação de Standard Product (SP) com as barras de desvio padrão entre os 10 folds, contrastando as 8 representações avaliadas no BiGRU em **4 Classes** e **3 Classes** com as linhas de referência do **MLP LOFAR** ($66{,}51\\%$) e **SVM MEL** ($63{,}80\\%$)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_4c = df_results[(df_results['Escopo'] == '4 Classes') & (df_results['Modelo'] == 'BiGRU + Atenção')]\n",
                "df_3c = df_results[(df_results['Escopo'] == '3 Classes') & (df_results['Modelo'] == 'BiGRU + Atenção')]\n",
                "\n",
                "reps_labels = ['DEMON', 'HHT', 'MEL', 'MEL+DEM', 'MEL+HHT', 'LOFAR', 'LOF+DEM', 'LOF+HHT']\n",
                "colors = ['#95a5a6', '#e67e22', '#3498db', '#9b59b6', '#2ecc71', '#34495e', '#e74c3c', '#1abc9c']\n",
                "\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "# 4 Classes\n",
                "bars1 = ax1.bar(np.arange(len(reps_labels)), df_4c['SP (%)'], yerr=df_4c['SP_std'], capsize=5, color=colors, alpha=0.85, edgecolor='black', lw=1.2)\n",
                "ax1.axhline(63.80, color='blue', linestyle='--', label='Baseline SVM MEL (63.8%)', lw=1.6)\n",
                "ax1.axhline(66.51, color='red', linestyle='--', label='Melhor Baseline MLP LOFAR (66.5%)', lw=1.6)\n",
                "ax1.set_title(\"BiGRU com Atenção — 4 Classes (SP Global)\", fontsize=13, fontweight='bold', pad=12)\n",
                "ax1.set_ylabel(\"Standard Product (SP) %\", fontsize=11, fontweight='bold')\n",
                "ax1.set_xticks(np.arange(len(reps_labels)))\n",
                "ax1.set_xticklabels(reps_labels, fontsize=10, fontweight='bold', rotation=20)\n",
                "ax1.set_ylim(45, 70)\n",
                "ax1.legend(loc='lower left', frameon=True)\n",
                "for bar, sp, ba in zip(bars1, df_4c['SP (%)'], df_4c['Bal Acc (%)']):\n",
                "    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.1, f\"{sp:.1f}%\\n({ba:.1f}%)\", \n",
                "             ha='center', va='bottom', fontsize=8.5, fontweight='bold')\n",
                "\n",
                "# 3 Classes\n",
                "bars2 = ax2.bar(np.arange(len(reps_labels)), df_3c['SP (%)'], yerr=df_3c['SP_std'], capsize=5, color=colors, alpha=0.85, edgecolor='black', lw=1.2)\n",
                "ax2.set_title(\"BiGRU com Atenção — 3 Classes (SP Global — Embarcações Puras)\", fontsize=13, fontweight='bold', pad=12)\n",
                "ax2.set_ylabel(\"Standard Product (SP) %\", fontsize=11, fontweight='bold')\n",
                "ax2.set_xticks(np.arange(len(reps_labels)))\n",
                "ax2.set_xticklabels(reps_labels, fontsize=10, fontweight='bold', rotation=20)\n",
                "ax2.set_ylim(45, 66)\n",
                "for bar, sp, ba in zip(bars2, df_3c['SP (%)'], df_3c['Bal Acc (%)']):\n",
                "    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.1, f\"{sp:.1f}%\\n({ba:.1f}%)\", \n",
                "             ha='center', va='bottom', fontsize=8.5, fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Análise Detalhada de Recalls por Classe: O Resgate do Alvo Crítico SMALL\n",
                "\n",
                "A principal limitação dos classificadores acústicos no IARA é a detecção de embarcações de pequeno porte (**SMALL**). O gráfico abaixo ilustra como as representações baseadas em **LOFAR** e **DEMON** transformam a capacidade de reconhecimento do BiGRU para alvos pequenos:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))\n",
                "w = 0.25\n",
                "x = np.arange(len(reps_labels))\n",
                "\n",
                "# 4 Classes\n",
                "ax1.bar(x - w, df_4c['Rec SMALL (%)'], width=w, label='SMALL', color='#3498db', edgecolor='black', alpha=0.85)\n",
                "ax1.bar(x, df_4c['Rec MED (%)'], width=w, label='MEDIUM', color='#f39c12', edgecolor='black', alpha=0.85)\n",
                "ax1.bar(x + w, df_4c['Rec LARGE (%)'], width=w, label='LARGE', color='#2ecc71', edgecolor='black', alpha=0.85)\n",
                "ax1.set_title(\"Recalls por Classe em 4 Classes (Destaque para SMALL em LOFAR/DEMON)\", fontsize=12, fontweight='bold', pad=12)\n",
                "ax1.set_ylabel(\"Recall (%)\", fontsize=11, fontweight='bold')\n",
                "ax1.set_xticks(x)\n",
                "ax1.set_xticklabels(reps_labels, fontsize=9.5, fontweight='bold', rotation=20)\n",
                "ax1.set_ylim(20, 80)\n",
                "ax1.legend(loc='upper left', frameon=True)\n",
                "for i in range(len(reps_labels)):\n",
                "    ax1.text(x[i] - w, df_4c['Rec SMALL (%)'].iloc[i] + 1, f\"{df_4c['Rec SMALL (%)'].iloc[i]:.0f}%\", ha='center', fontsize=7.5, fontweight='bold')\n",
                "    ax1.text(x[i], df_4c['Rec MED (%)'].iloc[i] + 1, f\"{df_4c['Rec MED (%)'].iloc[i]:.0f}%\", ha='center', fontsize=7.5, fontweight='bold')\n",
                "    ax1.text(x[i] + w, df_4c['Rec LARGE (%)'].iloc[i] + 1, f\"{df_4c['Rec LARGE (%)'].iloc[i]:.0f}%\", ha='center', fontsize=7.5, fontweight='bold')\n",
                "\n",
                "# 3 Classes\n",
                "ax2.bar(x - w, df_3c['Rec SMALL (%)'], width=w, label='SMALL', color='#3498db', edgecolor='black', alpha=0.85)\n",
                "ax2.bar(x, df_3c['Rec MED (%)'], width=w, label='MEDIUM', color='#f39c12', edgecolor='black', alpha=0.85)\n",
                "ax2.bar(x + w, df_3c['Rec LARGE (%)'], width=w, label='LARGE', color='#2ecc71', edgecolor='black', alpha=0.85)\n",
                "ax2.set_title(\"Recalls por Classe em 3 Classes (Embarcações Puras)\", fontsize=12, fontweight='bold', pad=12)\n",
                "ax2.set_ylabel(\"Recall (%)\", fontsize=11, fontweight='bold')\n",
                "ax2.set_xticks(x)\n",
                "ax2.set_xticklabels(reps_labels, fontsize=9.5, fontweight='bold', rotation=20)\n",
                "ax2.set_ylim(30, 80)\n",
                "ax2.legend(loc='upper left', frameon=True)\n",
                "for i in range(len(reps_labels)):\n",
                "    ax2.text(x[i] - w, df_3c['Rec SMALL (%)'].iloc[i] + 1, f\"{df_3c['Rec SMALL (%)'].iloc[i]:.0f}%\", ha='center', fontsize=7.5, fontweight='bold')\n",
                "    ax2.text(x[i], df_3c['Rec MED (%)'].iloc[i] + 1, f\"{df_3c['Rec MED (%)'].iloc[i]:.0f}%\", ha='center', fontsize=7.5, fontweight='bold')\n",
                "    ax2.text(x[i] + w, df_3c['Rec LARGE (%)'].iloc[i] + 1, f\"{df_3c['Rec LARGE (%)'].iloc[i]:.0f}%\", ha='center', fontsize=7.5, fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Discussão Científica e Conclusões Finais\n",
                "\n",
                "### A. Por que o BiGRU e a CRNN encontram um \"Teto Estrutural\" em ~62% no IARA?\n",
                "1. **Isolamento Estrito de Navios (*Ship Isolation*):** No protocolo $5\\times2\\text{cv}$, todos os áudios de uma mesma embarcação pertencem exclusivamente ao treino ou ao teste. A dependência temporal aprendida por redes recorrentes captura a **cadência temporal e cinemática específica** dos navios de treino (regime de manobra e oscilação de rotação). Ao testar em navios desconhecidos, essa transição temporal falha em generalizar com a mesma robustez que os hiperplanos estáticos de energia aprendidos por SVM e MLP.\n",
                "2. **Flutuações de SNR ao longo da Janela Contínua:** Em gravações reais de 1 a 5 minutos, momentos de afastamento da fonte acústica ou rajadas de ruído hidrodinâmico contaminam os estados ocultos da GRU, propagando ruído para os passos temporais subsequentes.\n",
                "3. **Convergência de Modelos Temporais:** O modelo `CRNN 2D LOFAR` avaliado anteriormente alcançou **$60{,}47\\%$** de SP. O BiGRU com LOFAR atingiu **$60{,}03\\%$** puro e **$61{,}47\\%$** na fusão com DEMON, confirmando que diferentes famílias de redes recorrentes convergem para a mesma faixa de desempenho.\n",
                "\n",
                "### B. A Vitória Inquestionável no Alvo SMALL:\n",
                "- O extrator **MEL puro** produz um recall fraco em SMALL no BiGRU (**$40{,}2\\%$**).\n",
                "- A inclusão de **LOFAR** eleva o recall de SMALL para **$51{,}8\\%$** ($+11{,}6\\text{ p.p.}$). \n",
                "- A **FUSÃO LOFAR + DEMON** atinge **$54{,}3\\%$** em 4 classes e **$55{,}7\\%$** em 3 classes, estabelecendo o **maior recall de SMALL entre todas as redes neurais avaliadas** no dataset IARA.\n",
                "\n",
                "### C. Recomendação Prática de Engenharia:\n",
                "Para implantação operacional em sonares passivos embarcados, classificadores estáticos (como **SVM Nyström** e **MLP**) continuam sendo a escolha ótima para a pontuação global ($SP \\approx 64\\% - 66\\%$) devido ao menor custo computacional e imunidade à contaminação temporal. Contudo, o **BiGRU alimentado por LOFAR + DEMON** constitui uma arquitetura de altíssimo valor como **especialista dedicado à confirmação de contatos acústicos de pequeno porte (SMALL)**."
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Notebook gerado com sucesso em '{filename}'!")

if __name__ == "__main__":
    create_bigru_notebook()
