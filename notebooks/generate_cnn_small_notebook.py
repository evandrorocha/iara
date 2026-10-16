"""
Script de geração do notebook 20: notebooks/20_analise_complementaridade_cnn_small.ipynb
Análise de Ortogonalidade, Efeito Oráculo e Grau de Consenso na CNN 2D para a Classe SMALL (Dataset IARA).
"""

import os
import json

def create_notebook(filename="notebooks/20_analise_complementaridade_cnn_small.ipynb"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA\n",
                "### Notebook 20: Análise de Ortogonalidade, Efeito Oráculo e Grau de Consenso na CNN 2D — Classe SMALL\n",
                "\n",
                "**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)\n",
                "**Data:** Outubro de 2026\n",
                "\n",
                "---\n",
                "\n",
                "## Resumo Executivo da Investigação com Redes Convolucionais 2D (CNN)\n",
                "\n",
                "A classe **`SMALL`** (embarcações de pequeno porte: lanchas rápidas e veleiros a motor com comprimento $\\\\le 15\\\\text{ m}$) foi avaliada em todas as **1.780 instâncias de teste** nos 10 folds oficiais do protocolo $5\\\\times2$cv com a arquitetura **CNN 2D PyTorch (GPU RTX 5060 Ti, blocos $32 \\\\times D$)** nas 4 representações (**MEL, LOFAR, HHT e DEMON**).\n",
                "\n",
                "O objetivo é confrontar os resultados de SVM (Notebook 10), MLP (Notebook 14) e Random Forest (Notebook 17) com a abordagem de representação espaço-temporal convolucional.\n",
                "\n",
                "### Principais Descobertas na CNN 2D:\n",
                "1. **Teto do Oráculo Atinge $73{,}54\\\\%$ (1.309 / 1.780 lanchas):**\n",
                "   - A combinação das 4 entradas na CNN resgata **1.309 lanchas**, superando o melhor modelo isolado (LOFAR com $47{,}02\\\\%$) em **$+26{,}5$ pontos percentuais**.\n",
                "   - O erro comum (onde as 4 CNNs erram simultaneamente) é de **$26{,}46\\\\%$ (471 áudios)**.\n",
                "2. **A Mais Baixa Unanimidade de Todo o Estudo ($1{,}29\\\\%$ — Apenas 23 áudios):**\n",
                "   - Apenas **23 de 1.780 lanchas** são acertadas simultaneamente pelas 4 CNNs. Convoluções espaciais pressupõem correlação e continuidade temporal que são severamente quebradas pelo marulho caótico e transientes rápidos de propulsores leves.\n",
                "3. **Ortogonalidade Extrema e Alta Dependência de Resgates Exclusivos ($34{,}89\\\\%$):**\n",
                "   - Mais de um terço de todas as lanchas salvas pela CNN dependem de **apenas UMA entrada** ($621$ áudios).\n",
                "   - O LOFAR lidera os resgates exclusivos na CNN com **254 lanchas ($14{,}27\\\\%$)**, seguido de perto pelo MEL com **211 lanchas ($11{,}85\\\\%$)**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Configuração do Ambiente e Imports\n",
                "import os\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "try:\n",
                "    from IPython.display import display\n",
                "except ImportError:\n",
                "    display = print\n",
                "\n",
                "# Estilização publication-ready (sem dependência externa)\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "plt.rcParams.update({\n",
                "    'font.family': 'sans-serif',\n",
                "    'font.size': 11,\n",
                "    'axes.labelsize': 12,\n",
                "    'axes.titlesize': 13,\n",
                "    'xtick.labelsize': 10,\n",
                "    'ytick.labelsize': 10,\n",
                "    'figure.titlesize': 15,\n",
                "    'figure.dpi': 150\n",
                "})\n",
                "\n",
                "print(\"Ambiente configurado com sucesso! Bibliotecas prontas.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Metodologia Experimental e Base de Dados de Teste\n",
                "\n",
                "Avaliação rigorosa sobre todas as **1.780 instâncias de teste** onde a classe real comprovada era **`SMALL` (Classe 0)** sob o protocolo oficial $5\\\\times2$cv nos 10 folds com **CNN 2D PyTorch** nas 4 representações (**MEL, LOFAR, HHT e DEMON**):\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Dados Consolidados das Predições da CNN 2D na Classe SMALL\n",
                "total_sml = 1780\n",
                "\n",
                "# Desempenho individual na CNN\n",
                "hits_data = {\n",
                "    'Representacao': ['DEMON (253-D)', 'HHT Puro (256-D)', 'MEL (256-D)', 'LOFAR (1024-D)', 'ORÁCULO CNN (União 4 Modos)'],\n",
                "    'Acertos': [273, 365, 728, 837, 1309],\n",
                "    'Recall_Pct': [15.34, 20.51, 40.90, 47.02, 73.54]\n",
                "}\n",
                "df_hits = pd.DataFrame(hits_data)\n",
                "\n",
                "# Grau de consenso na CNN\n",
                "consensus_data = {\n",
                "    'Nivel_Consenso': [\n",
                "        '4 de 4 (Unanimidade / Independente)',\n",
                "        '3 de 4 (Alto Consenso)',\n",
                "        '2 de 4 (Consenso Parcial / Dividido)',\n",
                "        '1 de 4 (Frágil / Resgate Exclusivo)',\n",
                "        '0 de 4 (Erro Comum / Incorrigível)'\n",
                "    ],\n",
                "    'Audios': [23, 160, 505, 621, 471],\n",
                "    'Pct_Total_Small': [1.29, 8.99, 28.37, 34.89, 26.46],\n",
                "    'Pct_Dentro_Oraculo': [1.76, 12.22, 38.58, 47.44, np.nan]\n",
                "}\n",
                "df_consensus = pd.DataFrame(consensus_data)\n",
                "\n",
                "# Resgates exclusivos por representação na CNN\n",
                "exclusive_data = {\n",
                "    'Representacao': ['DEMON Exclusivo', 'HHT Exclusivo', 'MEL Exclusivo', 'LOFAR Exclusivo'],\n",
                "    'Audios_Salvos': [72, 84, 211, 254],\n",
                "    'Percentual': [4.04, 4.72, 11.85, 14.27]\n",
                "}\n",
                "df_exclusive = pd.DataFrame(exclusive_data)\n",
                "\n",
                "# Efeito Reverso em Relação ao Baseline LOFAR (837 acertos, 943 erros)\n",
                "reverse_data = {\n",
                "    'Destino_Troca': ['Para MEL', 'Para HHT', 'Para DEMON'],\n",
                "    'Mantem_do_LOFAR': [431, 199, 150],\n",
                "    'Perde_do_LOFAR (Efeito Reverso)': [406, 638, 687],\n",
                "    'Pct_Perda_Acertos (%)': [48.51, 76.22, 82.08],\n",
                "    'Resgata_Erros_LOFAR': [297, 166, 123],\n",
                "    'Pct_Resgate_Erros (%)': [31.50, 17.60, 13.04],\n",
                "    'Saldo_Liquido_Audios': [-109, -472, -564]\n",
                "}\n",
                "df_reverse = pd.DataFrame(reverse_data)\n",
                "\n",
                "# Padrão de confusão quando erra em SMALL na CNN\n",
                "confusion_pattern_data = {\n",
                "    'Representacao': ['LOFAR (943 erros)', 'MEL (1052 erros)', 'HHT (1415 erros)', 'DEMON (1507 erros)'],\n",
                "    'Confundido_MEDIUM_Pct': [51.43, 57.79, 34.28, 50.56],\n",
                "    'Confundido_LARGE_Pct': [37.12, 26.43, 37.74, 31.06],\n",
                "    'Confundido_AMBIENTAL_Pct': [11.45, 15.78, 27.99, 18.38]\n",
                "}\n",
                "df_confusion_pattern = pd.DataFrame(confusion_pattern_data)\n",
                "\n",
                "display(df_hits)\n",
                "display(df_consensus)\n",
                "display(df_reverse)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. O Efeito Oráculo na CNN 2D: Recall Individual vs. Teto Teórico\n",
                "\n",
                "A CNN 2D atinge **$47{,}02\\\\%$ de recall em `SMALL`** com o LOFAR e **$40{,}90\\\\%$** com o MEL.\n",
                "\n",
                "Contudo, a combinação das 4 entradas via **Oráculo CNN** salta para **$73{,}54\\\\%$ (1.309 acertos)**:\n",
                "- Um ganho de **$+26{,}5$ pontos percentuais** sobre o LOFAR isolado;\n",
                "- O erro comum incorrigível é de **$26{,}46\\\\%$ (471 áudios)**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Gráfico: Desempenho Isolado vs Teto Teórico do Oráculo na CNN em SMALL\n",
                "plt.figure(figsize=(10, 5.5))\n",
                "colors = ['#95a5a6', '#9b59b6', '#2980b9', '#e67e22', '#27ae60']\n",
                "bars = plt.bar(df_hits['Representacao'], df_hits['Recall_Pct'], color=colors, width=0.55, edgecolor='black', linewidth=0.8)\n",
                "bars[-1].set_hatch('//')\n",
                "\n",
                "plt.axhline(73.54, color='#27ae60', linestyle='--', lw=1.5, alpha=0.7, label='Teto do Oráculo CNN (73.54%)')\n",
                "plt.axhline(47.02, color='#e67e22', linestyle=':', lw=1.5, alpha=0.7, label='Melhor Isolado LOFAR (47.02%)')\n",
                "\n",
                "plt.title('Recall CNN 2D na Classe SMALL: Modelos Isolados vs. Teto Teórico do Oráculo', fontweight='bold', pad=12)\n",
                "plt.ylabel('Recall em SMALL (%)')\n",
                "plt.ylim(0, 95)\n",
                "\n",
                "for bar in bars:\n",
                "    h = bar.get_height()\n",
                "    plt.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),\n",
                "                 xytext=(0, 4), textcoords=\"offset points\", ha='center', va='bottom',\n",
                "                 fontsize=10.5, fontweight='bold', color='#111')\n",
                "\n",
                "plt.legend(loc='upper left', frameon=True)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Análise de Grau de Consenso e Risco do Efeito Reverso em SMALL (CNN)\n",
                "\n",
                "### 4.1 Pirâmide de Consenso na CNN 2D\n",
                "- **Unanimidade (4/4 certas): $23$ áudios ($1{,}29\\\\%$)** — extrema fragilidade transversal;\n",
                "- **Alto Consenso (3/4 certas): $160$ áudios ($8{,}99\\\\%$)**;\n",
                "- **Consenso Dividido (2/4 certas): $505$ áudios ($28{,}37\\\\%$)**;\n",
                "- **Resgate Exclusivo (1/4 certa): $621$ áudios ($34{,}89\\\\%$)** — quase metade dos acertos totais dependem de apenas 1 modelo!\n",
                "- **Erro Comum (0/4 certas): $471$ áudios ($26{,}46\\\\%$)**.\n",
                "\n",
                "### 4.2 Votação por Consenso em SMALL (CNN):\n",
                "- **Unanimidade (4/4):** $1{,}29\\\\%$ ($23$ áudios);\n",
                "- **Maioria Estrita ($\\\\ge 3/4$):** $10{,}28\\\\%$ ($183$ áudios);\n",
                "- **Maioria Simples / Empate ($\\\\ge 2/4$):** $38{,}65\\\\%$ ($688$ áudios);\n",
                "- **Oráculo ($\\\\ge 1/4$):** **$73{,}54\\\\%$ ($1.309$ áudios)**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Visualização do Grau de Consenso e do Efeito Reverso em SMALL (CNN)\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "# Painel 1: Pirâmide de Consenso dos 1.780 Áudios\n",
                "labels_cons = ['4 Entradas\\n(Unânime)', '3 Entradas\\n(Alto Consenso)', '2 Entradas\\n(Dividido)', '1 Entrada\\n(Exclusivo/Frágil)', '0 Entradas\\n(Erro Comum)']\n",
                "counts_cons = [23, 160, 505, 621, 471]\n",
                "pcts_cons = [1.29, 8.99, 28.37, 34.89, 26.46]\n",
                "colors_cons = ['#27ae60', '#2980b9', '#f39c12', '#e74c3c', '#7f8c8d']\n",
                "\n",
                "bars1 = ax1.bar(labels_cons, counts_cons, color=colors_cons, edgecolor='black', linewidth=0.8, width=0.55)\n",
                "ax1.set_title('Grau de Consenso CNN em SMALL (Total: 1.780 Áudios no 5x2cv)', fontweight='bold', pad=12)\n",
                "ax1.set_ylabel('Quantidade de Áudios de SMALL')\n",
                "ax1.set_ylim(0, 720)\n",
                "\n",
                "for bar, pct in zip(bars1, pcts_cons):\n",
                "    h = bar.get_height()\n",
                "    ax1.annotate(f'{h} ({pct:.1f}%)', xy=(bar.get_x() + bar.get_width()/2, h),\n",
                "                 xytext=(0, 4), textcoords=\"offset points\", ha='center', va='bottom',\n",
                "                 fontsize=9.5, fontweight='bold')\n",
                "\n",
                "# Painel 2: Simulação do Efeito Reverso a partir do LOFAR (837 Acertos na CNN)\n",
                "destinos = ['Para MEL', 'Para HHT', 'Para DEMON']\n",
                "perdas = [406, 638, 687]\n",
                "resgates = [297, 166, 123]\n",
                "x = np.arange(len(destinos))\n",
                "w = 0.35\n",
                "\n",
                "r1 = ax2.bar(x - w/2, perdas, w, label='Perde Acertos do LOFAR (Efeito Reverso)', color='#c0392b', alpha=0.9, edgecolor='black')\n",
                "r2 = ax2.bar(x + w/2, resgates, w, label='Resgata dos Erros do LOFAR (Ganhos)', color='#27ae60', alpha=0.9, edgecolor='black')\n",
                "\n",
                "ax2.set_title('Balanço do Efeito Reverso na CNN ao Trocar de Entrada a partir do LOFAR', fontweight='bold', pad=12)\n",
                "ax2.set_ylabel('Quantidade de Áudios de SMALL')\n",
                "ax2.set_xticks(x)\n",
                "ax2.set_xticklabels(destinos, fontweight='bold', fontsize=11)\n",
                "ax2.set_ylim(0, 780)\n",
                "ax2.legend(loc='upper right', frameon=True)\n",
                "\n",
                "for rect in r1:\n",
                "    h = rect.get_height()\n",
                "    ax2.annotate(f'-{h}', xy=(rect.get_x() + rect.get_width()/2, h),\n",
                "                 xytext=(0, 3), textcoords=\"offset points\", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#900')\n",
                "\n",
                "for rect in r2:\n",
                "    h = rect.get_height()\n",
                "    ax2.annotate(f'+{h}', xy=(rect.get_x() + rect.get_width()/2, h),\n",
                "                 xytext=(0, 3), textcoords=\"offset points\", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#006400')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Resgates Exclusivos na CNN 2D em SMALL: Domínio do LOFAR e MEL\n",
                "\n",
                "Em **$621$ áudios de `SMALL` ($34{,}89\\\\%$)**, três entradas falharam e apenas um modelo de CNN acertou a lancha:\n",
                "\n",
                "1. **LOFAR Isolado (254 lanchas salvas exclusivamente — $14{,}27\\\\%$):**\n",
                "   - É o **campeão de resgates exclusivos na CNN em `SMALL`**!\n",
                "   - Filtros convolucionais no LOFAR detectam linhas espectrais contínuas ao longo do tempo que o DEMON e o HHT dispersam.\n",
                "2. **MEL Isolado (211 lanchas salvas exclusivamente — $11{,}85\\\\%$):**\n",
                "   - O filtro convolucional 2D capta bandas de energia em alta frequência da descarga subaquática.\n",
                "3. **HHT Puro (84 lanchas salvas exclusivamente — $4{,}72\\\\%$):**\n",
                "   - Decomposição instantânea de aceleração.\n",
                "4. **DEMON Isolado (72 lanchas salvas exclusivamente — $4{,}04\\\\%$):**\n",
                "   - Modulação de rotação de pás."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Gráfico: Resgates Exclusivos na Classe SMALL (CNN)\n",
                "plt.figure(figsize=(9, 4.5))\n",
                "colors_excl = ['#16a085', '#9b59b6', '#2980b9', '#e67e22']\n",
                "bars_excl = plt.barh(df_exclusive['Representacao'], df_exclusive['Audios_Salvos'], color=colors_excl, height=0.5, edgecolor='black', linewidth=0.8)\n",
                "\n",
                "plt.title('Resgates Exclusivos CNN 2D na Classe SMALL (Total: 621 Áudios Únicos)', fontweight='bold', pad=12)\n",
                "plt.xlabel('Número de Áudios Salvos Exclusivamente')\n",
                "plt.xlim(0, 310)\n",
                "\n",
                "for idx, (bar, pct) in enumerate(zip(bars_excl, df_exclusive['Percentual'])):\n",
                "    w = bar.get_width()\n",
                "    tag = \"  <-- CAMPEÃO DE RESGATES EM SMALL\" if idx == 3 else \"\"\n",
                "    plt.annotate(f'{w} áudios ({pct:.1f}%){tag}', xy=(w, bar.get_y() + bar.get_height()/2),\n",
                "                 xytext=(5, 0), textcoords=\"offset points\", ha='left', va='center',\n",
                "                 fontsize=9.5, fontweight='bold', color='#111' if idx != 3 else '#b03a2e')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Padrões de Erro em SMALL na CNN 2D\n",
                "\n",
                "| Entrada | Principal Erro | Porcentagem do Erro | Segundo Maior Erro | Diagnóstico Mecânico |\n",
                "| :--- | :---: | :---: | :---: | :--- |\n",
                "| **MEL** | **`MEDIUM`** | **$57{,}79\\\\%$** | `LARGE` ($26{,}43\\\\%$) | Convoluções temporais suavizam picos transitórios rápidos em padrões estáveis de rebocadores. |\n",
                "| **LOFAR** | **`MEDIUM`** | **$51{,}43\\\\%$** | `LARGE` ($37{,}12\\\\%$) | Raias tônicas espúrias de marulho são interpretadas como maquinário pesado. |\n",
                "| **HHT** | **`LARGE`** | **$37{,}74\\\\%$** | `MEDIUM` ($34{,}28\\\\%$) | Dispersão nas primeiras IMFs é interpretada como cavitação massiva de cargueiro. |\n",
                "| **DEMON** | **`MEDIUM`** | **$50{,}56\\\\%$** | `LARGE` ($31{,}06\\\\%$) | Na ausência de modulação limpa, a imagem 2D demodulada cai na classe intermediária. |"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Gráfico: Destino do Erro em SMALL por Representação (CNN)\n",
                "plt.figure(figsize=(10, 5.5))\n",
                "reps = df_confusion_pattern['Representacao']\n",
                "med_err = df_confusion_pattern['Confundido_MEDIUM_Pct']\n",
                "lrg_err = df_confusion_pattern['Confundido_LARGE_Pct']\n",
                "amb_err = df_confusion_pattern['Confundido_AMBIENTAL_Pct']\n",
                "\n",
                "w = 0.52\n",
                "p1 = plt.bar(reps, med_err, w, label='Confundido com MEDIUM (Rebocadores)', color='#f39c12', alpha=0.9, edgecolor='black', linewidth=0.6)\n",
                "p2 = plt.bar(reps, lrg_err, w, bottom=med_err, label='Confundido com LARGE (Cargueiros)', color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=0.6)\n",
                "p3 = plt.bar(reps, amb_err, w, bottom=np.array(med_err)+np.array(lrg_err), label='Confundido com AMBIENTAL (Ruído)', color='#95a5a6', alpha=0.9, edgecolor='black', linewidth=0.6)\n",
                "\n",
                "plt.title('Padrão de Erro CNN 2D na Classe SMALL: Para onde cada entrada envia a lancha? (%)', fontweight='bold', pad=12)\n",
                "plt.ylabel('Distribuição Percentual dos Erros (%)')\n",
                "plt.ylim(0, 118)\n",
                "plt.legend(loc='upper right', frameon=True)\n",
                "\n",
                "for i in range(len(reps)):\n",
                "    plt.text(i, med_err[i]/2, f'{med_err[i]:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')\n",
                "    plt.text(i, med_err[i] + lrg_err[i]/2, f'{lrg_err[i]:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')\n",
                "    if amb_err[i] > 3:\n",
                "        plt.text(i, med_err[i] + lrg_err[i] + amb_err[i]/2, f'{amb_err[i]:.1f}%', ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Painel Consolidado de Descoberta Científica (CNN 2D)\n",
                "\n",
                "Abaixo exibimos o painel visual oficial de 4 quadrantes para a classe `SMALL` com o modelo CNN 2D:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. Exibição do Gráfico Oficial Consolidado salvo em disco\n",
                "from IPython.display import Image, display\n",
                "\n",
                "img_path = 'reports/analise_complementaridade_cnn_small_oraculo.png'\n",
                "if os.path.exists(img_path):\n",
                "    display(Image(filename=img_path))\n",
                "else:\n",
                "    print(f\"Imagem {img_path} não encontrada. Execute 'scratch/plot_cnn_figures.py' para gerá-la.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Comparativo Quadrilateral em SMALL: SVM vs. MLP vs. RF vs. CNN 2D\n",
                "\n",
                "| Dimensão Analítica | SVM Nyström (NB 10) | MLP PyTorch (NB 14) | Random Forest (NB 17) | CNN 2D (NB 20) | Diagnóstico Científico |\n",
                "| :--- | :---: | :---: | :---: | :---: | :--- |\n",
                "| **Melhor Modelo Isolado** | $50{,}73\\\\%$ (MEL) | $46{,}35\\\\%$ (LOFAR) | **$58{,}15\\\\%$ (MEL)** | $47{,}02\\\\%$ (LOFAR) | O RF lidera com ampla vantagem em lanchas isoladas. |\n",
                "| **Teto do Oráculo** | $83{,}99\\\\%$ ($1.495$) | $76{,}85\\\\%$ ($1.368$) | **$86{,}52\\\\%$ ($1.540$)** | $73{,}54\\\\%$ ($1.309$) | A CNN apresenta o menor teto em SMALL devido à rigidez de filtros 2D. |\n",
                "| **Taxa de Unanimidade (4/4)** | $6{,}80\\\\%$ ($121$) | $5{,}34\\\\%$ ($95$) | **$16{,}80\\\\%$ ($299$)** | **$1{,}29\\\\%$ ($23$)** | Na CNN, o consenso é quase nulo: quase todas as lanchas dependem de resgates pontuais. |\n",
                "| **Erro Comum Incorrigível** | $16{,}01\\\\%$ ($285$) | $23{,}15\\\\%$ ($412$) | **$13{,}48\\\\%$ ($240$)** | $26{,}46\\\\%$ ($471$) | Mais de um quarto das lanchas é perdido na CNN. |\n",
                "| **Campeão de Resgates Exclusivos** | HHT ($207$) | DEMON ($155$) | DEMON ($178$) | **LOFAR ($254$)** | Na CNN, LOFAR e MEL salvam $465$ lanchas exclusivas somados. |\n",
                "| **Votação Simples ($\\\\ge 2/4$)** | $55{,}17\\\\%$ | $41{,}74\\\\%$ | **$65{,}62\\\\%$** | $38{,}65\\\\%$ | Votação majoritária falha na CNN em SMALL devido à baixa unanimidade. |"
            ]
        }
    ]

    nb_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=2, ensure_ascii=False)
    print(f"Notebook gerado com sucesso em '{filename}'!")

if __name__ == "__main__":
    create_notebook()
