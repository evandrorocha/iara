"""
Script de geração do notebook 22: notebooks/22_analise_complementaridade_cnn_large.ipynb
Análise de Ortogonalidade, Efeito Oráculo e Grau de Consenso na CNN 2D para a Classe LARGE (Dataset IARA).
"""

import os
import json

def create_notebook(filename="notebooks/22_analise_complementaridade_cnn_large.ipynb"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA\n",
                "### Notebook 22: Análise de Ortogonalidade, Efeito Oráculo e Grau de Consenso na CNN 2D — Classe LARGE\n",
                "\n",
                "**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)\n",
                "**Data:** Outubro de 2026\n",
                "\n",
                "---\n",
                "\n",
                "## Resumo Executivo da Investigação com Redes Convolucionais 2D (CNN)\n",
                "\n",
                "A classe **`LARGE`** (embarcações de grande porte: navios mercantes, graneleiros, petroleiros e porta-contêineres, $> 150\\text{ m}$ até mais de $300\\text{ m}$) foi avaliada em todas as **2.140 instâncias de teste** nos 10 folds oficiais do protocolo $5\\times2$cv com a arquitetura **CNN 2D PyTorch (GPU RTX 5060 Ti, blocos $32 \\times D$)** nas 4 representações (**MEL, LOFAR, HHT e DEMON**).\n",
                "\n",
                "O objetivo é confrontar os resultados de SVM (Notebook 13), MLP (Notebook 16) e Random Forest (Notebook 19) com a modelagem convolucional profunda em cargueiros de grande porte.\n",
                "\n",
                "### Principais Descobertas na CNN 2D:\n",
                "1. **Teto Teórico do Oráculo CNN Atinge $89{,}44\\%$ (1.914 / 2.140 cargueiros):**\n",
                "   - A combinação das 4 entradas na CNN resgata **1.914 navios de grande porte**, proporcionando um ganho de **$+24{,}86$ pontos percentuais** sobre o melhor modelo isolado (MEL com $64{,}58\\%$).\n",
                "   - O erro comum irrecuperável é de apenas **$10{,}56\\%$ (226 áudios)** — virtualmente idêntico ao recorde do Random Forest ($10{,}23\\%$).\n",
                "2. **A Unanimidade na CNN se Mantém Baixa ($7{,}34\\%$ — 157 navios):**\n",
                "   - Diferente dos classificadores rasos onde a unanimidade em cargueiros atingia $40\\%$ a $45\\%$, na CNN 2D a unanimidade é de apenas **$7{,}34\\%$**. As convoluções espaciais capturam características tão idiossincráticas de cada domínio espectral que as predições tornam-se altamente ortogonais.\n",
                "3. **A Votação por Maioria Simples (\\ge 2/4) Supera o Melhor Modelo Isolado:**\n",
                "   - O consenso majoritário simples (\\ge 2/4) atinge **$65{,}33\\%$ (1.398 navios)**, superando o MEL isolado ($64{,}58\\%$).\n",
                "4. **Poderoso Equilíbrio nos Resgates Exclusivos:**\n",
                "   - O MEL resgata sozinho **187 navios ($8{,}74\\%$)**, o DEMON salva **138 ($6{,}45\\%$)** e o HHT salva **123 ($5{,}75\\%$)**. As 4 entradas contribuem ativamente com resgates únicos para compor o teto de $89{,}44\\%$."
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
                "Avaliação sobre todas as **2.140 instâncias de teste** onde a classe real comprovada era **`LARGE` (Classe 2)** nos 10 folds do protocolo oficial $5\\times2$cv com a arquitetura **CNN 2D PyTorch** nas 4 representações (**MEL, LOFAR, HHT e DEMON**):\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Dados Consolidados das Predições da CNN 2D na Classe LARGE\n",
                "total_lrg = 2140\n",
                "\n",
                "# Desempenho individual na CNN\n",
                "hits_data = {\n",
                "    'Representacao': ['LOFAR (1024-D)', 'HHT Puro (256-D)', 'DEMON (253-D)', 'MEL (256-D)', 'ORÁCULO CNN (União 4 Modos)'],\n",
                "    'Acertos': [583, 1089, 1125, 1382, 1914],\n",
                "    'Recall_Pct': [27.24, 50.89, 52.57, 64.58, 89.44]\n",
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
                "    'Audios': [157, 553, 688, 516, 226],\n",
                "    'Pct_Total_Large': [7.34, 25.84, 32.15, 24.11, 10.56],\n",
                "    'Pct_Dentro_Oraculo': [8.20, 28.89, 35.95, 26.96, np.nan]\n",
                "}\n",
                "df_consensus = pd.DataFrame(consensus_data)\n",
                "\n",
                "# Resgates exclusivos por representação na CNN\n",
                "exclusive_data = {\n",
                "    'Representacao': ['LOFAR Exclusivo', 'HHT Exclusivo', 'DEMON Exclusivo', 'MEL Exclusivo'],\n",
                "    'Audios_Salvos': [68, 123, 138, 187],\n",
                "    'Percentual': [3.18, 5.75, 6.45, 8.74]\n",
                "}\n",
                "df_exclusive = pd.DataFrame(exclusive_data)\n",
                "\n",
                "# Efeito Reverso em Relação ao Baseline MEL (1382 acertos, 758 erros)\n",
                "reverse_data = {\n",
                "    'Destino_Troca': ['Para LOFAR', 'Para HHT', 'Para DEMON'],\n",
                "    'Mantem_do_MEL': [404, 790, 828],\n",
                "    'Perde_do_MEL (Efeito Reverso)': [978, 592, 554],\n",
                "    'Pct_Perda_Acertos (%)': [70.77, 42.84, 40.09],\n",
                "    'Resgata_Erros_MEL': [179, 299, 297],\n",
                "    'Pct_Resgate_Erros (%)': [23.61, 39.45, 39.18],\n",
                "    'Saldo_Liquido_Audios': [-799, -293, -257]\n",
                "}\n",
                "df_reverse = pd.DataFrame(reverse_data)\n",
                "\n",
                "# Padrão de confusão quando erra em LARGE na CNN\n",
                "confusion_pattern_data = {\n",
                "    'Representacao': ['MEL (758 erros)', 'DEMON (1015 erros)', 'HHT (1051 erros)', 'LOFAR (1557 erros)'],\n",
                "    'Confundido_MEDIUM_Pct': [54.62, 57.04, 49.19, 39.56],\n",
                "    'Confundido_SMALL_Pct': [37.07, 24.83, 26.26, 46.05],\n",
                "    'Confundido_AMBIENTAL_Pct': [8.31, 18.13, 24.55, 14.39]\n",
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
                "A CNN 2D atinge **$64{,}58\\%$ de recall em `LARGE`** com o MEL e **$52{,}57\\%$** com o DEMON.\n",
                "\n",
                "Contudo, a combinação das 4 entradas via **Oráculo CNN** salta para **$89{,}44\\%$ (1.914 acertos)**:\n",
                "- Um ganho extraordinário de **$+24{,}86$ pontos percentuais** sobre o MEL isolado;\n",
                "- O erro comum (onde as 4 CNNs erram simultaneamente) é de apenas **$10{,}56\\%$ (226 áudios)**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Gráfico: Desempenho Isolado vs Teto Teórico do Oráculo na CNN em LARGE\n",
                "plt.figure(figsize=(10, 5.5))\n",
                "colors = ['#e67e22', '#9b59b6', '#95a5a6', '#2980b9', '#27ae60']\n",
                "bars = plt.bar(df_hits['Representacao'], df_hits['Recall_Pct'], color=colors, width=0.55, edgecolor='black', linewidth=0.8)\n",
                "bars[-1].set_hatch('//')\n",
                "\n",
                "plt.axhline(89.44, color='#27ae60', linestyle='--', lw=1.5, alpha=0.7, label='Teto do Oráculo CNN (89.44%)')\n",
                "plt.axhline(64.58, color='#2980b9', linestyle=':', lw=1.5, alpha=0.7, label='Melhor Isolado MEL (64.58%)')\n",
                "\n",
                "plt.title('Recall CNN 2D na Classe LARGE: Modelos Isolados vs. Teto Teórico do Oráculo', fontweight='bold', pad=12)\n",
                "plt.ylabel('Recall em LARGE (%)')\n",
                "plt.ylim(0, 100)\n",
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
                "## 4. Análise de Grau de Consenso em LARGE na CNN 2D\n",
                "\n",
                "A pirâmide de consenso da CNN 2D em `LARGE`:\n",
                "- **Unanimidade (4/4 certas): $157$ áudios ($7{,}34\\%$)**;\n",
                "- **Alto Consenso (3/4 certas): $553$ áudios ($25{,}84\\%$)**;\n",
                "- **Consenso Dividido (2/4 certas): $688$ áudios ($32{,}15\\%$)**;\n",
                "- **Resgate Exclusivo (1/4 certa): $516$ áudios ($24{,}11\\%$)**;\n",
                "- **Erro Comum (0/4 certas): $226$ áudios ($10{,}56\\%$)**.\n",
                "\n",
                "### Votação por Consenso em LARGE (CNN 2D):\n",
                "- **Unanimidade (4/4):** $7{,}34\\%$ ($157$ áudios);\n",
                "- **Maioria Estrita (\\ge 3/4):** $33{,}18\\%$ ($710$ áudios);\n",
                "- **Maioria Simples / Empate (\\ge 2/4):** $65{,}33\\%$ ($1.398$ áudios) — **Supera o MEL isolado ($64{,}58\\%$)**;\n",
                "- **Oráculo (\\ge 1/4):** **$89{,}44\\%$ ($1.914$ áudios)**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Visualização do Grau de Consenso na CNN\n",
                "plt.figure(figsize=(9, 5))\n",
                "labels_cons = ['4 Entradas\\n(Unânime)', '3 Entradas\\n(Alto Consenso)', '2 Entradas\\n(Dividido)', '1 Entrada\\n(Exclusivo/Frágil)', '0 Entradas\\n(Erro Comum)']\n",
                "counts_cons = [157, 553, 688, 516, 226]\n",
                "pcts_cons = [7.34, 25.84, 32.15, 24.11, 10.56]\n",
                "colors_cons = ['#27ae60', '#2980b9', '#f39c12', '#e74c3c', '#7f8c8d']\n",
                "\n",
                "bars = plt.bar(labels_cons, counts_cons, color=colors_cons, edgecolor='black', linewidth=0.8, width=0.55)\n",
                "plt.title('Grau de Consenso CNN 2D em LARGE (Total: 2.140 Áudios no 5x2cv)', fontweight='bold', pad=12)\n",
                "plt.ylabel('Quantidade de Áudios de LARGE')\n",
                "plt.ylim(0, 800)\n",
                "\n",
                "for bar, pct in zip(bars, pcts_cons):\n",
                "    h = bar.get_height()\n",
                "    plt.annotate(f'{h} ({pct:.1f}%)', xy=(bar.get_x() + bar.get_width()/2, h),\n",
                "                 xytext=(0, 4), textcoords=\"offset points\", ha='center', va='bottom',\n",
                "                 fontsize=9.5, fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Resgates Exclusivos e Padrão de Erro na CNN 2D\n",
                "\n",
                "### Resgates Exclusivos em LARGE:\n",
                "- **MEL:** Salva sozinho **$187$ navios ($8{,}74\\%$)**;\n",
                "- **DEMON:** Salva sozinho **$138$ navios ($6{,}45\\%$)**;\n",
                "- **HHT:** Salva sozinho **$123$ navios ($5{,}75\\%$)**;\n",
                "- **LOFAR:** Salva sozinho **$68$ navios ($3{,}18\\%$)**.\n",
                "\n",
                "### Padrões de Confusão em LARGE na CNN:\n",
                "- **MEL, DEMON e HHT:** Confundem **predominantemente com `MEDIUM` ($49\\%$ a $57\\%$)**;\n",
                "- **LOFAR:** Confunde cargueiros **mais com `SMALL` ($46{,}05\\%$)** do que com `MEDIUM` ($39{,}56\\%$)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Gráfico de Resgates Exclusivos e Padrão de Confusão\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))\n",
                "\n",
                "# Resgates Exclusivos\n",
                "colors_exc = ['#e67e22', '#9b59b6', '#95a5a6', '#2980b9']\n",
                "bars_exc = ax1.bar(df_exclusive['Representacao'], df_exclusive['Audios_Salvos'], color=colors_exc, edgecolor='black', linewidth=0.8, width=0.55)\n",
                "ax1.set_title('Resgates Exclusivos por Entrada na CNN (LARGE)', fontweight='bold', pad=12)\n",
                "ax1.set_ylabel('Áudios Salvos Exclusivamente')\n",
                "ax1.set_ylim(0, 220)\n",
                "\n",
                "for bar, pct in zip(bars_exc, df_exclusive['Percentual']):\n",
                "    h = bar.get_height()\n",
                "    ax1.annotate(f'{h} ({pct:.1f}%)', xy=(bar.get_x() + bar.get_width()/2, h),\n",
                "                 xytext=(0, 4), textcoords=\"offset points\", ha='center', va='bottom',\n",
                "                 fontsize=9.5, fontweight='bold')\n",
                "\n",
                "# Padrão de Confusão (Stacked Bar)\n",
                "reps = ['MEL', 'DEMON', 'HHT', 'LOFAR']\n",
                "med_errs = [54.62, 57.04, 49.19, 39.56]\n",
                "sml_errs = [37.07, 24.83, 26.26, 46.05]\n",
                "amb_errs = [8.31, 18.13, 24.55, 14.39]\n",
                "\n",
                "x = np.arange(len(reps))\n",
                "width = 0.5\n",
                "\n",
                "ax2.bar(x, med_errs, width, label='Confundido com MEDIUM', color='#f39c12', edgecolor='black')\n",
                "ax2.bar(x, sml_errs, width, bottom=med_errs, label='Confundido com SMALL', color='#e74c3c', edgecolor='black')\n",
                "ax2.bar(x, amb_errs, width, bottom=np.array(med_errs) + np.array(sml_errs), label='Confundido com AMBIENTAL', color='#2ecc71', edgecolor='black')\n",
                "\n",
                "ax2.set_xticks(x)\n",
                "ax2.set_xticklabels(reps)\n",
                "ax2.set_title('Distribuição de Erros ao Classificar LARGE na CNN', fontweight='bold', pad=12)\n",
                "ax2.set_ylabel('Proporção de Erros (%)')\n",
                "ax2.set_ylim(0, 105)\n",
                "ax2.legend(loc='lower right', frameon=True)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Efeito Reverso em LARGE: O Risco de Chaveamento Baseado em Entrada\n",
                "\n",
                "Tomando o **MEL** como baseline ($1.382$ acertos e $758$ erros):\n",
                "- **Chavear para LOFAR:** Perde **$978$ cargueiros certos ($70{,}77\\%$)** e resgata apenas $179$ erros ($23{,}61\\%$), com saldo de **$-799$**;\n",
                "- **Chavear para HHT:** Perde **$592$ cargueiros certos ($42{,}84\\%$)** e resgata $299$ erros ($39{,}45\\%$), com saldo de **$-293$**;\n",
                "- **Chavear para DEMON:** Perde **$554$ cargueiros certos ($40{,}09\\%$)** e resgata $297$ erros ($39{,}18\\%$), com saldo de **$-257$**.\n",
                "\n",
                "O chaveamento cego é altamente destrutivo mesmo em cargueiros de grande porte, enfatizando a necessidade de abordagens integradas via votação ou fusão em espaço latente."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Visualização do Efeito Reverso na CNN em LARGE\n",
                "fig, ax = plt.subplots(figsize=(9, 5))\n",
                "\n",
                "x = np.arange(len(df_reverse['Destino_Troca']))\n",
                "width = 0.35\n",
                "\n",
                "bars1 = ax.bar(x - width/2, df_reverse['Perde_do_MEL (Efeito Reverso)'], width, label='Perde do MEL (Efeito Reverso)', color='#c0392b', edgecolor='black')\n",
                "bars2 = ax.bar(x + width/2, df_reverse['Resgata_Erros_MEL'], width, label='Resgata dos Erros do MEL', color='#27ae60', edgecolor='black')\n",
                "\n",
                "ax.set_title('Efeito Reverso do Chaveamento na CNN a partir do Baseline MEL (LARGE)', fontweight='bold', pad=12)\n",
                "ax.set_xticks(x)\n",
                "ax.set_xticklabels(df_reverse['Destino_Troca'])\n",
                "ax.set_ylabel('Quantidade de Áudios')\n",
                "ax.set_ylim(0, 1100)\n",
                "\n",
                "for bar in bars1:\n",
                "    h = bar.get_height()\n",
                "    ax.annotate(f'-{h}', xy=(bar.get_x() + bar.get_width()/2, h),\n",
                "                xytext=(0, 4), textcoords=\"offset points\", ha='center', va='bottom',\n",
                "                fontsize=10, fontweight='bold', color='#c0392b')\n",
                "\n",
                "for bar in bars2:\n",
                "    h = bar.get_height()\n",
                "    ax.annotate(f'+{h}', xy=(bar.get_x() + bar.get_width()/2, h),\n",
                "                xytext=(0, 4), textcoords=\"offset points\", ha='center', va='bottom',\n",
                "                fontsize=10, fontweight='bold', color='#27ae60')\n",
                "\n",
                "ax.legend(frameon=True)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Figura Consolidada Oficial e Painel Visual (CNN 2D)\n",
                "\n",
                "Abaixo, carregamos o painel consolidado oficial de 4 gráficos gerado para a classe `LARGE` na CNN 2D:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. Exibição da Figura Oficial do Relatório\n",
                "from IPython.display import Image, display\n",
                "\n",
                "img_path = 'reports/analise_complementaridade_cnn_large_oraculo.png'\n",
                "if not os.path.exists(img_path):\n",
                "    img_path = '../' + img_path\n",
                "if os.path.exists(img_path):\n",
                "    display(Image(filename=img_path, width=950))\n",
                "else:\n",
                "    print(f\"Imagem oficial '{img_path}' ainda não encontrada no caminho relativo.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Comparativo Quadrilateral em LARGE: SVM vs. MLP vs. RF vs. CNN 2D\n",
                "\n",
                "| Dimensão Analítica | SVM Nyström (NB 13) | MLP PyTorch (NB 16) | Random Forest (NB 19) | CNN 2D (NB 22) | Diagnóstico Científico |\n",
                "| :--- | :---: | :---: | :---: | :---: | :--- |\n",
                "| **Melhor Modelo Isolado** | $75{,}56\\%$ (LOFAR) | $72{,}15\\%$ (MEL) | **$75{,}65\\%$ (LOFAR)** | $64{,}58\\%$ (MEL) | Modelos tabulares com LOFAR exploram melhor as raias finas contínuas de cargueiros. |\n",
                "| **Teto do Oráculo** | $88{,}18\\%$ ($1.887$) | $88{,}83\\%$ ($1.901$) | **$89{,}77\\%$ ($1.921$)** | $89{,}44\\%$ ($1.914$) | Todas as 4 famílias convergem para um teto histórico comum próximo a $89\\% - 90\\%$. |\n",
                "| **Taxa de Unanimidade (4/4)** | $43{,}93\\%$ ($940$) | $41{,}31\\%$ ($884$) | **$44{,}91\\%$ ($961$)** | **$7{,}34\\%$ ($157$)** | Na CNN, a especialização dos filtros reduz a redundância unânime para $7{,}34\\%$. |\n",
                "| **Erro Comum Incorrigível** | $11{,}82\\%$ ($253$) | $11{,}17\\%$ ($239$) | **$10{,}23\\%$ ($219$)** | $10{,}56\\%$ ($226$) | Menor perda de todo o estudo (~$10\\%$) em todas as arquiteturas. |\n",
                "| **Campeão de Resgates Exclusivos** | DEMON ($117$) | DEMON ($126$) | DEMON ($130$) | **MEL ($187$)** | Na CNN, MEL e DEMON somam juntos $325$ resgates exclusivos em cargueiros. |\n",
                "| **Votação Simples (\\ge 2/4)** | $78{,}08\\%$ | $75{,}98\\%$ | **$78{,}27\\%$** | $65{,}33\\%$ | Na CNN, a maioria simples supera o melhor modelo isolado ($65{,}33\\%$ vs. $64{,}58\\%$). |"
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
