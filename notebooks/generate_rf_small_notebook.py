"""
Script de geração do notebook 17: notebooks/17_analise_complementaridade_rf_small.ipynb
Análise de Ortogonalidade, Efeito Oráculo e Grau de Consenso no Random Forest para a Classe SMALL (Dataset IARA).
"""

import os
import json

def create_notebook(filename="notebooks/17_analise_complementaridade_rf_small.ipynb"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA\n",
                "### Notebook 17: Análise de Ortogonalidade, Efeito Oráculo e Grau de Consenso no Random Forest — Classe SMALL\n",
                "\n",
                "**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)\n",
                "**Data:** Outubro de 2026\n",
                "\n",
                "---\n",
                "\n",
                "## Resumo Executivo da Investigação com Random Forest (RF)\n",
                "\n",
                "A classe **`SMALL`** (embarcações de pequeno porte: lanchas rápidas e veleiros a motor com comprimento $\\\\le 15\\\\text{ m}$) foi avaliada em todas as **1.780 instâncias de teste** nos 10 folds oficiais do protocolo $5\\\\times2$cv com a arquitetura **Random Forest (100 árvores com balanceamento de classes)** nas 4 representações (**MEL, LOFAR, HHT e DEMON**).\n",
                "\n",
                "O objetivo é confrontar os resultados do classificador SVM (Notebook 10) e do MLP (Notebook 14) com os do ensemble de árvores de decisão para investigar como uma abordagem baseada em particionamento ortogonal de espaço lida com a fragilidade acústica de embarcações leves.\n",
                "\n",
                "### Principais Descobertas no Random Forest:\n",
                "1. **O Maior Teto de Oráculo em SMALL de Todo o Estudo ($86{,}52\\\\%$ — 1.540 / 1.780 áudios):**\n",
                "   - O Oráculo do Random Forest em `SMALL` supera tanto o SVM ($83{,}99\\\\%$) quanto o MLP ($76{,}85\\\\%$).\n",
                "   - O erro comum irrecuperável cai para apenas **$13{,}48\\\\%$ (240 áudios)**, contra $16\\\\%$ no SVM e $23\\\\%$ no MLP.\n",
                "2. **Aumento Expressivo na Taxa de Unanimidade ($16{,}80\\\\%$ — 299 áudios):**\n",
                "   - No RF, a taxa de consenso 4/4 em `SMALL` salta para **$16{,}80\\\\%$**, mais que o dobro do SVM ($6{,}80\\\\%$) e o triplo do MLP ($5{,}34\\\\%$), demonstrando que ensembles de árvores encontram limiares de decisão mais estáveis no marulho.\n",
                "3. **DEMON Isolado Lidera com Folga os Resgates Exclusivos ($178$ lanchas — $10{,}00\\\\%$):**\n",
                "   - Sozinho, o DEMON recupera **178 lanchas** que MEL, LOFAR e HHT erraram simultaneamente. Em árvores de decisão, a cadência rápida de pás de motores de popa ($> 15\\\\text{ Hz}$) cria partições binárias muito limpas.\n",
                "4. **A Votação por Maioria Simples ($\\\\ge 2/4$) Supera com Força o Melhor Modelo Isolado:**\n",
                "   - Um ensemble cego por maioria simples ($\\\\ge 2/4$) atinge **$65{,}62\\\\%$ ($1.168$ áudios)**, superando o melhor modelo isolado (MEL com $58{,}15\\\\%$) em expressivos **$+7{,}47$ pontos percentuais**."
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
                "Avaliação rigorosa sobre todas as **1.780 instâncias de teste** onde a classe real comprovada era **`SMALL` (Classe 0)** sob o protocolo oficial $5\\\\times2$cv nos 10 folds com **Random Forest (100 árvores, class_weight='balanced')** nas 4 representações (**MEL, LOFAR, HHT e DEMON**):\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Dados Consolidados das Predições do Random Forest na Classe SMALL\n",
                "total_sml = 1780\n",
                "\n",
                "# Desempenho individual no RF\n",
                "hits_data = {\n",
                "    'Representacao': ['LOFAR (1024-D)', 'HHT Puro (256-D)', 'DEMON (253-D)', 'MEL (256-D)', 'ORÁCULO RF (União 4 Modos)'],\n",
                "    'Acertos': [866, 929, 951, 1035, 1540],\n",
                "    'Recall_Pct': [48.65, 52.19, 53.43, 58.15, 86.52]\n",
                "}\n",
                "df_hits = pd.DataFrame(hits_data)\n",
                "\n",
                "# Grau de consenso no RF\n",
                "consensus_data = {\n",
                "    'Nivel_Consenso': [\n",
                "        '4 de 4 (Unanimidade / Independente)',\n",
                "        '3 de 4 (Alto Consenso)',\n",
                "        '2 de 4 (Consenso Parcial / Dividido)',\n",
                "        '1 de 4 (Frágil / Resgate Exclusivo)',\n",
                "        '0 de 4 (Erro Comum / Incorrigível)'\n",
                "    ],\n",
                "    'Audios': [299, 475, 394, 372, 240],\n",
                "    'Pct_Total_Small': [16.80, 26.69, 22.13, 20.90, 13.48],\n",
                "    'Pct_Dentro_Oraculo': [19.42, 30.84, 25.58, 24.16, np.nan]\n",
                "}\n",
                "df_consensus = pd.DataFrame(consensus_data)\n",
                "\n",
                "# Resgates exclusivos por representação no RF\n",
                "exclusive_data = {\n",
                "    'Representacao': ['LOFAR Exclusivo', 'HHT Exclusivo', 'MEL Exclusivo', 'DEMON Exclusivo'],\n",
                "    'Audios_Salvos': [48, 63, 83, 178],\n",
                "    'Percentual': [2.70, 3.54, 4.66, 10.00]\n",
                "}\n",
                "df_exclusive = pd.DataFrame(exclusive_data)\n",
                "\n",
                "# Efeito Reverso em Relação ao Baseline MEL (1.035 acertos, 745 erros)\n",
                "reverse_data = {\n",
                "    'Destino_Troca': ['Para LOFAR', 'Para HHT', 'Para DEMON'],\n",
                "    'Mantem_do_MEL': [663, 699, 611],\n",
                "    'Perde_do_MEL (Efeito Reverso)': [372, 336, 424],\n",
                "    'Pct_Perda_Acertos (%)': [35.94, 32.46, 40.97],\n",
                "    'Resgata_Erros_MEL': [203, 230, 340],\n",
                "    'Pct_Resgate_Erros (%)': [27.25, 30.87, 45.64],\n",
                "    'Saldo_Liquido_Audios': [-169, -106, -84]\n",
                "}\n",
                "df_reverse = pd.DataFrame(reverse_data)\n",
                "\n",
                "# Padrão de confusão quando erra em SMALL no RF\n",
                "confusion_pattern_data = {\n",
                "    'Representacao': ['MEL (745 erros)', 'LOFAR (914 erros)', 'HHT (851 erros)', 'DEMON (829 erros)'],\n",
                "    'Confundido_MEDIUM_Pct': [51.41, 39.06, 34.90, 21.83],\n",
                "    'Confundido_LARGE_Pct': [44.03, 58.64, 55.82, 69.72],\n",
                "    'Confundido_AMBIENTAL_Pct': [4.56, 2.30, 9.28, 8.44]\n",
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
                "## 3. O Efeito Oráculo no Random Forest: Recall Individual vs. Teto Teórico\n",
                "\n",
                "O classificador Random Forest atinge **$58{,}15\\\\%$ de recall em `SMALL`** com o MEL, superando os melhores modelos isolados de SVM ($50{,}73\\\\%$) e MLP ($46{,}35\\\\%$).\n",
                "\n",
                "Contudo, a combinação das 4 entradas via **Oráculo RF** salta para **$86{,}52\\\\%$ (1.540 acertos)**:\n",
                "- Um ganho de **$+28{,}4$ pontos percentuais** sobre o MEL isolado;\n",
                "- O erro comum (onde todas as 4 florestas erram simultaneamente) cai para apenas **$13{,}48\\\\%$ (240 áudios)**, estabelecendo o novo recorde de menor perda irrecuperável na classe `SMALL`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Gráfico: Desempenho Isolado vs Teto Teórico do Oráculo no RF em SMALL\n",
                "plt.figure(figsize=(10, 5.5))\n",
                "colors = ['#e67e22', '#9b59b6', '#95a5a6', '#2980b9', '#27ae60']\n",
                "bars = plt.bar(df_hits['Representacao'], df_hits['Recall_Pct'], color=colors, width=0.55, edgecolor='black', linewidth=0.8)\n",
                "bars[-1].set_hatch('//')\n",
                "\n",
                "plt.axhline(86.52, color='#27ae60', linestyle='--', lw=1.5, alpha=0.7, label='Teto do Oráculo RF (86.52%)')\n",
                "plt.axhline(58.15, color='#2980b9', linestyle=':', lw=1.5, alpha=0.7, label='Melhor Isolado MEL (58.15%)')\n",
                "\n",
                "plt.title('Recall Random Forest na Classe SMALL: Modelos Isolados vs. Teto Teórico do Oráculo', fontweight='bold', pad=12)\n",
                "plt.ylabel('Recall em SMALL (%)')\n",
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
                "## 4. Análise de Grau de Consenso e Risco do Efeito Reverso em SMALL (RF)\n",
                "\n",
                "### 4.1 Pirâmide de Consenso no Random Forest\n",
                "- **Unanimidade (4/4 certas): $299$ áudios ($16{,}80\\\\%$)** — muito superior a SVM ($6{,}8\\\\%$) e MLP ($5{,}3\\\\%$);\n",
                "- **Alto Consenso (3/4 certas): $475$ áudios ($26{,}69\\\\%$)**;\n",
                "- **Consenso Dividido (2/4 certas): $394$ áudios ($22{,}13\\\\%$)**;\n",
                "- **Resgate Exclusivo (1/4 certa): $372$ áudios ($20{,}90\\\\%$)**;\n",
                "- **Erro Comum (0/4 certas): $240$ áudios ($13{,}48\\\\%$)**.\n",
                "\n",
                "### 4.2 O Poder da Votação por Maioria em SMALL no RF\n",
                "- **Unanimidade (4/4):** $16{,}80\\\\%$ ($299$ áudios);\n",
                "- **Maioria Estrita ($\\\\ge 3/4$):** $43{,}48\\\\%$ ($774$ áudios);\n",
                "- **Maioria Simples / Empate ($\\\\ge 2/4$):** **$65{,}62\\\\%$ ($1.168$ áudios)** $\\\\rightarrow$ **Supera o melhor modelo isolado (MEL com $58{,}15\\\\%$) em $+7{,}47$ p.p.!**\n",
                "- **Oráculo ($\\\\ge 1/4$):** **$86{,}52\\\\%$ ($1.540$ áudios)**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Visualização do Grau de Consenso e do Efeito Reverso em SMALL (RF)\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "# Painel 1: Pirâmide de Consenso dos 1.780 Áudios\n",
                "labels_cons = ['4 Entradas\\n(Unânime)', '3 Entradas\\n(Alto Consenso)', '2 Entradas\\n(Dividido)', '1 Entrada\\n(Exclusivo/Frágil)', '0 Entradas\\n(Erro Comum)']\n",
                "counts_cons = [299, 475, 394, 372, 240]\n",
                "pcts_cons = [16.80, 26.69, 22.13, 20.90, 13.48]\n",
                "colors_cons = ['#27ae60', '#2980b9', '#f39c12', '#e74c3c', '#7f8c8d']\n",
                "\n",
                "bars1 = ax1.bar(labels_cons, counts_cons, color=colors_cons, edgecolor='black', linewidth=0.8, width=0.55)\n",
                "ax1.set_title('Grau de Consenso RF em SMALL (Total: 1.780 Áudios no 5x2cv)', fontweight='bold', pad=12)\n",
                "ax1.set_ylabel('Quantidade de Áudios de SMALL')\n",
                "ax1.set_ylim(0, 560)\n",
                "\n",
                "for bar, pct in zip(bars1, pcts_cons):\n",
                "    h = bar.get_height()\n",
                "    ax1.annotate(f'{h} ({pct:.1f}%)', xy=(bar.get_x() + bar.get_width()/2, h),\n",
                "                 xytext=(0, 4), textcoords=\"offset points\", ha='center', va='bottom',\n",
                "                 fontsize=9.5, fontweight='bold')\n",
                "\n",
                "# Painel 2: Simulação do Efeito Reverso a partir do MEL (1.035 Acertos no RF)\n",
                "destinos = ['Para LOFAR', 'Para HHT', 'Para DEMON']\n",
                "perdas = [372, 336, 424]\n",
                "resgates = [203, 230, 340]\n",
                "x = np.arange(len(destinos))\n",
                "w = 0.35\n",
                "\n",
                "r1 = ax2.bar(x - w/2, perdas, w, label='Perde Acertos do MEL (Efeito Reverso)', color='#c0392b', alpha=0.9, edgecolor='black')\n",
                "r2 = ax2.bar(x + w/2, resgates, w, label='Resgata dos Erros do MEL (Ganhos)', color='#27ae60', alpha=0.9, edgecolor='black')\n",
                "\n",
                "ax2.set_title('Balanço do Efeito Reverso no RF ao Trocar de Entrada a partir do MEL', fontweight='bold', pad=12)\n",
                "ax2.set_ylabel('Quantidade de Áudios de SMALL')\n",
                "ax2.set_xticks(x)\n",
                "ax2.set_xticklabels(destinos, fontweight='bold', fontsize=11)\n",
                "ax2.set_ylim(0, 500)\n",
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
                "## 5. Resgates Exclusivos no Random Forest em SMALL: O Domínio do DEMON\n",
                "\n",
                "Em **$372$ áudios de `SMALL` ($20{,}90\\\\%$)**, três entradas falharam e apenas um modelo de floresta aleatória acertou a lancha:\n",
                "\n",
                "1. **DEMON Isolado (178 lanchas salvas exclusivamente — $10{,}00\\\\%$):**\n",
                "   - É o **campeão incontestável de resgates exclusivos em `SMALL`**!\n",
                "   - O DEMON demodula a alta frequência de rotação de eixos e batimento de pás de motores de popa de 2 ou 4 tempos ($2.500\\\\text{ a }5.000\\\\text{ RPM} \\\\implies 40\\\\text{ a }80\\\\text{ Hz}$). Árvores de decisão particionam essa faixa com altíssima pureza de Gini.\n",
                "2. **MEL Isolado (83 lanchas salvas exclusivamente — $4{,}66\\\\%$):**\n",
                "   - Captura a distribuição de energia em altas frequências acima de $2\\\\text{ kHz}$.\n",
                "3. **HHT Puro (63 lanchas salvas exclusivamente — $3{,}54\\\\%$):**\n",
                "   - Decompõe acelerações bruscas e não-linearidades do casco no marulho.\n",
                "4. **LOFAR Puro (48 lanchas salvas exclusivamente — $2{,}70\\\\%$):**\n",
                "   - Raias tônicas discretas isoladas de geradores de bordo."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Gráfico: Resgates Exclusivos na Classe SMALL (RF)\n",
                "plt.figure(figsize=(9, 4.5))\n",
                "colors_excl = ['#e67e22', '#9b59b6', '#2980b9', '#16a085']\n",
                "bars_excl = plt.barh(df_exclusive['Representacao'], df_exclusive['Audios_Salvos'], color=colors_excl, height=0.5, edgecolor='black', linewidth=0.8)\n",
                "\n",
                "plt.title('Resgates Exclusivos RF na Classe SMALL (Total: 372 Áudios Únicos)', fontweight='bold', pad=12)\n",
                "plt.xlabel('Número de Áudios Salvos Exclusivamente')\n",
                "plt.xlim(0, 230)\n",
                "\n",
                "for idx, (bar, pct) in enumerate(zip(bars_excl, df_exclusive['Percentual'])):\n",
                "    w = bar.get_width()\n",
                "    tag = \"  <-- CAMPEÃO DE RESGATES EM SMALL\" if idx == 3 else \"\"\n",
                "    plt.annotate(f'{w} áudios ({pct:.1f}%){tag}', xy=(w, bar.get_y() + bar.get_height()/2),\n",
                "                 xytext=(5, 0), textcoords=\"offset points\", ha='left', va='center',\n",
                "                 fontsize=9.5, fontweight='bold', color='#111' if idx != 3 else '#117864')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Padrões de Erro em SMALL: Para Onde Cada Entrada Envia a Lancha no RF?\n",
                "\n",
                "| Entrada | Principal Erro | Porcentagem do Erro | Segundo Maior Erro | Diagnóstico Mecânico |\n",
                "| :--- | :---: | :---: | :---: | :--- |\n",
                "| **MEL** | **`MEDIUM`** | **$51{,}41\\\\%$** | `LARGE` ($44{,}03\\\\%$) | Suavização espectral em bandas Mel cria sobreposição energética com rebocadores. |\n",
                "| **LOFAR** | **`LARGE`** | **$58{,}64\\\\%$** | `MEDIUM` ($39{,}06\\\\%$) | Ausência de raias tônicas subsônicas induz o particionamento do LOFAR a enviar o áudio para cargueiros. |\n",
                "| **HHT** | **`LARGE`** | **$55{,}82\\\\%$** | `MEDIUM` ($34{,}90\\\\%$) | Modulação de amplitude de ondas superficiais assemelha-se a ruído de cavitação de grandes eixos. |\n",
                "| **DEMON** | **`LARGE`** | **$69{,}72\\\\%$** | `MEDIUM` ($21{,}83\\\\%$) | Quando a cadência de motor de popa falha, a modulação residual de marulho cai em baixas frequências (< 2 Hz). |"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Gráfico: Destino do Erro em SMALL por Representação (RF)\n",
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
                "plt.title('Padrão de Erro RF na Classe SMALL: Para onde cada entrada envia a lancha? (%)', fontweight='bold', pad=12)\n",
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
                "## 7. Painel Consolidado de Descoberta Científica (RF)\n",
                "\n",
                "Abaixo exibimos o painel visual oficial de 4 quadrantes para a classe `SMALL` com o modelo Random Forest:"
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
                "img_path = 'reports/analise_complementaridade_rf_small_oraculo.png'\n",
                "if os.path.exists(img_path):\n",
                "    display(Image(filename=img_path))\n",
                "else:\n",
                "    print(f\"Imagem {img_path} não encontrada. Execute 'scratch/generate_rf_predictions_and_analysis.py' para gerá-la.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Comparativo Trilateral em SMALL: SVM Nyström vs. MLP PyTorch vs. Random Forest\n",
                "\n",
                "| Dimensão Analítica | SVM Nyström (Notebook 10) | MLP PyTorch (Notebook 14) | Random Forest (Notebook 17) | Diagnóstico Científico |\n",
                "| :--- | :---: | :---: | :---: | :--- |\n",
                "| **Melhor Modelo Isolado** | $50{,}73\\\\%$ (MEL) | $46{,}35\\\\%$ (LOFAR) | **$58{,}15\\\\%$ (MEL)** | **O Random Forest é o campeão absoluto isolado em `SMALL`!** |\n",
                "| **Teto do Oráculo** | $83{,}99\\\\%$ ($1.495$) | $76{,}85\\\\%$ ($1.368$) | **$86{,}52\\\\%$ ($1.540$)** | **O RF atinge o maior teto de `SMALL` de toda a pesquisa ($+2{,}5$ p.p. sobre SVM).** |\n",
                "| **Taxa de Unanimidade (4/4)** | $6{,}80\\\\%$ ($121$) | $5{,}34\\\\%$ ($95$) | **$16{,}80\\\\%$ ($299$)** | O RF amplia em quase $3\\\\times$ o núcleo estável de acerto em lanchas. |\n",
                "| **Erro Comum Incorrigível** | $16{,}01\\\\%$ ($285$) | $23{,}15\\\\%$ ($412$) | **$13{,}48\\\\%$ ($240$)** | Menor taxa de perda irrecuperável do dataset em embarcações leves. |\n",
                "| **Campeão de Resgates Exclusivos** | HHT ($207$ lanchas) | DEMON ($155$ lanchas) | **DEMON ($178$ lanchas)** | Validação da complementaridade do DEMON em lanchas rápidas. |\n",
                "| **Votação Simples ($\\\\ge 2/4$)** | $55{,}17\\\\%$ ($+4{,}4$ p.p.) | $41{,}74\\\\%$ | **$65{,}62\\\\%$ ($+7{,}47$ p.p.)** | A votação por maioria simples no RF salta para $65\\\\%$! |"
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
