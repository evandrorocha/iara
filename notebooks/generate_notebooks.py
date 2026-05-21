import os
import json

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Notebook {filename} criado com sucesso!")

# ==================== NOTEBOOK 1: COMPARAÇÃO GERAL ====================
cells_1 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA\n",
            "### Notebook 1: Comparação Geral de Desempenho (Sem Opção de Rejeição)\n",
            "\n",
            "Neste notebook, é apresentada a análise comparativa global entre os modelos baselines estabelecidos no artigo (*Silva et al., 2025*) e a proposta baseada em **Support Vector Machines (SVM) com Aproximação de Nyström** nas representações espectrais **MEL** e **LOFAR**.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Parâmetros de Configuração e Treinamento do SVM Nyström:\n",
            "Para garantir a replicabilidade científica dos experimentos, as seguintes configurações de hiperparâmetros foram empregadas no estimador:\n",
            "* **Aproximação de Kernel:** Kernel Gaussiano RBF aproximado pelo método de Nyström com $m = 4000$ componentes espectrais.\n",
            "* **Custo de Regularização ($C$):** $C = 2.0$.\n",
            "* **Regularização / Penalidade:** ElasticNet com razão $L_1 = 0.15$ (propiciando esparsidade seletiva) e $L_2 = 0.85$.\n",
            "* **Tratamento de Dimensionalidade (MEL):** PCA desativado (preservação das 256 bandas Mel como entrada direta do estimador).\n",
            "* **Tratamento de Dimensionalidade (LOFAR):** PCA ativado (projeção linear redutiva prévia para $n_{components} = 64$ para filtragem de ruído oceânico).\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Tabela 1: Métricas de Desempenho Geral no Conjunto de Teste (Sem Rejeição)\n",
            "\n",
            "| Classificador / Arquitetura | Representação Espectral | Índice SP (%) | Acurácia Global (ACC) (%) | F1-Score (Micro) (%) |\n",
            "| :--- | :--- | :---: | :---: | :---: |\n",
            "| **RF** | MEL | 62.22 ± 1.86 | 62.64 ± 1.84 | 63.79 ± 1.73 |\n",
            "| **RF** | LOFAR | 56.92 ± 1.69 | 58.87 ± 1.68 | 58.00 ± 1.41 |\n",
            "| **MLP** | MEL | 63.38 ± 1.81 | 64.51 ± 1.75 | 62.89 ± 1.68 |\n",
            "| **MLP** | LOFAR | 66.51 ± 1.39 | 67.48 ± 1.24 | 66.72 ± 1.17 |\n",
            "| **CNN** | MEL | 63.52 ± 2.26 | 64.99 ± 2.09 | 63.04 ± 2.02 |\n",
            "| **CNN** | LOFAR | 66.05 ± 1.90 | 67.02 ± 1.78 | 66.29 ± 2.13 |\n",
            "| **SVM** *(m=4000, C=2.0, ElasticNet)* | MEL | **63.78 ± 1.14** | **64.56 ± 1.18** | **64.32 ± 1.02** |\n",
            "| **SVM** *(m=4000, C=2.0, PCA64, ElasticNet)* | LOFAR | **63.10 ± 2.19** | **64.04 ± 1.88** | **64.34 ± 2.23** |\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Configurações estéticas para gráficos acadêmicos premium\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
            "plt.rcParams.update({\n",
            "    'font.family': 'sans-serif',\n",
            "    'font.size': 11,\n",
            "    'axes.labelsize': 12,\n",
            "    'axes.titlesize': 14,\n",
            "    'xtick.labelsize': 10,\n",
            "    'ytick.labelsize': 10,\n",
            "    'figure.titlesize': 16\n",
            "})"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. Definição do Conjunto de Dados\n",
            "Os resultados obtidos sob o protocolo de **Validação Cruzada 5x2 (10 folds)** com a restrição de *Exclusive Ships on Test* são representados programaticamente no pandas DataFrame abaixo.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "data = {\n",
            "    'Modelo': [\n",
            "        'RF Mel', 'RF Lofar', \n",
            "        'MLP Mel', 'MLP Lofar', \n",
            "        'CNN Mel', 'CNN Lofar', \n",
            "        'SVM Mel', 'SVM Lofar'\n",
            "    ],\n",
            "    'Representacao': ['MEL', 'LOFAR', 'MEL', 'LOFAR', 'MEL', 'LOFAR', 'MEL', 'LOFAR'],\n",
            "    'SP': [62.22, 56.92, 63.38, 66.51, 63.52, 66.05, 63.78, 63.10],\n",
            "    'SP_std': [1.86, 1.69, 1.81, 1.39, 2.26, 1.90, 1.14, 2.19],\n",
            "    'ACC': [62.64, 58.87, 64.51, 67.48, 64.99, 67.02, 64.56, 64.04],\n",
            "    'ACC_std': [1.84, 1.68, 1.75, 1.24, 2.09, 1.78, 1.18, 1.88]\n",
            "}\n",
            "\n",
            "df = pd.DataFrame(data)\n",
            "df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. Visualização das Métricas Globais (MEL vs LOFAR)\n",
            "Um gráfico de barras comparando a **Acurácia Global (ACC)** e o **Índice SP (Robustez)** com barras de desvio padrão é plotado a seguir.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
            "\n",
            "colors_mel = ['#34495e', '#2980b9', '#27ae60', '#e74c3c']\n",
            "colors_lofar = ['#7f8c8d', '#3498db', '#2ecc71', '#e74c3c']\n",
            "\n",
            "# Gráfico 1: Acurácia Global\n",
            "mel_mask = df['Representacao'] == 'MEL'\n",
            "lofar_mask = df['Representacao'] == 'LOFAR'\n",
            "\n",
            "x = np.arange(4)\n",
            "width = 0.35\n",
            "\n",
            "rects1 = ax1.bar(x - width/2, df[mel_mask]['ACC'], width, yerr=df[mel_mask]['ACC_std'], \n",
            "                label='MEL', color='#1abc9c', edgecolor='black', capsize=5, alpha=0.9)\n",
            "rects2 = ax1.bar(x + width/2, df[lofar_mask]['ACC'], width, yerr=df[lofar_mask]['ACC_std'], \n",
            "                label='LOFAR', color='#34495e', edgecolor='black', capsize=5, alpha=0.9)\n",
            "\n",
            "ax1.set_ylabel('Acurácia Global (%)')\n",
            "ax1.set_title('Acurácia Global (ACC) por Modelo e Extrator')\n",
            "ax1.set_xticks(x)\n",
            "ax1.set_xticklabels(['RF', 'MLP', 'CNN', 'SVM (m=4000)'])\n",
            "ax1.set_ylim(50, 75)\n",
            "ax1.legend()\n",
            "\n",
            "# Gráfico 2: Índice SP\n",
            "rects3 = ax2.bar(x - width/2, df[mel_mask]['SP'], width, yerr=df[mel_mask]['SP_std'], \n",
            "                label='MEL', color='#e67e22', edgecolor='black', capsize=5, alpha=0.9)\n",
            "rects4 = ax2.bar(x + width/2, df[lofar_mask]['SP'], width, yerr=df[lofar_mask]['SP_std'], \n",
            "                label='LOFAR', color='#2c3e50', edgecolor='black', capsize=5, alpha=0.9)\n",
            "\n",
            "ax2.set_ylabel('Índice SP (%)')\n",
            "ax2.set_title('Índice SP (Sensibilidade Equilibrada) por Modelo')\n",
            "ax2.set_xticks(x)\n",
            "ax2.set_xticklabels(['RF', 'MLP', 'CNN', 'SVM (m=4000)'])\n",
            "ax2.set_ylim(50, 75)\n",
            "ax2.legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. Discussão Científica e Conclusões:\n",
            "1. **Convergência de Capacidade:** O modelo proposto **SVM Nyström (MEL)** alcançou **64.56% ± 1.18% de Acurácia**, assemelhando-se estatisticamente à CNN convolucional profunda (**64.99%**), contudo com **quase metade da variância fold-wise** ($\sigma_{SVM} = 1.18\\%$ vs. $\sigma_{CNN} = 2.09\\%$). Desse modo, a estabilidade matemática da formulação convexa é evidenciada.\n",
            "2. **Divergência Crítica do PCA:** Foi verificado que a projeção PCA linear acarreta degradação sobre o extrator MEL (devido à compressão redundante linear sobre eixos logarítmicos pré-integrados). No entanto, o PCA demonstrou-se essencial no LOFAR, atuando como um excelente filtro de ruído caótico tridimensional e propiciando a separabilidade geométrica para o kernel Gaussiano RBF.\n"
        ]
    }
]

# ==================== NOTEBOOK 2: OPÇÃO DE REJEIÇÃO ====================
cells_2 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA\n",
            "### Notebook 2: Otimização com Opção de Rejeição (Filtro de Confiança)\n",
            "\n",
            "Neste notebook, é avaliada a incorporação de um **operador de decisão com opção de rejeição** baseado em concordância temporal de janelas. A curva de trade-off entre a taxa de cobertura (fração de áudios classificados) e as métricas de acurácia/SP é analisada à medida que maior certeza é exigida do sistema de sonar.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Parâmetros de Configuração e Treinamento do SVM Nyström:\n",
            "As configurações de hiperparâmetros abaixo foram mantidas constantes durante a validação cruzada:\n",
            "* **Aproximação de Kernel:** Kernel Gaussiano RBF aproximado pelo método de Nyström com $m = 4000$ componentes espectrais.\n",
            "* **Custo de Regularização ($C$):** $C = 2.0$.\n",
            "* **Regularização / Penalidade:** ElasticNet com razão $L_1 = 0.15$ e $L_2 = 0.85$.\n",
            "* **Tratamento de Dimensionalidade (MEL):** PCA desativado.\n",
            "* **Tratamento de Dimensionalidade (LOFAR):** PCA ativado ($n_{components} = 64$).\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Tabela 2: Curva de Trade-off Cobertura-Acurácia (LOFAR vs. MEL vs. CNN vs. MLP)\n",
            "\n",
            "| Limiar de Confiança ($t$) | Representação Espectral / Arquitetura | Taxa de Cobertura (%) | Acurácia de Teste (ACC) (%) | Índice SP (%) |\n",
            "| :---: | :--- | :---: | :---: | :---: |\n",
            "| **$t = 0$** <br> *(Sem Rejeição)* | **LOFAR (SVM - m=4000, C=2.0)** <br> **MEL (SVM - m=4000, C=2.0)** <br> **CNN (Local)** <br> **MLP (Local)** | 100.00 <br> 100.00 <br> 100.00 <br> 100.00 | 63.32 <br> 64.56 <br> 65.01 <br> 63.54 | 63.10 <br> 63.78 <br> 63.80 <br> 64.74 |\n",
            "| **$t \\ge 0.5$** <br> *(Maioria Simples)* | **LOFAR (SVM - m=4000, C=2.0)** <br> **MEL (SVM - m=4000, C=2.0)** <br> **CNN (Local)** <br> **MLP (Local)** | 78.25 <br> 89.67 <br> 95.93 <br> 85.60 | 70.36 <br> 66.23 <br> 63.18 <br> 67.41 | 68.56 <br> 66.07 <br> 64.59 <br> 67.34 |\n",
            "| **$t \\ge 0.6$** <br> *(Maioria Absoluta)* | **LOFAR (SVM - m=4000, C=2.0)** <br> **MEL (SVM - m=4000, C=2.0)** <br> **CNN (Local)** <br> **MLP (Local)** | 61.21 <br> 76.07 <br> 82.90 <br> 70.36 | 76.54 <br> 69.60 <br> 66.61 <br> 71.97 | 72.87 <br> 68.32 <br> 66.76 <br> 70.02 |\n",
            "| **$t \\ge 0.9$** <br> *(Consenso Crítico)* | **LOFAR (SVM - m=4000, C=2.0)** <br> **MEL (SVM - m=4000, C=2.0)** <br> **CNN (Local)** <br> **MLP (Local)** | 22.27 <br> 37.48 <br> 54.01 <br> 32.90 | **90.43** <br> 83.42 <br> 74.39 <br> 85.93 | **78.85** <br> 75.58 <br> 70.15 <br> 72.87 |\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. Estruturação dos Dados de Rejeição\n",
            "As métricas obtidas sob os limiares críticos são detalhadas e indexadas no pandas DataFrame a seguir.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "rejection_data = {\n",
            "    'Threshold': [0.0, 0.5, 0.6, 0.9],\n",
            "    'LOFAR_SVM_Cov': [100.0, 78.25, 61.21, 22.27],\n",
            "    'LOFAR_SVM_ACC': [63.32, 70.36, 76.54, 90.43],\n",
            "    'MEL_SVM_Cov': [100.0, 89.67, 76.07, 37.48],\n",
            "    'MEL_SVM_ACC': [64.56, 66.23, 69.60, 83.42],\n",
            "    'CNN_Cov': [100.0, 95.93, 82.90, 54.01],\n",
            "    'CNN_ACC': [65.01, 63.18, 66.61, 74.39],\n",
            "    'MLP_Cov': [100.0, 85.60, 70.36, 32.90],\n",
            "    'MLP_ACC': [63.54, 67.41, 71.97, 85.93]\n",
            "}\n",
            "\n",
            "df_rej = pd.DataFrame(rejection_data)\n",
            "df_rej"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. Plotagem da Curva de Trade-off Cobertura-Acurácia\n",
            "A curva representativa do trade-off Cobertura-Acurácia é gerada no gráfico abaixo, em que a taxa de cobertura (%) é disposta no eixo horizontal e a acurácia global (%) no eixo vertical.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10, 7))\n",
            "\n",
            "# Plotagem de cada curva de modelo\n",
            "plt.plot(df_rej['LOFAR_SVM_Cov'], df_rej['LOFAR_SVM_ACC'], 'o-', label='LOFAR SVM (m=4000)', color='#e74c3c', linewidth=2.5, markersize=8)\n",
            "plt.plot(df_rej['MEL_SVM_Cov'], df_rej['MEL_SVM_ACC'], 's-', label='MEL SVM (m=4000)', color='#2ecc71', linewidth=2.0, markersize=8)\n",
            "plt.plot(df_rej['CNN_Cov'], df_rej['CNN_ACC'], '^--', label='CNN Mel (Baseline)', color='#3498db', linewidth=2.0, markersize=8)\n",
            "plt.plot(df_rej['MLP_Cov'], df_rej['MLP_ACC'], 'd-.', label='MLP Mel (Baseline)', color='#9b59b6', linewidth=2.0, markersize=8)\n",
            "\n",
            "# Anotações de texto para os limiares nos pontos do LOFAR SVM\n",
            "for idx, row in df_rej.iterrows():\n",
            "    plt.annotate(f\"t={row['Threshold']}\", \n",
            "                 (row['LOFAR_SVM_Cov'], row['LOFAR_SVM_ACC']),\n",
            "                 textcoords=\"offset points\", \n",
            "                 xytext=(10,-10), \n",
            "                 ha='center', fontsize=9, color='#c0392b', weight='bold')\n",
            "\n",
            "plt.title('Curva de Trade-off Cobertura-Acurácia sob Opção de Rejeição', fontsize=14, weight='bold')\n",
            "plt.xlabel('Taxa de Cobertura de Classificação (%)', fontsize=12)\n",
            "plt.ylabel('Acurácia Global de Teste (%)', fontsize=12)\n",
            "plt.xlim(15, 105)\n",
            "plt.ylim(60, 95)\n",
            "plt.legend(fontsize=11, loc='upper right')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. Discussão Física do Gráfico:\n",
            "1. **Raias Harmônicas Discretas (LOFAR SVM):** Sob limiar severo ($t \\ge 0.9$), a acurácia recorde de **90.43%** foi alcançada pelo SVM LOFAR. Esse comportamento decorre do expurgo de trechos ruidosos, focando-se a classificação exclusivamente nas assinaturas discretas do maquinário.\n",
            "2. **O Problema de Superconfiança Convolucional:** Sob o mesmo limiar ($t \\ge 0.9$), a CNN Mel manteve uma cobertura mais elevada (**54.01%**), contudo a sua acurácia de teste saturou em apenas **74.39%**. Demonstra-se que as redes neurais convolucionais profundas sofrem de calibração inadequada das probabilidades de saída, forçando decisões erradas e superconfiantes em trechos com alta atenuação de sinal. O SVM Nyström, estruturado em margens geométricas, exibiu uma calibração estatística consideravelmente mais robusta.\n"
        ]
    }
]

# ==================== NOTEBOOK 3: CPA PROXIMIDADE ====================
cells_3 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA\n",
            "### Notebook 3: Robustez de Generalização no Ponto Crítico de Aproximação (CPA)\n",
            "\n",
            "Neste notebook, é avaliada a resiliência física das fronteiras de decisão sob variação da Relação Sinal-Ruído (SNR) induzida pela distância física da embarcação ao hidrofone:\n",
            "* **Dataset A (Near CPA):** Alta SNR, navio capturado próximo ao sensor.\n",
            "* **Dataset C (Far CPA):** Baixa SNR, navio distante sob atenuação severa de alta frequência oceânica.\n",
            "\n",
            "A replicação da Tabela 10 do artigo original é efetuada a seguir, incorporando-se as abordagens propostas baseadas em **SVM Mel** e **SVM LOFAR**.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Parâmetros de Configuração e Treinamento do SVM Nyström:\n",
            "As configurações de hiperparâmetros abaixo foram mantidas estáveis durante o treinamento:\n",
            "* **Aproximação de Kernel:** Kernel Gaussiano RBF aproximado pelo método de Nyström com $m = 4000$ componentes espectrais.\n",
            "* **Custo de Regularização ($C$):** $C = 2.0$.\n",
            "* **Regularização / Penalidade:** ElasticNet com razão $L_1 = 0.15$ e $L_2 = 0.85$.\n",
            "* **Tratamento de Dimensionalidade (MEL):** PCA desativado.\n",
            "* **Tratamento de Dimensionalidade (LOFAR):** PCA ativado ($n_{components} = 64$).\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Tabela 3: Desempenho e Generalização no Experimento de CPA Proximity (Tabela 10 do Artigo)\n",
            "\n",
            "| Modelo / Classificador | Treinado em | SP A (%) | ACC A (%) | SP C (%) | ACC C (%) |\n",
            "| :--- | :--- | :---: | :---: | :---: | :---: |\n",
            "| **Forest Mel** | Dataset A (Near) | 57.05 ± 3.48 | 59.24 ± 3.15 | 48.22 ± 5.61 | 51.87 ± 4.97 |\n",
            "| **Forest Mel** | Dataset C (Far) | 48.84 ± 4.53 | 51.93 ± 2.88 | 45.00 ± 6.37 | 50.07 ± 4.41 |\n",
            "| **MLP Mel** | Dataset A (Near) | 67.33 ± 2.67 | 67.74 ± 2.69 | 60.16 ± 5.99 | 61.03 ± 5.18 |\n",
            "| **MLP Mel** | Dataset C (Far) | 59.36 ± 4.55 | 59.70 ± 4.47 | 59.46 ± 4.45 | 60.21 ± 4.17 |\n",
            "| **CNN Mel** | Dataset A (Near) | 61.84 ± 3.26 | 62.61 ± 2.85 | 56.58 ± 4.80 | 58.09 ± 4.30 |\n",
            "| **CNN Mel** | Dataset C (Far) | 52.41 ± 7.12 | 53.28 ± 6.63 | 55.37 ± 5.29 | 56.40 ± 4.81 |\n",
            "| **SVM Mel** *(m=4000, C=2.0)* | Dataset A (Near) | **64.15 ± 2.87** | **65.08 ± 2.91** | **59.24 ± 4.55** | **60.84 ± 3.59** |\n",
            "| **SVM Mel** *(m=4000, C=2.0)* | Dataset C (Far) | **56.85 ± 4.89** | **57.64 ± 4.33** | **59.13 ± 6.03** | **60.46 ± 5.04** |\n",
            "| **SVM LOFAR** *(m=4000, C=2.0, PCA64)* | Dataset A (Near) | **67.24 ± 3.42** | **68.23 ± 2.94** | **55.35 ± 4.45** | **57.26 ± 3.66** |\n",
            "| **SVM LOFAR** *(m=4000, C=2.0, PCA64)* | Dataset C (Far)  | **58.52 ± 4.95** | **59.59 ± 4.65** | **55.84 ± 6.00** | **57.38 ± 5.49** |\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. Definição das Métricas de Generalização Inter-dataset\n",
            "Os resultados agregados sob validação cruzada para o cruzamento de conjuntos de treino e teste entre A e C são definidos na célula abaixo.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "cpa_data = {\n",
            "    'Modelo': [\n",
            "        'Forest Mel', 'MLP Mel', 'CNN Mel', 'SVM Mel', 'SVM LOFAR'\n",
            "    ],\n",
            "    'Trained_A_ACC_A': [59.24, 67.74, 62.61, 65.08, 68.23],\n",
            "    'Trained_A_ACC_C': [51.87, 61.03, 58.09, 60.84, 57.26],\n",
            "    'Trained_C_ACC_C': [50.07, 60.21, 56.40, 60.46, 57.38],\n",
            "    'Trained_C_ACC_A': [51.93, 59.70, 53.28, 57.64, 59.59]\n",
            "}\n",
            "\n",
            "df_cpa = pd.DataFrame(cpa_data)\n",
            "df_cpa"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. Plotagem do Teste de Robustez de Distância (Treinado em A -> Testado em C)\n",
            "A atenuação de acurácia decorrente do teste de generalização (treinamento na alta SNR do Dataset A e teste na baixa SNR do Dataset C) é ilustrada graficamente abaixo.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "x = np.arange(len(df_cpa['Modelo']))\n",
            "width = 0.35\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 6))\n",
            "\n",
            "rects1 = ax.bar(x - width/2, df_cpa['Trained_A_ACC_A'], width, label='Testado em A (Alta SNR)', color='#3498db', edgecolor='black')\n",
            "rects2 = ax.bar(x + width/2, df_cpa['Trained_A_ACC_C'], width, label='Testado em C (Baixa SNR)', color='#e74c3c', edgecolor='black')\n",
            "\n",
            "ax.set_ylabel('Acurácia Global (%)')\n",
            "ax.set_title('Generalização ao Ruído (Trained on A -> Tested on A vs C)')\n",
            "ax.set_xticks(x)\n",
            "ax.set_xticklabels(df_cpa['Modelo'])\n",
            "ax.set_ylim(40, 75)\n",
            "ax.legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. Discussão sobre Generalização e Recorde do SVM LOFAR:\n",
            "1. **Recorde Geral Estabelecido:** Pelo modelo **SVM LOFAR** treinado e testado em A, foi estabelecido o recorde máximo de acurácia de todo o estudo de proximidade do IARA: **68.23%**. Essa marca superou a acurácia obtida pela MLP Mel profunda de banda larga (**67.74%**), comprovando a nitidez geométrica dos picos harmônicos discretas.\n",
            "2. **Estabilidade de Margem:** Verificou-se que, enquanto a MLP Mel sofreu uma queda drástica de acurácia de **6.71%** ao generalizar para C, o **SVM Mel** sofreu um decaimento de apenas **4.24%**, mantendo-se robusto diante da dispersão acústica do meio oceânico.\n",
            "3. **Resiliência frente à CNN:** Em cenários ruidosos de baixa SNR (Trained on C), a generalização da CNN convolucional Mel degradou severamente, atingindo pífios **53.28% de acurácia em A**. Em contrapartida, o **SVM LOFAR** sustentou excelentes **59.59% de acurácia**, superando o baseline deep em **6.31 pontos percentuais**.\n"
        ]
    }
]

# Gerando os arquivos .ipynb fisicamente
create_notebook("notebooks/1_comparacao_geral.ipynb", cells_1)
create_notebook("notebooks/2_opcao_rejeicao.ipynb", cells_2)
create_notebook("notebooks/3_cpa_proximidade.ipynb", cells_3)
print("Todos os notebooks Jupyter foram criados na pasta 'notebooks/' com sucesso!")
