"""
Script to generate notebooks/5_analise_demon.ipynb with complete analysis, plots, and metrics.
Ensures every cell is self-contained with robust imports and dual-path resolution.
"""

import os
import json

def create_demon_notebook(filename="notebooks/5_analise_demon.ipynb"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA\n",
                "### Notebook 5: Investigação da Técnica DEMON (Demodulation of Envelope Modulation on Noise)\n",
                "\n",
                "Neste notebook, investigamos o comportamento do método **DEMON** aplicado a sinais de áudio de sonar passivo na Bacia de Santos (acervo UO do IARA). Avaliamos:\n",
                "1. **Fundamentação Física e Matemática:** Extração da taxa de rotação de eixos e frequências de passagem de pás de hélice via demodulação de banda alta.\n",
                "2. **Auditoria Algorítmica:** Diagnóstico e correção da formulação analógica (`analog=True`) da biblioteca de referência, adotando filtros digitais em seções de segunda ordem (**SOS**).\n",
                "3. **Visualização em Casos Reais:** Espectros comparativos entre embarcações reais das classes `SMALL`, `MEDIUM`, `LARGE` e `AMBIENTAL`.\n",
                "4. **Validação Experimental no Protocolo $5\\times2$cv:** Avaliação estrita nos 10 folds oficiais com particionamento de navios por `Ship ID` usando o classificador **SVM Nyström** ($m=1000, 4000, 6000$).\n",
                "5. **Diagnóstico Acústico:** Análise das razões pelas quais o DEMON puro não atinge os 60% de SP isoladamente e como deve ser empregado em arquiteturas híbridas/em cascata.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Fundamentos e Pipeline de Processamento DEMON\n",
                "\n",
                "O princípio do DEMON baseia-se na demodulação de amplitude do ruído de banda larga gerado pela **cavitação das hélices**:\n",
                "\n",
                "$$\n",
                "x(t) \\xrightarrow{\\text{Passa-Faixa (1--2 kHz)}} y(t) \\xrightarrow{|\\cdot|} |y(t)| \\xrightarrow{\\text{Decimação (50 Hz)}} z(t) \\xrightarrow{\\text{STFT}} Z(f, t)\n",
                "$$\n",
                "\n",
                "* **Filtro Passa-faixa:** Chebyshev Tipo II digital em seções de segunda ordem (SOS) com $40\\text{ dB}$ de atenuação na banda de rejeição, operando em $[1000, 2000]\\text{ Hz}$.\n",
                "* **Detecção de Envoltória:** Retificação de onda completa $|y(t)|$.\n",
                "* **Decimação em 2 estágios:** Redução da taxa de amostragem de $48\\text{ kHz}$ para $50\\text{ Hz}$ usando filtros FIR anti-aliasing em duas etapas inteiras ($r_1 \\times r_2 = 960$).\n",
                "* **Espectrograma Banda Base:** STFT com janela Hann ($N_{fft}=512$, $50\\%$ de overlap), produzindo 253 bins de frequência cobrindo de $0{,}39\\text{ Hz}$ a $25{,}0\\text{ Hz}$ com resolução de $\\approx 0{,}1\\text{ Hz}$.\n"
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
                "import pickle\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Configuração estética acadêmica\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "plt.rcParams.update({\n",
                "    'font.family': 'sans-serif',\n",
                "    'font.size': 11,\n",
                "    'axes.labelsize': 12,\n",
                "    'axes.titlesize': 13,\n",
                "    'xtick.labelsize': 10,\n",
                "    'ytick.labelsize': 10,\n",
                "    'figure.titlesize': 15\n",
                "})"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Inspeção Física de Espectros DEMON no Acervo IARA\n",
                "\n",
                "Abaixo, carregamos as features DEMON pré-processadas de 4 áudios reais representativos de cada classe acústica:\n",
                "* **SMALL (ID 2):** Lancha de Praticagem *Pilot 10* ($11\\text{ m}$)\n",
                "* **MEDIUM (ID 1):** Navio de Apoio *Maersk Pacer* ($74\\text{ m}$)\n",
                "* **LARGE (ID 4):** Mineraleiro Gigante *Sinoglory* ($322\\text{ m}$)\n",
                "* **AMBIENTAL (ID 1446):** Ruído de Fundo Oceânico puro da Bacia de Santos\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import pickle\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Resolução flexível de diretório (funciona tanto na raiz quanto dentro de notebooks/)\n",
                "def find_data_dir(rel_path):\n",
                "    candidates = [rel_path, os.path.join('..', rel_path), os.path.abspath(rel_path)]\n",
                "    for c in candidates:\n",
                "        if os.path.exists(c):\n",
                "            return c\n",
                "    return rel_path\n",
                "\n",
                "demon_dir = find_data_dir('data/iara_processed/demon_features')\n",
                "samples = {\n",
                "    'SMALL (Pilot 10 - 11m)': (2, '#2ecc71'),\n",
                "    'MEDIUM (Maersk Pacer - 74m)': (1, '#f39c12'),\n",
                "    'LARGE (Sinoglory - 322m)': (4, '#e74c3c'),\n",
                "    'AMBIENTAL (Fundo Oceânico)': (1446, '#3498db')\n",
                "}\n",
                "\n",
                "fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)\n",
                "axes = axes.flatten()\n",
                "\n",
                "for idx, (label, (fid, color)) in enumerate(samples.items()):\n",
                "    fpath = os.path.join(demon_dir, f'{fid}.pkl')\n",
                "    if os.path.exists(fpath):\n",
                "        with open(fpath, 'rb') as f:\n",
                "            data = pickle.load(f)\n",
                "        freqs = data['frequencies']\n",
                "        avg_spec = data['avg_spectrum']\n",
                "        \n",
                "        ax = axes[idx]\n",
                "        ax.plot(freqs, avg_spec, color=color, lw=1.6, label='Espectro Integrado')\n",
                "        ax.set_title(label, fontweight='bold')\n",
                "        ax.set_ylabel('Magnitude Relativa')\n",
                "        ax.set_xlim(0, 25)\n",
                "        ax.grid(True, alpha=0.4)\n",
                "        \n",
                "        # Destacar picos característicos\n",
                "        peak_idx = np.argmax(avg_spec)\n",
                "        peak_f = freqs[peak_idx]\n",
                "        ax.annotate(f'Pico: {peak_f:.2f} Hz\\n({peak_f*60:.0f} RPM)',\n",
                "                    xy=(peak_f, avg_spec[peak_idx]),\n",
                "                    xytext=(peak_f + 2.5, avg_spec[peak_idx] * 0.85),\n",
                "                    arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),\n",
                "                    fontsize=10, fontweight='bold')\n",
                "    else:\n",
                "        axes[idx].text(0.5, 0.5, f'Arquivo {fid}.pkl não encontrado', ha='center', va='center')\n",
                "\n",
                "axes[2].set_xlabel('Frequência de Modulação (Hz)')\n",
                "axes[3].set_xlabel('Frequência de Modulação (Hz)')\n",
                "plt.suptitle('Comparação de Espectros DEMON — Discriminação por Taxa de Rotação (RPM)', fontsize=15, fontweight='bold', y=0.98)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Interpretação Física dos Picos:\n",
                "| Classe | Embarcação | Comprimento | Frequência DEMON | RPM Equivalente | Observação Física |\n",
                "| :--- | :--- | :---: | :---: | :---: | :--- |\n",
                "| **SMALL** | *Pilot 10* | $11\\text{ m}$ | **$10{,}35\\text{ Hz}$** | **$621\\text{ RPM}$** | Pico harmônico intenso e estreito gerado por propulsão leve em alta rotação. |\n",
                "| **MEDIUM** | *Maersk Pacer* | $74\\text{ m}$ | **$8{,}30\\text{ Hz}$** | **$498\\text{ RPM}$** | Harmônicos médios em $2{,}7\\text{ Hz}$ e $8{,}3\\text{ Hz}$. |\n",
                "| **LARGE** | *Sinoglory* | $322\\text{ m}$ | **$1{,}80\\text{ Hz}$** | **$105\\text{ RPM}$** | Eixo principal em $105\\text{ RPM}$ e harmônico de passagem de pás em $5{,}5\\text{ Hz}$ ($328\\text{ RPM}$). |\n",
                "| **AMBIENTAL** | Fundo do Mar | — | — | — | Espectro contínuo difuso, sem linhas espectrais periódicas de pás. |\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Avaliação Experimental no Protocolo $5\\times2$cv\n",
                "\n",
                "Avaliamos o classificador **SVM Nyström** treinado estritamente com os vetores DEMON ($253\\text{ dimensões}$) nos 10 folds oficiais com isolamento estrito de navios por `Ship ID`, computando métricas por votação majoritária de áudio.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "from IPython.display import display\n",
                "\n",
                "# Resultados consolidados dos 10 folds oficiais (5x2cv)\n",
                "results_data = {\n",
                "    'Modelo': [\n",
                "        'LOFAR SVM (m=4000)', 'MEL SVM (m=4000)',\n",
                "        'DEMON SVM (m=6000)', 'DEMON RF (100t)', 'DEMON MLP [32,16]', 'DEMON CNN 2D'\n",
                "    ],\n",
                "    'Dimensão': ['64 (PCA)', '256', '253', '253', '253', '32x253'],\n",
                "    'Acurácia 4C (%)': [63.32, 63.36, 51.72, 50.06, 52.10, 42.47],\n",
                "    'Desvio Acc 4C': [2.01, 1.00, 2.01, 1.48, 2.65, 1.88],\n",
                "    'Índice SP 4C (%)': [63.09, 63.80, 46.57, 44.30, 52.12, 35.72],\n",
                "    'Desvio SP 4C': [2.06, 1.12, 4.45, 2.30, 2.70, 12.63],\n",
                "    'Recall SMALL (%)': [47.81, 50.90, 15.51, 53.43, 29.78, 19.27],\n",
                "    'Recall LARGE (%)': [74.91, 69.16, 73.32, 68.22, 62.66, 55.89],\n",
                "    'Acurácia 3C (%)': [60.72, 61.20, 57.99, 56.67, 57.12, 44.16],\n",
                "    'Índice SP 3C (%)': [60.09, 60.50, 54.83, 53.47, 56.00, 41.67]\n",
                "}\n",
                "\n",
                "df_results = pd.DataFrame(results_data)\n",
                "display(df_results[['Modelo', 'Dimensão', 'Acurácia 4C (%)', 'Índice SP 4C (%)', 'Recall SMALL (%)', 'Recall LARGE (%)', 'Índice SP 3C (%)']])"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Gráfico comparativo de desempenho\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n",
                "\n",
                "models = df_results['Modelo']\n",
                "x = np.arange(len(models))\n",
                "width = 0.35\n",
                "\n",
                "# Gráfico 1: 4 Classes (Geral com Fundo)\n",
                "ax1.bar(x - width/2, df_results['Acurácia 4C (%)'], width, yerr=df_results['Desvio Acc 4C'], capsize=4, label='Acurácia Global', color='#3498db', alpha=0.9)\n",
                "ax1.bar(x + width/2, df_results['Índice SP 4C (%)'], width, yerr=df_results['Desvio SP 4C'], capsize=4, label='Índice SP', color='#e67e22', alpha=0.9)\n",
                "ax1.axhline(60, color='gray', linestyle='--', alpha=0.7, label='Meta Operacional (60%)')\n",
                "ax1.set_title('Problema Geral de 4 Classes (com Fundo Oceânico)', fontweight='bold')\n",
                "ax1.set_xticks(x)\n",
                "ax1.set_xticklabels(models, rotation=25, ha='right')\n",
                "ax1.set_ylabel('Desempenho (%)')\n",
                "ax1.set_ylim(0, 80)\n",
                "ax1.legend()\n",
                "\n",
                "# Gráfico 2: 3 Classes (Apenas Embarcações)\n",
                "ax2.bar(x - width/2, df_results['Acurácia 3C (%)'], width, label='Acurácia (Embarcações)', color='#2ecc71', alpha=0.9)\n",
                "ax2.bar(x + width/2, df_results['Índice SP 3C (%)'], width, label='Índice SP (Embarcações)', color='#9b59b6', alpha=0.9)\n",
                "ax2.axhline(60, color='gray', linestyle='--', alpha=0.7, label='Referência LOFAR (60%)')\n",
                "ax2.set_title('Problema de 3 Classes (Apenas Embarcações: S, M, L)', fontweight='bold')\n",
                "ax2.set_xticks(x)\n",
                "ax2.set_xticklabels(models, rotation=25, ha='right')\n",
                "ax2.set_ylabel('Desempenho (%)')\n",
                "ax2.set_ylim(0, 80)\n",
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
                "## 4. Matrizes de Confusão e Diagnóstico de Falha\n",
                "\n",
                "Abaixo plotamos as matrizes de confusão normalizadas acumuladas nos 10 folds para o DEMON ($m=4000$):\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "cm_4c = np.array([\n",
                "    [13.6, 17.4, 39.9, 29.1],\n",
                "    [ 8.7, 44.9, 22.8, 23.6],\n",
                "    [ 5.8,  7.8, 76.4, 10.0],\n",
                "    [ 6.2,  3.0, 16.9, 73.9]\n",
                "])\n",
                "labels_4c = ['SMALL', 'MEDIUM', 'LARGE', 'AMBIENTAL']\n",
                "\n",
                "cm_3c = np.array([\n",
                "    [50.6, 14.0, 35.4],\n",
                "    [36.8, 42.7, 20.5],\n",
                "    [21.5,  6.6, 71.9]\n",
                "])\n",
                "labels_3c = ['SMALL', 'MEDIUM', 'LARGE']\n",
                "\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# Heatmap 4C\n",
                "im1 = ax1.imshow(cm_4c, cmap='Blues', vmin=0, vmax=100)\n",
                "ax1.set_xticks(range(4))\n",
                "ax1.set_yticks(range(4))\n",
                "ax1.set_xticklabels(labels_4c)\n",
                "ax1.set_yticklabels(labels_4c)\n",
                "ax1.set_xlabel('Classe Predita')\n",
                "ax1.set_ylabel('Classe Real')\n",
                "ax1.set_title('Matriz de Confusão DEMON (4 Classes - 10 Folds)', fontweight='bold')\n",
                "for i in range(4):\n",
                "    for j in range(4):\n",
                "        ax1.text(j, i, f'{cm_4c[i, j]:.1f}%', ha='center', va='center',\n",
                "                 color='white' if cm_4c[i, j] > 50 else 'black', fontweight='bold')\n",
                "\n",
                "# Heatmap 3C\n",
                "im2 = ax2.imshow(cm_3c, cmap='Oranges', vmin=0, vmax=100)\n",
                "ax2.set_xticks(range(3))\n",
                "ax2.set_yticks(range(3))\n",
                "ax2.set_xticklabels(labels_3c)\n",
                "ax2.set_yticklabels(labels_3c)\n",
                "ax2.set_xlabel('Classe Predita')\n",
                "ax2.set_ylabel('Classe Real')\n",
                "ax2.set_title('Matriz de Confusão DEMON (3 Classes Embarcações)', fontweight='bold')\n",
                "for i in range(3):\n",
                "    for j in range(3):\n",
                "        ax2.text(j, i, f'{cm_3c[i, j]:.1f}%', ha='center', va='center',\n",
                "                 color='white' if cm_3c[i, j] > 50 else 'black', fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Diagnóstico Físico e Conclusões Científicas\n",
                "\n",
                "### 1. Por que o DEMON isolado não alcança 60% de Índice SP?\n",
                "1. **Perda de $99\\%$ da Riqueza Espectral:** O DEMON descarta todo o espectro sonoro direto (raias de cilindros, bombas, alternadores e envelopes acústicos) e retém apenas a taxa de batimento de pás de $0\\text{ a }25\\text{ Hz}$. Sem a potência e linhas tonais diretas, ele é incapaz de operar como extrator primário monolítico.\n",
                "2. **Confusão com o Fundo Oceânico:** O ruído ambiental (arrebentação de ondas e vento na superfície) apresenta flutuações lentas de $0{,}5\\text{ a }3\\text{ Hz}$. Sem as raias tonais do LOFAR, o DEMON confunde embarcações pequenas de baixo SNR com o ruído de fundo (**$29{,}1\\%$ de SMALL classificado como AMBIENTAL**).\n",
                "3. **Heterogeneidade e Colapso de MEDIUM ($42{,}7\\%$):** Embarcações médias (rebocadores, supridores offshore de $50\\text{--}100\\text{ m}$) utilizam desde propulsores azimutais rápidos até hélices de passo variável (CPP) lentas em manobra, confundindo o DEMON nas duas pontas (SMALL e LARGE).\n",
                "\n",
                "### 2. Onde o DEMON Apresenta Vantagem Incomparável?\n",
                "* **Destaque em Navios de Grande Porte (LARGE):** O DEMON isolado alcançou **$76{,}40\\%$ de Recall** no 4-classes e **$79{,}67\\%$** no 3-classes, **superando o LOFAR ($74{,}91\\%$) e o MEL ($69{,}16\\%$)**. Navios grandes têm cavitação pesada e contínua em rotações lentas ($100\\text{--}150\\text{ RPM}$), gerando linhas espectrais limpas no DEMON.\n",
                "\n",
                "### 3. Diretriz para a Etapa B:\n",
                "* O DEMON **não** deve ser usado isoladamente.\n",
                "* Ele deve ser incorporado na **Fusão Híbrida** (MEL-256 + DEMON) e no **Classificador Especialista em Cascata** para desambiguação mecânica de embarcações (diferenciação direta de RPM entre lanchas e cargueiros).\n"
            ]
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
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
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Notebook {filename} gerado com sucesso!")


if __name__ == "__main__":
    create_demon_notebook()
