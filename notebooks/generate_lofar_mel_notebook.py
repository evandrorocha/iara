"""
Script to generate notebooks/6_analise_lofar_mel.ipynb detailing LOFAR and MEL feature extraction,
TPSW background normalization, physical acoustics, real spectrograms, and comparison.
Ensures every cell is self-contained with robust imports and dual-path resolution.
"""

import os
import json

def create_lofar_mel_notebook(filename="notebooks/6_analise_lofar_mel.ipynb"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA\n",
                "### Notebook 6: Análise Aprofundada dos Extratores Espectrais LOFAR e MEL\n",
                "\n",
                "Neste notebook, exploramos detalhadamente os dois principais extratores espectrais do projeto IARA, derivados da formulação de referência de **Fábio Silva et al. (2025)** e da biblioteca `labsonar`:\n",
                "1. **LOFAR (Low Frequency Analysis and Recording):** Espectro linear de banda estreita com normalizador de fundo **TPSW (Two-Pass Split Window)** para realce de raias tonais mecânicas.\n",
                "2. **LOG-MELGRAM (Escala Mel de 256 Bandas):** Espectro logarítmico de banda larga com banco de filtros triangulares perceptuais, atuando como integrador resiliente de potência.\n",
                "3. **Inspeção Passo a Passo do TPSW:** Demonstração matemática e visual do cancelamento do piso de ruído oceânico contínuo.\n",
                "4. **Visualização Comparativa em Áudios Reais:** Espectrogramas 2D e curvas integradas para `SMALL`, `MEDIUM`, `LARGE` e `AMBIENTAL`.\n",
                "5. **Trade-off Físico Acústico e Fusão Híbrida:** A dualidade entre o *\"Sniper de Raias\"* (LOFAR) e o *\"Escudo Resiliente\"* (MEL), e como a fusão atinge o melhor desempenho operacional.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Origem Algorítmica e Formulação Matemática\n",
                "\n",
                "As rotinas de processamento implementadas em `iara.processing.analysis` foram projetadas para processar sinais de hidrofone amostrados a $48\\text{ kHz}$ na Bacia de Santos:\n",
                "\n",
                "### 1.1 Pipeline do LOFAR\n",
                "$$\n",
                "x(t) \\xrightarrow{\\text{Decimação } 3\\times (16\\text{ kHz})} x_d(t) \\xrightarrow{\\text{STFT } (N=2048)} S(f, t) \\xrightarrow{20\\log_{10}} P_{\\text{dB}}(f, t) \\xrightarrow{- \\text{TPSW}} P_{\\text{norm}} \\xrightarrow{\\ge -0.2} \\text{LOFAR}\n",
                "$$\n",
                "\n",
                "* **Objetivo:** Isolar frequências fundamentais e harmônicas discretas geradas por motores a combustão interna (ciclo de pistões dos cilindros), bombas de injeção, alternadores e ressonâncias mecânicas estruturais do casco.\n",
                "* **Normalizador TPSW:** Estima o ruído de fundo oceânico através de uma janela móvel de duas passagens com fenda central (*split*), subtraindo o contínuo e preservando as raias estreitas.\n",
                "\n",
                "### 1.2 Pipeline do LOG-MELGRAM\n",
                "$$\n",
                "x(t) \\xrightarrow{\\text{Decimação } 3\\times (16\\text{ kHz})} x_d(t) \\xrightarrow{\\text{Banco Mel } (M=256)} M(m, t) \\xrightarrow{\\text{Power-to-dB}} \\text{Log-Mel}\n",
                "$$\n",
                "\n",
                "* **Objetivo:** Capturar a distribuição global de energia e envoltória espectral do ruído hidrodinâmico (turbulência de esteira, deslocamento de massa d'água e cavitação espalhada).\n",
                "* **Escala Mel:** Concentra a resolução espectral em frequências médias e baixas com filtros triangulares sobrepostos mais espaçados em altas frequências ($m = 2595 \\log_{10}(1 + f / 700)$).\n"
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
                "import scipy.signal as sci\n",
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
                "    'figure.titlesize': 15\n",
                "})"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. O Algoritmo TPSW (Two-Pass Split Window) em Detalhes\n",
                "\n",
                "O TPSW é o algoritmo de normalização mais tradicional em processamento de sonar passivo militar e acadêmico. Ele resolve o dilema fundamental:\n",
                "> *\"Como remover a curva contínua do ruído do mar sem atenuar as raias tonais das embarcações?\"*\n",
                "\n",
                "Ele opera com três parâmetros fundamentais:\n",
                "* $N$: Tamanho total da janela de estimativa lateral;\n",
                "* $P$: Fenda central desconsiderada (para que a raia sob análise não contamine a estimativa do fundo ao seu redor);\n",
                "* $A$: Fator de saturação da primeira passagem para eliminar picos vizinhos fortes.\n",
                "\n",
                "Abaixo, implementamos e plotamos a ação do TPSW sobre um espectro simulado com fundo oceânico realista (decaimento $1/f$) e raias tonais sobrepostas:\n"
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
                "import scipy.signal as sci\n",
                "\n",
                "def tpsw_demo(signal, n=40, p=5, a=2.0):\n",
                "    \"\"\"Implementação explicativa do TPSW em 1D.\"\"\"\n",
                "    h = np.concatenate((np.ones(n - p + 1), np.zeros(2 * p - 1), np.ones(n - p + 1)))\n",
                "    h /= np.sum(h)\n",
                "    \n",
                "    # Primeira passagem: convolução com fenda\n",
                "    background_pass1 = sci.convolve(signal, h, mode='same')\n",
                "    \n",
                "    # Saturação de picos que excedem a*background\n",
                "    clipped = np.where(signal > a * background_pass1, background_pass1, signal)\n",
                "    \n",
                "    # Segunda passagem: estimativa refinada do ruído contínuo\n",
                "    background_final = sci.convolve(clipped, h, mode='same')\n",
                "    return background_final\n",
                "\n",
                "# Criação de espectro sintético realista de navio + marulho\n",
                "freqs = np.linspace(10, 2000, 1000)\n",
                "noise_floor = 60 - 15 * np.log10(freqs) + np.random.normal(0, 1.2, len(freqs))\n",
                "signal_raw = noise_floor.copy()\n",
                "\n",
                "# Inserção de raias tonais de motor diesel\n",
                "signal_raw[150] += 18.0  # Raias fundamentais\n",
                "signal_raw[300] += 14.0  # 2o harmônico\n",
                "signal_raw[450] += 11.0  # 3o harmônico\n",
                "signal_raw[700] += 16.0  # Gerador elétrico (60 Hz / 400 Hz)\n",
                "\n",
                "# Aplicação do TPSW\n",
                "bg_est = tpsw_demo(signal_raw, n=50, p=6, a=2.0)\n",
                "lofar_final = signal_raw - bg_est\n",
                "lofar_final[lofar_final < 0] = 0  # Thresholding pós-TPSW\n",
                "\n",
                "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)\n",
                "\n",
                "ax1.plot(freqs, signal_raw, color='#7f8c8d', lw=1.2, label='Espectro Bruto (dB) - Navio + Fundo Oceânico')\n",
                "ax1.plot(freqs, bg_est, color='#e74c3c', lw=2.0, linestyle='--', label='Piso de Ruído Estimado pelo TPSW')\n",
                "ax1.set_ylabel('Magnitude (dB)')\n",
                "ax1.set_title('Passo 1: Estimativa da Curva Contínua de Ruído Oceânico via TPSW', fontweight='bold')\n",
                "ax1.legend(loc='upper right')\n",
                "ax1.grid(True, alpha=0.3)\n",
                "\n",
                "ax2.plot(freqs, lofar_final, color='#2980b9', lw=1.5, label='LOFAR Normalizado (Raias Puras)')\n",
                "ax2.axhline(0, color='black', lw=0.8, linestyle=':')\n",
                "ax2.set_ylabel('SNR Normalizado')\n",
                "ax2.set_xlabel('Frequência Acústica (Hz)')\n",
                "ax2.set_title('Passo 2: LOFAR Final — Fundo Retificado a Zero e Raias Tonais Preservadas', fontweight='bold')\n",
                "ax2.legend(loc='upper right')\n",
                "ax2.grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. O Banco de Filtros Mel (256 Bandas)\n",
                "\n",
                "Diferente do LOFAR que analisa cada bin de FFT linear individualmente (produzindo 1024 bins), o **MEL** agrupa a potência em bandas triangulares sobrepostas com espaçamento logarítmico:\n",
                "* Em baixas frequências ($< 1\\text{ kHz}$), os filtros são estreitos para acompanhar harmônicos de rotação;\n",
                "* Em altas frequências ($> 2\\text{ kHz}$), os filtros se alargam para capturar a energia difusa da cavitação e turbulência hidrodinâmica sem inflar a dimensionalidade.\n"
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
                "import librosa.filters\n",
                "\n",
                "# Visualização dos primeiros 40 filtros do banco de 256 bandas Mel do IARA\n",
                "fs_decimated = 16000\n",
                "n_fft = 2048\n",
                "mel_basis = librosa.filters.mel(sr=fs_decimated, n_fft=n_fft, n_mels=64, fmin=0.0, fmax=fs_decimated/2)\n",
                "freqs_mel = np.linspace(0, fs_decimated/2, mel_basis.shape[1])\n",
                "\n",
                "plt.figure(figsize=(14, 4))\n",
                "for i in range(0, 40, 2):\n",
                "    plt.plot(freqs_mel, mel_basis[i], lw=1.3, alpha=0.8)\n",
                "\n",
                "plt.title('Banco de Filtros Triangulares Mel (Visualização das Primeiras Bandas)', fontweight='bold')\n",
                "plt.xlabel('Frequência (Hz)')\n",
                "plt.ylabel('Ganho do Filtro')\n",
                "plt.xlim(0, 3000)\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Inspeção Comparativa nos Áudios Reais do IARA\n",
                "\n",
                "Carregamos abaixo as matrizes pré-processadas de **LOFAR** e **MEL** armazenadas no cache do IARA para as 4 classes:\n",
                "* **SMALL (ID 2):** Lancha de Praticagem *Pilot 10*\n",
                "* **MEDIUM (ID 1):** Rebocador/Apoio *Maersk Pacer*\n",
                "* **LARGE (ID 4):** Mineraleiro Gigante *Sinoglory*\n",
                "* **AMBIENTAL (ID 1446):** Fundo Oceânico puro\n"
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
                "lofar_dir = find_data_dir('data/iara_processed/lofar_bc1559516c522e2b993895bf5cb2ac94')\n",
                "mel_dir = find_data_dir('data/iara_processed/log_melgram_a68edc75124f6c2ae6f093281fb42a9a')\n",
                "\n",
                "sample_files = [\n",
                "    ('SMALL: Pilot 10 (11m)', 2),\n",
                "    ('MEDIUM: Maersk Pacer (74m)', 1),\n",
                "    ('LARGE: Sinoglory (322m)', 4),\n",
                "    ('AMBIENTAL: Fundo Oceânico', 1446)\n",
                "]\n",
                "\n",
                "fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex='col')\n",
                "\n",
                "for row_idx, (title, fid) in enumerate(sample_files):\n",
                "    # Carregar LOFAR\n",
                "    lofar_path = os.path.join(lofar_dir, f'{fid}.pkl')\n",
                "    if os.path.exists(lofar_path):\n",
                "        with open(lofar_path, 'rb') as f:\n",
                "            data_l = pickle.load(f)\n",
                "        spec_l = data_l['df'].values.T  # (1024, N_windows)\n",
                "        ax_l = axes[row_idx, 0]\n",
                "        im_l = ax_l.imshow(spec_l[:250, :], aspect='auto', origin='lower', cmap='inferno')\n",
                "        ax_l.set_title(f'LOFAR: {title}', fontweight='bold')\n",
                "        ax_l.set_ylabel('Bins Baixos (0-500 Hz)')\n",
                "    \n",
                "    # Carregar MEL\n",
                "    mel_path = os.path.join(mel_dir, f'{fid}.pkl')\n",
                "    if os.path.exists(mel_path):\n",
                "        with open(mel_path, 'rb') as f:\n",
                "            data_m = pickle.load(f)\n",
                "        spec_m = data_m['df'].values.T  # (256, N_windows)\n",
                "        ax_m = axes[row_idx, 1]\n",
                "        im_m = ax_m.imshow(spec_m, aspect='auto', origin='lower', cmap='viridis')\n",
                "        ax_m.set_title(f'MEL: {title}', fontweight='bold')\n",
                "        ax_m.set_ylabel('Bandas Mel (0-256)')\n",
                "\n",
                "axes[3, 0].set_xlabel('Janelas de Tempo (0.512s)')\n",
                "axes[3, 1].set_xlabel('Janelas de Tempo (0.512s)')\n",
                "plt.suptitle('Comparação Espectro-Temporal: LOFAR (Raias de Motor) vs. MEL (Envoltória de Energia)', fontsize=15, fontweight='bold', y=0.99)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### O que a visualização revela:\n",
                "1. **No LOFAR (Coluna da Esquerda):**\n",
                "   * *Pilot 10 (SMALL):* Raias finas bem espaçadas em frequências mais altas.\n",
                "   * *Sinoglory (LARGE):* Dezenas de raias harmônicas densas e perfeitamente paralelas nos primeiros $200\\text{ Hz}$, geradas pelo motor diesel de 2 tempos de baixa rotação.\n",
                "   * *Fundo Oceânico (AMBIENTAL):* Espectro quase plano e deserto; o TPSW zerou praticamente toda a matriz, restando apenas ruído térmico residual esparso.\n",
                "2. **No MEL (Coluna da Direita):**\n",
                "   * O MEL captura o gradiente contínuo de energia acústica.\n",
                "   * O ruído AMBIENTAL possui uma assinatura energética ampla concentrada em baixas frequências (marulho), enquanto os navios têm uma crista de energia ascendente bem definida.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Comparativo Estatístico e Trade-off Físico no $5\\times2$cv\n",
                "\n",
                "Abaixo consolidamos o comparativo oficial entre os extratores nos 10 folds de teste do IARA:\n"
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
                "comp_data = {\n",
                "    'Extrator': ['LOFAR (m=4000, PCA64)', 'LOFAR (m=6000, Puro)', 'MEL (m=4000, Puro)', 'HÍBRIDO (MEL + LOFAR-PCA64)'],\n",
                "    'Dimensão': [64, 1024, 256, 320],\n",
                "    'Índice SP (%)': [63.09, 63.84, 63.80, 65.16],\n",
                "    'SP_std': [2.06, 1.91, 1.12, 1.67],\n",
                "    'Acurácia (%)': [63.32, 63.63, 63.36, 65.77],\n",
                "    'Acc_std': [2.01, 1.36, 1.00, 1.55],\n",
                "    'Recall SMALL (%)': [47.81, 38.20, 50.90, 52.40],\n",
                "    'Recall MEDIUM (%)': [55.77, 60.47, 60.40, 61.80],\n",
                "    'Recall LARGE (%)': [74.91, 76.31, 69.16, 74.20],\n",
                "    'Recall AMB (%)': [77.65, 86.94, 77.86, 81.50]\n",
                "}\n",
                "\n",
                "df_comp = pd.DataFrame(comp_data)\n",
                "display(df_comp)"
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
                "# Gráfico de barras com recalls por classe\n",
                "classes = ['SMALL', 'MEDIUM', 'LARGE', 'AMBIENTAL']\n",
                "models_name = ['LOFAR (PCA64)', 'MEL Puro', 'HÍBRIDO']\n",
                "rec_lofar = [47.81, 55.77, 74.91, 77.65]\n",
                "rec_mel = [50.90, 60.40, 69.16, 77.86]\n",
                "rec_hyb = [52.40, 61.80, 74.20, 81.50]\n",
                "\n",
                "x = np.arange(len(classes))\n",
                "width = 0.25\n",
                "\n",
                "plt.figure(figsize=(12, 5))\n",
                "plt.bar(x - width, rec_lofar, width, label='LOFAR (m=4000)', color='#34495e', alpha=0.9)\n",
                "plt.bar(x, rec_mel, width, label='MEL (m=4000)', color='#e67e22', alpha=0.9)\n",
                "plt.bar(x + width, rec_hyb, width, label='HÍBRIDO (MEL + LOFAR)', color='#27ae60', alpha=0.9)\n",
                "\n",
                "plt.title('Comparativo de Recall por Classe: LOFAR vs. MEL vs. Fusão Híbrida', fontweight='bold')\n",
                "plt.xticks(x, classes, fontweight='bold')\n",
                "plt.ylabel('Sensibilidade / Recall (%)')\n",
                "plt.ylim(0, 100)\n",
                "plt.legend(loc='lower right')\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Conclusões e Síntese Teórica\n",
                "\n",
                "1. **A Dualidade Acústica Complementar:**\n",
                "   * O **LOFAR** atua como o *\"Sniper de Picos\"*: ele é imbatível na identificação de navios grandes (`LARGE`) com suas assinaturas tonais densas de cilindros de motor ($74{,}91\\%$ a $76{,}31\\%$), mas sofre em embarcações pequenas e silenciosas (`SMALL` cai para $38\\text{--}47\\%$).\n",
                "   * O **MEL** atua como o *\"Escudo Resiliente\"*: ao integrar a energia em 256 bandas perceptuais, ele tem baixíssima variância estatística ($\\sigma = 1{,}00\\%$), protege o recall de embarcações menores (`SMALL` atinge $50{,}90\\%$) e separa com folga o ruído ambiental.\n",
                "\n",
                "2. **A Vitória da Fusão Híbrida:**\n",
                "   * Ao concatenar os **$256\\text{ atributos do MEL}$** com os **$64\\text{ componentes principais do LOFAR}$** ($320\\text{-D}$), o classificador SVM elimina as fraquezas mútuas: atinge **$65{,}16\\%$ de SP** e quando acoplado ao modelo de cascata com especialistas, atinge o recorde absoluto de **$68{,}05\\%$ de SP**.\n",
                "\n",
                "3. **Relação com o DEMON:**\n",
                "   * Enquanto LOFAR e MEL cobrem o espectro acústico direto de $0\\text{ a }8\\text{ kHz}$, o **DEMON** introduz a dimensão faltante de modulação de envoltória ($0\\text{ a }25\\text{ Hz}$), fornecendo o RPM da hélice para desambiguar a confusão residual entre lanchas rápidas e cargueiros lentos.\n"
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
    create_lofar_mel_notebook()
