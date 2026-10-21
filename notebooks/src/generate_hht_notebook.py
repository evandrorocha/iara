"""
Script to generate notebooks/9_analise_hht_teste_cego.ipynb with complete analysis, plots, and metrics.
Covers:
1. HHT theory and mathematical foundations (EMD, IMFs, Hilbert Transform, Marginal Spectrum).
2. Distinction between HHT Pura (autonomous category) vs. LOFAR vs. MEL vs. DEMON.
3. Blind test analysis across the 4 canonical IARA audios (SMALL, MEDIUM, LARGE, AMBIENTAL).
4. Physical diagnosis: Mode mixing, noise sensitivity, and VMD (Variational Mode Decomposition).
5. Quantitative comparison tables and interactive plotting cells.
"""

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iara_results as R  # noqa: E402  (módulo único de métricas)

REPO_ROOT = R.ROOT

# Experimentos HHT em results/trainings/multirep (scratch/eval_hht_models_5x2cv.py)
EXP_RF_4C, EXP_RF_3C = 'rf_hht_n100_balanced_4c', 'rf_hht_n100_balanced_3c'
EXP_MLP_4C, EXP_MLP_3C = 'mlp_hht_h32-16_e30_noseed_4c', 'mlp_hht_h32-16_e30_noseed_3c'
EXP_SVM_4C, EXP_SVM_3C = 'svm_nystroem_6000_hht_C2.0_elasticnet_l1r0.15_4c', 'svm_nystroem_6000_hht_C2.0_elasticnet_l1r0.15_3c'
EXP_CNN_4C, EXP_CNN_3C = 'cnn2d_hht_w32_b32_e25_4c', 'cnn2d_hht_w32_b32_e25_3c'
EXP_MLP_SEED = 'mlp_hht_h32-16_e30_4c'          # semente 42 (gate), outro experimento
EXP_RF_LOFAR = 'rf_lofar5s_n100_balanced_4c'     # RF LOFAR local (cache de 5,12 s)
HHT_EXPS_4C = [(EXP_RF_4C, '4c'), (EXP_MLP_4C, '4c'), (EXP_SVM_4C, '4c'), (EXP_CNN_4C, '4c')]
HHT_EXPS_3C = [(EXP_RF_3C, '3c'), (EXP_MLP_3C, '3c'), (EXP_SVM_3C, '3c'), (EXP_CNN_3C, '3c')]


def pt(x, nd=2):
    return f"{x:.{nd}f}".replace('.', ',')


def fmtc(ms, nd=2):
    return f"{pt(ms[0], nd)} ± {pt(ms[1], nd)}"


def pval(p):
    return 'p < 0,001' if p < 0.001 else f"p = {pt(p, 3)}"

def create_hht_notebook(filename="notebooks/9_analise_hht_teste_cego.ipynb"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA\n",
                "### Notebook 9: Investigação da Transformada de Hilbert-Huang (HHT) e Teste Cego\n",
                "\n",
                "Neste notebook, investigamos a aplicação da **Transformada de Hilbert-Huang (HHT)** (*Huang et al.*, 1998) em sinais de áudio de sonar passivo na Bacia de Santos (acervo UO do dataset IARA). Avaliamos:\n",
                "\n",
                "1. **Fundamentação Matemática e Física:** Decomposição adaptativa não-linear e não-estacionária orientada a dados via **EMD** (*Empirical Mode Decomposition*) e frequências instantâneas via **Transformada de Hilbert**.\n",
                "2. **Taxonomia de Sonar Passivo:** Esclarecimento definitivo da HHT como uma **categoria independente de representação** (concorrente do **LOFAR** e do **MEL**), distinta do **DEMON** (que é um pipeline físico de demodulação de cavitação).\n",
                "3. **Teste Cego com os 4 Áudios Canônicos:** Avaliação empírica minuciosa em gravações reais representativas:\n",
                "   - `A-0002.wav`: **SMALL** (*Pilot 10*, 11m, lancha rápida de praticagem);\n",
                "   - `A-0001.wav`: **MEDIUM** (*Maersk Pacer*, 74m, rebocador/supridor offshore);\n",
                "   - `A-0004.wav`: **LARGE** (*Sinoglory*, 322m, graneleiro pesado de grande porte);\n",
                "   - `E-1446.wav`: **AMBIENTAL** (Ruído de fundo marinho puro).\n",
                "4. **Diagnóstico dos Fenômenos Observados:** Fenômeno de *Mode Mixing* (mistura de modos), modos induzidos por ruído no EMD e estabilidade espectral.\n",
                "5. **VMD (Variational Mode Decomposition) sobre a envoltória demodulada:** frequências centrais dos modos encontradas pelo VMD no áudio SMALL (valores na Seção 5, calculados na execução) e o que se pode — e o que não se pode — concluir delas.\n",
                "6. **Diretrizes para Machine Learning:** Justificativa da dimensionalidade recomendada de **$256\\text{ bins}$** e perspectivas para classificação supervisionada.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Fundamentos Teóricos da Transformada de Hilbert-Huang (HHT)\n",
                "\n",
                "A Transformada de Fourier tradicional (e a STFT do LOFAR) assume linearidade e periodicidade estritamente estacionária por janelas, sofrendo com o princípio de incerteza de Gabor-Heisenberg ($\\Delta f \\cdot \\Delta t \\ge 1/2$).\n",
                "\n",
                "A **HHT** contorna essas restrições operando em duas etapas sequenciais:\n",
                "\n",
                "### 1.1. Decomposição em Modos Empíricos (EMD)\n",
                "O sinal temporal $x(t)$ é decomposto em uma soma finita de $K$ **Funções de Modo Intrínseco** (IMFs — *Intrinsic Mode Functions*) e um resíduo monótono ou de baixa frequência $r(t)$:\n",
                "\n",
                "$$\n",
                "x(t) = \\sum_{k=1}^K \\text{IMF}_k(t) + r(t)\n",
                "$$\n",
                "\n",
                "Cada IMF satisfaz duas condições essenciais:\n",
                "1. O número de extremos (máximos e mínimos) e o número de cruzamentos por zero diferem no máximo por um em toda a duração;\n",
                "2. O valor médio dos envelopes superior $e_{\\text{up}}(t)$ e inferior $e_{\\text{low}}(t)$, construídos por interpolação de splines cúbicas dos extremos locais, é zero em qualquer ponto.\n",
                "\n",
                "### 1.2. Transformada de Hilbert e Espectro Marginal (HMS)\n",
                "Para cada $\\text{IMF}_k(t)$, calcula-se o sinal analítico complexo:\n",
                "\n",
                "$$\n",
                "z_k(t) = \\text{IMF}_k(t) + j \\mathcal{H}\\{\\text{IMF}_k(t)\\} = a_k(t) e^{j \\theta_k(t)}\n",
                "$$\n",
                "\n",
                "onde:\n",
                "- **Amplitude Instantânea:** $a_k(t) = \\sqrt{\\text{IMF}_k^2(t) + \\mathcal{H}^2\\{\\text{IMF}_k(t)\\}}$\n",
                "- **Fase Instantânea:** $\\theta_k(t) = \\text{unwrap}\\left(\\arctan\\left(\\frac{\\mathcal{H}\\{\\text{IMF}_k(t)\\}}{\\text{IMF}_k(t)}\\right)\\right)$\n",
                "- **Frequência Instantânea:** $f_k(t) = \\frac{1}{2\\pi} \\frac{d\\theta_k(t)}{dt}$\n",
                "\n",
                "O **Espectro Marginal de Hilbert (HMS)** acumula a densidade de energia associada a cada frequência instantânea ao longo do tempo $T$:\n",
                "\n",
                "$$\n",
                "h(f) = \\int_0^T H(f, t) \\, dt\n",
                "$$\n",
                "\n",
                "Fornecendo uma medida de densidade de potência sem os limites de discretização da FFT.\n"
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
                "import time\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import scipy.signal as scipy_sig\n",
                "import scipy.io.wavfile as wavfile\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Configuração de caminhos\n",
                "ROOT_DIR = os.path.abspath('..') if os.path.exists('../data') else os.path.abspath('.')\n",
                "sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))\n",
                "\n",
                "# Garantir que PyEMD e vmdpy estão instalados no kernel\n",
                "try:\n",
                "    from PyEMD import EMD\n",
                "    from vmdpy import VMD\n",
                "except ImportError:\n",
                "    print(\"Instalando bibliotecas de análise modal (EMD-signal e vmdpy)...\", flush=True)\n",
                "    import subprocess\n",
                "    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"EMD-signal\", \"vmdpy\"])\n",
                "    from PyEMD import EMD\n",
                "    from vmdpy import VMD\n",
                "\n",
                "# Estilo visual dos gráficos\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "plt.rcParams.update({\n",
                "    'font.family': 'sans-serif',\n",
                "    'font.size': 11,\n",
                "    'axes.labelsize': 12,\n",
                "    'axes.titlesize': 13,\n",
                "    'figure.titlesize': 15\n",
                "})\n",
                "\n",
                "print(f\"[OK] Ambiente configurado com sucesso! ROOT_DIR: {ROOT_DIR}\")\n",
                "print(f\"[OK] PyEMD e vmdpy carregados e disponíveis para análise modal adaptativa.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Taxonomia Comparativa: HHT vs. LOFAR vs. MEL vs. DEMON\n",
                "\n",
                "| Categoria | Tipo de Processamento | Base Matemática | Faixa Espectral Típica | Alvo Físico Primário |\n",
                "| :--- | :---: | :--- | :---: | :--- |\n",
                "| **LOFAR** | *Representação* | STFT (Fourier) + TPSW | $7{,}8\\text{--}8000\\text{ Hz}$ (1024 bins, $\\Delta f = 7{,}81$ Hz) | Raias tônicas estreitas de eixos, bombas e motores diesel. |\n",
                "| **MEL** | *Representação* | STFT em Escala Mel | $0\\text{--}8\\text{ kHz}$ (após decimação para 16 kHz) | Energia contínua audível de cavitação e combustão. |\n",
                "| **HHT Pura** | *Representação* | **EMD + Hilbert (Adaptativo)** | $0\\text{--}1000\\text{ Hz}$ | **Modos intrínsecos de oscilação e frequências instantâneas.** |\n",
                "| **DEMON** | *Pipeline Físico* | Chebyshev + Envelope + STFT | $0{,}39\\text{--}25\\text{ Hz}$ | Frequência fundamental do hélice e passagem de pás. |\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Teste Cego: Extração da HHT Pura nos 4 Áudios Canônicos\n",
                "\n",
                "Carregamos as 4 gravações reais representativas e aplicamos a HHT pura diretamente no sinal de pressão acústica (decimado para $f_s = 2000\\text{ Hz}$, cobrindo a banda de $1\\text{ a }1000\\text{ Hz}$). No teste cego diagnótico abaixo, utilizamos alta resolução espectral (**500 bins**, `win_len=2.56s`, `hop=1.28s` e `MAX_ITERATION=100`). Na extração massiva do dataset completo (`cache_hht.pkl`, **256 bins**), utilizamos `win_len=2.56s` com `hop=5.12s` e `MAX_ITERATION=30` para garantir escalabilidade computacional sobre todas as 1.706 gravações. **Atenção:** com passo (5,12 s) igual ao dobro da janela (2,56 s), a extração massiva analisa só metade do áudio: 2,56 s de cada 5,12 s são descartados (`scratch/extract_all_hht.py`).\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "canonical_samples = [\n",
                "    ('ÁUDIO A (SMALL: Pilot 10, 11m)', os.path.join(ROOT_DIR, 'data/iara/A/A-0002.wav'), '#1f77b4'),\n",
                "    ('ÁUDIO B (MEDIUM: Maersk Pacer, 74m)', os.path.join(ROOT_DIR, 'data/iara/A/A-0001.wav'), '#ff7f0e'),\n",
                "    ('ÁUDIO C (LARGE: Sinoglory, 322m)', os.path.join(ROOT_DIR, 'data/iara/A/A-0004.wav'), '#2ca02c'),\n",
                "    ('ÁUDIO D (AMBIENTAL: Fundo do Mar)', os.path.join(ROOT_DIR, 'data/iara/E/E-1446.wav'), '#7f7f7f'),\n",
                "]\n",
                "\n",
                "def extract_hht_pure(filepath, duration_sec=10.0, fs_target=2000, n_bins=500, max_freq_hz=1000.0):\n",
                "    fs, data = wavfile.read(filepath)\n",
                "    if data.ndim > 1: data = data[:, 0]\n",
                "    data = data.astype(np.float32)\n",
                "    \n",
                "    total_sec = len(data) / fs\n",
                "    s_idx = int(max(0, (total_sec - duration_sec) / 2) * fs)\n",
                "    e_idx = s_idx + int(duration_sec * fs)\n",
                "    segment = data[s_idx:e_idx]\n",
                "    \n",
                "    decim = int(round(fs / fs_target))\n",
                "    sig_dec = scipy_sig.decimate(segment, decim, ftype='fir', zero_phase=True)\n",
                "    actual_fs = fs / decim\n",
                "    \n",
                "    sig_dec = sig_dec - np.mean(sig_dec)\n",
                "    std_val = np.std(sig_dec)\n",
                "    if std_val > 0: sig_dec /= std_val\n",
                "    \n",
                "    win_len = int(2.56 * actual_fs)\n",
                "    hop = int(win_len // 2)\n",
                "    n_wins = max(1, (len(sig_dec) - win_len) // hop + 1)\n",
                "    \n",
                "    freq_grid = np.linspace(1.0, max_freq_hz, n_bins)\n",
                "    df = freq_grid[1] - freq_grid[0]\n",
                "    \n",
                "    hms_accum = np.zeros(n_bins)\n",
                "    emd = EMD()\n",
                "    emd.MAX_ITERATION = 100\n",
                "    dt = 1.0 / actual_fs\n",
                "    \n",
                "    first_imfs = None\n",
                "    t0 = time.time()\n",
                "    for w in range(n_wins):\n",
                "        w_sig = sig_dec[w*hop : w*hop + win_len]\n",
                "        if len(w_sig) < win_len: break\n",
                "        imfs = emd(w_sig)\n",
                "        if first_imfs is None: first_imfs = imfs\n",
                "        \n",
                "        for imf in imfs:\n",
                "            analytic = scipy_sig.hilbert(imf)\n",
                "            amp = np.abs(analytic)\n",
                "            phase = np.unwrap(np.angle(analytic))\n",
                "            inst_freq = np.gradient(phase, dt) / (2.0 * np.pi)\n",
                "            \n",
                "            mask = (inst_freq >= 1.0) & (inst_freq <= max_freq_hz)\n",
                "            bin_idx = np.clip(np.floor((inst_freq[mask] - 1.0) / df).astype(int), 0, n_bins - 1)\n",
                "            en = amp[mask] ** 2\n",
                "            for b, e in zip(bin_idx, en):\n",
                "                hms_accum[b] += e\n",
                "                \n",
                "    elapsed = time.time() - t0\n",
                "    if np.max(hms_accum) > 0: hms_accum /= np.max(hms_accum)\n",
                "    \n",
                "    centroid = np.sum(freq_grid * hms_accum) / max(1e-9, np.sum(hms_accum))\n",
                "    cum = np.cumsum(hms_accum) / max(1e-9, np.sum(hms_accum))\n",
                "    rolloff_idx = np.where(cum >= 0.85)[0]\n",
                "    rolloff = freq_grid[rolloff_idx[0]] if len(rolloff_idx) > 0 else max_freq_hz\n",
                "    peak_freq = freq_grid[np.argmax(hms_accum)]\n",
                "    \n",
                "    return {\n",
                "        'freq_grid': freq_grid,\n",
                "        'hms': hms_accum,\n",
                "        'centroid': centroid,\n",
                "        'rolloff': rolloff,\n",
                "        'peak_freq': peak_freq,\n",
                "        'imfs': first_imfs,\n",
                "        'sig_sample': sig_dec[:win_len],\n",
                "        'fs': actual_fs,\n",
                "        'time_elapsed': elapsed\n",
                "    }\n",
                "\n",
                "hht_results = {}\n",
                "print(\"Executando teste cego de HHT Pura (500 bins, MAX_ITERATION=100, 0--1000 Hz)...\\n\")\n",
                "for label, path, color in canonical_samples:\n",
                "    res = extract_hht_pure(path, duration_sec=10.0, n_bins=500)\n",
                "    hht_results[label] = (res, color)\n",
                "    print(f\"--> {label}:\")\n",
                "    print(f\"    Pico Dominante: {res['peak_freq']:.1f} Hz | Centróide: {res['centroid']:.1f} Hz | Rolloff 85%: {res['rolloff']:.1f} Hz | Tempo: {res['time_elapsed']:.2f}s\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.1. Tabela Comparativa de Métricas Espectrais da HHT Pura\n",
                "\n",
                "A tabela abaixo resume os atributos extraídos de forma totalmente cega a partir do Espectro Marginal de Hilbert:\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "table_rows = []\n",
                "for label, (res, _) in hht_results.items():\n",
                "    table_rows.append({\n",
                "        'Áudio / Classe': label.split('(')[0].strip() + ' (' + label.split('(')[1].split(')')[0] + ')',\n",
                "        'Pico Dominante (HMS)': f\"{res['peak_freq']:.1f} Hz\",\n",
                "        'Centróide Espectral': f\"{res['centroid']:.1f} Hz\",\n",
                "        'Rolloff (85% Energia)': f\"{res['rolloff']:.1f} Hz\",\n",
                "        'Total de IMFs': res['imfs'].shape[0],\n",
                "        'Tempo EMD (10s)': f\"{res['time_elapsed']:.2f} s\"\n",
                "    })\n",
                "\n",
                "df_metrics = pd.DataFrame(table_rows)\n",
                "df_metrics"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.2. Visualização: Espectro Marginal de Hilbert (HMS 1D) nos 4 Áudios\n",
                "\n",
                "Plotamos a densidade de energia espectral acumulada pelo HMS ao longo de toda a banda de interesse ($0\\text{ a }1000\\text{ Hz}$):\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)\n",
                "fig.suptitle(\"Teste Cego: Espectro Marginal de Hilbert (HHT Pura - 500 Bins, 0--1000 Hz)\", fontsize=14, fontweight='bold', y=0.99)\n",
                "\n",
                "for i, (label, (res, color)) in enumerate(hht_results.items()):\n",
                "    ax = axes[i]\n",
                "    fg = res['freq_grid']\n",
                "    hms = res['hms']\n",
                "    \n",
                "    ax.plot(fg, hms, color=color, lw=1.6, label=label)\n",
                "    p_idx = np.argmax(hms)\n",
                "    ax.plot(fg[p_idx], hms[p_idx], 'ro', markersize=5)\n",
                "    ax.annotate(f\"Pico: {res['peak_freq']:.1f} Hz\\nCentróide: {res['centroid']:.1f} Hz\",\n",
                "                (fg[p_idx], hms[p_idx]), textcoords=\"offset points\", xytext=(15, -15),\n",
                "                fontsize=8.5, fontweight='bold', color='darkred',\n",
                "                bbox=dict(boxstyle=\"round,pad=0.2\", fc=\"yellow\", alpha=0.7),\n",
                "                arrowprops=dict(arrowstyle=\"->\", color='darkred'))\n",
                "    \n",
                "    ax.axvline(res['rolloff'], color='black', ls=':', lw=1.2, alpha=0.7)\n",
                "    ax.text(res['rolloff'] + 5, 0.7, f\"Rolloff 85%:\\n{res['rolloff']:.0f} Hz\", fontsize=8, color='black', alpha=0.8)\n",
                "    \n",
                "    ax.set_ylabel(\"Energia Normalizada\", fontsize=9)\n",
                "    ax.set_title(f\"{label} | EMD: {res['imfs'].shape[0]} IMFs ({res['time_elapsed']:.2f}s)\", fontsize=10, fontweight='bold')\n",
                "    ax.grid(True, alpha=0.3, ls='--')\n",
                "    ax.set_xlim(0, 1000)\n",
                "    ax.set_ylim(-0.05, 1.15)\n",
                "    ax.legend(loc='upper right', fontsize=8.5)\n",
                "\n",
                "axes[-1].set_xlabel(\"Frequência Instantânea Hilbert (Hz)\", fontsize=11, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.3. Visualização: Comparativo das Primeiras 6 IMFs no Domínio do Tempo\n",
                "\n",
                "Abaixo inspecionamos como o algoritmo EMD decompôs localmente as ondas acústicas de cada classe das oscilações mais rápidas (IMF 1) até as lentas (IMF 6):\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(6, 4, figsize=(18, 11), sharex=True)\n",
                "fig.suptitle(\"Decomposição EMD: Comparativo das Primeiras 6 IMFs (Teste Cego)\", fontsize=13, fontweight='bold', y=0.99)\n",
                "\n",
                "col_names = [\"ÁUDIO A (SMALL)\", \"ÁUDIO B (MEDIUM)\", \"ÁUDIO C (LARGE)\", \"ÁUDIO D (AMBIENTAL)\"]\n",
                "sample_res = list(hht_results.values())[0][0]\n",
                "t_axis = np.arange(len(sample_res['sig_sample'])) / sample_res['fs']\n",
                "\n",
                "for col, (label, (res, color)) in enumerate(hht_results.items()):\n",
                "    imfs = res['imfs']\n",
                "    axes[0, col].set_title(col_names[col], fontsize=10.5, fontweight='bold', color=color)\n",
                "    for row in range(6):\n",
                "        ax = axes[row, col]\n",
                "        if row < imfs.shape[0]:\n",
                "            ax.plot(t_axis, imfs[row], color=color, lw=0.9)\n",
                "            ax.grid(True, alpha=0.25)\n",
                "        if col == 0:\n",
                "            ax.set_ylabel(f\"IMF {row+1}\", fontsize=9, fontweight='bold')\n",
                "        if row == 5:\n",
                "            ax.set_xlabel(\"Tempo (s)\", fontsize=8.5)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "@@SECTION4@@",
                "### 4.1. Limitações Críticas Identificadas no EMD de Huang\n",
                "* **Mode Mixing (Mistura de Modos):** O ruído de banda larga marinho causa quebra na interpolação de splines cúbicas, fazendo com que uma mesma oscilação física vaze entre múltiplas IMFs.\n",
                "* **Modos Falsos no Ruído Gaussiano:** Em ruído ambiental puro, o algoritmo de *sifting* força a extração de oscilações locais até o resíduo ser monótono, criando picos espúrios que não existem fisicamente.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. A Solução Moderna: VMD (Variational Mode Decomposition) sobre a Envoltória de Cavitação\n",
                "\n",
                "Como alternativa ao EMD, que sofre de mistura de modos (*mode mixing*), avaliamos o **VMD** (*Dragomiretskiy & Zosso*, IEEE Trans. Signal Process., 2014) atuando especificamente sobre a **envoltória de cavitação demodulada** de `SMALL` (`A-0002.wav`).\n",
                "\n",
                "Diferente da HHT pura de banda larga ($0\\text{--}1000\\text{ Hz}$ sem demodulação avaliada na Seção 3), o código da célula a seguir aplica primeiro um **filtro passa-faixa Chebyshev II ($1\\text{--}2\\text{ kHz}$)** seguido de **retificação e decimação da envoltória de Hilbert para $50\\text{ Hz}$** (pipeline físico de demodulação de cavitação), e então executa o VMD ($K=4$ modos, $\\alpha=2000$) sobre essa envoltória demodulada. As frequências centrais obtidas são impressas pela célula e interpretadas logo abaixo dela.\n",
                "\n",
                "O VMD formula a decomposição como um **problema variacional convexo bem-posto**, determinando simultaneamente as bandas e as frequências centrais $\\omega_k$ de cada modo:\n",
                "\n",
                "$$\n",
                "\\min_{\\{u_k\\}, \\{\\omega_k\\}} \\left\\{ \\sum_{k=1}^K \\left\\| \\partial_t \\left[ \\left( \\delta(t) + \\frac{j}{\\pi t} \\right) * u_k(t) \\right] e^{-j \\omega_k t} \\right\\|_2^2 \\right\\} \\quad \\text{sujeito a } \\sum_{k=1}^K u_k(t) = f(t)\n",
                "$$\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Teste do VMD na envoltória de cavitação demodulada do áudio SMALL\n",
                "small_wav = os.path.join(ROOT_DIR, 'data/iara/A/A-0002.wav')\n",
                "fs_s, data_s = wavfile.read(small_wav)\n",
                "if data_s.ndim > 1: data_s = data_s[:, 0]\n",
                "seg_s = data_s[:int(30 * fs_s)].astype(np.float32)\n",
                "\n",
                "# Demodulação na banda de cavitação (1-2 kHz) -> Envoltória decimada para 50 Hz\n",
                "nyq = fs_s / 2.0\n",
                "N, wc = scipy_sig.cheb2ord([1000/nyq, 2000/nyq], [700/nyq, 2300/nyq], 0.5, 40.0)\n",
                "sos = scipy_sig.cheby2(N, rs=40, Wn=wc, btype='bandpass', output='sos')\n",
                "filtered = scipy_sig.sosfilt(sos, seg_s)\n",
                "env = np.abs(filtered)\n",
                "env_dec = scipy_sig.decimate(env, 24, ftype='fir', zero_phase=True)\n",
                "env_dec = scipy_sig.decimate(env_dec, 40, ftype='fir', zero_phase=True)\n",
                "fs_dem = (fs_s // 24) // 40  # 50 Hz\n",
                "env_dec = env_dec - np.mean(env_dec)\n",
                "env_dec /= np.max(np.abs(env_dec))\n",
                "\n",
                "# Executa VMD com K=4 modos e alpha=2000 sobre a envoltória demodulada\n",
                "u, u_hat, omega = VMD(env_dec, alpha=2000, tau=0, K=4, DC=0, init=1, tol=1e-7)\n",
                "vmd_freqs_hz = omega[-1] * fs_dem\n",
                "vmd_freqs_rpm = vmd_freqs_hz * 60.0\n",
                "\n",
                "print(\"=== RESULTADO DO VMD NA ENVOLTÓRIA DEMODULADA DE SMALL (Pilot 10, 11m) ===\")\n",
                "for m in range(len(vmd_freqs_hz)):\n",
                "    print(f\"  Modo {m+1}: {vmd_freqs_hz[m]:.2f} Hz ({vmd_freqs_rpm[m]:.1f} RPM)\")\n",
                "\n",
                "f_modos = sorted(vmd_freqs_hz)\n",
                "f_osc = [f for f in f_modos if f >= 1.0]  # descarta o modo quase-DC\n",
                "if len(f_osc) >= 2:\n",
                "    razoes = ', '.join(f'{f / f_osc[0]:.2f}' for f in f_osc)\n",
                "    print(f\"\\nModos oscilatórios (>= 1 Hz): {', '.join(f'{f:.2f}' for f in f_osc)} Hz; razão para o menor: {razoes}\")\n",
                "print(\"Obs.: sem RPM medido do Pilot 10, a associação dos modos à rotação do eixo / passagem de pás é hipótese, não comprovação.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["@@VMD_INTERP@@"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Conclusões e Decisão Arquitetural\n",
                "\n",
                "1. **Autonomia da HHT:**\n",
                "   A HHT Pura é uma categoria independente que substitui a FFT do LOFAR na decomposição adaptativa de sinais hidroacústicos.\n",
                "2. **Dimensionalidade Recomendada e Parâmetros de Extração Massiva (`cache_hht.pkl`):**\n",
                "   Adotamos **$N_{\\text{bins}} = 256$** como padrão oficial da HHT (grade de 1 a 1000 Hz, resolução de $999/255 \\approx 3{,}92\\text{ Hz/bin}$), garantindo:\n",
                "   - Paridade dimensional com o MEL ($256\\text{-D}$);\n",
                "   - Filtragem natural de microflutuações da frequência instantânea;\n",
                "   - Na extração em larga escala para os **1.706 áudios** (`cache_hht.pkl`, 256 bins), adotamos janelas de `win_len=2.56s` com passo `hop=5.12s` e `MAX_ITERATION=30` no EMD para viabilizar a escalabilidade computacional em todo o acervo. **Com isso metade de cada gravação (2,56 s a cada 5,12 s) não entra nas features**; o desempenho da Seção 7 refere-se a essa extração subamostrada.\n"
            ]
        },
        "@@SECTION7@@"
    ]

    # ------------------------------------------------------------------------------------------
    # Números do markdown: calculados aqui, executando as MESMAS células de código do notebook
    # (teste cego HHT e VMD) e lendo os experimentos 5x2cv via iara_results.
    # ------------------------------------------------------------------------------------------
    code_src = [''.join(c['source']) for c in cells if isinstance(c, dict) and c['cell_type'] == 'code']
    ns = {}
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        import matplotlib
        matplotlib.use('Agg')
        exec(code_src[0], ns)   # setup
        exec(code_src[1], ns)   # teste cego (HHT pura nos 4 áudios)
        vmd_code = next(c for c in code_src if 'VMD(env_dec' in c)
        exec(vmd_code, ns)
    finally:
        os.chdir(cwd)
    import matplotlib.pyplot as _plt
    _plt.close('all')

    res = {lab.split('(')[1].split(':')[0]: r for lab, (r, _) in ns['hht_results'].items()}
    order = ['SMALL', 'MEDIUM', 'LARGE', 'AMBIENTAL']

    def band(r, lo, hi):
        fg, h = r['freq_grid'], r['hms']
        return h[(fg >= lo) & (fg <= hi)].sum() / h.sum() * 100

    def sub_peak(r, lo=5.0, hi=50.0):
        fg, h = r['freq_grid'], r['hms']
        m = (fg >= lo) & (fg <= hi)
        j = np.argmax(h[m])
        return fg[m][j], h[m][j]

    cen_rank = sorted(order, key=lambda c: -res[c]['centroid'])
    rol_rank = sorted(order, key=lambda c: -res[c]['rolloff'])
    lf, la = sub_peak(res['LARGE'])
    sf, sa = sub_peak(res['SMALL'])
    sec4 = [
        "## 4. Diagnóstico dos Resultados do Teste Cego\n",
        "\n",
        "*(Valores abaixo calculados na geração do notebook executando as mesmas células acima; conferir com a tabela da Seção 3.1. "
        "São 10 s centrais de **uma** gravação por classe: servem para ilustrar o método, não para caracterizar as classes.)*\n",
        "\n",
        "| Áudio | Pico dominante | Centróide | Rolloff 85% | Energia < 50 Hz | Energia 150–350 Hz | Energia 450–550 Hz |\n",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n",
    ]
    for c in order:
        r = res[c]
        sec4.append(f"| {c} | {pt(r['peak_freq'], 1)} Hz | {pt(r['centroid'], 1)} Hz | {pt(r['rolloff'], 1)} Hz | "
                    f"{pt(band(r, 1, 50), 1)}% | {pt(band(r, 150, 350), 1)}% | {pt(band(r, 450, 550), 1)}% |\n")
    sec4 += [
        "\n",
        f"1. **Centróide e rolloff:** a ordem do centróide é {' > '.join(cen_rank)} e a do rolloff é {' > '.join(rol_rank)}. "
        f"O LARGE (*Sinoglory*) tem o pico dominante em {pt(res['LARGE']['peak_freq'], 1)} Hz; atribuí-lo a maquinário auxiliar específico "
        "(geradores, bombas) é hipótese, sem medida de bordo que a confirme.\n",
        f"2. **Faixa subsônica (5–50 Hz):** no LARGE o máximo local está em {pt(lf, 1)} Hz, com {pt(la * 100, 0)}% da energia do pico global "
        f"do HMS; no SMALL o máximo local nessa faixa está em {pt(sf, 1)} Hz, com {pt(sa * 100, 0)}%. "
        f"A energia abaixo de 50 Hz é {pt(band(res['SMALL'], 1, 50), 1)}% no SMALL e {pt(band(res['LARGE'], 1, 50), 1)}% no LARGE: "
        "portanto **não** se sustenta dizer que o SMALL \"não exibe linha subsônica\".\n",
        f"3. **MEDIUM (*Maersk Pacer*):** pico dominante em {pt(res['MEDIUM']['peak_freq'], 1)} Hz. A leitura como rotação de motor diesel médio é hipótese.\n",
        f"4. **AMBIENTAL (fundo do mar):** rolloff de {pt(res['AMBIENTAL']['rolloff'], 1)} Hz "
        + ("(o menor dos quatro)" if rol_rank[-1] == 'AMBIENTAL' else "") +
        f" e pico dominante em {pt(res['AMBIENTAL']['peak_freq'], 1)} Hz: o EMD também gera picos no ruído de fundo (ver 4.1).\n",
        "\n",
        "--- \n",
        "\n",
    ]

    fm = sorted(ns['vmd_freqs_hz'])
    fo = [f for f in fm if f >= 1.0]
    vmd_interp = [
        "### Leitura do resultado do VMD\n",
        "\n",
        f"* Frequências centrais dos {len(fm)} modos (ordem crescente): " + ", ".join(f"{pt(f)} Hz" for f in fm) + ".\n",
        (f"* Descartando o modo quase-DC, os modos oscilatórios ({', '.join(pt(f) + ' Hz' for f in fo)}) estão em razões "
         + ", ".join(pt(f / fo[0]) for f in fo) + " em relação ao menor, isto é, formam uma série próxima de múltiplos de "
         + pt(fo[0]) + " Hz.\n") if len(fo) >= 2 else "* Menos de dois modos oscilatórios acima de 1 Hz.\n",
        "* Isso é **compatível** com uma modulação periódica da envoltória (rotação do eixo e/ou passagem de pás), mas **não a comprova**: "
        "não há RPM medido do *Pilot 10* para comparar, o VMD foi aplicado em só 30 s de uma gravação e $K=4$ foi fixado a priori. "
        "Também não se mediu aqui a mistura de modos do EMD na mesma envoltória, então não se pode afirmar que o VMD \"resolveu\" o *mode mixing*.\n",
    ]

    # Seção 7 -------------------------------------------------------------------------------
    S7 = {e: R.summary(e, m) for e, m in HHT_EXPS_4C + HHT_EXPS_3C + [(EXP_RF_LOFAR, '4c'), (EXP_MLP_SEED, '4c')]}
    cnn_folds = R.fold_metrics(EXP_CNN_4C, '4c')
    best4 = max(HHT_EXPS_4C, key=lambda em: S7[em[0]]['SP'][0])[0]
    best3 = max(HHT_EXPS_3C, key=lambda em: S7[em[0]]['SP'][0])[0]
    second4 = sorted(HHT_EXPS_4C, key=lambda em: -S7[em[0]]['SP'][0])[1][0]
    second3 = sorted(HHT_EXPS_3C, key=lambda em: -S7[em[0]]['SP'][0])[1][0]
    f_b4 = R.f_test_5x2(best4, second4)
    f_b3 = R.f_test_5x2(best3, second3, mode_a='3c', mode_b='3c')
    f_rf_lofar = R.f_test_5x2(EXP_RF_4C, EXP_RF_LOFAR)
    names = {EXP_RF_4C: 'RF', EXP_MLP_4C: 'MLP', EXP_SVM_4C: 'SVM m=6000', EXP_CNN_4C: 'CNN 2D',
             EXP_RF_3C: 'RF', EXP_MLP_3C: 'MLP', EXP_SVM_3C: 'SVM m=6000', EXP_CNN_3C: 'CNN 2D'}

    sec7 = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Avaliação Experimental nos 10 Folds Oficiais (Protocolo 5x2cv)\n",
                "\n",
                "A extração massiva de HHT (`cache_hht.pkl`, 256 bins de 1 a 1000 Hz, `win_len=2.56s`, `hop=5.12s` — metade de cada áudio descartada —, `MAX_ITERATION=30`) foi feita para os **1.706 áudios UO** do IARA. "
                "Os modelos foram avaliados por `scratch/eval_hht_models_5x2cv.py` nos 10 folds oficiais do $5\\times2$cv (particionamento estratificado **por gravação**, `exclusive_header='ID'`, como no artigo; o mesmo navio pode aparecer em treino e teste), com voto majoritário por arquivo.\n",
                "\n",
                "A tabela é calculada na execução por `notebooks/iara_results.py` a partir de `results/trainings/multirep/<exp>/eval/fold_k/`. **ACC = acurácia balanceada** (média dos recalls); `ACC_micro` = fração de arquivos corretos.\n",
                "* **4 classes**: modelo treinado e avaliado com SMALL, MEDIUM, LARGE e BACKGROUND.\n",
                "* **3 classes**: modelo **retreinado** sem BACKGROUND (experimentos `*_3c`).\n",
                "* O MLP é a versão sem semente fixa (`" + EXP_MLP_4C + "`), a do script deste notebook. A versão com semente 42 usada no gate (`" + EXP_MLP_SEED + "`) dá SP " + fmtc(S7[EXP_MLP_SEED]['SP']) + ".\n",
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys\n",
                "from IPython.display import display\n",
                "for _p in ('.', 'src', 'notebooks', 'notebooks/src'):\n",
                "    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:\n",
                "        sys.path.insert(0, os.path.abspath(_p))\n",
                "import iara_results as R\n",
                "\n",
                "linhas_4c = [\n",
                "    ('Random Forest (100 árvores) — 4C', '" + EXP_RF_4C + "', '4c'),\n",
                "    ('MLP PyTorch [32, 16] — 4C', '" + EXP_MLP_4C + "', '4c'),\n",
                "    ('SVM Nyström (m=6000, RBF) — 4C', '" + EXP_SVM_4C + "', '4c'),\n",
                "    ('CNN 2D PyTorch [64, 32] — 4C (COLAPSADA)', '" + EXP_CNN_4C + "', '4c'),\n",
                "]\n",
                "linhas_3c = [\n",
                "    ('Random Forest (100 árvores) — 3C', '" + EXP_RF_3C + "', '3c'),\n",
                "    ('MLP PyTorch [32, 16] — 3C', '" + EXP_MLP_3C + "', '3c'),\n",
                "    ('SVM Nyström (m=6000, RBF) — 3C', '" + EXP_SVM_3C + "', '3c'),\n",
                "    ('CNN 2D PyTorch [64, 32] — 3C', '" + EXP_CNN_3C + "', '3c'),\n",
                "]\n",
                "display(R.table(linhas_4c, metrics=('SP', 'ACC', 'ACC_micro')))\n",
                "display(R.table(linhas_3c, metrics=('SP', 'ACC', 'ACC_micro')))\n",
                "\n",
                "cnn = R.fold_metrics('" + EXP_CNN_4C + "', '4c')\n",
                "print('CNN 2D 4C, SP por fold:', cnn['SP'].round(1).tolist())\n",
                "print('CNN 2D 4C, recall SMALL por fold:', cnn['R_SMALL'].round(1).tolist())\n",
                "print()\n",
                "print('F-test 5x2cv (SP):')\n",
                "for a, b, m in [('" + best4 + "', '" + second4 + "', '4c'), ('" + best3 + "', '" + second3 + "', '3c'),\n",
                "                ('" + EXP_RF_4C + "', '" + EXP_RF_LOFAR + "', '4c')]:\n",
                "    F, p = R.f_test_5x2(a, b, mode_a=m, mode_b=m)\n",
                "    print(f'  {a} vs {b}: F = {F:.2f}, p = {p:.3f}')\n",
                "print('RF LOFAR local (cache LOFAR de 5,12 s):', R.fmt(R.summary('" + EXP_RF_LOFAR + "')['SP']))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7.1 Leitura dos resultados\n",
                "\n",
                "*(Números calculados dos CSVs na geração; conferir com a saída acima.)*\n",
                "\n",
                f"1. **4 classes:** o melhor HHT é o {names[best4]} (SP {fmtc(S7[best4]['SP'])}), seguido do {names[second4]} (SP {fmtc(S7[second4]['SP'])}); "
                + ("a diferença é significativa" if f_b4[1] < 0.05 else "a diferença **não** é significativa") + f" (F-test 5×2cv, {pval(f_b4[1])}).\n",
                f"2. **3 classes (retreinado):** o melhor é o {names[best3]} (SP {fmtc(S7[best3]['SP'])}), seguido do {names[second3]} (SP {fmtc(S7[second3]['SP'])}); "
                + ("diferença significativa" if f_b3[1] < 0.05 else "diferença **não** significativa") + f" ({pval(f_b3[1])}).\n",
                f"3. **CNN 2D colapsada:** em 4 classes o SP é {fmtc(S7[EXP_CNN_4C]['SP'])}, com SP por fold de {pt(cnn_folds['SP'].min(), 1)} a {pt(cnn_folds['SP'].max(), 1)} "
                f"e recall de SMALL por fold de {pt(cnn_folds['R_SMALL'].min(), 1)} a {pt(cnn_folds['R_SMALL'].max(), 1)}%. "
                "Com `hop=5.12s`, uma imagem de 32 janelas cobre ~164 s: com passo de 16 janelas, cada gravação (de 2 a 5 min) gera só 1 ou 2 imagens, e as que têm menos de 32 janelas são completadas por *wrap padding* — pouquíssimas amostras de treino para uma CNN. "
                "Esse modelo não deve servir de baseline nem de argumento contra CNNs em HHT.\n",
                f"4. **HHT × LOFAR no mesmo classificador:** o RF com HHT (SP {fmtc(S7[EXP_RF_4C]['SP'])}) fica **abaixo** do RF com LOFAR local "
                f"(`{EXP_RF_LOFAR}`, SP {fmtc(S7[EXP_RF_LOFAR]['SP'])}; {pval(f_rf_lofar[1])}). Todos os modelos HHT ficam abaixo do MLP LOFAR do artigo "
                f"(SP {pt(R.ARTICLE_TABLE7[('MLP', 'LOFAR')]['SP'][0])}).\n",
            ]
        },
    ]

    def subst(c):
        if c == "@@SECTION7@@":
            return sec7
        if isinstance(c, dict) and c['cell_type'] == 'markdown':
            src = []
            for line in c['source']:
                if line == "@@SECTION4@@":
                    src.extend(sec4)
                elif line == "@@VMD_INTERP@@":
                    src.extend(vmd_interp)
                else:
                    src.append(line)
            c = dict(c, source=src)
        return [c]

    cells = [x for c in cells for x in subst(c)]
    assert not any('@@' in ''.join(c['source']) for c in cells)

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)

    print(f"[OK] Notebook gerado com sucesso em: {filename}")

if __name__ == "__main__":
    create_hht_notebook()
