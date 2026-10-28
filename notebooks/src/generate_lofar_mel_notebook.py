"""
Script to generate notebooks/5_analise_lofar_mel.ipynb detailing LOFAR and MEL feature extraction,
TPSW background normalization, physical acoustics, real spectrograms, and comparison.
Ensures every cell is self-contained with robust imports and dual-path resolution.
"""

import os
import sys
import json
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iara_results as R  # noqa: E402  (módulo único de métricas)

# Experimentos em disco usados neste notebook (results/trainings/tests/<exp>)
EXP_LOFAR_PCA = 'svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0'
EXP_LOFAR_6000 = 'svm_nystroem_6000_lofar_elasticnet_l1r0.15'
EXP_MEL = 'svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0'
EXP_HYB = 'svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5'
EXP_HYB_C2 = 'svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C2.0'
EXP_MLP_LOFAR = 'mlp_lofar'

# Código da reconstrução da cascata: o MESMO texto vira célula do notebook e é executado aqui
# no gerador para que os números citados no markdown saiam do mesmo cálculo.
CASCADE_CODE = r'''import os
from collections import Counter
import numpy as np
import pandas as pd
import iara_results as R

MC = 'svm_nystroem_6000_lofar_elasticnet_l1r0.15'                           # estágio 1 (4 classes)
SL = 'svm_nystroem_6000_hybrid_binary_small_large_pca8_elasticnet_l1r0.15'  # especialista SMALL x LARGE
SL_DIR_NOVO = R.exp_dir(SL)                                    # retreinado em 20/10/2026 (split alinhado ao 4 classes)
SL_DIR_ANTIGO = os.path.join(SL_DIR_NOVO, '20261020-174746')   # versão de jun/2026, guardada pelo --override
CASC = 'cascade_file_lofarmc6000_sl6000pca8_4c'                # scratch/eval_cascade_by_ship.py com o especialista retreinado


def votos_sl(d, subset):
    # voto por arquivo do especialista (mesma regra Counter.most_common do pipeline iara)
    out = {}
    for k in range(10):
        df = pd.read_csv(os.path.join(d, 'eval', f'fold_{k}', f'{SL}_multiclass_{subset}.csv'),
                         usecols=['File', 'Prediction'])
        out[k] = {f: Counter(g).most_common(1)[0][0] for f, g in df.groupby('File')['Prediction']}
    return out


def teste_no_treino_sl(d):
    # quantos arquivos SMALL/LARGE de teste do 4 classes estão no TREINO do especialista do mesmo fold
    trn = votos_sl(d, 'trn')
    n = t = 0
    for k, df in R.file_predictions(MC).items():
        alvo = df[df.Target.isin([0, 2])]
        t += len(alvo)
        n += int(alvo.File.isin(list(trn[k])).sum())
    return n, t


def cascata_historica():
    # Reconstrói, dos CSVs do especialista de jun/2026, a cascata reportada antes: arquivo votado LARGE pelo
    # estágio 1 recebe o voto do especialista do mesmo fold. MEDIUM/BG votados LARGE não constam desses CSVs
    # (só SMALL/LARGE) e ficam como LARGE (já é erro).
    sl = {s: votos_sl(SL_DIR_ANTIGO, s) for s in ('trn', 'val', 'test')}
    rows = []
    for k, df in R.file_predictions(MC).items():
        pred = []
        for fid, p in zip(df.File, df.Prediction):
            if p == 2:
                v = next((sl[s][k][fid] for s in sl if fid in sl[s][k]), None)
                p = p if v is None else (0 if v == 0 else 2)
            pred.append(p)
        rows.append({'fold': k, **R._metrics(df.Target.values, np.array(pred))})
    return pd.DataFrame(rows).set_index('fold')


cascata_stats = {'antigo': teste_no_treino_sl(SL_DIR_ANTIGO), 'novo': teste_no_treino_sl(SL_DIR_NOVO)}
cascata = {'Estágio 1 (LOFAR SVM m=6000)': R.fold_metrics(MC),
           'Cascata, especialista de jun/2026 (VAZADO)': cascata_historica(),
           'Cascata, especialista retreinado (20/10/2026)': R.fold_metrics(CASC)}
'''

_ns = {}
exec(CASCADE_CODE, _ns)
CASC, CASC_ST = _ns['cascata'], _ns['cascata_stats']
K_S1, K_LEAK, K_FREE = list(CASC)
VZ_OLD, VZ_NEW = CASC_ST['antigo'], CASC_ST['novo']
assert VZ_NEW[0] == 0, 'o especialista retreinado não deveria ter arquivos de teste no treino'

# Cascatas por JANELA (test_scripts/svm.py --cascade) re-executadas com o especialista SL retreinado.
JANELA_CODE = r'''import os
from collections import Counter
import numpy as np
import pandas as pd
import iara_results as R

PFX = 'svm_nystroem_4000_hybrid_cascade_pretrained_sm6000pca64_smallalso_fbmc_large_sl6000pca8_'
PFX_ANTIGO = ('svm_nystroem_4000_hybrid_cascade_pretrained_svm_nystroem_6000_hybrid_binary_small_medium_pca64_'
              'elasticnet_l1r0.15_smallalso_fbmc_large_svm_nystroem_6000_hybrid_binary_small_large_pca8_elasticnet_l1r0.15_')
DIR_ANTIGO = os.path.join(R.ROOT, 'results', 'trainings', '_cascatas_com_sl_vazado_pre20261020')
LOFAR_MC = 'svm_nystroem_6000_lofar_elasticnet_l1r0.15'
HYB_MC = 'svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C2.0'

# (rótulo, estágio 1, opções de roteamento do svm.py, experimento novo, experimento antigo com SL vazado)
JANELA = [
    ('lofarmc6000_nosm', LOFAR_MC, '--disable_sm_routing (só o especialista SL)',
     PFX + 'lofarmc6000_nosm_elasticnet_l1r0.15', PFX + 'lofarmc6000_nosm_elasticnet_l1r0.15'),
    ('lofarmc6000_smonly', LOFAR_MC, '--medium_only_routing',
     PFX + 'lofarmc6000_smonly_elasticnet_l1r0.15', PFX + 'lofarmc6000_smonly_elasticnet_l1r0.15'),
    ('hybmc4000C2.0', HYB_MC, 'SM + SL',
     PFX + 'hybmc4000C2.0_elasticnet_l1r0.15', PFX_ANTIGO + 'hybmc4000C2.0'),
    ('hybmc4000C2.0_slro', HYB_MC, '--large_specialist_runner_up_only',
     PFX + 'hybmc4000C2.0_slro_elasticnet_l1r0.15', PFX_ANTIGO + 'hybmc4000C2.0_slro'),
]


def sp_antigo(nome):
    # SP por fold da versão com especialista vazado (pasta movida em 20/10/2026); None se incompleta
    sps = []
    for k in range(10):
        f = os.path.join(DIR_ANTIGO, nome, 'eval', f'fold_{k}', f'{nome}_multiclass_test.csv')
        if not os.path.exists(f):
            return None
        df = pd.read_csv(f, usecols=['File', 'Target', 'Prediction'])
        g = df.groupby('File').agg(Target=('Target', 'first'),
                                   Prediction=('Prediction', lambda v: Counter(v).most_common(1)[0][0]))
        sps.append(R._metrics(g.Target.values, g.Prediction.values)['SP'])
    return np.array(sps)


tab_janela = []
for rot, s1, rota, novo, antigo in JANELA:
    sn, ss1 = R.summary(novo), R.summary(s1)
    old = sp_antigo(antigo)
    tab_janela.append({
        'Cascata': rot, 'Estágio 1': s1, 'Roteamento': rota,
        'SP (SL vazado, antes)': 'incompleto' if old is None else f'{old.mean():.2f} ± {old.std(ddof=0):.2f}',
        'SP (SL retreinado)': R.fmt(sn['SP']), 'SP estágio 1': R.fmt(ss1['SP']),
        **{c: round(sn['R_' + c][0], 2) for c in R.CLASSES},
        'p vs estágio 1': round(R.f_test_5x2(novo, s1)[1], 3),
        'p vs mlp_lofar': round(R.f_test_5x2(novo, 'mlp_lofar')[1], 3),
        '_sp': sn['SP'][0], '_old': None if old is None else old.mean(), '_exp': novo,
    })
tab_janela = pd.DataFrame(tab_janela).set_index('Cascata')
'''
_nsj = {}
exec(JANELA_CODE, _nsj)
TJ = _nsj['tab_janela']
TJ_BEST = TJ['_sp'].idxmax()


def pt(x, nd=2):
    """Número com vírgula decimal (padrão dos notebooks)."""
    return f"{x:.{nd}f}".replace('.', ',')


def fmtc(ms, nd=2):
    """'média ± desvio' com vírgula decimal."""
    return f"{pt(ms[0], nd)} ± {pt(ms[1], nd)}"


def pval(p):
    return 'p < 0,001' if p < 0.001 else f"p = {pt(p, 3)}"


def casc_sum(key, metric):
    s = CASC[key][metric]
    return s.mean(), s.std(ddof=0)


S = {e: R.summary(e) for e in (EXP_LOFAR_PCA, EXP_LOFAR_6000, EXP_MEL, EXP_HYB, EXP_HYB_C2, EXP_MLP_LOFAR)}
ART = R.ARTICLE_TABLE7[("MLP", "LOFAR")]
F_HYB_MEL = R.f_test_5x2(EXP_HYB, EXP_MEL)
F_HYB_L6 = R.f_test_5x2(EXP_HYB, EXP_LOFAR_6000)
F_HYB_MLP = R.f_test_5x2(EXP_HYB, EXP_MLP_LOFAR)
F_MEL_L6 = R.f_test_5x2(EXP_MEL, EXP_LOFAR_6000)
F_LEAK_MLP = R.f_test_5x2(CASC[K_LEAK]['SP'].values, EXP_MLP_LOFAR)
F_FREE_MLP = R.f_test_5x2(CASC[K_FREE]['SP'].values, EXP_MLP_LOFAR)
F_FREE_S1 = R.f_test_5x2(CASC[K_FREE]['SP'].values, CASC[K_S1]['SP'].values)
F_LEAK_S1 = R.f_test_5x2(CASC[K_LEAK]['SP'].values, CASC[K_S1]['SP'].values)

# Resolução do LOFAR exibido (src/iara/default.py: n_pts=1024 -> nfft=2048, decimação 3 de 48 kHz)
FS_DEC = 48000 / 3
DF_LOFAR = FS_DEC / 2048
F_250 = 250 * DF_LOFAR  # o bin 0 é f = Δf (o DC é descartado em analysis.spectrogram)

# Fração de zeros do LOFAR (limiar -0,2 após TPSW) nos 4 arquivos exibidos
_lofar_dir = os.path.join(R.ROOT, 'data', 'iara_processed', 'lofar_bc1559516c522e2b993895bf5cb2ac94')
ZEROS = {}
for _fid in (2, 1, 4, 1446):
    _p = os.path.join(_lofar_dir, f'{_fid}.pkl')
    if os.path.exists(_p):
        with open(_p, 'rb') as _f:
            _v = pickle.load(_f)['df'].values
        ZEROS[_fid] = ((_v == 0).mean() * 100, (_v[:, :250] == 0).mean() * 100)


def _zeros_txt(fid):
    if fid not in ZEROS:
        return '(cache LOFAR indisponível na geração)'
    return f"{pt(ZEROS[fid][0], 1)}% de zeros na matriz inteira ({pt(ZEROS[fid][1], 1)}% nos 250 bins exibidos)"


def create_lofar_mel_notebook(filename="notebooks/5_analise_lofar_mel.ipynb"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA\n",
                "### Notebook 5: Análise Aprofundada dos Extratores Espectrais LOFAR e MEL\n",
                "\n",
                "Neste notebook, exploramos detalhadamente os dois principais extratores espectrais do projeto IARA, derivados da formulação de referência de **Fábio Silva et al. (2025)** e da biblioteca `labsonar`:\n",
                "1. **LOFAR (Low Frequency Analysis and Recording):** Espectro linear de banda estreita com normalizador de fundo **TPSW (Two-Pass Split Window)** para realce de raias tonais mecânicas.\n",
                "2. **LOG-MELGRAM (Escala Mel de 256 Bandas):** Espectro logarítmico de banda larga com banco de filtros triangulares perceptuais, atuando como integrador resiliente de potência.\n",
                "3. **Inspeção Passo a Passo do TPSW:** Demonstração matemática e visual do cancelamento do piso de ruído oceânico contínuo.\n",
                "4. **Visualização Comparativa em Áudios Reais:** Espectrogramas 2D e curvas integradas para `SMALL`, `MEDIUM`, `LARGE` e `AMBIENTAL`.\n",
                "5. **Trade-off Físico Acústico e Fusão Híbrida:** A dualidade entre o *\"Sniper de Raias\"* (LOFAR) e o *\"Escudo Resiliente\"* (MEL), e o que os dados (com F-test 5×2cv) dizem sobre a fusão híbrida e a cascata com especialista.\n"
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
                "mel_basis = librosa.filters.mel(sr=fs_decimated, n_fft=n_fft, n_mels=256, fmin=0.0, fmax=fs_decimated/2)\n",
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
                "# Resolução do LOFAR (src/iara/default.py: n_pts=1024 -> nfft=2048; 48 kHz decimado 3x -> 16 kHz).\n",
                "# O bin DC é descartado em analysis.spectrogram, então a coluna i corresponde a (i+1)*Δf.\n",
                "DF_LOFAR = (48000 / 3) / 2048  # 7,8125 Hz\n",
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
                "        ax_l.set_ylabel(f'Bins 1–250 ({DF_LOFAR:.2f}–{250 * DF_LOFAR:.0f} Hz)')\n",
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
                "        ax_m.set_ylabel('Bandas Mel 0–255 (0–8 kHz)')\n",
                "\n",
                "axes[3, 0].set_xlabel('Janelas de tempo (0,512 s cada)')\n",
                "axes[3, 1].set_xlabel('Janelas de tempo (0,512 s cada)')\n",
                "plt.suptitle('Comparação Espectro-Temporal: LOFAR (Raias de Motor) vs. MEL (Envoltória de Energia)', fontsize=15, fontweight='bold', y=0.99)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### O que a visualização revela (leitura qualitativa das figuras):\n",
                "1. **No LOFAR (coluna da esquerda, bins 1–250 = " + pt(DF_LOFAR) + "–" + pt(F_250, 0) + " Hz, Δf = " + pt(DF_LOFAR) + " Hz):**\n",
                "   * *Pilot 10 (SMALL)* e *Sinoglory (LARGE)*: raias tonais visíveis como linhas horizontais; a atribuição física (ex.: motor diesel de baixa rotação no LARGE) é uma hipótese interpretativa, não medida neste notebook.\n",
                "   * *Fundo Oceânico (AMBIENTAL)*: o TPSW **não** zera o fundo. A fração de zeros após o limiar de $-0{,}2$ é semelhante à dos navios: AMBIENTAL (ID 1446) " + _zeros_txt(1446) + "; SMALL (ID 2) " + _zeros_txt(2) + "; MEDIUM (ID 1) " + _zeros_txt(1) + "; LARGE (ID 4) " + _zeros_txt(4) + ". O TPSW remove a tendência contínua do espectro, mas o resíduo de banda larga permanece.\n",
                "2. **No MEL (coluna da direita, 256 bandas de 0 a 8 kHz):**\n",
                "   * O MEL captura o gradiente contínuo de energia acústica ao longo da banda inteira; as diferenças entre as classes nas figuras são qualitativas.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Comparativo Estatístico no $5\\times2$cv (calculado dos CSVs em disco)\n",
                "\n",
                "A tabela abaixo é calculada na execução por `notebooks/iara_results.py` a partir das predições por janela de cada experimento (`results/trainings/tests/<exp>/eval/fold_k/`), com voto majoritário por arquivo e média ± desvio populacional sobre os 10 folds.\n",
                "* **ACC** = acurácia balanceada (média dos recalls, como a coluna ACC da Tabela 7 do artigo); **ACC_micro** = fração de arquivos corretos. As duas ficam em colunas separadas.\n",
                "* O **Híbrido** reportado usa **C = 0,5** com normalização (`" + EXP_HYB + "`). A variante com **C = 2,0** (`" + EXP_HYB_C2 + "`) dá SP " + fmtc(S[EXP_HYB_C2]['SP']) + ". Como o C foi escolhido olhando o teste, o número do Híbrido é otimista.\n",
                "* Para referência, incluímos o MLP LOFAR re-executado localmente (`mlp_lofar`) e o MLP LOFAR da Tabela 7 do artigo (SP " + pt(ART['SP'][0]) + " ± " + pt(ART['SP'][1]) + ", ACC " + pt(ART['ACC'][0]) + " ± " + pt(ART['ACC'][1]) + ").\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys\n",
                "import pandas as pd\n",
                "from IPython.display import display\n",
                "for _p in ('.', 'src', 'notebooks', 'notebooks/src'):\n",
                "    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:\n",
                "        sys.path.insert(0, os.path.abspath(_p))\n",
                "import iara_results as R\n",
                "\n",
                "linhas = [\n",
                "    ('LOFAR SVM (m=4000, PCA64, C=2,0)', '" + EXP_LOFAR_PCA + "'),\n",
                "    ('LOFAR SVM (m=6000, 1024-D)', '" + EXP_LOFAR_6000 + "'),\n",
                "    ('MEL SVM (m=4000, 256-D, C=2,0)', '" + EXP_MEL + "'),\n",
                "    ('HÍBRIDO SVM (MEL + LOFAR-PCA64, C=0,5)', '" + EXP_HYB + "'),\n",
                "    ('HÍBRIDO SVM (MEL + LOFAR-PCA64, C=2,0)', '" + EXP_HYB_C2 + "'),\n",
                "    ('MLP LOFAR (≈ artigo)', '" + EXP_MLP_LOFAR + "'),\n",
                "]\n",
                "df_comp = R.table(linhas, metrics=('SP', 'ACC', 'ACC_micro', 'F1'))\n",
                "art = R.ARTICLE_TABLE7[('MLP', 'LOFAR')]\n",
                "df_comp.loc['MLP LOFAR (Tabela 7 do artigo)'] = {'SP': R.fmt(art['SP']), 'ACC': R.fmt(art['ACC']), 'F1': R.fmt(art['F1']),\n",
                "                                                 'Experimento': 'artigo (sem predições por fold)'}\n",
                "display(df_comp)\n",
                "\n",
                "print('F-test 5x2cv (SP), Híbrido C=0,5 contra:')\n",
                "for rot, exp in linhas[:3] + linhas[5:]:\n",
                "    F, p = R.f_test_5x2('" + EXP_HYB + "', exp)\n",
                "    print(f'  {rot:38s} F = {F:5.2f}  p = {p:.3f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.1 Cascata no nível de arquivo (LOFAR-MC → especialista SMALL × LARGE)\n",
                "\n",
                "> ⚠️ **O número da cascata reportado antes (SP " + pt(casc_sum(K_LEAK, 'SP')[0]) + ") tinha vazamento.** O especialista `svm_nystroem_6000_hybrid_binary_small_large_pca8_elasticnet_l1r0.15` de junho/2026 usava um `StratifiedKFold` só sobre {SMALL, LARGE}, que gera partições diferentes das do modelo de 4 classes: **" + f"{VZ_OLD[0]}/{VZ_OLD[1]} = " + pt(100 * VZ_OLD[0] / VZ_OLD[1], 1) + "%** dos arquivos SMALL/LARGE de teste estavam no **treino** do especialista do mesmo fold.\n",
                "\n",
                "Em **20/10/2026** o especialista foi **retreinado** com `test_scripts/svm_binary_specialist_large.py --pca_components 8 --override` (mesmos hiperparâmetros; partições derivadas das do 4 classes, com `assert` de interseção vazia): agora **" + f"{VZ_NEW[0]}/{VZ_NEW[1]}" + "** arquivos de teste estão no treino. A cascata foi re-executada com `scratch/eval_cascade_by_ship.py`, que passa pelo especialista **todas** as janelas dos arquivos votados `LARGE` (inclusive os MEDIUM/BACKGROUND), e as predições estão em `results/trainings/multirep/cascade_file_lofarmc6000_sl6000pca8_4c`. O especialista antigo foi preservado em `.../20261020-174746/` e aparece abaixo só como registro histórico.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in CASCADE_CODE.rstrip("\n").split("\n")] + [
                "\n",
                "for versao, (n, t) in cascata_stats.items():\n",
                "    print(f\"Especialista {versao}: arquivos SMALL/LARGE de teste no TREINO do mesmo fold = {n}/{t} ({100 * n / t:.1f}%)\")\n",
                "\n",
                "tab = []\n",
                "for nome, fm in cascata.items():\n",
                "    linha = {'Versão': nome}\n",
                "    for m in ('SP', 'ACC', 'ACC_micro', 'F1'):\n",
                "        linha[m] = f\"{fm[m].mean():.2f} ± {fm[m].std(ddof=0):.2f}\"\n",
                "    for c in R.CLASSES:\n",
                "        linha[c] = round(fm['R_' + c].mean(), 2)\n",
                "    tab.append(linha)\n",
                "display(pd.DataFrame(tab).set_index('Versão'))\n",
                "\n",
                "s1 = cascata['Estágio 1 (LOFAR SVM m=6000)']['SP'].values\n",
                "print('F-test 5x2cv (SP):')\n",
                "for nome in list(cascata)[1:]:\n",
                "    sp = cascata[nome]['SP'].values\n",
                "    for rot, ref in (('estágio 1', s1), ('mlp_lofar (local)', 'mlp_lofar')):\n",
                "        F, p = R.f_test_5x2(sp, ref)\n",
                "        print(f'  {nome:38s} vs {rot:18s} F = {F:5.2f}  p = {p:.3f}')\n",
                "a = R.ARTICLE_TABLE7[('MLP', 'LOFAR')]['SP']\n",
                "print(f\"Artigo, MLP LOFAR (Tabela 7): SP {a[0]:.2f} ± {a[1]:.2f} (sem predições por fold: não admite F-test)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.2 Cascatas por janela (`test_scripts/svm.py --cascade`) com o especialista retreinado\n",
                "\n",
                "As quatro cascatas por janela que usavam o especialista SMALL × LARGE vazado foram re-executadas em 20/10/2026 com o especialista retreinado (mesmos estágios 1 e mesmo especialista SMALL × MEDIUM, `svm_nystroem_6000_hybrid_binary_small_medium_pca64_elasticnet_l1r0.15`, que não tem vazamento). As versões antigas foram movidas para `results/trainings/_cascatas_com_sl_vazado_pre20261020/` e aparecem só na coluna \"antes\". A cascata `..._smallalso_large_..._pca8` (sem `fbmc`) veio de uma versão antiga do `svm.py` e não foi re-executada.\n",
                "\n",
                "A melhor delas, **`" + TJ_BEST + "`**, atinge SP **" + TJ.loc[TJ_BEST, 'SP (SL retreinado)'].replace('.', ',') + "** (antes: " + TJ.loc[TJ_BEST, 'SP (SL vazado, antes)'].replace('.', ',') + "), com recall `SMALL` " + pt(TJ.loc[TJ_BEST, 'SMALL']) + "% e `LARGE` " + pt(TJ.loc[TJ_BEST, 'LARGE']) + "%. Contra o próprio estágio 1, " + pval(TJ.loc[TJ_BEST, 'p vs estágio 1']) + "; contra o MLP LOFAR local, " + pval(TJ.loc[TJ_BEST, 'p vs mlp_lofar']) + ". **Nenhuma** das quatro difere significativamente do MLP LOFAR (menor p entre as quatro: " + pval(TJ['p vs mlp_lofar'].min()) + ").\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in JANELA_CODE.rstrip("\n").split("\n")] + [
                "\n",
                "display(tab_janela.drop(columns=['_sp', '_old', '_exp']))"
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
                "# Recalls por classe lidos dos experimentos (não digitados)\n",
                "classes = ['SMALL', 'MEDIUM', 'LARGE', 'BACKGROUND']\n",
                "series = [\n",
                "    ('LOFAR (m=4000, PCA64)', R.summary('" + EXP_LOFAR_PCA + "'), '#34495e'),\n",
                "    ('MEL (m=4000)', R.summary('" + EXP_MEL + "'), '#e67e22'),\n",
                "    ('HÍBRIDO (C=0,5)', R.summary('" + EXP_HYB + "'), '#27ae60'),\n",
                "]\n",
                "\n",
                "x = np.arange(len(classes))\n",
                "width = 0.25\n",
                "plt.figure(figsize=(12, 5))\n",
                "for i, (rot, s, cor) in enumerate(series):\n",
                "    plt.bar(x + (i - 1) * width, [s['R_' + c][0] for c in classes], width, label=rot, color=cor, alpha=0.9)\n",
                "plt.title('Recall por classe: LOFAR vs. MEL vs. Híbrido (média de 10 folds)', fontweight='bold')\n",
                "plt.xticks(x, classes, fontweight='bold')\n",
                "plt.ylabel('Recall (%)')\n",
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
                "## 6. Conclusões e Síntese\n",
                "\n",
                "*(Números calculados dos CSVs em disco na geração deste notebook; conferir com as saídas acima.)*\n",
                "\n",
                "1. **LOFAR × MEL:**\n",
                "   * O **LOFAR** tem o maior recall de `LARGE` entre os SVMs (" + pt(S[EXP_LOFAR_PCA]['R_LARGE'][0]) + "% com PCA64; " + pt(S[EXP_LOFAR_6000]['R_LARGE'][0]) + "% com m=6000), mas o recall de `SMALL` fica entre " + pt(S[EXP_LOFAR_6000]['R_SMALL'][0]) + "% (m=6000) e " + pt(S[EXP_LOFAR_PCA]['R_SMALL'][0]) + "% (PCA64).\n",
                "   * O **MEL** (m=4000) tem o menor desvio de SP entre os SVMs (σ = " + pt(S[EXP_MEL]['SP'][1]) + " p.p.) e recall de `SMALL` de " + pt(S[EXP_MEL]['R_SMALL'][0]) + "%. O recall de `BACKGROUND` do MEL (" + pt(S[EXP_MEL]['R_BACKGROUND'][0]) + "%) é **menor** que o do LOFAR m=6000 (" + pt(S[EXP_LOFAR_6000]['R_BACKGROUND'][0]) + "%).\n",
                "   * Em SP, MEL (" + fmtc(S[EXP_MEL]['SP']) + ") e LOFAR m=6000 (" + fmtc(S[EXP_LOFAR_6000]['SP']) + ") são **equivalentes** (F-test 5×2cv, " + pval(F_MEL_L6[1]) + ").\n",
                "\n",
                "2. **Fusão Híbrida:**\n",
                "   * O Híbrido MEL + LOFAR-PCA64 (C = 0,5) atinge SP " + fmtc(S[EXP_HYB]['SP']) + " (ACC balanceada " + fmtc(S[EXP_HYB]['ACC']) + "), com recalls " + "/".join(pt(S[EXP_HYB]['R_' + c][0], 1) for c in R.CLASSES) + " (SMALL/MEDIUM/LARGE/BACKGROUND).\n",
                "   * A diferença **não é estatisticamente significativa** contra o MEL (" + pval(F_HYB_MEL[1]) + "), o LOFAR m=6000 (" + pval(F_HYB_L6[1]) + ") ou o MLP LOFAR local (" + pval(F_HYB_MLP[1]) + "). Além disso, o C = 0,5 foi escolhido pelo teste (com C = 2,0 o SP é " + pt(S[EXP_HYB_C2]['SP'][0]) + "). Portanto o Híbrido é **equivalente** às representações isoladas, e não uma \"vitória\".\n",
                "\n",
                "3. **Cascata no nível de arquivo (LOFAR-MC → especialista SMALL × LARGE):**\n",
                "   * Com o especialista **retreinado sem vazamento**, a cascata atinge SP **" + fmtc(casc_sum(K_FREE, 'SP')) + "**, ACC " + fmtc(casc_sum(K_FREE, 'ACC')) + ", recall `SMALL` " + pt(CASC[K_FREE]['R_SMALL'].mean()) + "% (estágio 1: " + pt(CASC[K_S1]['R_SMALL'].mean()) + "%) e `LARGE` " + pt(CASC[K_FREE]['R_LARGE'].mean()) + "% (estágio 1: " + pt(CASC[K_S1]['R_LARGE'].mean()) + "%): o ganho em `SMALL` é pago com parte do `LARGE`.\n",
                "   * Contra o estágio 1 (SP " + fmtc(casc_sum(K_S1, 'SP')) + "), o F-test dá " + pval(F_FREE_S1[1]) + (" (diferença significativa a 5%)" if F_FREE_S1[1] < 0.05 else " (sem significância a 5%)") + ". Contra o MLP LOFAR local (" + fmtc(S[EXP_MLP_LOFAR]['SP']) + "), " + pval(F_FREE_MLP[1]) + (": diferença significativa" if F_FREE_MLP[1] < 0.05 else ": **equivalente**") + ". Contra o MLP LOFAR do artigo (" + pt(ART['SP'][0]) + " ± " + pt(ART['SP'][1]) + "), fica **dentro do desvio** (o artigo não traz predições por fold, então não há F-test). A cascata **não supera** o melhor baseline.\n",
                "   * O valor reportado antes, SP " + fmtc(casc_sum(K_LEAK, 'SP')) + ", vinha do especialista com vazamento (" + pt(100 * VZ_OLD[0] / VZ_OLD[1], 1) + "% dos arquivos de teste no treino dele) e superestimava a cascata em " + pt(casc_sum(K_LEAK, 'SP')[0] - casc_sum(K_FREE, 'SP')[0]) + " p.p. **Não deve ser citado.**\n",
                "   * **Cascatas por janela** (seção 5.2), também com o especialista retreinado: a melhor, `" + TJ_BEST + "`, dá SP " + TJ.loc[TJ_BEST, "SP (SL retreinado)"].replace(".", ",") + " (antes, com o especialista vazado: " + TJ.loc[TJ_BEST, "SP (SL vazado, antes)"].replace(".", ",") + "), " + ("significativamente acima do" if TJ.loc[TJ_BEST, "p vs estágio 1"] < 0.05 else "sem diferença significativa em relação ao") + " seu estágio 1 (" + pval(TJ.loc[TJ_BEST, "p vs estágio 1"]) + ") e **equivalente** ao MLP LOFAR (" + pval(TJ.loc[TJ_BEST, "p vs mlp_lofar"]) + "). O valor de 69,48% citado para essa cascata em `presentation/apresentacao-resultados.tex` vinha do especialista vazado e não se sustenta.\n",
                "\n",
                "4. **Relação com o DEMON:**\n",
                "   * LOFAR e MEL cobrem o espectro direto de 0 a 8 kHz (após decimação para 16 kHz); o **DEMON** descreve a modulação de envoltória (0,39 a 25 Hz), ligada à rotação do hélice. Se essa informação ajuda a separar `SMALL` de `LARGE` é uma hipótese a ser testada nos notebooks de DEMON, não uma conclusão deste notebook.\n"
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
