"""
Script to generate notebooks/11_comparacao_modelos_hht.ipynb detailing the multi-model comparison
on pure HHT features (Random Forest, MLP, SVM Nystroem, CNN 2D GPU) alongside LOFAR, MEL, and DEMON baselines.

Todos os números vêm de notebooks/iara_results.py (predições por janela gravadas em
results/trainings/{multirep,tests}/<exp>/eval/fold_k/): as tabelas e figuras são células de código
que leem os experimentos na execução, e os números citados no markdown são calculados aqui, na geração.
Valores do artigo vêm de R.ARTICLE_TABLE7 (Tabela 7 de Silva et al., 2025).
"""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iara_results as R  # noqa: E402  (módulo único de métricas)

# ------------------------------------------------------------------------------------------------
# Experimentos
# ------------------------------------------------------------------------------------------------
HHT = {  # rótulo: (exp 4C, exp 3C retreinado)
    'Random Forest (100 árvores)': ('rf_hht_n100_balanced_4c', 'rf_hht_n100_balanced_3c'),
    'MLP PyTorch [32, 16]': ('mlp_hht_h32-16_e30_noseed_4c', 'mlp_hht_h32-16_e30_noseed_3c'),
    'SVM Nyström (m=6000)': ('svm_nystroem_6000_hht_C2.0_elasticnet_l1r0.15_4c', 'svm_nystroem_6000_hht_C2.0_elasticnet_l1r0.15_3c'),
    'CNN 2D PyTorch [64, 32] (COLAPSADA)': ('cnn2d_hht_w32_b32_e25_4c', 'cnn2d_hht_w32_b32_e25_3c'),
}
HHT_LABELS = list(HHT)
EXP_BIGRU_HHT = ('bigru_att_hht_h64_s15-5_e20_4c', 'bigru_att_hht_h64_s15-5_e20_3c')
EXP_SVM_LOFAR = 'svm_nystroem_6000_lofar_elasticnet_l1r0.15'
EXP_SVM_MEL = 'svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0'
EXP_MLP_LOFAR = 'mlp_lofar'
EXP_RF_LOFAR = 'rf_lofar5s_n100_balanced_4c'
EXP_RF_MEL = 'rf_mel5s_n100_balanced_4c'
EXP_DEMON = ('mlp_demon_h32-16_e35_4c', 'mlp_demon_h32-16_e35_3c')
# Pares HHT x DEMON por arquitetura (mesmo classificador; diferenças de configuração declaradas)
PAIRS_DEMON = [
    ('Random Forest', 'rf_hht_n100_balanced_4c', 'rf_demon_n100_balanced_4c', 'mesma configuração'),
    ('MLP [32, 16]', 'mlp_hht_h32-16_e30_4c', 'mlp_demon_h32-16_e30_4c', 'mesma configuração (30 épocas, semente 42)'),
    ('SVM Nyström m=6000', 'svm_nystroem_6000_hht_C2.0_elasticnet_l1r0.15_4c', 'svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000_4c',
     'HHT: max_iter=200 c/ early stopping; DEMON: max_iter=1000 sem early stopping. O SVM DEMON 4C é instável (desvio alto; com m=4000 o SP é '
     + '{:.2f} ± {:.2f}'.format(*R.summary('svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_4c')['SP']).replace('.', ',') + ')'),
    ('BiGRU + atenção', 'bigru_att_hht_h64_s15-5_e20_4c', 'bigru_att_demon_h64_s15-5_e20_4c', 'mesma configuração'),
]
# SVMs do estudo de complementaridade do Notebook 10 (mesmos experimentos)
NB10_SVMS = {
    'MEL': 'svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0',
    'LOFAR': 'svm_nystroem_6000_lofar_elasticnet_l1r0.15',
    'HHT': 'svm_nystroem_4000_hht_C2.0_elasticnet_l1r0.15_4c',
    'DEMON': 'svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_4c',
}


def pt(x, nd=2):
    return f"{x:.{nd}f}".replace('.', ',')


def fmtc(ms, nd=2):
    return f"{pt(ms[0], nd)} ± {pt(ms[1], nd)}"


def pval(p):
    return 'p < 0,001' if p < 0.001 else f"p = {pt(p, 3)}"


def signif(p, sim='diferença significativa', nao='**equivalentes** (diferença não significativa)'):
    return sim if p < 0.05 else nao


# Funções usadas no gerador E copiadas para o notebook (mesmo texto)
HELPERS = r'''import numpy as np
import pandas as pd

def matriz_confusao(exp, classes):
    """Matriz de confusão por arquivo somada nos 10 folds, normalizada por linha (%)."""
    df = pd.concat(R.file_predictions(exp).values(), ignore_index=True)
    cm = pd.crosstab(df.Target, df.Prediction).reindex(index=classes, columns=classes, fill_value=0)
    return cm.div(cm.sum(axis=1), axis=0).values * 100, int(cm.values.sum())

def exclusivos_small(exps):
    """Instâncias arquivo x fold de SMALL acertadas por UM único modelo do conjunto."""
    out = {k: 0 for k in exps}
    total = 0
    for k, df in R.aligned(list(exps.values())).items():
        d = df[df.Target == 0]
        total += len(d)
        hit = np.column_stack([d[e].values == 0 for e in exps.values()])
        so_um = hit.sum(axis=1) == 1
        for i, nome in enumerate(exps):
            out[nome] += int((so_um & hit[:, i]).sum())
    return out, total
'''
_ns = {'R': R}
exec(HELPERS, _ns)
matriz_confusao, exclusivos_small = _ns['matriz_confusao'], _ns['exclusivos_small']

# ------------------------------------------------------------------------------------------------
# Números para o markdown
# ------------------------------------------------------------------------------------------------
S4 = {lab: R.summary(e4, '4c') for lab, (e4, _) in HHT.items()}
S3 = {lab: R.summary(e3, '3c') for lab, (_, e3) in HHT.items()}
REF = {
    'svm_lofar': R.summary(EXP_SVM_LOFAR), 'svm_lofar_sbg': R.summary(EXP_SVM_LOFAR, '4c_sem_bg'),
    'svm_mel': R.summary(EXP_SVM_MEL), 'svm_mel_sbg': R.summary(EXP_SVM_MEL, '4c_sem_bg'),
    'mlp_lofar': R.summary(EXP_MLP_LOFAR), 'rf_lofar': R.summary(EXP_RF_LOFAR), 'rf_mel': R.summary(EXP_RF_MEL),
    'demon4': R.summary(EXP_DEMON[0]), 'demon3': R.summary(EXP_DEMON[1], '3c'),
    'bigru4': R.summary(EXP_BIGRU_HHT[0]), 'bigru3': R.summary(EXP_BIGRU_HHT[1], '3c'),
}
ART = R.ARTICLE_TABLE7
rank4 = sorted(HHT_LABELS, key=lambda lab: -S4[lab]['SP'][0])
rank3 = sorted(HHT_LABELS, key=lambda lab: -S3[lab]['SP'][0])
F4 = R.f_test_5x2(HHT[rank4[0]][0], HHT[rank4[1]][0])
F3 = R.f_test_5x2(HHT[rank3[0]][1], HHT[rank3[1]][1], mode_a='3c', mode_b='3c')
F_RF_LOFAR = R.f_test_5x2(HHT['Random Forest (100 árvores)'][0], EXP_RF_LOFAR)
F_RF_MEL = R.f_test_5x2(HHT['Random Forest (100 árvores)'][0], EXP_RF_MEL)
F_BEST3_DEMON = R.f_test_5x2(HHT[rank3[0]][1], EXP_DEMON[1], mode_a='3c', mode_b='3c')
F_BEST4_DEMON = R.f_test_5x2(HHT[rank4[0]][0], EXP_DEMON[0])
PAIR_RES = [(arq, R.summary(h)['SP'], R.summary(d)['SP'], R.f_test_5x2(h, d), obs) for arq, h, d, obs in PAIRS_DEMON]
CNN_FOLDS = R.fold_metrics(HHT['CNN 2D PyTorch [64, 32] (COLAPSADA)'][0])
rec_small_rank = sorted(HHT_LABELS, key=lambda lab: -S4[lab]['R_SMALL'][0])
rec_bg_rank = sorted(HHT_LABELS, key=lambda lab: -S4[lab]['R_BACKGROUND'][0])
large_vals = [S4[lab]['R_LARGE'][0] for lab in HHT_LABELS if 'CNN' not in lab]

CM4 = {lab: matriz_confusao(e4, [0, 1, 2, 3]) for lab, (e4, _) in HHT.items()}
CM3 = {lab: matriz_confusao(e3, [0, 1, 2]) for lab, (_, e3) in HHT.items()}
N_AVAL = CM4[HHT_LABELS[0]][1]

# Erro do SVM MEL em SMALL (o mesmo modelo do Notebook 10) e sobreposição com o RF HHT
_mel = pd.concat(R.file_predictions(NB10_SVMS["MEL"]).values(), ignore_index=True)
_ms = _mel[_mel.Target == 0]
MEL_SMALL_N, MEL_SMALL_ERR = len(_ms), int((_ms.Prediction != 0).sum())
MEL_SMALL_TO_MED = int((_ms.Prediction == 1).sum())
_rf = HHT['Random Forest (100 árvores)'][0]
_both = pd.concat(R.aligned([NB10_SVMS["MEL"], _rf]).values(), ignore_index=True)
_mm = _both[(_both.Target == 0) & (_both[NB10_SVMS['MEL']] == 1)]
MEL_TO_MED_N, RF_ALSO_MED, RF_OK = len(_mm), int((_mm[_rf] == 1).sum()), int((_mm[_rf] == 0).sum())
ORA4 = R.oracle_recall(list(NB10_SVMS.values()))
EXCL, EXCL_TOT = exclusivos_small(NB10_SVMS)
REC_SMALL_NB10 = {k: R.summary(v)['R_SMALL'][0] for k, v in NB10_SVMS.items()}
REC_SMALL_HHT6000 = S4['SVM Nyström (m=6000)']['R_SMALL'][0]

SETUP = [
    "import os\n",
    "import sys\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "try:\n",
    "    from IPython.display import display, HTML\n",
    "except ImportError:\n",
    "    display = print\n",
    "for _p in ('.', 'src', 'notebooks', 'notebooks/src'):\n",
    "    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:\n",
    "        sys.path.insert(0, os.path.abspath(_p))\n",
    "import iara_results as R  # módulo único de métricas (lê os CSVs por janela em results/trainings)\n",
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
    "HHT = " + repr(HHT) + "\n",
    "EXP_SVM_LOFAR, EXP_SVM_MEL, EXP_MLP_LOFAR = " + repr((EXP_SVM_LOFAR, EXP_SVM_MEL, EXP_MLP_LOFAR)) + "\n",
    "EXP_RF_LOFAR, EXP_RF_MEL = " + repr((EXP_RF_LOFAR, EXP_RF_MEL)) + "\n",
    "EXP_DEMON = " + repr(EXP_DEMON) + "\n",
    "EXP_BIGRU_HHT = " + repr(EXP_BIGRU_HHT) + "\n",
    "NB10_SVMS = " + repr(NB10_SVMS) + "\n",
    "\n",
] + [line + "\n" for line in HELPERS.rstrip("\n").split("\n")] + [
    "\n",
    "print('Ambiente configurado com sucesso! Bibliotecas prontas.')\n",
]


def create_comparacao_hht_notebook(filename="notebooks/11_comparacao_modelos_hht.ipynb"):
    lab_rf, lab_mlp, lab_svm, lab_cnn = HHT_LABELS
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA\n",
                "### Notebook 11: Comparativo Multimodelo com o Extrator HHT (Random Forest, MLP, SVM e CNN)\n",
                "\n",
                "Neste notebook, investigamos o comportamento da **Transformada de Hilbert-Huang (HHT Pura)** ($256\\text{ bins}$ de frequência instantânea, de $1{,}0\\text{ Hz}$ a $1000\\text{ Hz}$) através de **quatro famílias distintas de aprendizado de máquina**:\n",
                "\n",
                "1. **Random Forest (100 árvores de decisão):** sem limite de profundidade, com balanceamento de pesos de classe.\n",
                "2. **Multi-Layer Perceptron (MLP PyTorch `[32, 16]`):** rede feedforward com `BatchNorm1d` e `Dropout 0.2`, 30 épocas sem early stopping e sem semente fixa.\n",
                "3. **SVM Nyström Gaussiano ($m=6000$):** núcleo RBF aproximado com penalidade ElasticNet (`max_iter=200`, early stopping).\n",
                "4. **CNN 2D PyTorch `[64, 32]`:** convoluções 2D em imagens de $32 \\times 256$ (32 janelas HHT consecutivas). **Este modelo colapsou** (ver Seção 2).\n",
                "\n",
                "Todos os modelos foram avaliados por `scratch/eval_hht_models_5x2cv.py` nos **10 folds oficiais do $5\\times2$cv** com **particionamento estratificado por gravação de áudio (`ID`)** sobre os **1.706 áudios de UO** do IARA (o mesmo protocolo do artigo de *Silva et al.*, IEEE Access 2025; o isolamento é por gravação, não por navio), com voto majoritário por arquivo.\n",
                "\n",
                "> **Limitação da feature:** o `cache_hht.pkl` usa janelas de 2,56 s com passo de 5,12 s; **metade de cada gravação não é analisada**.\n",
                "\n",
                "Todas as tabelas e figuras são calculadas na execução por `notebooks/iara_results.py` a partir das predições por janela em `results/trainings/{multirep,tests}/<exp>/eval/fold_k/`. **ACC = acurácia balanceada** (média dos recalls), como a coluna ACC da Tabela 7 do artigo.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": SETUP
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Tabela Mestra Consolidada: HHT vs. DEMON vs. LOFAR/MEL (local e artigo)\n",
                "\n",
                "* Linhas **\"(Silva et al., 2025)\"**: valores da Tabela 7 do artigo (4 classes; o artigo não traz recalls nem 3 classes).\n",
                "* Linhas **\"(local)\"**: re-execuções locais lidas dos CSVs. `RF LOFAR/MEL (local)` usam caches de **5,12 s** (`lofar5s`/`mel5s`), resolução temporal 10× menor que a do artigo.\n",
                "* Colunas **3C**: para os modelos HHT/DEMON/BiGRU, modelo **retreinado** sem BACKGROUND (`*_3c`); para os SVMs LOFAR/MEL, o modelo de 4 classes avaliado só nos navios, contando predição BACKGROUND como erro (`4c_sem_bg`). Os dois tipos **não** são diretamente comparáveis (coluna `Tipo 3C`).\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def linha(modelo, rep, exp4=None, exp3=None, modo3='3c', artigo=None):\n",
                "    d = {'Modelo': modelo, 'Rep': rep}\n",
                "    if artigo is not None:\n",
                "        a = R.ARTICLE_TABLE7[artigo]\n",
                "        d.update({'Acc 4C (%)': a['ACC'][0], 'SP 4C (%)': a['SP'][0], 'SP 4C std': a['SP'][1]})\n",
                "    if exp4 is not None:\n",
                "        s = R.summary(exp4, '4c')\n",
                "        d.update({'Acc 4C (%)': s['ACC'][0], 'SP 4C (%)': s['SP'][0], 'SP 4C std': s['SP'][1],\n",
                "                  'Rec SMALL (%)': s['R_SMALL'][0], 'Rec MED (%)': s['R_MEDIUM'][0],\n",
                "                  'Rec LARGE (%)': s['R_LARGE'][0], 'Rec AMB (%)': s['R_BACKGROUND'][0]})\n",
                "    if exp3 is not None:\n",
                "        s3 = R.summary(exp3, modo3)\n",
                "        d.update({'Acc 3C (%)': s3['ACC'][0], 'SP 3C (%)': s3['SP'][0],\n",
                "                  'Tipo 3C': 'retreinado' if modo3 == '3c' else '4C sem BG'})\n",
                "    d['Fonte'] = 'Tabela 7 do artigo' if artigo is not None else (exp4 or exp3)\n",
                "    return d\n",
                "\n",
                "tabela = [\n",
                "    linha('MLP (Silva et al., 2025)', 'LOFAR', artigo=('MLP', 'LOFAR')),\n",
                "    linha('CNN (Silva et al., 2025)', 'LOFAR', artigo=('CNN', 'LOFAR')),\n",
                "    linha('RF (Silva et al., 2025)', 'LOFAR', artigo=('RF', 'LOFAR')),\n",
                "    linha('CNN (Silva et al., 2025)', 'MEL', artigo=('CNN', 'MEL')),\n",
                "    linha('MLP (Silva et al., 2025)', 'MEL', artigo=('MLP', 'MEL')),\n",
                "    linha('RF (Silva et al., 2025)', 'MEL', artigo=('RF', 'MEL')),\n",
                "    linha('MLP (local)', 'LOFAR', EXP_MLP_LOFAR),\n",
                "    linha('SVM m=6000 (local)', 'LOFAR', EXP_SVM_LOFAR, EXP_SVM_LOFAR, '4c_sem_bg'),\n",
                "    linha('SVM m=4000 (local)', 'MEL', EXP_SVM_MEL, EXP_SVM_MEL, '4c_sem_bg'),\n",
                "    linha('RF 100t (local, cache 5,12 s)', 'LOFAR', EXP_RF_LOFAR),\n",
                "    linha('RF 100t (local, cache 5,12 s)', 'MEL', EXP_RF_MEL),\n",
                "] + [linha(rot, 'HHT', e4, e3) for rot, (e4, e3) in HHT.items()] + [\n",
                "    linha('BiGRU + atenção (local)', 'HHT', *EXP_BIGRU_HHT),\n",
                "    linha('MLP [32, 16], 35 épocas (local)', 'DEMON', *EXP_DEMON),\n",
                "]\n",
                "df_master = pd.DataFrame(tabela).set_index('Modelo').round(2)\n",
                "display(df_master)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Comparativo Global: O Desempenho da HHT Pura (4 Classes vs. 3 Classes)\n",
                "\n",
                "No gráfico abaixo, comparamos a **Acurácia Balanceada** e o **Índice SP** de cada uma das 4 arquiteturas no problema de 4 classes e no de 3 classes (modelos retreinados só com embarcações), com linhas de referência:\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "hht_models = list(HHT)\n",
                "s4 = [R.summary(e4, '4c') for e4, _ in HHT.values()]\n",
                "s3 = [R.summary(e3, '3c') for _, e3 in HHT.values()]\n",
                "x = np.arange(len(hht_models))\n",
                "w = 0.35\n",
                "art = R.ARTICLE_TABLE7[('MLP', 'LOFAR')]['SP'][0]\n",
                "mel = R.summary(EXP_SVM_MEL)['SP'][0]\n",
                "dem4 = R.summary(EXP_DEMON[0])['SP'][0]\n",
                "dem3 = R.summary(EXP_DEMON[1], '3c')['SP'][0]\n",
                "\n",
                "# Gráfico 1: 4 Classes\n",
                "ax1.bar(x - w/2, [s['ACC'][0] for s in s4], w, yerr=[s['ACC'][1] for s in s4], capsize=4, label='Acurácia Balanceada (%)', color='#2980b9', alpha=0.9)\n",
                "rects2 = ax1.bar(x + w/2, [s['SP'][0] for s in s4], w, yerr=[s['SP'][1] for s in s4], capsize=4, label='Índice SP (%)', color='#9b59b6', alpha=0.9)\n",
                "ax1.axhline(art, color='#27ae60', linestyle='--', lw=2, label=f'MLP LOFAR (artigo, {art:.2f}%)')\n",
                "ax1.axhline(mel, color='#e67e22', linestyle=':', lw=2, label=f'SVM MEL local ({mel:.2f}%)')\n",
                "ax1.axhline(dem4, color='#95a5a6', linestyle='-.', lw=1.5, label=f'MLP DEMON local ({dem4:.2f}%)')\n",
                "ax1.set_title('Problema de 4 Classes (com AMBIENTAL)', fontweight='bold')\n",
                "ax1.set_xticks(x)\n",
                "ax1.set_xticklabels(hht_models, rotation=20, ha='right', fontweight='bold')\n",
                "ax1.set_ylabel('Taxa (%)')\n",
                "ax1.set_ylim(20, 75)\n",
                "ax1.legend(loc='lower left', frameon=True, fontsize=9)\n",
                "for rect in rects2:\n",
                "    h = rect.get_height()\n",
                "    ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 7), textcoords='offset points',\n",
                "                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#6c3483')\n",
                "\n",
                "# Gráfico 2: 3 Classes (retreinados)\n",
                "ax2.bar(x - w/2, [s['ACC'][0] for s in s3], w, yerr=[s['ACC'][1] for s in s3], capsize=4, label='Acurácia Balanceada (embarcações)', color='#27ae60', alpha=0.9)\n",
                "rects4 = ax2.bar(x + w/2, [s['SP'][0] for s in s3], w, yerr=[s['SP'][1] for s in s3], capsize=4, label='Índice SP (embarcações)', color='#d35400', alpha=0.9)\n",
                "ax2.axhline(dem3, color='#95a5a6', linestyle='-.', lw=1.5, label=f'MLP DEMON 3C retreinado ({dem3:.2f}%)')\n",
                "ax2.set_title('Problema de 3 Classes (modelos retreinados sem AMBIENTAL)', fontweight='bold')\n",
                "ax2.set_xticks(x)\n",
                "ax2.set_xticklabels(hht_models, rotation=20, ha='right', fontweight='bold')\n",
                "ax2.set_ylabel('Taxa (%)')\n",
                "ax2.set_ylim(20, 75)\n",
                "ax2.legend(loc='lower left', frameon=True, fontsize=9)\n",
                "for rect in rects4:\n",
                "    h = rect.get_height()\n",
                "    ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 7), textcoords='offset points',\n",
                "                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#ba4a00')\n",
                "\n",
                "plt.suptitle('Comparativo de Modelos com a HHT Pura (10 Folds Oficiais 5x2cv)', fontsize=15, fontweight='bold', y=0.98)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print('F-test 5x2cv (SP):')\n",
                "for a, b, m in [(HHT['" + rank4[0] + "'][0], HHT['" + rank4[1] + "'][0], '4c'),\n",
                "                (HHT['" + rank3[0] + "'][1], HHT['" + rank3[1] + "'][1], '3c'),\n",
                "                (HHT['" + lab_rf + "'][0], EXP_RF_LOFAR, '4c'), (HHT['" + lab_rf + "'][0], EXP_RF_MEL, '4c'),\n",
                "                (HHT['" + rank4[0] + "'][0], EXP_DEMON[0], '4c'), (HHT['" + rank3[0] + "'][1], EXP_DEMON[1], '3c')]:\n",
                "    F, p = R.f_test_5x2(a, b, mode_a=m, mode_b=m)\n",
                "    print(f'  {a} vs {b}: F = {F:.2f}, p = {p:.3f}')\n",
                "cnn = R.fold_metrics(HHT['" + lab_cnn + "'][0])\n",
                "print('CNN 2D 4C — SP por fold:', cnn['SP'].round(1).tolist())\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Principais Observações do Comparativo Global:\n",
                "*(Números calculados dos CSVs na geração; conferir com as saídas acima.)*\n",
                "\n",
                f"1. **4 classes:** o melhor modelo HHT é o {rank4[0]} (SP {fmtc(S4[rank4[0]]['SP'])}), seguido do {rank4[1]} (SP {fmtc(S4[rank4[1]]['SP'])}); "
                f"são {signif(F4[1])} (F-test 5×2cv, {pval(F4[1])}).\n",
                f"2. **HHT não supera o LOFAR nem o MEL no mesmo classificador:** o RF com HHT (SP {fmtc(S4[lab_rf]['SP'])}) fica abaixo do RF LOFAR local "
                f"(SP {fmtc(REF['rf_lofar']['SP'])}; {pval(F_RF_LOFAR[1])}, " + ("significativo" if F_RF_LOFAR[1] < 0.05 else "não significativo")
                + f") e é {'significativamente diferente' if F_RF_MEL[1] < 0.05 else 'equivalente'} ao RF MEL local (SP {fmtc(REF['rf_mel']['SP'])}; {pval(F_RF_MEL[1])}). "
                f"A comparação anterior, contra o RF LOFAR da **Tabela 7 do artigo** (SP {pt(ART[('RF', 'LOFAR')]['SP'][0])}), misturava pipelines diferentes "
                "(features de 0,512 s e outros hiperparâmetros no artigo) e levava à conclusão oposta.\n",
                f"3. **HHT × DEMON:** o melhor HHT em 4 classes ({rank4[0]}) contra o MLP DEMON local (SP {fmtc(REF['demon4']['SP'])}): {pval(F_BEST4_DEMON[1])} "
                + ("(diferença significativa). " if F_BEST4_DEMON[1] < 0.05 else "(diferença não significativa). ")
                + "A comparação por arquitetura está na Seção 5.2.\n",
                f"4. **3 classes (retreinados):** o melhor é o {rank3[0]} (SP {fmtc(S3[rank3[0]]['SP'])}), seguido do {rank3[1]} (SP {fmtc(S3[rank3[1]]['SP'])}; {pval(F3[1])}). "
                f"O SVM LOFAR m=6000 avaliado sem BACKGROUND (modelo 4C, `4c_sem_bg`) dá SP {fmtc(REF['svm_lofar_sbg']['SP'])}, mas não é um modelo retreinado e não serve de \"teto\" direto.\n",
                f"5. **CNN 2D colapsada:** SP {fmtc(S4[lab_cnn]['SP'])} em 4 classes, com SP por fold de {pt(CNN_FOLDS['SP'].min(), 1)} a {pt(CNN_FOLDS['SP'].max(), 1)} "
                f"e recall de SMALL de {pt(CNN_FOLDS['R_SMALL'].min(), 1)} a {pt(CNN_FOLDS['R_SMALL'].max(), 1)}% por fold. "
                "Com o passo de 5,12 s, uma imagem de 32 janelas cobre ~164 s e cada gravação gera só 1 ou 2 imagens (as curtas, completadas por *wrap padding*). "
                "O resultado reflete esse desenho de treino, não uma propriedade da HHT; o modelo não deve ser usado como baseline nem para concluir que \"convoluções borram as raias\".\n",
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Análise de Sensibilidade por Classe (Recalls)\n",
                "\n",
                "Abaixo avaliamos o recall de cada modelo em cada classe do problema de 4 classes com HHT:\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "classes = ['SMALL', 'MEDIUM', 'LARGE', 'AMBIENTAL']\n",
                "models_subset = list(HHT)\n",
                "recalls_matrix = np.array([[R.summary(e4)['R_' + c][0] for c in R.CLASSES] for e4, _ in HHT.values()])\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(13, 6))\n",
                "x = np.arange(len(classes))\n",
                "width = 0.20\n",
                "colors = ['#27ae60', '#3498db', '#8e44ad', '#e74c3c']\n",
                "\n",
                "for i, (m_name, color) in enumerate(zip(models_subset, colors)):\n",
                "    offset = (i - 1.5) * width\n",
                "    rects = ax.bar(x + offset, recalls_matrix[i], width, label=m_name, color=color, alpha=0.9)\n",
                "    for rect in rects:\n",
                "        h = rect.get_height()\n",
                "        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),\n",
                "                    xytext=(0, 2), textcoords=\"offset points\", ha='center', va='bottom', fontsize=8.5, fontweight='bold')\n",
                "\n",
                "ax.set_title('Sensibilidade por Classe (Recall %) no Problema de 4 Classes com HHT Pura', fontweight='bold', pad=15)\n",
                "ax.set_xticks(x)\n",
                "ax.set_xticklabels(classes, fontsize=12, fontweight='bold')\n",
                "ax.set_ylabel('Recall (%)', fontsize=12)\n",
                "ax.set_ylim(0, 95)\n",
                "ax.legend(title='Arquitetura', loc='upper left', frameon=True)\n",
                "ax.grid(True, axis='y', alpha=0.4)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Leitura da Sensibilidade por Classe:\n",
                f"1. **SMALL:** o maior recall é o do {rec_small_rank[0]} ({pt(S4[rec_small_rank[0]]['R_SMALL'][0])}%), seguido do {rec_small_rank[1]} ({pt(S4[rec_small_rank[1]]['R_SMALL'][0])}%). "
                "A explicação de que as árvores \"capturam transientes de alta frequência\" é uma hipótese, não testada aqui.\n",
                f"2. **AMBIENTAL:** o maior recall é o do {rec_bg_rank[0]} ({pt(S4[rec_bg_rank[0]]['R_BACKGROUND'][0])}%). Atribuir isso à `BatchNorm1d` não é sustentado por nenhum experimento de ablação.\n",
                f"3. **LARGE:** nos três modelos que não colapsaram o recall vai de {pt(min(large_vals))}% a {pt(max(large_vals))}%. "
                "O máximo local em ~19 Hz visto no Notebook 9 vem de **uma** gravação LARGE e não basta para explicar o desempenho na classe.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Matrizes de Confusão Acumuladas nos 10 Folds ($5\\times2$cv)\n",
                "\n",
                "### 4.1 Problema de 4 Classes (SMALL, MEDIUM, LARGE, AMBIENTAL)\n",
                f"Matrizes por arquivo, somadas nos 10 folds ({N_AVAL} avaliações de teste = instâncias arquivo × fold) e normalizadas por linha:\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "classes = ['SMALL', 'MEDIUM', 'LARGE', 'AMBIENTAL']\n",
                "fig, axes = plt.subplots(2, 2, figsize=(15, 13))\n",
                "axes = axes.flatten()\n",
                "\n",
                "for idx, (rot, (e4, _)) in enumerate(HHT.items()):\n",
                "    cm, n = matriz_confusao(e4, [0, 1, 2, 3])\n",
                "    sp = R.summary(e4)['SP'][0]\n",
                "    ax = axes[idx]\n",
                "    im = ax.imshow(cm, cmap='Purples', vmin=0, vmax=100)\n",
                "    ax.set_title(f'{rot} — SP: {sp:.2f}% (n={n})', fontsize=12, fontweight='bold', pad=10)\n",
                "    ax.set_xticks(range(4))\n",
                "    ax.set_yticks(range(4))\n",
                "    ax.set_xticklabels(classes, fontsize=10, fontweight='bold')\n",
                "    ax.set_yticklabels(classes, fontsize=10, fontweight='bold')\n",
                "    ax.set_xlabel('Classe Predita', fontsize=11)\n",
                "    ax.set_ylabel('Classe Real', fontsize=11)\n",
                "    for i in range(4):\n",
                "        for j in range(4):\n",
                "            val = cm[i, j]\n",
                "            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color='white' if val > 45 else 'black',\n",
                "                    fontsize=11, fontweight='bold' if i == j else 'normal')\n",
                "    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)\n",
                "    cbar.set_label('Taxa (%)', fontsize=10)\n",
                "\n",
                "plt.suptitle('Matrizes de Confusão Normalizadas — HHT Pura [4 Classes]', fontsize=15, fontweight='bold', y=0.99)\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4.2 Matrizes de Confusão no Problema de 3 Classes (modelos retreinados: SMALL, MEDIUM, LARGE)\n",
                "\n",
                "Ao excluirmos a classe `AMBIENTAL` (e retreinarmos os modelos), avaliamos a capacidade do espectro de Hilbert-Huang de discriminar os portes de embarcação:\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "classes_3c = ['SMALL', 'MEDIUM', 'LARGE']\n",
                "fig, axes = plt.subplots(2, 2, figsize=(15, 13))\n",
                "axes = axes.flatten()\n",
                "\n",
                "for idx, (rot, (_, e3)) in enumerate(HHT.items()):\n",
                "    cm, n = matriz_confusao(e3, [0, 1, 2])\n",
                "    sp = R.summary(e3, '3c')['SP'][0]\n",
                "    ax = axes[idx]\n",
                "    im = ax.imshow(cm, cmap='BuPu', vmin=0, vmax=100)\n",
                "    ax.set_title(f'{rot} — SP: {sp:.2f}% (n={n})', fontsize=12, fontweight='bold', pad=10)\n",
                "    ax.set_xticks(range(3))\n",
                "    ax.set_yticks(range(3))\n",
                "    ax.set_xticklabels(classes_3c, fontsize=11, fontweight='bold')\n",
                "    ax.set_yticklabels(classes_3c, fontsize=11, fontweight='bold')\n",
                "    ax.set_xlabel('Classe Predita', fontsize=11)\n",
                "    ax.set_ylabel('Classe Real', fontsize=11)\n",
                "    for i in range(3):\n",
                "        for j in range(3):\n",
                "            val = cm[i, j]\n",
                "            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color='white' if val > 45 else 'black',\n",
                "                    fontsize=12, fontweight='bold' if i == j else 'normal')\n",
                "    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)\n",
                "    cbar.set_label('Taxa (%)', fontsize=10)\n",
                "\n",
                "plt.suptitle('Matrizes de Confusão Normalizadas — HHT Pura [3 Classes: modelos retreinados]', fontsize=15, fontweight='bold', y=0.99)\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Análise do Padrão de Erro em SMALL e o que ela permite concluir:\n",
                f"1. **Direção da confusão de SMALL no HHT (3 classes):** SMALL → LARGE em {pt(CM3[lab_rf][0][0, 2], 1)}% das instâncias de SMALL no RF "
                f"e {pt(CM3[lab_cnn][0][0, 2], 1)}% na CNN (colapsada); SMALL → MEDIUM em {pt(CM3[lab_rf][0][0, 1], 1)}% no RF.\n",
                f"2. **SVM MEL (4 classes, o mesmo modelo do Notebook 10):** dos {MEL_SMALL_ERR} erros em {MEL_SMALL_N} instâncias de SMALL, "
                f"{MEL_SMALL_TO_MED} ({pt(100 * MEL_SMALL_TO_MED / MEL_SMALL_ERR, 1)}%) foram para MEDIUM ({pt(100 * MEL_SMALL_TO_MED / MEL_SMALL_N, 2)}% de todas as instâncias de SMALL).\n",
                f"3. **Não é \"complementaridade perfeita\":** nas {MEL_TO_MED_N} instâncias em que o SVM MEL errou SMALL → MEDIUM, o RF HHT (4 classes) acertou {RF_OK} "
                f"({pt(100 * RF_OK / MEL_TO_MED_N, 1)}%) e errou **para a mesma classe** (MEDIUM) em {RF_ALSO_MED} ({pt(100 * RF_ALSO_MED / MEL_TO_MED_N, 1)}%).\n",
                f"4. **Oráculo do Notebook 10 (SVMs MEL, LOFAR, HHT e DEMON):** recall de SMALL do oráculo observado = {pt(ORA4.iloc[0, 0], 1)}%, "
                f"**abaixo** do esperado se os erros fossem independentes ({pt(ORA4.iloc[0, 1], 1)}% = $1-\\prod_i(1-r_i)$). Ou seja, os erros são positivamente correlacionados. "
                "Além disso, um oráculo (acerta se *algum* modelo acerta) não é um sistema realizável: precisa saber a resposta para escolher o modelo.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Conferência dos números da análise acima\n",
                "print('Oráculo dos 4 SVMs do Notebook 10 (observado vs. esperado com erros independentes):')\n",
                "display(R.oracle_recall(list(NB10_SVMS.values())).round(2))\n",
                "excl, tot = exclusivos_small(NB10_SVMS)\n",
                "print(f'Instâncias arquivo x fold de SMALL acertadas por um único SVM (de {tot}):', excl)\n",
                "print('Recall SMALL de cada SVM:', {k: round(R.summary(v)['R_SMALL'][0], 2) for k, v in NB10_SVMS.items()})\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Diagnóstico e Conclusões\n",
                "\n",
                f"### 5.1 O melhor modelo HHT: {rank4[0]} (SP {fmtc(S4[rank4[0]]['SP'])})\n",
                f"* É o melhor HHT em SP médio, mas a diferença para o segundo ({rank4[1]}) "
                + ("é significativa" if F4[1] < 0.05 else "**não** é significativa") + f" ({pval(F4[1])}).\n",
                "* As explicações mecânicas (resiliência a *mode mixing* por cortes ortogonais; invariância a transformações monotônicas) são **hipóteses plausíveis**, não testadas neste notebook.\n",
                "\n",
                "### 5.2 O Confronto HHT vs. DEMON vs. LOFAR\n",
                "**HHT × DEMON no mesmo classificador** (SP em 4 classes, F-test 5×2cv):\n",
                "\n",
                "| Arquitetura | SP HHT | SP DEMON | Δ (p.p.) | p | Observação |\n",
                "| :--- | :---: | :---: | :---: | :---: | :--- |\n",
            ] + [
                f"| {arq} | {fmtc(sh)} | {fmtc(sd)} | {pt(sh[0] - sd[0])} | {pt(ft[1], 3)} | {obs} |\n"
                for arq, sh, sd, ft, obs in PAIR_RES
            ] + [
                "\n",
                f"* A HHT tem SP médio maior que o DEMON nos {len(PAIR_RES)} pares, mas a diferença só é significativa a 5% em "
                + (", ".join(arq for arq, _, _, ft, _ in PAIR_RES if ft[1] < 0.05) or "nenhum") + ". A CNN não entra na comparação porque as duas versões (HHT e DEMON) colapsaram.\n",
                "* As faixas de frequência (HHT: 1–1000 Hz; DEMON: 0,39–25 Hz de modulação) são compatíveis com essa diferença, mas não a explicam por si.\n",
                f"* **LOFAR e MEL seguem à frente:** MLP LOFAR do artigo SP {pt(ART[('MLP', 'LOFAR')]['SP'][0])} (local: {fmtc(REF['mlp_lofar']['SP'])}); SVM MEL local {fmtc(REF['svm_mel']['SP'])}; "
                f"no mesmo classificador (RF), LOFAR local {fmtc(REF['rf_lofar']['SP'])} vs. HHT {fmtc(S4[lab_rf]['SP'])} ({pval(F_RF_LOFAR[1])}).\n",
                "\n",
                "### 5.3 O Papel da HHT no Projeto IARA\n",
                f"1. **SMALL:** no conjunto de SVMs do Notebook 10, o SVM HHT ($m=4000$) tem recall de SMALL de {pt(REC_SMALL_NB10['HHT'])}% "
                f"(o SVM HHT $m=6000$ da Tabela Mestra: {pt(REC_SMALL_HHT6000)}%), contra {pt(REC_SMALL_NB10['MEL'])}% do MEL. "
                f"Em acertos exclusivos de SMALL (instâncias arquivo × fold, de {EXCL_TOT}), HHT = {EXCL['HHT']}, MEL = {EXCL['MEL']}, DEMON = {EXCL['DEMON']}, LOFAR = {EXCL['LOFAR']}: "
                + ("HHT e MEL ficam praticamente empatados" if abs(EXCL['HHT'] - EXCL['MEL']) <= 0.05 * max(EXCL['HHT'], EXCL['MEL']) else
                   ("a HHT tem mais acertos exclusivos" if EXCL['HHT'] > EXCL['MEL'] else "o MEL tem mais acertos exclusivos"))
                + ". Essas contagens são instâncias de fold (a mesma gravação aparece em vários folds) e, como mostra o oráculo acima, não indicam erros independentes.\n",
                "2. **Gating / roteamento dinâmico:** usar a HHT como especialista num sistema de roteamento é uma hipótese a testar com um roteador validado fora do teste e F-test contra a melhor entrada isolada; os dados deste notebook não a demonstram.\n"
            ]
        }
    ]

    notebook_content = {
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
        json.dump(notebook_content, f, indent=2, ensure_ascii=False)

    print(f"Notebook {filename} gerado com sucesso!")


if __name__ == "__main__":
    create_comparacao_hht_notebook()
