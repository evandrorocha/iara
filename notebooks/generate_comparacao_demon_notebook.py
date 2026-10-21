"""
Gera notebooks/7_comparacao_modelos_demon.ipynb: comparação multimodelo com o extrator DEMON puro
(SVM Nyström, Random Forest, MLP, CNN 2D ad hoc) ao lado das referências MEL/LOFAR.

Todos os números (tabelas, matrizes de confusão e valores citados no markdown) são recalculados a partir das
predições por janela em results/trainings/{multirep,tests} por notebooks/iara_results.py. O MESMO código que vai
para as células é executado aqui para gerar os números do markdown.
"""

import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")


def br(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pm(ms, nd=2):
    return f"{br(ms[0], nd)} ± {br(ms[1], nd)}"


def pv(p):
    return "p < 0,001" if p < 0.001 else f"p = {br(p, 3)}"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


SETUP_CODE = r'''
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print

sys.path.insert(0, os.path.abspath('.') if os.path.exists('iara_results.py') else os.path.abspath('notebooks'))
import iara_results as R

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 15, 'figure.dpi': 120})
'''

TABLES_CODE = r'''
# ---- Experimentos em disco (sufixo _4c = 4 classes; _3c = modelo RETREINADO sem BACKGROUND) ----
DEMON = [  # results/trainings/multirep; DEMON 253-D, um vetor a cada 5,12 s
    ('MLP [32,16] 25 ép.', 'mlp_demon_h32-16_e25'),                                # eval_demon_baselines_5x2cv.py
    ('SVM Nyström m=6000', 'svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000'),  # eval_demon_svm_5x2cv.py
    ('SVM Nyström m=4000', 'svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_it1000'),
    ('Random Forest 100 árv.', 'rf_demon_n100_balanced'),                          # eval_demon_baselines_5x2cv.py
    ('SVM Nyström m=1000', 'svm_nystroem_1000_demon_C2.0_elasticnet_l1r0.15_it1000'),
    ('CNN 2D ad hoc', 'cnn2d_demon_w32_b32_e20'),                                  # eval_demon_baselines_5x2cv.py
]
EXTRA = [  # outras configurações DEMON que aparecem em outros notebooks (só para rastreabilidade)
    ('MLP [32,16] 30 ép., seed 42 (NBs 14–16, gate)', 'mlp_demon_h32-16_e30_4c'),     # generate_mlp_predictions_and_analysis.py
    ('MLP [32,16] 35 ép. (grade da fusão, NB8)', 'mlp_demon_h32-16_e35_4c'),           # eval_fusion_mel_demon_5x2cv.py
    ('SVM m=4000 early stopping [INSTÁVEL] (NBs 10–13)', 'svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_4c'),
]
LOCAL_REF = [  # referências MEL/LOFAR do pipeline iara oficial (results/trainings/tests, janelas de 0,512 s)
    ('SVM m=6000 | LOFAR', 'svm_nystroem_6000_lofar_elasticnet_l1r0.15'),
    ('SVM m=4000 PCA64 | LOFAR', 'svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0'),
    ('SVM m=4000 | MEL', 'svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0'),
    ('MLP (oficial) | MEL', 'mlp'),
    ('MLP (oficial) | LOFAR', 'mlp_lofar'),
    ('CNN (oficial) | MEL', 'cnn'),
]
MET = ('SP', 'ACC', 'F1')
ROWS_4C = [(l, e) for l, e in LOCAL_REF] + [(f'DEMON | {l}', e + '_4c') for l, e in DEMON] + \
          [(f'DEMON | {l}', e) for l, e in EXTRA]
ROWS_3C = [(f'DEMON | {l}', e + '_3c', '3c') for l, e in DEMON] + \
          [('DEMON | MLP [32,16] 35 ép. (grade da fusão, NB8)', 'mlp_demon_h32-16_e35_3c', '3c')]

# Tabela 7 do artigo (Silva et al., IEEE Access 2025): só SP, ACC e F1 em 4 classes
tab_artigo = pd.DataFrame([{'Modelo': f'{m} | {r} (Tabela 7 do artigo)',
                            **{k: R.fmt(v) for k, v in R.ARTICLE_TABLE7[(m, r)].items()}}
                           for (m, r) in R.ARTICLE_TABLE7]).set_index('Modelo')
tab_4c = R.table(ROWS_4C, mode='4c', metrics=MET)
tab_sem_bg = R.table(ROWS_4C, mode='4c_sem_bg', metrics=MET)
tab_3c = R.table(ROWS_3C, metrics=MET)

S4 = {l: R.summary(e, '4c') for l, e in ROWS_4C}
SB = {l: R.summary(e, '4c_sem_bg') for l, e in ROWS_4C}
S3 = {l: R.summary(e, m) for l, e, m in ROWS_3C}
E4 = dict(ROWS_4C)
SP_FOLDS = {l: R.fold_metrics(e)['SP'].round(1).tolist() for l, e in ROWS_4C}
SP3_FOLDS = {l: R.fold_metrics(e, m)['SP'].round(1).tolist() for l, e, m in ROWS_3C}
'''

PLOT_GLOBAL_CODE = r'''
lbl4 = [f'DEMON | {l}' for l, _ in DEMON]
short = [l for l, _ in DEMON]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(lbl4)); w = 0.35
for ax, S, title in [(ax1, S4, '4 classes (modelos 4c)'), (ax2, S3, 'Só navios — modelos RETREINADOS sem BG (3c)')]:
    acc = [S[l]['ACC'][0] for l in lbl4]; sp = [S[l]['SP'][0] for l in lbl4]
    ax.bar(x - w / 2, acc, w, yerr=[S[l]['ACC'][1] for l in lbl4], capsize=3, label='ACC (acurácia balanceada)', color='#2980b9', alpha=0.9)
    r = ax.bar(x + w / 2, sp, w, yerr=[S[l]['SP'][1] for l in lbl4], capsize=3, label='SP', color='#e67e22', alpha=0.9)
    for rect in r:
        ax.annotate(f'{rect.get_height():.1f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=25, ha='right', fontweight='bold')
    ax.set_title(title, fontweight='bold'); ax.set_ylabel('Média ± desvio nos 10 folds (%)'); ax.set_ylim(0, 75)
mlp_lofar_art = R.ARTICLE_TABLE7[('MLP', 'LOFAR')]['SP'][0]
svm_mel = S4['SVM m=4000 | MEL']['SP'][0]
ax1.axhline(mlp_lofar_art, color='#27ae60', ls='--', lw=2, label=f'MLP LOFAR, Tabela 7 do artigo ({mlp_lofar_art:.2f}% SP)')
ax1.axhline(svm_mel, color='#8e44ad', ls=':', lw=2, label=f'SVM MEL m=4000 local ({svm_mel:.2f}% SP)')
ax1.legend(loc='lower left', fontsize=9)
ax2.legend(loc='lower left', fontsize=9)
ax2.text(0.5, 0.97, 'Não há referência MEL/LOFAR retreinada sem BG', transform=ax2.transAxes, ha='center', va='top', fontsize=9, style='italic')
plt.suptitle('Modelos com o extrator DEMON (10 folds, 5x2cv oficial)', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.show()
'''

PLOT_RECALL_CODE = r'''
classes = ['SMALL', 'MEDIUM', 'LARGE', 'BACKGROUND']
subset = ['DEMON | MLP [32,16] 25 ép.', 'DEMON | SVM Nyström m=6000', 'DEMON | Random Forest 100 árv.', 'DEMON | CNN 2D ad hoc']
recalls_matrix = np.array([[S4[m][f'R_{c}'][0] for c in classes] for m in subset])
display(pd.DataFrame(recalls_matrix.round(2), index=subset, columns=classes))
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(classes)); width = 0.20
for i, (m, color) in enumerate(zip(subset, ['#3498db', '#9b59b6', '#e67e22', '#e74c3c'])):
    rects = ax.bar(x + (i - 1.5) * width, recalls_matrix[i], width, label=m.replace('DEMON | ', ''), color=color, alpha=0.9)
    for rect in rects:
        ax.annotate(f'{rect.get_height():.1f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_title('Recall por classe (média nos 10 folds) — 4 classes, DEMON', fontweight='bold', pad=15)
ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=12, fontweight='bold')
ax.set_ylabel('Recall (%)'); ax.set_ylim(0, 95)
ax.legend(title='Arquitetura', loc='upper right', frameon=True)
plt.tight_layout()
plt.show()
'''

CONF_CODE = r'''
CLS = ['SMALL', 'MEDIUM', 'LARGE', 'BACKGROUND']


def confusion(exp):
    """Matriz de confusão por arquivo, ACUMULADA nas 10 dobras (instâncias arquivo x fold), normalizada por linha (%)."""
    df = pd.concat(R.file_predictions(exp).values(), ignore_index=True)
    rows = sorted(df.Target.unique())
    cols = sorted(set(rows) | set(df.Prediction.unique()))
    cm = pd.crosstab(df.Target, df.Prediction).reindex(index=rows, columns=cols, fill_value=0)
    cmn = 100 * cm.div(cm.sum(axis=1), axis=0)
    cmn.index = [CLS[i] for i in rows]; cmn.columns = [CLS[i] for i in cols]
    return cmn, int(cm.values.sum())


CM_MODELS = [('SVM Nyström m=6000', 'svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000'),
             ('Random Forest 100 árv.', 'rf_demon_n100_balanced'),
             ('MLP [32,16] 25 ép.', 'mlp_demon_h32-16_e25'),
             ('CNN 2D ad hoc', 'cnn2d_demon_w32_b32_e20')]
CM4 = {l: confusion(e + '_4c') for l, e in CM_MODELS}
CM3 = {l: confusion(e + '_3c') for l, e in CM_MODELS}
N_FILES = len(set().union(*[set(d.File) for d in R.file_predictions(CM_MODELS[0][1] + '_4c').values()]))


def plot_cms(CMS, S, prefix, cmap, suptitle):
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    for ax, (l, (cm, n)) in zip(axes.flatten(), CMS.items()):
        sp = S[f'DEMON | {l}']['SP']
        im = ax.imshow(cm.values, cmap=cmap, vmin=0, vmax=100)
        ax.set_title(f'{l} — SP {sp[0]:.2f} ± {sp[1]:.2f} % (n = {n})', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(range(cm.shape[1])); ax.set_xticklabels(cm.columns, fontweight='bold')
        ax.set_yticks(range(cm.shape[0])); ax.set_yticklabels(cm.index, fontweight='bold')
        ax.set_xlabel('Classe predita'); ax.set_ylabel('Classe real')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm.values[i, j]
                ax.text(j, i, f'{v:.1f}%', ha='center', va='center', color='white' if v > 45 else 'black',
                        fontweight='bold' if i == j else 'normal')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Taxa (%)')
    plt.suptitle(suptitle, fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.show()
'''

TESTS_CODE = r'''
# F-test 5x2cv combinado (Alpaydin), como no artigo. Modelos 4c; recall de SMALL é idêntico em 4c e 4c_sem_bg.
PAIRS = [
    ('DEMON | MLP [32,16] 25 ép.', 'MLP (oficial) | LOFAR', 'SP'),
    ('DEMON | MLP [32,16] 25 ép.', 'SVM m=4000 | MEL', 'SP'),
    ('DEMON | MLP [32,16] 25 ép.', 'DEMON | SVM Nyström m=6000', 'SP'),
    ('DEMON | MLP [32,16] 25 ép.', 'DEMON | Random Forest 100 árv.', 'SP'),
    ('DEMON | MLP [32,16] 25 ép.', 'DEMON | MLP [32,16] 35 ép. (grade da fusão, NB8)', 'SP'),
    ('DEMON | Random Forest 100 árv.', 'SVM m=6000 | LOFAR', 'R_SMALL'),
    ('DEMON | Random Forest 100 árv.', 'SVM m=4000 PCA64 | LOFAR', 'R_SMALL'),
    ('DEMON | Random Forest 100 árv.', 'MLP (oficial) | LOFAR', 'R_SMALL'),
    ('DEMON | Random Forest 100 árv.', 'SVM m=4000 | MEL', 'R_SMALL'),
]
rows = []
for a, b, met in PAIRS:
    F, p = R.f_test_5x2(E4[a], E4[b], metric=met)
    rows.append({'A': a, 'B': b, 'métrica': met, 'média A': round(S4[a][met][0], 2), 'média B': round(S4[b][met][0], 2),
                 'F': round(F, 2), 'p': round(p, 4), 'p < 0,05': p < 0.05})
df_tests = pd.DataFrame(rows)
display(df_tests)
'''


def build(filename=os.path.join(HERE, "7_comparacao_modelos_demon.ipynb")):
    ns = {}
    exec(SETUP_CODE, ns)
    exec(TABLES_CODE, ns)
    exec(CONF_CODE, ns)
    ns["display"] = lambda *a, **k: None
    exec(TESTS_CODE, ns)
    R = ns["R"]
    S4, SB, S3, T = ns["S4"], ns["SB"], ns["S3"], ns["df_tests"]
    CM4, CM3 = ns["CM4"], ns["CM3"]

    def t(a, b, met):
        r = T[(T.A == a) & (T.B == b) & (T["métrica"] == met)].iloc[0]
        return f"F = {br(r.F)}, {pv(r.p)}", r.p

    mlp25, mlp30, mlp35 = ("DEMON | MLP [32,16] 25 ép.", "DEMON | MLP [32,16] 30 ép., seed 42 (NBs 14–16, gate)",
                           "DEMON | MLP [32,16] 35 ép. (grade da fusão, NB8)")
    svm6, svm4, svm1 = "DEMON | SVM Nyström m=6000", "DEMON | SVM Nyström m=4000", "DEMON | SVM Nyström m=1000"
    rf, cnn = "DEMON | Random Forest 100 árv.", "DEMON | CNN 2D ad hoc"
    uns = "DEMON | SVM m=4000 early stopping [INSTÁVEL] (NBs 10–13)"
    lof6, lof4, mel4 = "SVM m=6000 | LOFAR", "SVM m=4000 PCA64 | LOFAR", "SVM m=4000 | MEL"
    mlpl, mlpm, cnnm = "MLP (oficial) | LOFAR", "MLP (oficial) | MEL", "CNN (oficial) | MEL"
    art_mlpl = R.ARTICLE_TABLE7[("MLP", "LOFAR")]["SP"]
    demon4 = [f"DEMON | {l}" for l, _ in ns["DEMON"]]
    best = max(demon4, key=lambda l: S4[l]["SP"][0])
    allD = demon4 + [mlp30, mlp35]
    bestall = max(allD, key=lambda l: S4[l]["SP"][0])
    cm_rf, cm_mlp, cm_svm = CM4["Random Forest 100 árv."][0], CM4["MLP [32,16] 25 ép."][0], CM4["SVM Nyström m=6000"][0]
    c3 = {k: v[0] for k, v in CM3.items()}
    N4 = CM4["SVM Nyström m=6000"][1]
    fcnn = ns["SP_FOLDS"][cnn]
    fcnn3 = ns["SP3_FOLDS"][cnn]

    bigru_sp = R.summary('bigru_att_demon_h64_s15-5_e20_4c')['SP']
    t_lofar, p_lofar = t(mlp25, mlpl, "SP")
    t_mel, p_mel = t(mlp25, mel4, "SP")
    t_svm, p_svm = t(mlp25, svm6, "SP")
    t_rf, p_rf = t(mlp25, rf, "SP")
    t_35, p_35 = t(mlp25, mlp35, "SP")

    def sig(p):
        return "significativa" if p < 0.05 else "**não** significativa"

    cells = []
    cells.append(md(r"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 7: Comparativo Multimodelo com o Extrator DEMON (SVM, Random Forest, MLP e CNN)

Neste notebook, investigamos o comportamento do extrator **DEMON** (*Demodulation of Envelope Modulation on Noise*) puro ($0{,}39\text{ Hz}$ a $25\text{ Hz}$) através de **quatro famílias de aprendizado de máquina**:

1. **SVM Nyström ($m=1000, 4000, 6000$):** núcleo RBF aproximado por *landmark points* com penalidade ElasticNet (`max_iter=1000`, sem early stopping).
2. **Random Forest (100 árvores):** ensemble de árvores com `class_weight='balanced'`.
3. **Multi-Layer Perceptron (MLP PyTorch `[32, 16]`):** rede feedforward com `BatchNorm1d` e `Dropout 0.2`.
4. **CNN 2D ad hoc (PyTorch):** convoluções 2D sobre blocos de 32 vetores DEMON consecutivos ($32 \times 253$).

Todos os modelos foram avaliados sob o protocolo oficial de **$5\times2$ validação cruzada (10 folds)** com **particionamento por gravação de áudio (`ID`, `exclusive_header='ID'` em `src/iara/ml/experiment.py`)** — janelas de um mesmo arquivo nunca ficam em treino e teste ao mesmo tempo — sobre os **__NFILES__ áudios de UO** do IARA (*Silva et al.*, IEEE Access 2025). O particionamento **não** isola navios: gravações diferentes do mesmo navio (`Ship ID`) podem cair em treino e teste.

> Todos os números são recalculados das predições por janela em `results/trainings/` por `notebooks/iara_results.py` (voto majoritário por arquivo, SP de `src/iara/ml/metrics.py`, **ACC = acurácia balanceada**, média ± desvio populacional nos 10 folds).
"""))
    cells[0]["source"] = [x.replace("__NFILES__", br(ns["N_FILES"], 0)) for x in cells[0]["source"]]
    cells.append(code(SETUP_CODE))
    cells.append(md(f"""
## 1. Tabela Mestra: DEMON vs. Referências MEL/LOFAR

> **Notas metodológicas:**
> * **ACC = acurácia balanceada** (média dos recalls) em todas as tabelas, inclusive na Tabela 7 do artigo. Versões anteriores deste notebook misturavam acurácia micro (UACC) e balanceada na mesma coluna.
> * **Três modos, sempre rotulados:** `4c` (4 classes); `4c_sem_bg` (o mesmo modelo 4c avaliado só em navios, prever BACKGROUND = erro; os recalls por classe são os mesmos do 4c); `3c` (modelo **retreinado** sem BACKGROUND — só existe para o DEMON). Comparações entre DEMON e MEL/LOFAR "em 3 classes" só são válidas no modo `4c_sem_bg`.
> * **Referências locais MEL/LOFAR** vêm do pipeline iara oficial (`results/trainings/tests`, janelas de 0,512 s); o DEMON usa um vetor a cada 5,12 s. O recall de `SMALL` do `SVM m=6000 | LOFAR` é **{br(S4[lof6]['R_SMALL'][0])}%** (versões anteriores traziam 42,58%).
> * **Três MLP DEMON [32,16] distintos** existem no projeto: **25 épocas** (`mlp_demon_h32-16_e25_*`, `eval_demon_baselines_5x2cv.py`, usado aqui; SP 4c {pm(S4[mlp25]['SP'])}%), **30 épocas com seed 42** (`mlp_demon_h32-16_e30_4c`, `generate_mlp_predictions_and_analysis.py`, usado nos NBs 14–16 e no gate; SP {pm(S4[mlp30]['SP'])}%) e **35 épocas** (`mlp_demon_h32-16_e35_*`, `eval_fusion_mel_demon_5x2cv.py`, usado no NB8; SP {pm(S4[mlp35]['SP'])}%). A diferença 25 × 35 épocas não é significativa (F-test 5x2cv: {t_35}).
> * O `SVM m=4000` com early stopping (`svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_4c`, usado nos NBs 10–13) é **instável** (SP {pm(S4[uns]['SP'])}%; folds: {', '.join(br(x, 1) for x in ns['SP_FOLDS'][uns])}) e aparece só para rastreabilidade.
"""))
    cells.append(code(TABLES_CODE + r'''
print('Tabela 7 do artigo (4 classes):'); display(tab_artigo)
print('Avaliação local, 4 classes (modelos 4c):'); display(tab_4c)
print('Avaliação local, só navios, modelos 4c (prever BACKGROUND = erro):'); display(tab_sem_bg)
print('Só navios, modelos DEMON RETREINADOS sem BACKGROUND (3c):'); display(tab_3c)
'''))
    cells.append(md(r"""
## 2. Comparativo Global: o Teto do DEMON Isolado

Índice SP e acurácia balanceada (média ± desvio nos 10 folds) no problema de 4 classes e nos modelos retreinados sem BACKGROUND (3c):
"""))
    cells.append(code(PLOT_GLOBAL_CODE))
    cells.append(md(r"""
## 3. Recall por Classe

Cada modelo extrai comportamentos distintos do envelope DEMON (recall médio nos 10 folds, modelos 4c):
"""))
    cells.append(code(PLOT_RECALL_CODE))
    cells.append(md(f"""
### Principais Fenômenos Observados (valores calculados acima e nas matrizes da seção 4):
1. **Random Forest e SMALL ({br(S4[rf]['R_SMALL'][0])}%):** o RF tem o maior recall de SMALL entre os modelos DEMON 4c. *Hipótese:* cortes ortogonais em bins isolados não são diluídos pelos demais bins, como ocorre num produto escalar global.
2. **O colapso do RF em BACKGROUND ({br(S4[rf]['R_BACKGROUND'][0])}%):** o RF classifica **{br(cm_rf.loc['BACKGROUND', 'SMALL'], 1)}%** das instâncias de fundo como SMALL (matriz acumulada). Parte do recall de SMALL do RF vem, portanto, de um viés para SMALL, não de discriminação.
3. **O equilíbrio do MLP ({br(S4[mlp25]['SP'][0])}% SP):** o MLP mantém **{br(S4[mlp25]['R_BACKGROUND'][0])}% de BACKGROUND** e envia {br(cm_mlp.loc['SMALL', 'LARGE'], 1)}% dos SMALL para LARGE (contra {br(cm_svm.loc['SMALL', 'LARGE'], 1)}% no SVM m=6000). A vantagem de SP do MLP sobre o SVM m=6000 é {sig(p_svm)} ({t_svm}); sobre o RF é {sig(p_rf)} ({t_rf}).
"""))
    cells.append(md(f"""
## 4. Matrizes de Confusão Acumuladas nos 10 Folds ($5\\times2$cv)

### 4.1 Problema de 4 Classes (SMALL, MEDIUM, LARGE, BACKGROUND)
Matrizes por arquivo, **acumuladas** nas 10 dobras (n = {br(N4, 0)} instâncias arquivo × fold; cada arquivo aparece em média {br(N4 / ns['N_FILES'])} vezes no teste) e normalizadas por linha. A diagonal acumulada difere ligeiramente do recall médio por fold da tabela.
"""))
    cells.append(code(CONF_CODE + r'''
plot_cms(CM4, S4, 'DEMON', 'Blues', 'Matrizes de confusão normalizadas — DEMON isolado [4 classes]')
'''))
    cells.append(md(r"""
### 4.2 Modelos Retreinados sem BACKGROUND (3c: SMALL, MEDIUM, LARGE)

Modelos **retreinados** só com as três classes de navio (experimentos `*_3c`). Isto mede a capacidade do envelope DEMON de separar os portes quando o fundo é removido do treino **e** do teste:
"""))
    cells.append(code(r'''
plot_cms(CM3, S3, 'DEMON', 'YlGnBu', 'Matrizes de confusão normalizadas — DEMON isolado [3c, retreinado sem BACKGROUND]')
'''))
    sm4 = {k: CM4[k][0].loc["SMALL", "BACKGROUND"] for k in CM4}
    cells.append(md(f"""
### Análise Comparativa do Comportamento em 3 Classes (3c):
1. **SMALL sobe quando o BACKGROUND é retirado do problema:**
   - No 4c, a fração de SMALL classificada como BACKGROUND é {br(sm4['MLP [32,16] 25 ép.'], 1)}% (MLP), {br(sm4['SVM Nyström m=6000'], 1)}% (SVM m=6000), {br(sm4['Random Forest 100 árv.'], 1)}% (RF) e {br(sm4['CNN 2D ad hoc'], 1)}% (CNN).
   - Nos modelos 3c, o recall de SMALL é {br(S3[mlp25]['R_SMALL'][0])}% (MLP; {br(S4[mlp25]['R_SMALL'][0])}% no 4c), {br(S3['DEMON | Random Forest 100 árv.']['R_SMALL'][0])}% (RF; {br(S4[rf]['R_SMALL'][0])}% no 4c) e {br(S3[svm6]['R_SMALL'][0])}% (SVM m=6000; {br(S4[svm6]['R_SMALL'][0])}% no 4c). É outro problema (3 classes), não um ganho do DEMON comparável aos modelos 4c.
2. **A confusão SMALL → LARGE persiste:**
   - No 3c, {br(c3['MLP [32,16] 25 ép.'].loc['SMALL', 'LARGE'], 1)}% (MLP), {br(c3['SVM Nyström m=6000'].loc['SMALL', 'LARGE'], 1)}% (SVM m=6000) e {br(c3['Random Forest 100 árv.'].loc['SMALL', 'LARGE'], 1)}% (RF) dos SMALL ainda são preditos como LARGE.
   - *Hipótese física (não testada aqui):* em baixa velocidade, lanchas produzem pouca cavitação na banda de 1–2 kHz e o envelope passa a ser dominado por modulações lentas, parecidas com as de navios grandes.
3. **MEDIUM é atraído para as duas pontas:** no 3c, {br(min(c3[k].loc['MEDIUM', 'SMALL'] for k in ['MLP [32,16] 25 ép.', 'SVM Nyström m=6000', 'Random Forest 100 árv.']), 1)}% a {br(max(c3[k].loc['MEDIUM', 'SMALL'] for k in ['MLP [32,16] 25 ép.', 'SVM Nyström m=6000', 'Random Forest 100 árv.']), 1)}% dos MEDIUM vão para SMALL e {br(min(c3[k].loc['MEDIUM', 'LARGE'] for k in ['MLP [32,16] 25 ép.', 'SVM Nyström m=6000', 'Random Forest 100 árv.']), 1)}% a {br(max(c3[k].loc['MEDIUM', 'LARGE'] for k in ['MLP [32,16] 25 ép.', 'SVM Nyström m=6000', 'Random Forest 100 árv.']), 1)}% para LARGE (MLP, SVM m=6000 e RF).
"""))
    cells.append(md(r"""
## 5. Testes Estatísticos (F-test 5x2cv)

Toda afirmação de "supera" abaixo depende deste teste (Alpaydin, 1999; o mesmo usado no artigo):
"""))
    cells.append(code(TESTS_CODE))

    t_s6, p_s6 = t(rf, lof6, "R_SMALL")
    t_s4, p_s4 = t(rf, lof4, "R_SMALL")
    t_sm, p_sm = t(rf, mlpl, "R_SMALL")
    t_smel, p_smel = t(rf, mel4, "R_SMALL")
    _sig = [n for n, p in [('SVM LOFAR m=6000', p_s6), ('SVM LOFAR m=4000 PCA64', p_s4), ('MLP LOFAR local', p_sm), ('SVM MEL m=4000', p_smel)] if p < 0.05]
    sig_small = ("Nenhuma dessas diferenças é significativa no F-test 5x2cv (p ≥ 0,05): o recall de SMALL do RF DEMON é **equivalente** ao das referências, dentro da variação entre folds" if not _sig
                 else "No F-test 5x2cv, a diferença é significativa (p < 0,05) apenas contra " + ", ".join(_sig) + "; contra as demais referências o RF DEMON é equivalente")
    cells.append(md(f"""
## 6. Diagnóstico Científico e Conclusões

### 6.1 A CNN 2D ad hoc colapsou no DEMON
* SP 4c = **{pm(S4[cnn]['SP'])}%** (folds: {', '.join(br(x, 1) for x in fcnn)}), recall SMALL {br(S4[cnn]['R_SMALL'][0])}%; no 3c, SP = {pm(S3[cnn]['SP'])}% (folds: {', '.join(br(x, 1) for x in fcnn3)}). Com esse desempenho e essa variância, o modelo deve ser tratado como **colapsado** (20 épocas sem early stopping, conforme `config.json`), e não como evidência sobre a adequação de convoluções ao DEMON.
* *Hipótese (não testada):* o espectro DEMON é formado por raias finas e esparsas, e convoluções com pooling tenderiam a suavizá-las.

### 6.2 O teto do DEMON isolado
* O melhor SP 4c entre os modelos DEMON desta tabela é **{pm(S4[best]['SP'])}%** ({best.replace('DEMON | ', '')}); a melhor configuração DEMON em disco nas 4 famílias é {bestall.replace('DEMON | ', '')} com {pm(S4[bestall]['SP'])}%. Fora dessas 4 famílias, a BiGRU com atenção sobre sequências DEMON (`bigru_att_demon_h64_s15-5_e20_4c`, Notebook 23) chega a {pm(bigru_sp)}%.
* Isso fica **{br(art_mlpl[0] - S4[best]['SP'][0], 1)} p.p.** abaixo da MLP LOFAR da Tabela 7 do artigo ({pm(art_mlpl)}%), {br(S4[mlpl]['SP'][0] - S4[best]['SP'][0], 1)} p.p. abaixo da MLP LOFAR local ({pm(S4[mlpl]['SP'])}%; F-test {t_lofar}) e {br(S4[mel4]['SP'][0] - S4[best]['SP'][0], 1)} p.p. abaixo do SVM MEL m=4000 ({pm(S4[mel4]['SP'])}%; F-test {t_mel}).
* Como a diferença se repete nas 4 famílias, os dados **sugerem** que a limitação está na representação: o DEMON descarta o espectro direto (raias de máquinas, energia de banda larga) e retém só a envoltória de 1–2 kHz entre 0,39 e 25 Hz.

### 6.3 O papel do DEMON na arquitetura IARA
1. **Não usar monoliticamente:** o DEMON não deve ser o classificador primário de 4 classes.
2. **Fusão Multimodal (`MEL + DEMON`):** avaliada no Notebook 8.
3. **Especialista entre embarcações — o que os dados permitem dizer:**
   - Comparação válida em SMALL (modelos 4c, mesmo modo de avaliação): RF DEMON {br(S4[rf]['R_SMALL'][0])}% contra {br(S4[lof6]['R_SMALL'][0])}% (SVM LOFAR m=6000; {t_s6}), {br(S4[lof4]['R_SMALL'][0])}% (SVM LOFAR m=4000 PCA64; {t_s4}), {br(S4[mlpl]['R_SMALL'][0])}% (MLP LOFAR local; {t_sm}) e {br(S4[mel4]['R_SMALL'][0])}% (SVM MEL m=4000; {t_smel}). {sig_small}, e o recall de SMALL do RF vem junto com {br(cm_rf.loc['BACKGROUND', 'SMALL'], 1)}% do fundo classificado como SMALL (seção 3).
   - Uma versão anterior afirmava que o DEMON "supera com folga o LOFAR" em SMALL, comparando o recall de modelos DEMON **retreinados sem BG** ({br(S3[svm6]['R_SMALL'][0])}% SVM m=6000, {br(S3['DEMON | Random Forest 100 árv.']['R_SMALL'][0])}% RF) com o recall de um LOFAR **4c**. Essa comparação mistura problemas diferentes e foi retirada. Para sustentar o DEMON como especialista seria preciso retreinar também MEL/LOFAR sem BG e aplicar o F-test.
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py",
                              "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python",
                              "pygments_lexer": "ipython3", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Notebook {filename} gerado com sucesso!")


if __name__ == "__main__":
    build()
