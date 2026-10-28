"""
Gera notebooks/12_comparacao_multimodal_mel_demon_lofar.ipynb: comparação de MEL, DEMON, LOFAR e FUSÃO (MEL+DEMON)
em MLP, SVM Nyström, Random Forest e CNN 2D sob o protocolo 5x2cv.

Todos os números (tabelas, matrizes de confusão, testes e valores citados no markdown) são recalculados a partir
das predições por janela em results/trainings/{multirep,tests} por notebooks/iara_results.py. O MESMO código que vai
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

for _p in ('.', 'src', 'notebooks', 'notebooks/src'):
    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))
import iara_results as R

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 15, 'figure.dpi': 130})
'''

TABLES_CODE = r'''
MODELS = ['MLP', 'SVM Nyström', 'Random Forest', 'CNN 2D']
REPS = ['DEMON', 'MEL', 'LOFAR', 'FUSÃO (MEL+DEMON)']
# (modelo, representação) -> (rótulo, experimento 4c, experimento 3c retreinado ou None, grade temporal, observação)
GRID = {
    ('MLP', 'DEMON'): ('MLP [32,16] 35 ép. | DEMON', 'mlp_demon_h32-16_e35_4c', 'mlp_demon_h32-16_e35_3c', '5,12 s', 'eval_fusion_mel_demon_5x2cv.py'),
    ('MLP', 'MEL'): ('MLP [32,16] 35 ép. | MEL 5,12 s', 'mlp_mel5s_h32-16_e35_4c', 'mlp_mel5s_h32-16_e35_3c', '5,12 s', 'eval_fusion_mel_demon_5x2cv.py'),
    ('MLP', 'LOFAR'): ('MLP ≈ artigo | LOFAR', 'mlp_lofar', None, '0,512 s', 'pipeline iara (early stopping); arquitetura do artigo, mas paciência 16 em vez de 8 e, dropout 0,2 em vez de 0'),
    ('MLP', 'FUSÃO (MEL+DEMON)'): ('MLP [64,32] 35 ép. | FUSÃO', 'mlp_mel5s-demon_h64-32_e35_4c', 'mlp_mel5s-demon_h64-32_e35_3c', '5,12 s', 'eval_fusion_mel_demon_5x2cv.py (rede MAIOR)'),
    ('SVM Nyström', 'DEMON'): ('SVM m=6000 | DEMON', 'svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000_4c', 'svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000_3c', '5,12 s', 'eval_demon_svm_5x2cv.py'),
    ('SVM Nyström', 'MEL'): ('SVM m=4000 | MEL', 'svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0', None, '0,512 s', 'pipeline iara (m=4000, não 6000)'),
    ('SVM Nyström', 'LOFAR'): ('SVM m=6000 | LOFAR', 'svm_nystroem_6000_lofar_elasticnet_l1r0.15', None, '0,512 s', 'pipeline iara'),
    ('SVM Nyström', 'FUSÃO (MEL+DEMON)'): ('SVM m=6000 | FUSÃO', 'svm_nystroem_6000_mel5s-demon_C2.0_elasticnet_l1r0.15_4c', 'svm_nystroem_6000_mel5s-demon_C2.0_elasticnet_l1r0.15_3c', '5,12 s', 'eval_fusion_svm_cnn_5x2cv.py'),
    ('Random Forest', 'DEMON'): ('RF 100 árv. | DEMON', 'rf_demon_n100_balanced_4c', 'rf_demon_n100_balanced_3c', '5,12 s', 'eval_demon_baselines_5x2cv.py'),
    ('Random Forest', 'MEL'): ('RF 100 árv. | MEL 5,12 s', 'rf_mel5s_n100_balanced_4c', None, '5,12 s', 'generate_rf_predictions_and_analysis.py'),
    ('Random Forest', 'LOFAR'): ('RF 100 árv. | LOFAR 5,12 s', 'rf_lofar5s_n100_balanced_4c', None, '5,12 s', 'generate_rf_predictions_and_analysis.py'),
    ('Random Forest', 'FUSÃO (MEL+DEMON)'): ('RF 100 árv. | FUSÃO', 'rf_mel5s-demon_n100_balanced_4c', 'rf_mel5s-demon_n100_balanced_3c', '5,12 s', 'eval_rf_fusion_5x2cv.py'),
    ('CNN 2D', 'DEMON'): ('CNN 2D ad hoc | DEMON', 'cnn2d_demon_w32_b32_e20_4c', 'cnn2d_demon_w32_b32_e20_3c', '5,12 s', 'eval_demon_baselines_5x2cv.py [COLAPSADA]'),
    ('CNN 2D', 'MEL'): ('CNN repo | MEL', 'cnn', None, '0,512 s', 'pipeline iara'),
    ('CNN 2D', 'LOFAR'): ('CNN 2D ad hoc | LOFAR 5,12 s', 'cnn2d_lofar5s_w32_b64_e15_4c', None, '5,12 s', 'generate_cnn_predictions_and_analysis.py [COLAPSADA]'),
    ('CNN 2D', 'FUSÃO (MEL+DEMON)'): ('CNN 2D ad hoc | FUSÃO', 'cnn2d_mel5s-demon_w32_b32_e25_4c', 'cnn2d_mel5s-demon_w32_b32_e25_3c', '5,12 s', 'eval_fusion_svm_cnn_5x2cv.py [COLAPSADA]'),
}
S4 = {k: R.summary(v[1], '4c') for k, v in GRID.items()}
SB = {k: R.summary(v[1], '4c_sem_bg') for k, v in GRID.items()}
S3 = {k: R.summary(v[2], '3c') for k, v in GRID.items() if v[2]}


def row(k, S, extra=None):
    lbl, e4, e3, grade, obs = GRID[k]
    s = S[k]
    d = {'Modelo': k[0], 'Representação': k[1], 'Configuração': lbl, 'Grade': grade,
         'SP': R.fmt(s['SP']), 'ACC (bal.)': R.fmt(s['ACC']), 'F1': R.fmt(s['F1']),
         'Rec SMALL': round(s['R_SMALL'][0], 2), 'Rec MEDIUM': round(s['R_MEDIUM'][0], 2), 'Rec LARGE': round(s['R_LARGE'][0], 2)}
    if 'R_BACKGROUND' in s:
        d['Rec BG'] = round(s['R_BACKGROUND'][0], 2)
    d['Origem'] = obs
    d['Experimento'] = e3 if S is S3 else e4
    return d


KEYS = [(m, r) for m in MODELS for r in REPS]
df_master = pd.DataFrame([row(k, S4) for k in KEYS])
df_sem_bg = pd.DataFrame([row(k, SB) for k in KEYS])
df_3c = pd.DataFrame([row(k, S3) for k in KEYS if k in S3])

tab_artigo = pd.DataFrame([{'Par': f'{m} | {r}', **{k: R.fmt(v) for k, v in R.ARTICLE_TABLE7[(m, r)].items()}}
                           for (m, r) in R.ARTICLE_TABLE7]).set_index('Par')
LOCAL_OF_ARTICLE = {('RF', 'MEL'): ('Random Forest', 'MEL'), ('RF', 'LOFAR'): ('Random Forest', 'LOFAR'),
                    ('MLP', 'MEL'): None, ('MLP', 'LOFAR'): ('MLP', 'LOFAR'),
                    ('CNN', 'MEL'): ('CNN 2D', 'MEL'), ('CNN', 'LOFAR'): ('CNN 2D', 'LOFAR')}
rast = []
for (m, r), v in R.ARTICLE_TABLE7.items():
    if LOCAL_OF_ARTICLE[(m, r)] is None:
        lbl, exp, grade = 'MLP ≈ artigo | MEL', 'mlp', '0,512 s'
    else:
        lbl, exp, _, grade, _ = GRID[LOCAL_OF_ARTICLE[(m, r)]]
    s = R.summary(exp)
    rast.append({'Par': f'{m} | {r}', 'Artigo SP': R.fmt(v['SP']), 'Artigo ACC': R.fmt(v['ACC']),
                 'Local': lbl, 'Grade local': grade, 'Local SP': R.fmt(s['SP']), 'Local ACC': R.fmt(s['ACC']),
                 'Local Rec SMALL': round(s['R_SMALL'][0], 2), 'Experimento': exp})
df_rast = pd.DataFrame(rast).set_index('Par')

MLP_DEMON = [('25 ép. (NB9)', 'mlp_demon_h32-16_e25_4c'), ('30 ép., seed 42 (NBs 16–18, gate)', 'mlp_demon_h32-16_e30_4c'),
             ('35 ép. (grade da fusão, este NB)', 'mlp_demon_h32-16_e35_4c')]
tab_mlp_demon = R.table([(f'MLP [32,16] DEMON {l}', e) for l, e in MLP_DEMON], metrics=('SP', 'ACC', 'F1'))
SP_FOLDS = {k: R.fold_metrics(GRID[k][1])['SP'].round(1).tolist() for k in KEYS}
'''

PLOT_SP_CODE = r'''
fig, ax = plt.subplots(figsize=(15, 7))
x = np.arange(len(MODELS)); width = 0.20
colors = {'DEMON': '#95a5a6', 'MEL': '#3498db', 'LOFAR': '#e67e22', 'FUSÃO (MEL+DEMON)': '#27ae60'}
for i, r in enumerate(REPS):
    vals = [S4[(m, r)]['SP'][0] for m in MODELS]
    errs = [S4[(m, r)]['SP'][1] for m in MODELS]
    rects = ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=3, label=r, color=colors[r], alpha=0.9)
    for m, rect in zip(MODELS, rects):
        h = rect.get_height()
        coll = 'COLAPSADA' in GRID[(m, r)][4]
        ax.annotate(f'{h:.1f}' + ('*' if coll else ''), xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                    textcoords='offset points', ha='center', va='bottom', fontsize=9, color='#c0392b' if coll else 'black')
ax.set_title('SP (média ± desvio, 10 folds) — 4 modelos × 4 representações, 4 classes\n* = modelo ad hoc colapsado (não usar como baseline)',
             fontweight='bold', fontsize=13, pad=15)
ax.set_ylabel('SP (%)', fontweight='bold'); ax.set_xticks(x); ax.set_xticklabels(MODELS, fontweight='bold')
ax.set_ylim(0, 78); ax.legend(loc='lower left', frameon=True, fontsize=10.5)
plt.tight_layout()
plt.show()
'''

PLOT_SMALL_CODE = r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
m3 = MODELS[:3]
x = np.arange(len(m3)); w = 0.2
for ax, S, title in [(ax1, S4, 'Recall SMALL — modelos 4c (%)'), (ax2, S3, 'Recall SMALL — modelos RETREINADOS sem BG (3c) (%)')]:
    for i, (r, c) in enumerate([('DEMON', '#95a5a6'), ('MEL', '#3498db'), ('LOFAR', '#e67e22'), ('FUSÃO (MEL+DEMON)', '#27ae60')]):
        vals = [S[(m, r)]['R_SMALL'][0] if (m, r) in S else np.nan for m in m3]
        rects = ax.bar(x + (i - 1.5) * w, vals, w, label=r, color=c)
        for rect, v in zip(rects, vals):
            ax.annotate('n/d' if np.isnan(v) else f'{v:.1f}', xy=(rect.get_x() + rect.get_width() / 2, 0 if np.isnan(v) else v),
                        xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    ax.set_title(title, fontweight='bold'); ax.set_xticks(x); ax.set_xticklabels(m3, fontweight='bold')
    ax.set_ylim(0, 75); ax.legend(fontsize=9)
ax2.text(0.5, 0.97, 'n/d = não há modelo retreinado sem BG para esse par', transform=ax2.transAxes, ha='center', va='top', fontsize=9, style='italic')
plt.tight_layout()
plt.show()
'''

CONF_CODE = r'''
CLS = ['SMALL', 'MEDIUM', 'LARGE', 'BACKGROUND']


def confusion(exp):
    """Matriz por arquivo, ACUMULADA nas 10 dobras (instâncias arquivo x fold), normalizada por linha (%)."""
    df = pd.concat(R.file_predictions(exp).values(), ignore_index=True)
    rows = sorted(df.Target.unique()); cols = sorted(set(rows) | set(df.Prediction.unique()))
    cm = pd.crosstab(df.Target, df.Prediction).reindex(index=rows, columns=cols, fill_value=0)
    cmn = 100 * cm.div(cm.sum(axis=1), axis=0)
    cmn.index = [CLS[i] for i in rows]; cmn.columns = [CLS[i] for i in cols]
    return cmn


CM_KEYS = [('MLP', 'FUSÃO (MEL+DEMON)'), ('SVM Nyström', 'FUSÃO (MEL+DEMON)')]
CMS = {k: confusion(GRID[k][1]) for k in CM_KEYS}
'''

PLOT_CONF_CODE = r'''
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, k, cmap in zip(axes, CM_KEYS, ['Blues', 'Greens']):
    cm = CMS[k]
    ax.imshow(cm.values, cmap=cmap, vmin=0, vmax=100)
    sp = S4[k]['SP']
    ax.set_title(f'{GRID[k][0]} (SP = {sp[0]:.2f} ± {sp[1]:.2f} %)', fontweight='bold')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(cm.columns, fontweight='bold'); ax.set_yticklabels(cm.index, fontweight='bold')
    ax.set_xlabel('Classe predita'); ax.set_ylabel('Classe real')
    for i in range(4):
        for j in range(4):
            v = cm.values[i, j]
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center', color='white' if v > 45 else 'black',
                    fontweight='bold' if i == j else 'normal')
plt.tight_layout()
plt.show()
'''

TESTS_CODE = r'''
# F-test 5x2cv (Alpaydin), como no artigo: FUSÃO contra as entradas isoladas, por família (modelos 4c)
F = 'FUSÃO (MEL+DEMON)'
PAIRS = [(('MLP', F), ('MLP', 'LOFAR')), (('MLP', F), ('MLP', 'MEL')), (('MLP', F), ('MLP', 'DEMON')),
         (('SVM Nyström', F), ('SVM Nyström', 'LOFAR')), (('SVM Nyström', F), ('SVM Nyström', 'MEL')),
         (('Random Forest', F), ('Random Forest', 'LOFAR')), (('Random Forest', F), ('Random Forest', 'MEL'))]
rows = []
for a, b in PAIRS:
    for met in ('SP', 'R_SMALL'):
        f, p = R.f_test_5x2(GRID[a][1], GRID[b][1], metric=met)
        rows.append({'A': GRID[a][0], 'B': GRID[b][0], 'métrica': met, 'média A': round(S4[a][met][0], 2),
                     'média B': round(S4[b][met][0], 2), 'F': round(f, 2), 'p': round(p, 4), 'p < 0,05': p < 0.05,
                     '_a': a, '_b': b})
df_tests = pd.DataFrame(rows)
# Extra: MLP MEL ≈ artigo (0,512 s) contra MLP MEL 5,12 s, para medir o efeito da grade temporal
f, p = R.f_test_5x2('mlp', 'mlp_mel5s_h32-16_e35_4c')
grid_effect = {'F': f, 'p': p, 'oficial': R.summary('mlp')['SP'], 'mel5s': S4[('MLP', 'MEL')]['SP']}
display(df_tests.drop(columns=['_a', '_b']))
print(f"MLP MEL ≈ artigo (0,512 s) vs MLP MEL 5,12 s: F = {f:.2f}, p = {p:.4f}")
'''


NOTEBOOKS_DIR = os.path.abspath(os.path.join(HERE, ".."))

def build(filename=os.path.join(NOTEBOOKS_DIR, "12_comparacao_multimodal_mel_demon_lofar.ipynb")):
    ns = {}
    exec(SETUP_CODE, ns)
    exec(TABLES_CODE, ns)
    exec(CONF_CODE, ns)
    ns["display"] = lambda *a, **k: None
    ns["print"] = lambda *a, **k: None
    exec(TESTS_CODE, ns)
    R = ns["R"]
    S4, S3, GRID, T = ns["S4"], ns["S3"], ns["GRID"], ns["df_tests"]
    CMS = ns["CMS"]
    FU = "FUSÃO (MEL+DEMON)"

    def t(a, b, met):
        r = T[(T._a == a) & (T._b == b) & (T["métrica"] == met)].iloc[0]
        return f"F = {br(r.F)}, {pv(r.p)}", r.p

    def eq(p):
        return "diferença **significativa**" if p < 0.05 else "**equivalentes**"

    k_mf, k_ml, k_mm, k_md = ("MLP", FU), ("MLP", "LOFAR"), ("MLP", "MEL"), ("MLP", "DEMON")
    k_sf, k_sl, k_sm = ("SVM Nyström", FU), ("SVM Nyström", "LOFAR"), ("SVM Nyström", "MEL")
    k_rf, k_rl, k_rm = ("Random Forest", FU), ("Random Forest", "LOFAR"), ("Random Forest", "MEL")
    k_cf = ("CNN 2D", FU)
    art_mlpl = R.ARTICLE_TABLE7[("MLP", "LOFAR")]["SP"]
    art_cnnl = R.ARTICLE_TABLE7[("CNN", "LOFAR")]["SP"]
    art_rfl = R.ARTICLE_TABLE7[("RF", "LOFAR")]["SP"]
    ge = ns["grid_effect"]
    sm_lofar = [S4[(m, "LOFAR")]["R_SMALL"][0] for m in ("MLP", "SVM Nyström", "Random Forest")]
    sm_mel = [S4[(m, "MEL")]["R_SMALL"][0] for m in ("MLP", "SVM Nyström", "Random Forest")]
    tab_md = ns["tab_mlp_demon"]
    md_sp = {l: R.summary(e)["SP"] for l, e in ns["MLP_DEMON"]}
    cnn_f_folds = ns["SP_FOLDS"][k_cf]

    _fp = R.file_predictions('mlp_lofar')
    n_files = len(set().union(*[set(d.File) for d in _fp.values()]))
    cells = []
    cells.append(md(r"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 12: Comparativo Multimodal Global — MEL, DEMON, LOFAR e FUSÃO (MEL + DEMON)
#### Modelos Avaliados: Multi-Layer Perceptron (MLP), SVM Nyström, Random Forest e CNN 2D

Este notebook avalia a interação entre **quatro representações acústicas** e **quatro famílias de aprendizado de máquina** sob o protocolo oficial de **$5\times2$ validação cruzada (10 folds)** com **particionamento por gravação de áudio (`ID`, `exclusive_header='ID'` em `src/iara/ml/experiment.py`)** sobre os **__NFILES__ áudios de UO** (*Silva et al.*, IEEE Access 2025, Seção V-D: *"samples from the same audio recording were not placed in different sets"*). O particionamento não isola navios: gravações diferentes do mesmo navio podem cair em treino e teste.

> **Nota metodológica:**
> * **ACC = acurácia balanceada** (média dos recalls; a coluna `ACC` da Tabela 7 do artigo) em todas as tabelas. Acurácia micro não é usada.
> * Todos os números locais são recalculados das predições por janela em `results/trainings/` por `notebooks/iara_results.py`. Os valores do artigo vêm de `R.ARTICLE_TABLE7` (Tabela 7, só 4 classes) e aparecem em tabela **separada**.
> * **Duas grades temporais:** os experimentos do pipeline iara oficial (`results/trainings/tests`: MLP/CNN/SVM com MEL ou LOFAR) usam janelas de **0,512 s**; o DEMON, a FUSÃO e os caches `mel5s`/`lofar5s` usam **um vetor a cada 5,12 s** (MEL/LOFAR com média de 10 frames, resolução temporal 10× menor). A coluna `Grade` indica qual é qual.
> * **"3 classes"** aparece em dois modos, sempre rotulados: `4c_sem_bg` (modelo 4c avaliado só em navios; prever BG = erro) e `3c` (modelo **retreinado** sem BG, só onde existe).

---

### Representações Espectrais Avaliadas:
1. **MEL (256 filtros log-triangulares):** energia de banda larga de **$0\text{ a }8\text{ kHz}$** (Nyquist do sinal decimado a $16\text{ kHz}$).
2. **DEMON (253 bins de modulação):** envoltória da banda de 1–2 kHz demodulada em **$0{,}39\text{ a }25\text{ Hz}$** (Nyquist da envoltória decimada a $50\text{ Hz}$). A FFT de $N_{\text{FFT}}=512$ tem 257 bins; os **4** primeiros ($0$; $0{,}098$; $0{,}195$; $0{,}293\text{ Hz}$) são descartados (`discard_dc_bins = 4` em `scratch/extract_all_demon.py`).
3. **LOFAR (1024 bins de banda estreita linear):** espectrograma de **$0\text{ a }8\text{ kHz}$** ($N_{\text{FFT}}=2048$ sobre o sinal a $16\text{ kHz}$).
4. **FUSÃO MULTIMODAL (509 atributos):** concatenação $\text{MEL}_{5{,}12\text{ s}} (256) + \text{DEMON} (253)$ na grade de $5{,}12\text{ s}$.

---

### Famílias de Aprendizado de Máquina:
1. **MLP (PyTorch):** camadas densas com `BatchNorm1d` e `Dropout`. **Atenção:** a MLP da FUSÃO usa `[64, 32]`; as MLP de MEL 5,12 s e DEMON usam `[32, 16]`.
2. **SVM Nyström:** kernel RBF aproximado com penalização ElasticNet; $m=6000$ para DEMON, LOFAR e FUSÃO e **$m=4000$ para MEL** (o único SVM MEL oficial em disco).
3. **Random Forest (100 árvores):** `class_weight='balanced'`.
4. **CNN 2D:** para MEL, a CNN do repositório (`test_scripts/cnn.py`: conv 128/64, ReLU, lr 1e-5, imagens sem sobreposição — não é a arquitetura do artigo); CNNs **ad hoc** (PyTorch) para DEMON, LOFAR 5,12 s e FUSÃO — estas três colapsaram e não devem ser usadas como baseline.
"""))
    cells[0]["source"] = [x.replace("__NFILES__", br(n_files, 0)) for x in cells[0]["source"]]
    cells.append(code(SETUP_CODE))
    cells.append(md(f"""
## 1. Tabela Mestra Comparativa Global ($4 \\times 4$)

Os 16 pares (Modelo × Representação), com o experimento em disco, a grade temporal e a origem de cada linha. Abaixo dela: a mesma matriz em `4c_sem_bg`, os modelos retreinados sem BG (`3c`), a Tabela 7 do artigo e a rastreabilidade artigo × avaliação local.

* O recall SMALL do `SVM m=6000 | LOFAR` é **{br(S4[k_sl]['R_SMALL'][0])}%** (uma versão anterior trazia 42,58%, provavelmente um valor 3C).
* A linha SVM | MEL é o modelo **m=4000** (uma versão anterior a rotulava como m=6000).
* **Três MLP DEMON [32,16]** distintos existem no projeto (tabela abaixo): 25 épocas ({pm(md_sp['25 ép. (NB9)'])}%, NB9), 30 épocas com seed 42 ({pm(md_sp['30 ép., seed 42 (NBs 16–18, gate)'])}%, NBs 16–18 e gate) e 35 épocas ({pm(md_sp['35 ép. (grade da fusão, este NB)'])}%, usada aqui por estar na mesma grade e no mesmo script da FUSÃO).
"""))
    cells.append(code(TABLES_CODE + r'''
print('4 classes (modelos 4c):'); display(df_master)
print('Só navios, modelos 4c (prever BACKGROUND = erro):'); display(df_sem_bg)
print('Só navios, modelos RETREINADOS sem BACKGROUND (3c):'); display(df_3c)
print('Tabela 7 do artigo (Silva et al., 2025; 4 classes):'); display(tab_artigo)
print('Rastreabilidade: Tabela 7 do artigo vs. avaliação local:'); display(df_rast)
print('Os três MLP DEMON [32,16] do projeto (4c):'); display(tab_mlp_demon)
'''))
    cells.append(md(r"""
## 2. Gráfico 1: Índice SP por Modelo e Representação

Os modelos marcados com `*` são as CNNs ad hoc colapsadas (SP baixo e desvio alto entre folds):
"""))
    cells.append(code(PLOT_SP_CODE))
    cells.append(md(f"""
## 3. Gráfico 2: Recall da Classe Crítica `SMALL`

Nos modelos 4c (MLP, SVM, RF), o recall de SMALL vai de {br(min(sm_lofar))}% a {br(max(sm_lofar))}% com LOFAR e de {br(min(sm_mel))}% a {br(max(sm_mel))}% com MEL. Abaixo, o efeito da fusão com DEMON (à esquerda, modelos 4c; à direita, modelos retreinados sem BG, só onde existem):
"""))
    cells.append(code(PLOT_SMALL_CODE))
    cells.append(md(r"""
## 4. Matrizes de Confusão da Fusão (MLP e SVM)

Matrizes por arquivo, acumuladas nas 10 dobras e normalizadas por linha:
"""))
    cells.append(code(CONF_CODE))
    cells.append(code(PLOT_CONF_CODE))
    cells.append(md(r"""
## 5. Testes Estatísticos (F-test 5x2cv)

FUSÃO contra cada entrada isolada da mesma família (SP e recall SMALL por fold). "Supera" só quando p < 0,05:
"""))
    cells.append(code(TESTS_CODE))

    tm_l, pm_l = t(k_mf, k_ml, "SP")
    tm_m, pm_m = t(k_mf, k_mm, "SP")
    tm_d, pm_d = t(k_mf, k_md, "SP")
    ts_l, ps_l = t(k_sf, k_sl, "SP")
    ts_m, ps_m = t(k_sf, k_sm, "SP")
    tr_l, pr_l = t(k_rf, k_rl, "SP")
    tr_m, pr_m = t(k_rf, k_rm, "SP")
    trs_l, prs_l = t(k_rf, k_rl, "R_SMALL")
    tms_l, pms_l = t(k_mf, k_ml, "R_SMALL")
    tss_l, pss_l = t(k_sf, k_sl, "R_SMALL")
    cm_m = CMS[k_mf]
    headline = ('A fusão MEL + DEMON é estatisticamente equivalente ao LOFAR em SP nas três famílias vetoriais — não o supera.'
                if max(pm_l, ps_l, pr_l) >= 0.05 and min(pm_l, ps_l, pr_l) >= 0.05 else
                'Fusão MEL + DEMON × LOFAR: resultado misto (ver p de cada família).')
    cells.append(md(f"""
## 6. Conclusões

1. **{headline}**
   * **MLP:** FUSÃO {pm(S4[k_mf]['SP'])}% × MLP LOFAR (≈ artigo) {pm(S4[k_ml]['SP'])}% ({tm_l}: {eq(pm_l)}); a MLP LOFAR da Tabela 7 do artigo tem {pm(art_mlpl)}%. A comparação **favorece a FUSÃO**: a MLP da FUSÃO tem rede maior (`[64, 32]` contra `[32, 16]`) e outra grade temporal (5,12 s contra 0,512 s). Contra a MLP MEL 5,12 s `[32,16]` (mesma grade e script), {tm_m}: {eq(pm_m)}; contra a MLP DEMON, {tm_d}: {eq(pm_d)}.
   * **SVM:** FUSÃO (m=6000) {pm(S4[k_sf]['SP'])}% × SVM LOFAR m=6000 {pm(S4[k_sl]['SP'])}% ({ts_l}: {eq(ps_l)}) e × SVM MEL m=4000 {pm(S4[k_sm]['SP'])}% ({ts_m}: {eq(ps_m)}).
   * **Random Forest:** FUSÃO {pm(S4[k_rf]['SP'])}% × RF LOFAR 5,12 s local {pm(S4[k_rl]['SP'])}% ({tr_l}: {eq(pr_l)}) e × RF MEL 5,12 s {pm(S4[k_rm]['SP'])}% ({tr_m}: {eq(pr_m)}). O RF LOFAR local tem SP média **{'maior' if S4[k_rl]['SP'][0] > S4[k_rf]['SP'][0] else 'menor'}** que a RF FUSÃO. (O RF LOFAR da Tabela 7 do artigo, {pm(art_rfl)}%, não é comparável: grade e hiperparâmetros diferentes.)
   * **Recall SMALL:** a FUSÃO eleva o recall de SMALL em relação ao LOFAR na RF ({br(S4[k_rf]['R_SMALL'][0])}% × {br(S4[k_rl]['R_SMALL'][0])}%; {trs_l}: {eq(prs_l)}); na MLP ({br(S4[k_mf]['R_SMALL'][0])}% × {br(S4[k_ml]['R_SMALL'][0])}%; {tms_l}: {eq(pms_l)}) e no SVM ({br(S4[k_sf]['R_SMALL'][0])}% × {br(S4[k_sl]['R_SMALL'][0])}%; {tss_l}: {eq(pss_l)}).
   * A FUSÃO tem metade da dimensão do LOFAR (509 × 1024) com SP estatisticamente equivalente; isto é o que os dados sustentam.

2. **A CNN 2D ad hoc colapsa na fusão:** SP {pm(S4[k_cf]['SP'])}% (folds: {', '.join(br(x, 1) for x in cnn_f_folds)}). A CNN ad hoc também colapsa em DEMON ({pm(S4[('CNN 2D', 'DEMON')]['SP'])}%) e em LOFAR 5,12 s ({pm(S4[('CNN 2D', 'LOFAR')]['SP'])}%, contra {pm(art_cnnl)}% da CNN LOFAR da Tabela 7 do artigo), então o colapso não pode ser atribuído à fusão. *Hipótese (não testada):* a fronteira artificial entre os 256 canais MEL e os 253 canais DEMON não tem continuidade física para convoluções 2D.

3. **MLP FUSÃO:** tem a maior SP média local ({pm(S4[k_mf]['SP'])}%, recall SMALL {br(S4[k_mf]['R_SMALL'][0])}%, {br(cm_m.loc['SMALL', 'MEDIUM'], 1)}% dos SMALL indo para MEDIUM na matriz acumulada), mas **sem diferença estatística** para a MLP LOFAR (≈ artigo). O efeito da grade temporal também precisa ser considerado: MLP MEL ≈ artigo (0,512 s) {pm(ge['oficial'])}% × MLP MEL 5,12 s {pm(ge['mel5s'])}% (F = {br(ge['F'])}, {pv(ge['p'])}).
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py",
                              "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python",
                              "pygments_lexer": "ipython3", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Notebook gerado com sucesso em: {filename}")


if __name__ == "__main__":
    build()
