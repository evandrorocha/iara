"""
Gera os notebooks de comparação multimodelo com os extratores do artigo:

    notebooks/6_comparacao_modelos_lofar.ipynb   (LOFAR: MLP, SVM Nyström, Random Forest, CNN 2D, BiGRU)
    notebooks/7_comparacao_modelos_mel.ipynb     (MEL:   MLP, CNN, SVM Nyström, Random Forest, BiGRU)

Mesma estrutura dos notebooks 9 (DEMON) e 11 (HHT). Todos os números (tabelas, figuras, matrizes de confusão
e valores citados no markdown) são recalculados das predições por janela em results/trainings/{multirep,tests}
por notebooks/src/iara_results.py. O MESMO código que vai para as células é executado aqui para gerar os
números do markdown.

Uso (na raiz do repositório): python notebooks/src/generate_comparacao_lofar_mel_notebooks.py [lofar|mel]
"""

import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")
NOTEBOOKS_DIR = os.path.abspath(os.path.join(HERE, ".."))


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


# ------------------------------------------------------------------------------------------------
# Experimentos de cada representação
#   PRINCIPAIS: (rótulo curto, experimento 4c, experimento 3c retreinado ou None, grade temporal, origem)
#   SVM_SWEEP:  (rótulo, experimento) — varredura de hiperparâmetros do SVM Nyström no pipeline oficial
#   OUTRA:      a outra representação, para comparar no mesmo classificador: (arquitetura, exp desta, exp da outra)
# ------------------------------------------------------------------------------------------------
REPS = {
    "LOFAR": {
        "numero": 6,
        "arquivo": "6_comparacao_modelos_lofar.ipynb",
        "titulo": "Comparativo Multimodelo com o Extrator LOFAR (MLP, SVM, Random Forest, CNN 2D e BiGRU)",
        "descricao": (
            "o **LOFAR** (*Low Frequency Analysis and Recording*): espectro de magnitude com `n_pts = 1024` bins "
            "(resolução de ~7,8 Hz até 8 kHz), normalizado pelo TPSW (*Two-Pass Split Window*), que realça as raias "
            "tonais de máquinas e hélice sobre o ruído de fundo (detalhes no Notebook 5)"),
        "principais": [
            ("MLP (≈ artigo)", "mlp_lofar", None, "0,512 s", "pipeline iara, `test_scripts/mlp_lofar.py`: como o MLP MEL, mas com dropout 0,2 (artigo: 0); paciência 16 e até 512 épocas (artigo: 8 e 50)"),
            ("SVM Nyström m=6000", "svm_nystroem_6000_lofar_elasticnet_l1r0.15", None, "0,512 s",
             "pipeline iara (`test_scripts/svm.py`), ElasticNet l1_ratio=0,15"),
            ("Random Forest 100 árv.", "rf_lofar5s_n100_balanced_4c", None, "5,12 s",
             "`scratch/generate_rf_predictions_and_analysis.py`, média de 10 quadros de 0,512 s"),
            ("CNN 2D ad hoc", "cnn2d_lofar5s_w32_b64_e15_4c", None, "5,12 s",
             "`scratch/generate_cnn_predictions_and_analysis.py`, imagens de 32 vetores LOFAR de 5,12 s"),
            ("BiGRU + atenção", "bigru_att_lofar5s_h128_s15-5_e15_4c", "bigru_att_lofar5s_h128_s15-5_e15_3c", "5,12 s",
             "`scratch/eval_bigru_lofar_multiclass_5x2cv.py`, sequências de 15 vetores"),
        ],
        "svm_sweep": [
            ("m=1000", "svm_nystroem_1000_lofar"),
            ("m=1000, C=2.0", "svm_nystroem_1000_lofar_C2.0"),
            ("m=1000, C=2.0, PCA64", "svm_nystroem_1000_lofar_pca64_C2.0"),
            ("m=2000", "svm_nystroem_2000_lofar"),
            ("m=2000, C=0.1", "svm_nystroem_2000_lofar_C0.1"),
            ("m=2000, C=2.0", "svm_nystroem_2000_lofar_C2.0"),
            ("m=4000, C=2.0", "svm_nystroem_4000_lofar_C2.0"),
            ("m=4000, C=0.5, ElasticNet", "svm_nystroem_4000_lofar_elasticnet_l1r0.15_C0.5"),
            ("m=4000, C=2.0, PCA64, ElasticNet", "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0"),
            ("m=4000, C=0.5, PCA64, ElasticNet", "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C0.5"),
            ("m=6000, ElasticNet", "svm_nystroem_6000_lofar_elasticnet_l1r0.15"),
            ("m=6000, C=0.5, ElasticNet", "svm_nystroem_6000_lofar_elasticnet_l1r0.15_C0.5"),
            ("m=10000, ElasticNet", "svm_nystroem_10000_lofar_elasticnet_l1r0.15"),
            ("m=10000, C=0.5, ElasticNet", "svm_nystroem_10000_lofar_elasticnet_l1r0.15_C0.5"),
        ],
        "outra": "MEL",
        "pares_outra": [
            ("MLP (≈ artigo, 0,512 s)", "mlp_lofar", "mlp"),
            ("SVM Nyström (melhor de cada)", "svm_nystroem_6000_lofar_elasticnet_l1r0.15",
             "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0"),
            ("Random Forest (5,12 s)", "rf_lofar5s_n100_balanced_4c", "rf_mel5s_n100_balanced_4c"),
            ("BiGRU + atenção (5,12 s)", "bigru_att_lofar5s_h128_s15-5_e15_4c", "bigru_att_mel5s_h64_s15-5_e20_4c"),
        ],
        "cmap": ("Greens", "YlGn"),
    },
    "MEL": {
        "numero": 7,
        "arquivo": "7_comparacao_modelos_mel.ipynb",
        "titulo": "Comparativo Multimodelo com o Extrator MEL (MLP, CNN, SVM, Random Forest e BiGRU)",
        "descricao": (
            "o **log-MEL** (*log-melgram*): energia em `n_mels = 256` bandas na escala Mel, em escala logarítmica, "
            "que comprime a resolução nas altas frequências e preserva a envoltória espectral de banda larga "
            "(detalhes no Notebook 5)"),
        "principais": [
            ("MLP (≈ artigo)", "mlp", None, "0,512 s", "pipeline iara, `test_scripts/mlp.py`: arquitetura do artigo ([32,16], dropout 0,2, BatchNorm, ReLU, Sigmoid, lr 1e-4), mas até 512 épocas com paciência 16 (grid search do artigo: 50 épocas, paciência 8)"),
            ("CNN (repo)", "cnn", None, "0,512 s", "pipeline iara, `test_scripts/cnn.py`: conv 128/64, ReLU, lr 1e-5, imagens sem sobreposição — não é a arquitetura do artigo"),
            ("SVM Nyström m=4000", "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0", None, "0,512 s",
             "pipeline iara (`test_scripts/svm.py`), C=2,0, ElasticNet l1_ratio=0,15"),
            ("Random Forest 100 árv.", "rf_mel5s_n100_balanced_4c", None, "5,12 s",
             "`scratch/generate_rf_predictions_and_analysis.py`, média de 10 quadros de 0,512 s"),
            ("MLP [32,16] 35 ép.", "mlp_mel5s_h32-16_e35_4c", "mlp_mel5s_h32-16_e35_3c", "5,12 s",
             "`scratch/eval_fusion_mel_demon_5x2cv.py` (linha de base só MEL da fusão do Notebook 12)"),
            ("BiGRU + atenção", "bigru_att_mel5s_h64_s15-5_e20_4c", "bigru_att_mel5s_h64_s15-5_e20_3c", "5,12 s",
             "`scratch/eval_bigru_multiclass_5x2cv.py`, sequências de 15 vetores"),
        ],
        "svm_sweep": [
            ("m=300", "svm_nystroem"),
            ("m=1000", "svm_nystroem_1000"),
            ("m=1000, norm, PCA64", "svm_nystroem_1000_log_melgram_norm_pca64"),
            ("m=2000", "svm_nystroem_2000_log_melgram"),
            ("m=2000, C=10", "svm_nystroem_2000_log_melgram_C10.0"),
            ("m=3000", "svm_nystroem_3000_log_melgram"),
            ("m=4000, C=0.5, ElasticNet", "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C0.5"),
            ("m=4000, C=2.0, ElasticNet", "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0"),
            ("m=4000, C=2.0, PCA64, ElasticNet", "svm_nystroem_4000_log_melgram_pca64_elasticnet_l1r0.15_C2.0"),
            ("m=4000, landmarks k-means, ElasticNet", "svm_nystroem_4000_log_melgram_kmeans_elasticnet_l1r0.15"),
        ],
        "outra": "LOFAR",
        "pares_outra": [
            ("MLP (≈ artigo, 0,512 s)", "mlp", "mlp_lofar"),
            ("SVM Nyström (melhor de cada)", "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0",
             "svm_nystroem_6000_lofar_elasticnet_l1r0.15"),
            ("Random Forest (5,12 s)", "rf_mel5s_n100_balanced_4c", "rf_lofar5s_n100_balanced_4c"),
            ("BiGRU + atenção (5,12 s)", "bigru_att_mel5s_h64_s15-5_e20_4c", "bigru_att_lofar5s_h128_s15-5_e15_4c"),
        ],
        "cmap": ("Oranges", "YlOrBr"),
    },
}

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
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 15, 'figure.dpi': 120})
'''

TABLES_CODE = r'''
# ---- Experimentos em disco (results/trainings/{tests,multirep}); _3c = modelo RETREINADO sem BACKGROUND ----
REP = __REP__
OUTRA = __OUTRA__
PRINCIPAIS = __PRINCIPAIS__  # (rótulo, exp 4c, exp 3c ou None, grade temporal, origem)
SVM_SWEEP = __SVM_SWEEP__
PARES_OUTRA = __PARES_OUTRA__  # (arquitetura, exp com REP, exp com OUTRA)
MET = ('SP', 'ACC', 'F1')

E4 = {l: e for l, e, *_ in PRINCIPAIS}
E3 = {l: e3 for l, _, e3, *_ in PRINCIPAIS if e3}
S4 = {l: R.summary(e, '4c') for l, e in E4.items()}
SB = {l: R.summary(e, '4c_sem_bg') for l, e in E4.items()}
S3 = {l: R.summary(e, '3c') for l, e in E3.items()}
FOLDS = {l: R.fold_metrics(e) for l, e in E4.items()}
# Colapso: em algum fold, alguma classe com recall < 5% (o SP do fold despenca e a média deixa de ser informativa)
COLAPSADO = {l: bool((FOLDS[l][[f'R_{c}' for c in R.CLASSES]].min(axis=1) < 5).any()) for l in E4}

# Tabela 7 do artigo (Silva et al., IEEE Access 2025): só SP, ACC e F1 em 4 classes
tab_artigo = pd.DataFrame([{'Modelo': f'{m} | {r} (Tabela 7 do artigo)',
                            **{k: R.fmt(v) for k, v in R.ARTICLE_TABLE7[(m, r)].items()}}
                           for (m, r) in R.ARTICLE_TABLE7 if r == REP]).set_index('Modelo')
ROWS = [(l + (' [COLAPSADA]' if COLAPSADO[l] else ''), e) for l, e in E4.items()]
tab_4c = R.table(ROWS, mode='4c', metrics=MET)
tab_4c.insert(len(MET), 'Grade', [g for *_, g, _ in PRINCIPAIS])
tab_sem_bg = R.table(ROWS, mode='4c_sem_bg', metrics=MET)
tab_3c = R.table([(l, e, '3c') for l, e in E3.items()], metrics=MET) if E3 else None
def _completo(e):
    try:
        R.fold_metrics(e)
        return True
    except FileNotFoundError:   # sem os 10 folds (ou CSVs com outro nome) em disco
        return False


SWEEP_FORA = [(l, e) for l, e in SVM_SWEEP if not _completo(e)]
SVM_SWEEP = [(l, e) for l, e in SVM_SWEEP if _completo(e)]
SWEEP = {l: R.summary(e) for l, e in SVM_SWEEP}
tab_sweep = R.table(SVM_SWEEP, metrics=MET)
'''

PLOT_GLOBAL_CODE = r'''
lbl = list(E4)
fig, ax = plt.subplots(figsize=(15, 6))
x = np.arange(len(lbl)); w = 0.35
ax.bar(x - w / 2, [S4[l]['ACC'][0] for l in lbl], w, yerr=[S4[l]['ACC'][1] for l in lbl], capsize=3,
       label='ACC (acurácia balanceada)', color='#2980b9', alpha=0.9)
r = ax.bar(x + w / 2, [S4[l]['SP'][0] for l in lbl], w, yerr=[S4[l]['SP'][1] for l in lbl], capsize=3,
           label='SP', color='#e67e22', alpha=0.9)
for rect in r:
    ax.annotate(f'{rect.get_height():.1f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')
for (m, rr), cor, ls in zip([k for k in R.ARTICLE_TABLE7 if k[1] == REP], ['#27ae60', '#8e44ad', '#7f8c8d'], ['--', ':', '-.']):
    v = R.ARTICLE_TABLE7[(m, rr)]['SP'][0]
    ax.axhline(v, color=cor, ls=ls, lw=1.8, label=f'{m} {REP}, Tabela 7 do artigo ({v:.2f}% SP)')
ax.set_xticks(x); ax.set_xticklabels([l + ('\n[COLAPSADA]' if COLAPSADO[l] else '') + f'\n({g})' for l, (*_, g, _) in zip(lbl, PRINCIPAIS)],
                                     fontweight='bold')
ax.set_ylabel('Média ± desvio nos 10 folds (%)'); ax.set_ylim(0, 80)
ax.legend(loc='lower left', fontsize=9, ncol=2, frameon=True)
plt.title(f'Modelos com o extrator {REP} — 4 classes (10 folds, 5x2cv oficial)', fontweight='bold')
plt.tight_layout()
plt.show()
'''

PLOT_SWEEP_CODE = r'''
lbls = [l for l, _ in SVM_SWEEP]
fig, ax = plt.subplots(figsize=(14, 5.5))
sp = [SWEEP[l]['SP'][0] for l in lbls]; sd = [SWEEP[l]['SP'][1] for l in lbls]
best_sw = int(np.argmax(sp))
bars = ax.bar(range(len(lbls)), sp, yerr=sd, capsize=3, color=['#c0392b' if i == best_sw else '#95a5a6' for i in range(len(lbls))])
for b in bars:
    ax.annotate(f'{b.get_height():.1f}', xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                textcoords='offset points', ha='center', va='bottom', fontsize=8.5)
ax.set_xticks(range(len(lbls))); ax.set_xticklabels(lbls, rotation=30, ha='right')
ax.set_ylim(40, 75); ax.set_ylabel('SP (%) — média ± desvio nos 10 folds')
ax.set_title(f'Varredura do SVM Nyström com {REP} (pipeline oficial, janelas de 0,512 s)', fontweight='bold')
plt.tight_layout()
plt.show()
'''

PLOT_RECALL_CODE = r'''
classes = R.CLASSES
recalls_matrix = np.array([[S4[l][f'R_{c}'][0] for c in classes] for l in E4])
display(pd.DataFrame(recalls_matrix.round(2), index=list(E4), columns=classes))
fig, ax = plt.subplots(figsize=(14, 6))
n = len(E4); width = 0.8 / n
cores = ['#3498db', '#9b59b6', '#e67e22', '#e74c3c', '#16a085', '#7f8c8d']
for i, (l, color) in enumerate(zip(E4, cores)):
    rects = ax.bar(np.arange(4) + (i - (n - 1) / 2) * width, recalls_matrix[i], width, label=l, color=color, alpha=0.9)
    for rect in rects:
        ax.annotate(f'{rect.get_height():.0f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=8)
ax.set_title(f'Recall por classe (média nos 10 folds) — 4 classes, {REP}', fontweight='bold', pad=15)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(classes, fontsize=12, fontweight='bold')
ax.set_ylabel('Recall (%)'); ax.set_ylim(0, 100)
ax.legend(title='Modelo', loc='upper left', frameon=True, fontsize=9, ncol=2)
plt.tight_layout()
plt.show()
'''

CONF_CODE = r'''
def confusion(exp):
    """Matriz de confusão por arquivo, ACUMULADA nas 10 dobras (instâncias arquivo x fold), normalizada por linha (%)."""
    df = pd.concat(R.file_predictions(exp).values(), ignore_index=True)
    rows = sorted(df.Target.unique())
    cols = sorted(set(rows) | set(df.Prediction.unique()))
    cm = pd.crosstab(df.Target, df.Prediction).reindex(index=rows, columns=cols, fill_value=0)
    cmn = 100 * cm.div(cm.sum(axis=1), axis=0)
    cmn.index = [R.CLASSES[i] for i in rows]; cmn.columns = [R.CLASSES[i] for i in cols]
    return cmn, int(cm.values.sum())


CM4 = {l: confusion(e) for l, e in E4.items()}
CM3 = {l: confusion(e) for l, e in E3.items()}
N_FILES = len(set().union(*[set(d.File) for d in R.file_predictions(E4[next(iter(E4))]).values()]))


def plot_cms(CMS, S, cmap, suptitle):
    n = len(CMS); ncol = min(n, 3); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.6 * nrow), squeeze=False)
    for ax in axes.flatten()[n:]:
        ax.axis('off')
    for ax, (l, (cm, nn)) in zip(axes.flatten(), CMS.items()):
        sp = S[l]['SP']
        im = ax.imshow(cm.values, cmap=cmap, vmin=0, vmax=100)
        ax.set_title(f'{l}\nSP {sp[0]:.2f} ± {sp[1]:.2f} % (n = {nn})', fontsize=11, fontweight='bold')
        ax.set_xticks(range(cm.shape[1])); ax.set_xticklabels([c[:5] for c in cm.columns], fontweight='bold')
        ax.set_yticks(range(cm.shape[0])); ax.set_yticklabels([c[:5] for c in cm.index], fontweight='bold')
        ax.set_xlabel('Classe predita'); ax.set_ylabel('Classe real')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm.values[i, j]
                ax.text(j, i, f'{v:.1f}%', ha='center', va='center', color='white' if v > 50 else 'black',
                        fontweight='bold' if i == j else 'normal', fontsize=9.5)
    plt.suptitle(suptitle, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()
'''

TESTS_CODE = r'''
# F-test 5x2cv combinado (Alpaydin, 1999), como no artigo, sobre o SP por fold (modelos 4c)
ORDEM = sorted(E4, key=lambda l: -S4[l]['SP'][0])
MELHOR = ORDEM[0]
rows = []
for b in ORDEM[1:]:
    F, p = R.f_test_5x2(E4[MELHOR], E4[b])
    rows.append({'A': MELHOR, 'B': b, 'SP A': round(S4[MELHOR]['SP'][0], 2), 'SP B': round(S4[b]['SP'][0], 2),
                 'F': round(F, 2), 'p': round(p, 4), 'p < 0,05': p < 0.05})
df_tests = pd.DataFrame(rows)
print(f'Melhor modelo {REP} (SP médio) contra os demais:')
display(df_tests)

rows = []
for arq, a, b in PARES_OUTRA:
    sa, sb = R.summary(a), R.summary(b)
    F, p = R.f_test_5x2(a, b)
    Fs, ps = R.f_test_5x2(a, b, metric='R_SMALL')
    rows.append({'Arquitetura': arq, f'SP {REP}': R.fmt(sa['SP']), f'SP {OUTRA}': R.fmt(sb['SP']),
                 'Δ SP (p.p.)': round(sa['SP'][0] - sb['SP'][0], 2), 'p (SP)': round(p, 4),
                 f'R_SMALL {REP}': round(sa['R_SMALL'][0], 2), f'R_SMALL {OUTRA}': round(sb['R_SMALL'][0], 2),
                 'p (R_SMALL)': round(ps, 4), 'exp A': a, 'exp B': b})
df_outra = pd.DataFrame(rows).set_index('Arquitetura')
print(f'{REP} × {OUTRA} no mesmo classificador:')
display(df_outra.drop(columns=['exp A', 'exp B']))
'''


def build(rep):
    cfg = REPS[rep]
    filename = os.path.join(NOTEBOOKS_DIR, cfg["arquivo"])
    tables = (TABLES_CODE.replace("__REP__", repr(rep)).replace("__OUTRA__", repr(cfg["outra"]))
              .replace("__PRINCIPAIS__", "[\n" + "".join(f"    {t!r},\n" for t in cfg["principais"]) + "]")
              .replace("__SVM_SWEEP__", "[\n" + "".join(f"    {t!r},\n" for t in cfg["svm_sweep"]) + "]")
              .replace("__PARES_OUTRA__", "[\n" + "".join(f"    {t!r},\n" for t in cfg["pares_outra"]) + "]"))
    ns = {}
    exec(SETUP_CODE, ns)
    exec(tables, ns)
    exec(CONF_CODE, ns)
    ns["display"] = lambda *a, **k: None
    ns["print"] = lambda *a, **k: None
    exec(TESTS_CODE, ns)
    R = ns["R"]
    E4, E3, S4, SB, S3 = ns["E4"], ns["E3"], ns["S4"], ns["SB"], ns["S3"]
    FOLDS, COL, SWEEP = ns["FOLDS"], ns["COLAPSADO"], ns["SWEEP"]
    CM4, CM3 = ns["CM4"], ns["CM3"]
    T, TO = ns["df_tests"], ns["df_outra"]
    ORDEM, MELHOR = ns["ORDEM"], ns["MELHOR"]
    outra = cfg["outra"]
    art = {m: R.ARTICLE_TABLE7[(m, rep)] for m, r in R.ARTICLE_TABLE7 if r == rep}
    art_mlpl = R.ARTICLE_TABLE7[("MLP", "LOFAR")]
    oficiais = [l for l in E4 if "oficial" in l]
    validos = [l for l in E4 if not COL[l]]
    sw_best = max(SWEEP, key=lambda l: SWEEP[l]["SP"][0])
    sw_worst = min(SWEEP, key=lambda l: SWEEP[l]["SP"][0])
    sw_exp = dict(cfg["svm_sweep"])
    n_perto = sum(SWEEP[sw_best]["SP"][0] - SWEEP[l]["SP"][0] < SWEEP[sw_best]["SP"][1] for l in SWEEP)
    fora = ns["SWEEP_FORA"]
    fora_txt = ("" if not fora else "\n* Ficaram fora da varredura, por não terem as predições dos 10 folds com o nome padrão em disco: "
                + ", ".join(f"{l} (`{e}`)" for l, e in fora) + ".")
    F_sw = R.f_test_5x2(sw_exp[sw_best], sw_exp[sw_worst])
    nsig = int(T["p < 0,05"].sum())
    _ns = T[~T["p < 0,05"]]
    ns_gap = None if _ns.empty else max(((r["B"], r["SP A"] - r["SP B"]) for _, r in _ns.iterrows()), key=lambda t: t[1])
    iguais = T.loc[~T["p < 0,05"], "B"].tolist()
    difer = T.loc[T["p < 0,05"], "B"].tolist()
    rec_small = max(validos, key=lambda l: S4[l]["R_SMALL"][0])
    rec_bg = max(validos, key=lambda l: S4[l]["R_BACKGROUND"][0])
    pior_classe = {l: min(("SMALL", "MEDIUM", "LARGE", "BACKGROUND"), key=lambda c: S4[l][f"R_{c}"][0]) for l in validos}

    def cm(l, a, b, d=CM4):
        return br(d[l][0].loc[a, b], 1)

    def sig(p):
        return "significativa" if p < 0.05 else "**não** significativa"

    num = cfg["numero"]
    cells = []
    cells.append(md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook {num}: {cfg['titulo']}

Neste notebook comparamos as famílias de aprendizado de máquina avaliadas no projeto com {cfg['descricao']}. É o mesmo formato dos notebooks 9 (DEMON) e 11 (HHT), com o {rep} no lugar do extrator.

| Modelo | Experimento | Grade temporal | Origem |
| :--- | :--- | :---: | :--- |
""" + "".join(f"| {l} | `{e}` | {g} | {o} |\n" for l, e, _, g, o in cfg["principais"]) + f"""
Todos os modelos foram avaliados no protocolo oficial de **$5\\times2$ validação cruzada (10 folds)** com **particionamento por gravação de áudio (`ID`, `exclusive_header='ID'` em `src/iara/ml/experiment.py`)** — janelas de um mesmo arquivo nunca ficam em treino e teste ao mesmo tempo — sobre os **{br(ns['N_FILES'], 0)} áudios de UO** do IARA (*Silva et al.*, IEEE Access 2025). O particionamento **não** isola navios: gravações diferentes do mesmo navio (`Ship ID`) podem cair em treino e teste.

> **Duas grades temporais.** Os modelos do pipeline oficial (MLP, CNN e SVM) classificam janelas de **0,512 s**; os modelos ad hoc de `scratch/` (RF, CNN 2D, MLP 5,12 s, BiGRU) usam o cache de **5,12 s** (média de 10 quadros), o mesmo dos notebooks DEMON e HHT. A predição por arquivo é sempre o voto majoritário das janelas, mas modelos de grades diferentes não diferem só no classificador.

> Todos os números são recalculados das predições por janela em `results/trainings/` por `notebooks/src/iara_results.py` (voto majoritário por arquivo, SP de `src/iara/ml/metrics.py`, **ACC = acurácia balanceada**, média ± desvio populacional nos 10 folds).
"""))
    cells.append(code(SETUP_CODE))

    col_txt = ""
    if any(COL.values()):
        col_txt = ("\n> * **[COLAPSADA]** marca modelos com algum fold em que uma classe tem recall < 5%: " +
                   "; ".join(f"{l} (SP por fold: {', '.join(br(x, 1) for x in FOLDS[l]['SP'])})" for l in E4 if COL[l]) +
                   ". A média desses modelos não é informativa e eles ficam fora das conclusões.")
    cells.append(md(f"""
## 1. Tabela Mestra: {rep} no Artigo e nas Re-execuções Locais

> **Notas metodológicas:**
> * **ACC = acurácia balanceada** (média dos recalls) em todas as tabelas, inclusive na Tabela 7 do artigo.
> * **Três modos, sempre rotulados:** `4c` (4 classes); `4c_sem_bg` (o mesmo modelo 4c avaliado só em navios, prever BACKGROUND = erro; os recalls por classe são os mesmos do 4c); `3c` (modelo **retreinado** sem BACKGROUND — só existe para {', '.join(E3) if E3 else 'nenhum modelo'}). Comparar `3c` com `4c_sem_bg` mistura problemas diferentes.
> * O artigo publica só SP, ACC e F1 médios (sem predições por fold); por isso não há F-test entre as re-execuções locais e o artigo.{col_txt}
"""))
    cells.append(code(tables + r'''
print('Tabela 7 do artigo (4 classes):'); display(tab_artigo)
print('Avaliação local, 4 classes (modelos 4c):'); display(tab_4c)
print('Avaliação local, só navios, modelos 4c (prever BACKGROUND = erro):'); display(tab_sem_bg)
if tab_3c is not None:
    print('Só navios, modelos RETREINADOS sem BACKGROUND (3c):'); display(tab_3c)
'''))

    rep_art = "; ".join(f"{m}: {pm(v['SP'])}" for m, v in art.items())
    loc_art, reproduz = [], True
    for l in oficiais:
        fam = l.split(" ")[0]
        if fam in art:
            reproduz &= abs(S4[l]['SP'][0] - art[fam]['SP'][0]) <= max(S4[l]['SP'][1], art[fam]['SP'][1])
            loc_art.append(f"{l} {pm(S4[l]['SP'])}% contra {pm(art[fam]['SP'])}% no artigo "
                           f"(diferença de {br(S4[l]['SP'][0] - art[fam]['SP'][0])} p.p., dentro de um desvio)"
                           if abs(S4[l]['SP'][0] - art[fam]['SP'][0]) <= max(S4[l]['SP'][1], art[fam]['SP'][1]) else
                           f"{l} {pm(S4[l]['SP'])}% contra {pm(art[fam]['SP'])}% no artigo "
                           f"(diferença de {br(S4[l]['SP'][0] - art[fam]['SP'][0])} p.p., **maior que um desvio**)")
    cells.append(md(f"""
### Leitura da Tabela Mestra
1. **Artigo:** SP com {rep} na Tabela 7 — {rep_art}.
2. **Reprodução local do pipeline oficial:** {'; '.join(loc_art)}.
3. **Ranking local (SP médio, 4c):** {', '.join(f"{l} ({br(S4[l]['SP'][0])})" for l in ORDEM)}.
"""))

    cells.append(md(f"""
## 2. Comparativo Global

Índice SP e acurácia balanceada (média ± desvio nos 10 folds) no problema de 4 classes, com as linhas de referência da Tabela 7 do artigo para o {rep}:
"""))
    cells.append(code(PLOT_GLOBAL_CODE))

    cells.append(md(f"""
## 3. Varredura do SVM Nyström

O SVM Nyström (núcleo RBF aproximado por *landmarks*, `test_scripts/svm.py`) foi a família mais explorada no projeto. A tabela e o gráfico abaixo trazem todas as configurações com {rep} puro que têm os 10 folds em disco (cascatas e híbridos ficam no Notebook 5 e em `reports/comparacao_modelos.md`):
"""))
    cells.append(code(PLOT_SWEEP_CODE + "\ndisplay(tab_sweep)\n"))
    cells.append(md(f"""
* A melhor configuração é **{sw_best}** (SP {pm(SWEEP[sw_best]['SP'])}%) e a pior, **{sw_worst}** (SP {pm(SWEEP[sw_worst]['SP'])}%). A diferença entre elas é {sig(F_sw[1])} no F-test 5x2cv (F = {br(F_sw[0])}, {pv(F_sw[1])}).
* {n_perto - 1} das outras {len(SWEEP) - 1} configurações ficam a menos de um desvio ({br(SWEEP[sw_best]['SP'][1])} p.p.) do SP médio da melhor; escolher uma "configuração campeã" só pelo SP médio não tem suporte estatístico sem o F-test.{fora_txt}
"""))

    cells.append(md(f"""
## 4. Recall por Classe

Recall médio nos 10 folds (modelos 4c):
"""))
    cells.append(code(PLOT_RECALL_CODE))
    cells.append(md(f"""
### Principais Fenômenos Observados (valores calculados acima e nas matrizes da seção 5):
1. **SMALL:** o maior recall entre os modelos não colapsados é o do {rec_small} ({br(S4[rec_small]['R_SMALL'][0])}%). Classe de menor recall em cada modelo: {'; '.join(f"{l}: {pior_classe[l]} ({br(S4[l]['R_' + pior_classe[l]][0])}%)" for l in validos)}.
2. **BACKGROUND:** o maior recall é o do {rec_bg} ({br(S4[rec_bg]['R_BACKGROUND'][0])}%); no {MELHOR}, {cm(MELHOR, 'BACKGROUND', 'SMALL')}% do fundo vai para SMALL.
3. **Confusão SMALL → MEDIUM / LARGE** (matrizes acumuladas): {'; '.join(f"{l}: {cm(l, 'SMALL', 'MEDIUM')}% / {cm(l, 'SMALL', 'LARGE')}%" for l in validos)}.
"""))

    N4 = CM4[MELHOR][1]
    cells.append(md(f"""
## 5. Matrizes de Confusão Acumuladas nos 10 Folds ($5\\times2$cv)

### 5.1 Problema de 4 Classes (SMALL, MEDIUM, LARGE, BACKGROUND)
Matrizes por arquivo, **acumuladas** nas 10 dobras (n = {br(N4, 0)} instâncias arquivo × fold; cada arquivo aparece em média {br(N4 / ns['N_FILES'])} vezes no teste) e normalizadas por linha. A diagonal acumulada difere ligeiramente do recall médio por fold da tabela.
"""))
    cells.append(code(CONF_CODE + f'''
plot_cms(CM4, S4, {cfg['cmap'][0]!r}, f'Matrizes de confusão normalizadas — {{REP}} [4 classes]')
'''))
    if E3:
        cells.append(md(f"""
### 5.2 Modelos Retreinados sem BACKGROUND (3c: SMALL, MEDIUM, LARGE)

Só {', '.join(E3)} {'tem' if len(E3) == 1 else 'têm'} versão retreinada sem BACKGROUND com {rep}. É outro problema (3 classes): os números não são comparáveis aos modelos 4c nem ao modo `4c_sem_bg`.
"""))
        cells.append(code(f'''
plot_cms(CM3, S3, {cfg['cmap'][1]!r}, f'Matrizes de confusão normalizadas — {{REP}} [3c, retreinado sem BACKGROUND]')
'''))
        cells.append(md("* " + "\n* ".join(
            f"**{l}:** SP 3c {pm(S3[l]['SP'])}% (no 4c: {pm(S4[l]['SP'])}%); recall de SMALL {br(S3[l]['R_SMALL'][0])}% "
            f"(no 4c: {br(S4[l]['R_SMALL'][0])}%); SMALL → LARGE {cm(l, 'SMALL', 'LARGE', CM3)}% e SMALL → MEDIUM {cm(l, 'SMALL', 'MEDIUM', CM3)}%."
            for l in E3) + "\n"))

    cells.append(md(f"""
## 6. Testes Estatísticos (F-test 5x2cv)

Toda afirmação de "supera" abaixo depende deste teste (Alpaydin, 1999; o mesmo usado no artigo). Primeiro, o melhor modelo {rep} em SP médio contra os demais; depois, {rep} × {outra} no mesmo classificador:
"""))
    cells.append(code(TESTS_CODE))

    linhas_outra = "".join(
        f"| {arq} | {pm(R.summary(r['exp A'])['SP'])} | {pm(R.summary(r['exp B'])['SP'])} | {br(r['Δ SP (p.p.)'])} | {br(r['p (SP)'], 3)} | "
        f"{br(r[f'R_SMALL {rep}'])} | {br(r[f'R_SMALL {outra}'])} | {br(r['p (R_SMALL)'], 3)} |\n"
        for arq, r in TO.iterrows())
    sig_outra = [arq for arq, r in TO.iterrows() if r["p (SP)"] < 0.05]
    sig_small = [f"{arq} ({rep} {br(r[f'R_SMALL {rep}'])}% × {outra} {br(r[f'R_SMALL {outra}'])}%, {pv(r['p (R_SMALL)'])})"
                 for arq, r in TO.iterrows() if r["p (R_SMALL)"] < 0.05]
    small_txt = ("* No recall de SMALL, a diferença é significativa em: " + "; ".join(sig_small) + "." if sig_small
                 else "* No recall de SMALL, nenhuma diferença é significativa.")
    melhor_txt = (f"empata estatisticamente com todos os demais (F-test 5x2cv, p ≥ 0,05)" if nsig == 0 else
                  f"é significativamente melhor que {', '.join(difer)} e equivalente a {', '.join(iguais) if iguais else 'nenhum'}")
    mlp_l = R.summary("mlp_lofar")["SP"]
    F_m = R.f_test_5x2(E4[MELHOR], "mlp_lofar") if E4[MELHOR] != "mlp_lofar" else None
    # Melhores sistemas compostos do projeto contra o melhor modelo desta representação
    COMPOSTOS = [("fusão MEL + DEMON (MLP, Notebook 12)", "mlp_mel5s-demon_h64-32_e35_4c"),
                 ("cascata LOFAR-MC → especialista SMALL × LARGE (Notebook 5)",
                  "svm_nystroem_4000_hybrid_cascade_pretrained_sm6000pca64_smallalso_fbmc_large_sl6000pca8_lofarmc6000_nosm_elasticnet_l1r0.15")]
    comp_txt = "; ".join(f"{n}: SP {pm(R.summary(e)['SP'])}%, {pv(R.f_test_5x2(e, E4[MELHOR])[1])}" for n, e in COMPOSTOS)
    cells.append(md(f"""
## 7. Diagnóstico Científico e Conclusões

### 7.1 O melhor modelo {rep}
* Em SP médio, o melhor é **{MELHOR}** ({pm(S4[MELHOR]['SP'])}%), que {melhor_txt}.
""" + (f"* A maior diferença de SP médio **sem** significância é de {br(ns_gap[1])} p.p. ({MELHOR} × {ns_gap[0]}): a ordem do ranking da seção 1 **não** deve ser lida como hierarquia.\n" if ns_gap else "") + f"""
""" + (f"* Referência do projeto: a MLP LOFAR (artigo {pm(art_mlpl['SP'])}%; local {pm(mlp_l)}%). O melhor {rep} contra a MLP LOFAR local: F-test {pv(F_m[1])} ({sig(F_m[1])}).\n" if F_m else
       f"* É também a referência do projeto (MLP LOFAR do artigo: {pm(art_mlpl['SP'])}%). Os melhores sistemas compostos contra ele (F-test 5x2cv): {comp_txt}.\n") + f"""
### 7.2 {rep} × {outra} no mesmo classificador

| Arquitetura | SP {rep} | SP {outra} | Δ (p.p.) | p (SP) | R_SMALL {rep} | R_SMALL {outra} | p (R_SMALL) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{linhas_outra}
* A diferença de SP entre {rep} e {outra} é significativa (p < 0,05) em: {', '.join(sig_outra) if sig_outra else '**nenhuma** arquitetura'}.
{small_txt}
* No artigo, a ordem entre LOFAR e MEL também depende do classificador (Tabela 7: RF favorece o MEL; MLP e CNN favorecem o LOFAR).

### 7.3 O que estes resultados permitem dizer
1. {'Os números do pipeline oficial reproduzem os do artigo dentro do desvio entre folds (seção 1), o que dá base para as comparações locais.' if reproduz else 'Pelo menos um modelo do pipeline oficial difere do artigo por mais de um desvio (seção 1); as comparações locais devem ser lidas com essa ressalva.'}
2. Os modelos ad hoc de 5,12 s não são uma reprodução do artigo: mudam a grade temporal **e** a implementação; servem para comparar representações (LOFAR, MEL, DEMON, HHT) no mesmo classificador, como nos Notebooks 9, 11 e 12.
3. Afirmações de superioridade entre modelos {rep} só valem quando o F-test da seção 6 dá p < 0,05.
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
    for r in (sys.argv[1:] or ["lofar", "mel"]):
        build(r.upper())
