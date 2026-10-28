"""
Gera notebooks/8_analise_demon.ipynb (análise do extrator DEMON).

Todos os números do notebook (tabelas, matrizes de confusão, picos espectrais e valores citados no markdown)
são calculados a partir dos dados em disco:
  * features DEMON por arquivo: data/iara_processed/demon_features/<ID>.pkl;
  * predições por janela dos 10 folds: results/trainings/{multirep,tests}/<exp>/eval/fold_k/*.csv,
    lidas por notebooks/iara_results.py.
O MESMO código que vai para as células é executado aqui para gerar os números do markdown.
"""

import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def br(x, nd=2):
    """Número em formato brasileiro (vírgula decimal)."""
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
import os, sys, pickle
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
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 15})
'''

# --------------------------------------------------------------------------------------------------
# Espectros DEMON de 4 arquivos (picos calculados a partir de demon_features/<ID>.pkl)
# --------------------------------------------------------------------------------------------------
PEAKS_CODE = r'''
from scipy.signal import find_peaks

DEMON_DIR = os.path.join(R.ROOT, 'data', 'iara_processed', 'demon_features')
SAMPLES = {   # rótulo: (ID do arquivo, cor). Nome/comprimento conforme src/iara/dataset_info/iara.csv
    'SMALL (PILOT 10 - 11 m)': (2, '#2ecc71'),
    'MEDIUM (MAERSK PACER - 74 m)': (1, '#f39c12'),
    'LARGE (SINOGLORY - 322 m)': (4, '#e74c3c'),
    'BACKGROUND (fundo oceânico)': (1446, '#3498db'),
}


def demon_peaks(fid, k=3):
    """Espectro médio do arquivo, argmax e os k picos locais de maior proeminência (scipy.signal.find_peaks)."""
    with open(os.path.join(DEMON_DIR, f'{fid}.pkl'), 'rb') as f:
        d = pickle.load(f)
    freqs, spec = np.asarray(d['frequencies']), np.asarray(d['avg_spectrum'])
    idx, props = find_peaks(spec, prominence=0)
    top = idx[np.argsort(props['prominences'])[::-1][:k]]
    return {'freqs': freqs, 'spec': spec, 'argmax_hz': float(freqs[np.argmax(spec)]),
            'peaks_hz': [float(freqs[i]) for i in top], 'n_frames': len(d['times'])}


PEAKS = {lbl: demon_peaks(fid) for lbl, (fid, _) in SAMPLES.items()}
df_peaks = pd.DataFrame([{
    'Classe / embarcação': lbl, 'ID': SAMPLES[lbl][0], 'frames DEMON': p['n_frames'],
    'argmax (Hz)': round(p['argmax_hz'], 2),
    'picos mais proeminentes (Hz)': ', '.join(f'{x:.2f}' for x in p['peaks_hz']),
    'picos × 60 (ciclos/min)': ', '.join(f'{x * 60:.0f}' for x in p['peaks_hz']),
} for lbl, p in PEAKS.items()]).set_index('Classe / embarcação')
'''

PLOT_SPECTRA_CODE = r'''
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
for ax, (lbl, (fid, color)) in zip(axes.flatten(), SAMPLES.items()):
    p = PEAKS[lbl]
    ax.plot(p['freqs'], p['spec'], color=color, lw=1.6, label='Espectro médio do arquivo')
    for j, fpk in enumerate(p['peaks_hz']):
        y = p['spec'][np.argmin(np.abs(p['freqs'] - fpk))]
        ax.plot(fpk, y, 'kv', ms=7)
        ax.annotate(f'{fpk:.2f} Hz', xy=(fpk, y), xytext=(4, 6 + 10 * j), textcoords='offset points', fontsize=9)
    ax.axvline(p['argmax_hz'], color='gray', ls=':', lw=1, label=f"argmax = {p['argmax_hz']:.2f} Hz")
    ax.set_title(f'{lbl} — ID {fid}', fontweight='bold')
    ax.set_ylabel('Magnitude média')
    ax.set_xlim(0, 25)
    ax.legend(fontsize=9, loc='upper right')
for ax in axes[1]:
    ax.set_xlabel('Frequência de modulação (Hz)')
plt.suptitle('Espectros DEMON médios (triângulos: 3 picos de maior proeminência)', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.show()
display(df_peaks)
'''

# --------------------------------------------------------------------------------------------------
# Métricas 5x2cv
# --------------------------------------------------------------------------------------------------
METRICS_CODE = r'''
# Experimentos em disco. Sufixo _4c = 4 classes; _3c = modelo RETREINADO sem BACKGROUND.
REF = [  # referências MEL/LOFAR do pipeline iara oficial (results/trainings/tests, janelas de 0,512 s)
    ('SVM LOFAR (m=4000, PCA64)', 'svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0'),
    ('SVM LOFAR (m=6000)', 'svm_nystroem_6000_lofar_elasticnet_l1r0.15'),
    ('SVM MEL (m=4000)', 'svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0'),
]
DEMON = [  # DEMON (results/trainings/multirep; frames de 5,12 s). SVM com max_iter=1000, sem early stopping
    ('DEMON SVM (m=1000)', 'svm_nystroem_1000_demon_C2.0_elasticnet_l1r0.15_it1000'),
    ('DEMON SVM (m=4000)', 'svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_it1000'),
    ('DEMON SVM (m=6000)', 'svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000'),
    ('DEMON RF (100 árv.)', 'rf_demon_n100_balanced'),
    ('DEMON MLP [32,16] (25 ép.)', 'mlp_demon_h32-16_e25'),
    ('DEMON CNN 2D ad hoc', 'cnn2d_demon_w32_b32_e20'),
]
# SVM DEMON m=4000 com early stopping (max_iter=200): instável, mantido só como alerta
UNSTABLE = ('DEMON SVM (m=4000, early stopping) [INSTÁVEL]', 'svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_4c')

ROWS_4C = REF + [(l, e + '_4c') for l, e in DEMON] + [UNSTABLE]
ROWS_3C = [(l, e + '_3c', '3c') for l, e in DEMON]
MET = ('SP', 'ACC', 'F1')

tab_4c = R.table(ROWS_4C, mode='4c', metrics=MET)
tab_sem_bg = R.table(ROWS_4C, mode='4c_sem_bg', metrics=MET)
tab_3c = R.table(ROWS_3C, metrics=MET)

S4 = {l: R.summary(e, '4c') for l, e in ROWS_4C}
SB = {l: R.summary(e, '4c_sem_bg') for l, e in ROWS_4C}
S3 = {l: R.summary(e, m) for l, e, m in ROWS_3C}
SP_FOLDS = {l: R.fold_metrics(e)['SP'].round(1).tolist() for l, e in ROWS_4C}
'''

PLOT_METRICS_CODE = r'''
labels_4c = [l for l, _ in ROWS_4C]
labels_3c = [l for l, _, _ in ROWS_3C]
fig, axes = plt.subplots(1, 3, figsize=(20, 5.8), sharey=True)
w = 0.38
for ax, labels, S, title in [
        (axes[0], labels_4c, S4, '4 classes (modelos 4c)'),
        (axes[1], labels_4c, SB, 'Só navios — modelos 4c (prever BG = erro)'),
        (axes[2], labels_3c, S3, 'Só navios — modelos RETREINADOS sem BG (3c)')]:
    x = np.arange(len(labels))
    ax.bar(x - w / 2, [S[l]['ACC'][0] for l in labels], w, yerr=[S[l]['ACC'][1] for l in labels], capsize=3,
           label='ACC (acurácia balanceada)', color='#3498db', alpha=0.9)
    ax.bar(x + w / 2, [S[l]['SP'][0] for l in labels], w, yerr=[S[l]['SP'][1] for l in labels], capsize=3,
           label='SP', color='#e67e22', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, 80)
axes[0].set_ylabel('Média ± desvio nos 10 folds (%)')
axes[0].legend(loc='lower left')
plt.tight_layout()
plt.show()
'''

CONF_CODE = r'''
CLS = ['SMALL', 'MEDIUM', 'LARGE', 'BACKGROUND']


def confusion(exp, mode='4c'):
    """Matriz de confusão por arquivo, ACUMULADA nas 10 dobras (instâncias arquivo x fold), normalizada por linha (%)."""
    df = pd.concat(R.file_predictions(exp).values(), ignore_index=True)
    if mode == '4c_sem_bg':
        df = df[df.Target != 3]
    rows = sorted(df.Target.unique())
    cols = sorted(set(rows) | set(df.Prediction.unique()))
    cm = pd.crosstab(df.Target, df.Prediction).reindex(index=rows, columns=cols, fill_value=0)
    cmn = 100 * cm.div(cm.sum(axis=1), axis=0)
    cmn.index = [CLS[i] for i in rows]
    cmn.columns = [CLS[i] for i in cols]
    return cmn, int(cm.values.sum())


CMS = {
    ('m=4000', '4c'): confusion('svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_it1000_4c'),
    ('m=4000', '3c'): confusion('svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_it1000_3c'),
    ('m=6000', '4c'): confusion('svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000_4c'),
    ('m=6000', '3c'): confusion('svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000_3c'),
}
N_EVAL = CMS[('m=4000', '4c')][1]
N_FILES = len(set().union(*[set(d.File) for d in R.file_predictions('svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_it1000_4c').values()]))
'''

PLOT_CONF_CODE = r'''
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, (m, mode) in zip(axes.flatten(), CMS):
    cm, n = CMS[(m, mode)]
    lbl = f'DEMON SVM ({m})'
    sp = (S4 if mode == '4c' else S3)[lbl]['SP']
    ax.imshow(cm.values, cmap='Blues' if mode == '4c' else 'Oranges', vmin=0, vmax=100)
    ax.set_xticks(range(cm.shape[1])); ax.set_xticklabels(cm.columns, fontweight='bold')
    ax.set_yticks(range(cm.shape[0])); ax.set_yticklabels(cm.index, fontweight='bold')
    ax.set_xlabel('Classe predita'); ax.set_ylabel('Classe real')
    tag = '4 classes' if mode == '4c' else '3 classes (retreinado sem BG)'
    ax.set_title(f'{lbl} — {tag}\nSP = {sp[0]:.2f} ± {sp[1]:.2f} % ; n = {n}', fontweight='bold')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center', color='white' if v > 50 else 'black', fontweight='bold')
plt.tight_layout()
plt.show()
'''

TESTS_CODE = r'''
# F-test 5x2cv (Alpaydin), igual ao artigo, sobre o recall de LARGE por fold (modelos 4c)
PAIRS = [('DEMON SVM (m=6000)', 'SVM LOFAR (m=6000)'), ('DEMON SVM (m=6000)', 'SVM LOFAR (m=4000, PCA64)'),
         ('DEMON SVM (m=6000)', 'SVM MEL (m=4000)'), ('DEMON SVM (m=4000)', 'SVM LOFAR (m=4000, PCA64)'),
         ('DEMON SVM (m=4000)', 'SVM MEL (m=4000)')]
E4 = dict(ROWS_4C)
tests = []
for a, b in PAIRS:
    for met in ('R_LARGE', 'SP'):
        F, p = R.f_test_5x2(E4[a], E4[b], metric=met)
        tests.append({'A': a, 'B': b, 'métrica': met, 'média A': round(S4[a][met][0], 2),
                      'média B': round(S4[b][met][0], 2), 'F': round(F, 2), 'p': round(p, 4)})
df_tests = pd.DataFrame(tests)
display(df_tests)
'''


NOTEBOOKS_DIR = os.path.abspath(os.path.join(HERE, ".."))

def build(filename=os.path.join(NOTEBOOKS_DIR, "8_analise_demon.ipynb")):
    import iara_results as R  # noqa: F401  (usado pelos blocos executados)
    ns = {}
    os.environ.setdefault("MPLBACKEND", "Agg")
    exec(SETUP_CODE, ns)
    exec(PEAKS_CODE, ns)
    exec(METRICS_CODE, ns)
    exec(CONF_CODE, ns)
    ns["display"] = lambda *a, **k: None
    exec(TESTS_CODE, ns)

    P, S4, SB, S3 = ns["PEAKS"], ns["S4"], ns["SB"], ns["S3"]
    CMS, N_EVAL, T = ns["CMS"], ns["N_EVAL"], ns["df_tests"]
    N_FILES = ns["N_FILES"]
    lbl_large = "LARGE (SINOGLORY - 322 m)"
    lbl_small = "SMALL (PILOT 10 - 11 m)"
    lbl_med = "MEDIUM (MAERSK PACER - 74 m)"
    lbl_bg = "BACKGROUND (fundo oceânico)"

    def ratio(lbl):
        sp = P[lbl]["spec"]
        return float(sp.max() / sorted(sp)[len(sp) // 2])

    def peaks_txt(lbl):
        return " / ".join(br(x) for x in P[lbl]["peaks_hz"])

    cm44, cm43 = CMS[("m=4000", "4c")][0], CMS[("m=4000", "3c")][0]
    cm64, cm63 = CMS[("m=6000", "4c")][0], CMS[("m=6000", "3c")][0]

    def t(a, b, met):
        r = T[(T.A == a) & (T.B == b) & (T["métrica"] == met)].iloc[0]
        return r.F, r.p

    d6 = "DEMON SVM (m=6000)"
    d4 = "DEMON SVM (m=4000)"
    d1 = "DEMON SVM (m=1000)"
    lof6, lof4, mel4 = "SVM LOFAR (m=6000)", "SVM LOFAR (m=4000, PCA64)", "SVM MEL (m=4000)"
    best_large_lbl = max([d1, d4, d6], key=lambda l: S4[l]["R_LARGE"][0])
    best_large_3c = max([d1, d4, d6], key=lambda l: S3[l]["R_LARGE"][0])
    demon_lbls = [l for l, _ in ns["DEMON"]]
    best_sp_lbl = max(demon_lbls, key=lambda l: S4[l]["SP"][0])
    unstable = ns["UNSTABLE"][0]
    sp_folds_uns = ns["SP_FOLDS"][unstable]
    sp_folds_m1000 = ns["SP_FOLDS"][d1]
    sp_folds_cnn = ns["SP_FOLDS"]["DEMON CNN 2D ad hoc"]

    def any_sig(a, met):
        out = []
        for b in (lof6, lof4, mel4):
            m = T[(T.A == a) & (T.B == b) & (T["métrica"] == met)]
            if len(m):
                out.append(f"{b}: F = {br(m.iloc[0].F)}, {pv(m.iloc[0].p)}")
        return "; ".join(out)

    def sig_txt(a, met):
        sig, eq = [], []
        for b in (lof6, lof4, mel4):
            m = T[(T.A == a) & (T.B == b) & (T["métrica"] == met)]
            if len(m):
                (sig if m.iloc[0].p < 0.05 else eq).append(b)
        txt = ""
        if sig:
            txt += "Diferença significativa (p < 0,05) apenas contra " + ", ".join(sig) + ". "
        if eq:
            txt += "Contra " + ", ".join(eq) + " o DEMON é **equivalente** em LARGE (dentro da variação entre folds), não superior."
        return txt

    cells = []
    cells.append(md(r"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 8: Investigação da Técnica DEMON (Demodulation of Envelope Modulation on Noise)

Neste notebook, investigamos o comportamento do método **DEMON** aplicado a sinais de áudio de sonar passivo na Bacia de Santos (acervo UO do IARA). Avaliamos:
1. **Fundamentação Física e Matemática:** Extração da taxa de rotação de eixos e frequências de passagem de pás de hélice via demodulação de banda alta.
2. **Auditoria Algorítmica:** Diagnóstico e correção da formulação analógica (`analog=True`) da biblioteca de referência, adotando filtros digitais em seções de segunda ordem (**SOS**).
3. **Visualização em Casos Reais:** Espectros DEMON médios de 4 arquivos (`SMALL`, `MEDIUM`, `LARGE` e `BACKGROUND`), com os picos calculados diretamente das features.
4. **Validação Experimental no Protocolo $5\times2$cv:** 10 folds oficiais com **particionamento por gravação de áudio (`ID`, `exclusive_header='ID'`)** — janelas de um mesmo arquivo nunca ficam em treino e teste ao mesmo tempo. **Atenção:** o particionamento **não** é por navio: o mesmo navio (`Ship ID`) pode ter gravações diferentes no treino e no teste (ex.: a *PILOT 10* tem dezenas de gravações). Classificadores: **SVM Nyström** ($m=1000, 4000, 6000$), RF, MLP e uma CNN 2D ad hoc.
5. **Diagnóstico Acústico:** Por que o DEMON puro fica abaixo do MEL/LOFAR isoladamente e como pode ser usado em arquiteturas híbridas/em cascata.

> Todos os números deste notebook são recalculados das predições por janela em `results/trainings/` por `notebooks/iara_results.py` (voto majoritário por arquivo, SP como em `src/iara/ml/metrics.py`, **ACC = acurácia balanceada**, média ± desvio populacional nos 10 folds).
"""))
    cells.append(md(r"""
## 1. Fundamentos e Pipeline de Processamento DEMON

O princípio do DEMON baseia-se na demodulação de amplitude do ruído de banda larga gerado pela **cavitação das hélices**:

$$
x(t) \xrightarrow{\text{Passa-Faixa (1--2 kHz)}} y(t) \xrightarrow{|\cdot|} |y(t)| \xrightarrow{\text{Decimação (50 Hz)}} z(t) \xrightarrow{\text{STFT}} Z(f, t)
$$

* **Filtro Passa-faixa:** Chebyshev Tipo II digital em seções de segunda ordem (SOS) com $40\text{ dB}$ de atenuação na banda de rejeição, operando em $[1000, 2000]\text{ Hz}$.
* **Detecção de Envoltória:** Retificação de onda completa $|y(t)|$.
* **Decimação em 2 estágios:** Redução da taxa de amostragem de $48\text{ kHz}$ para $50\text{ Hz}$ usando filtros FIR anti-aliasing em duas etapas inteiras ($r_1 \times r_2 = 960$).
* **Espectrograma Banda Base:** STFT com janela Hann ($N_{fft}=512$, $50\%$ de overlap, ou seja, janelas de $10{,}24\text{ s}$ com passo de $5{,}12\text{ s}$). Dos 257 bins da FFT, os 4 primeiros (DC e sub-hertz, $0$ a $0{,}29\text{ Hz}$) são descartados (`discard_dc_bins = 4` em `scratch/extract_all_demon.py`), restando **253 bins** de $0{,}39\text{ Hz}$ a $25{,}0\text{ Hz}$ (Nyquist do envelope a $50\text{ Hz}$) com resolução de $50/512 \approx 0{,}098\text{ Hz}$.
"""))
    cells.append(code(SETUP_CODE))
    cells.append(md(r"""
## 2. Inspeção Física de Espectros DEMON no Acervo IARA

Carregamos as features DEMON pré-processadas (`data/iara_processed/demon_features/<ID>.pkl`) de 4 arquivos. Nomes e comprimentos vêm de `src/iara/dataset_info/iara.csv`:
* **SMALL (ID 2):** *PILOT 10* (Pilot Vessel, $11\text{ m}$)
* **MEDIUM (ID 1):** *MAERSK PACER* (Anchor Handling Vessel, $74\text{ m}$)
* **LARGE (ID 4):** *SINOGLORY* (Ore Carrier, $322\text{ m}$)
* **BACKGROUND (ID 1446):** ruído de fundo (dataset E, sem navio)

Os picos marcados são calculados na célula abaixo: o **argmax** do espectro médio e os **3 máximos locais de maior proeminência** (`scipy.signal.find_peaks`).
"""))
    cells.append(code(PEAKS_CODE))
    cells.append(code(PLOT_SPECTRA_CODE))
    cells.append(md(f"""
### Interpretação Física dos Picos

Valores calculados acima (ID: argmax; 3 picos mais proeminentes):

| Classe | Embarcação | Comprimento | argmax (Hz) | Picos mais proeminentes (Hz) | Observação |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **SMALL** | *PILOT 10* | 11 m | {br(P[lbl_small]['argmax_hz'])} | {peaks_txt(lbl_small)} | Pico estreito e dominante, {br(ratio(lbl_small), 1)}× a mediana do espectro. |
| **MEDIUM** | *MAERSK PACER* | 74 m | {br(P[lbl_med]['argmax_hz'])} | {peaks_txt(lbl_med)} | Pico máximo {br(ratio(lbl_med), 1)}× a mediana; vários picos de amplitude semelhante. |
| **LARGE** | *SINOGLORY* | 322 m | {br(P[lbl_large]['argmax_hz'])} | {peaks_txt(lbl_large)} | O argmax cai no **primeiro bin** retido ({br(P[lbl_large]['freqs'][0])} Hz), ou seja, energia de muito baixa frequência sem pico definido; os picos locais mais proeminentes estão acima de 7 Hz. |
| **BACKGROUND** | Fundo do mar | — | {br(P[lbl_bg]['argmax_hz'])} | {peaks_txt(lbl_bg)} | Espectro de baixa amplitude, sem pico dominante. |

> **Limitações.** (i) Um único arquivo por classe não caracteriza a classe. (ii) Converter um pico em RPM ($f \\times 60$, coluna `picos × 60` da tabela) só é válido se o pico for a rotação do eixo; sem saber o número de pás e qual harmônico é o fundamental, a distinção entre taxa de eixo e frequência de passagem de pás (BPF) não pode ser feita a partir destes espectros. Uma versão anterior deste notebook atribuía ao LARGE um eixo em 1,80 Hz (105 RPM) e BPF em 5,5 Hz — esses picos **não existem** em `demon_features/4.pkl` e foram removidos.
"""))
    cells.append(md(r"""
## 3. Avaliação Experimental no Protocolo $5\times2$cv

Classificadores treinados só com os vetores DEMON (253 dimensões, um vetor a cada 5,12 s) nos 10 folds oficiais (particionamento por gravação), métricas por voto majoritário por arquivo. As referências MEL/LOFAR são os SVMs do pipeline iara oficial (`results/trainings/tests`, janelas de 0,512 s).

Três modos de avaliação, **sempre rotulados**:
* **4c** — problema de 4 classes (SMALL, MEDIUM, LARGE, BACKGROUND);
* **4c_sem_bg** — o mesmo modelo 4c avaliado só nos arquivos de navio; prever BACKGROUND conta como erro (nenhum arquivo é descartado pela predição). Os recalls por classe são idênticos aos do modo 4c; mudam SP/ACC/F1;
* **3c** — modelo **retreinado** sem BACKGROUND (experimentos `*_3c`; só existem para o DEMON, não para as referências MEL/LOFAR).

**ACC = acurácia balanceada** (média dos recalls) em todas as tabelas. A linha "SVM DEMON m=4000 (early stopping)" é o modelo usado nos notebooks de complementaridade; ela é mostrada só como alerta, porque é instável (ver SP por fold abaixo).
"""))
    cells.append(code(METRICS_CODE + r'''
print('4 classes (modelos 4c):'); display(tab_4c)
print('Só navios, modelos 4c (prever BACKGROUND = erro):'); display(tab_sem_bg)
print('Só navios, modelos RETREINADOS sem BACKGROUND (3c):'); display(tab_3c)
print('SP por fold (4c):')
display(pd.DataFrame(SP_FOLDS, index=[f'fold {k}' for k in range(10)]).T)
'''))
    cells.append(code(PLOT_METRICS_CODE))
    cells.append(md(f"""
**Leitura da tabela (valores calculados acima):**
* Melhor SP 4c entre os modelos DEMON: **{best_sp_lbl}** com **{pm(S4[best_sp_lbl]['SP'])}%**, contra {pm(S4[lof6]['SP'])}% ({lof6}), {pm(S4[lof4]['SP'])}% ({lof4}) e {pm(S4[mel4]['SP'])}% ({mel4}).
* **Modelos instáveis:** {unstable} tem SP {pm(S4[unstable]['SP'])}% (SP por fold: {', '.join(br(x, 1) for x in sp_folds_uns)}); o SVM DEMON m=1000 também tem desvio alto ({pm(S4[d1]['SP'])}%; folds: {', '.join(br(x, 1) for x in sp_folds_m1000)}). Um fold com SP = 0 significa que alguma classe teve recall zero naquele fold. A CNN 2D ad hoc fica em {pm(S4['DEMON CNN 2D ad hoc']['SP'])}% (recall SMALL {br(S4['DEMON CNN 2D ad hoc']['R_SMALL'][0])}%) e deve ser tratada como **colapsada**, não como baseline.
* O recall SMALL do SVM LOFAR m=6000 é **{br(S4[lof6]['R_SMALL'][0])}%** (uma versão anterior trazia 42,58%).
"""))
    cells.append(md(f"""
## 4. Matrizes de Confusão (`m=4000` vs. `m=6000`, max_iter=1000)

Matrizes **acumuladas** nas 10 dobras (n = {br(N_EVAL, 0).replace(',', '.')} instâncias arquivo × fold de teste nos modelos 4c; cada um dos {br(N_FILES, 0)} arquivos aparece em média {br(N_EVAL / N_FILES)} vezes no teste), normalizadas por linha. A diagonal acumulada difere ligeiramente do recall médio por fold da tabela.
"""))
    cells.append(code(CONF_CODE))
    cells.append(code(PLOT_CONF_CODE))
    cells.append(code(TESTS_CODE))

    f_l6, p_l6 = t(d6, lof6, "R_LARGE")
    cells.append(md(f"""
## 5. Diagnóstico Físico e Conclusões

### 5.1 Por que o DEMON isolado fica abaixo do MEL/LOFAR?
1. **Informação descartada:** o DEMON retém só a envoltória da banda de 1–2 kHz, modulada entre 0,39 e 25 Hz; descarta o espectro direto (linhas tonais de máquinas e a distribuição de energia de banda larga) que o MEL e o LOFAR usam.
2. **Confusão com o fundo:** nas matrizes acumuladas, **{br(cm44.loc['SMALL', 'BACKGROUND'], 1)}%** (m=4000) e **{br(cm64.loc['SMALL', 'BACKGROUND'], 1)}%** (m=6000) dos arquivos SMALL são classificados como BACKGROUND, e **{br(cm44.loc['SMALL', 'LARGE'], 1)}%** / **{br(cm64.loc['SMALL', 'LARGE'], 1)}%** como LARGE. *Hipótese (não testada aqui):* lanchas com pouca cavitação na banda de 1–2 kHz produzem envoltórias de baixa modulação, parecidas com o fundo.
3. **MEDIUM:** recall de MEDIUM {br(cm43.loc['MEDIUM', 'MEDIUM'], 1)}% (m=4000) e {br(cm63.loc['MEDIUM', 'MEDIUM'], 1)}% (m=6000) no modelo 3c, com {br(cm43.loc['MEDIUM', 'SMALL'], 1)}% / {br(cm63.loc['MEDIUM', 'SMALL'], 1)}% indo para SMALL. *Hipótese:* a classe MEDIUM é heterogênea (rebocadores, supridores, propulsão azimutal ou CPP), o que espalha as rotações nas duas direções.

### 5.2 Onde o DEMON se destaca?
* **LARGE:** o maior recall de LARGE entre os SVMs DEMON 4c é **{br(S4[best_large_lbl]['R_LARGE'][0])}%** ({best_large_lbl}); referências: {br(S4[lof6]['R_LARGE'][0])}% ({lof6}), {br(S4[lof4]['R_LARGE'][0])}% ({lof4}) e {br(S4[mel4]['R_LARGE'][0])}% ({mel4}). F-test 5x2cv sobre o recall de LARGE ({d6} contra cada referência): {any_sig(d6, 'R_LARGE')}. {sig_txt(d6, 'R_LARGE')} No modelo 3c, o maior recall de LARGE é {br(S3[best_large_3c]['R_LARGE'][0])}% ({best_large_3c}), mas não há referência MEL/LOFAR retreinada sem BG para comparar.
* **SMALL no modelo 3c:** retreinado sem BACKGROUND, o recall de SMALL do SVM DEMON vai a **{br(S3[d6]['R_SMALL'][0])}%** (m=6000) e {br(S3[d4]['R_SMALL'][0])}% (m=4000), contra {br(S4[d6]['R_SMALL'][0])}% / {br(S4[d4]['R_SMALL'][0])}% no 4c. Esse aumento decorre de **mudar o problema** (retreino sem a classe BACKGROUND, que no 4c absorvia {br(cm64.loc['SMALL', 'BACKGROUND'], 1)}% dos SMALL no m=6000) e não é comparável a recalls de modelos 4c nem às referências MEL/LOFAR, que não foram retreinadas sem BG.

### 5.3 Diretriz para a Etapa B
* O DEMON **não** deve ser usado isoladamente.
* Pode ser testado na **Fusão Híbrida** (MEL + DEMON, Notebook 12) e como entrada de um **especialista em cascata** entre embarcações. Qualquer ganho dessas arquiteturas precisa ser demonstrado com o F-test 5x2cv contra o melhor baseline.
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py",
                              "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python",
                              "pygments_lexer": "ipython3", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Notebook {filename} gerado com sucesso!")


if __name__ == "__main__":
    build()
