"""
Gera notebooks/35_eda_lofar_<classe>.ipynb: análise exploratória das frequências LOFAR das gravações de uma classe
que os modelos acertam e erram (Tarefa 1 da reunião de 26/10/2026).

Etapas cobertas aqui:
  0. Rótulos de acerto/erro por gravação (CNN LOFAR do artigo; consenso com MLP e SVM LOFAR para robustez).
  (a) Espectro LOFAR médio: acertadas × erradas, faixa a faixa (Mann-Whitney por navio + FDR de Benjamini-Hochberg),
      semelhança das erradas com as outras classes e estratificação por subconjunto de gravação (A–D).
  (b) Linhas persistentes por gravação, ocupação por navio, pico grave (20–250 Hz) e harmônicos.
  (c) Banda grave × tipo de navio, mesmo navio em gravações acertadas e erradas, condições (vento, mar, chuva),
      regressão logística com validação agrupada por navio.
  (d) Validação em metades de navios (100 sorteios × 2 direções): banda escolhida às cegas numa metade, medida na outra.
  (e) lofar1hz (scripts/extrair_lofar1hz.py, faixas de 0,98 Hz até 1 kHz): componente larga × raias em 31–86 Hz e
      raias estreitas abaixo de 31 Hz. Cache: data/iara_processed/cache_lofar1hz_resumo.pkl.

Fontes: data/iara_processed/lofar_bc1559516c522e2b993895bf5cb2ac94 (LOFAR = log-espectrograma − TPSW, sem normalização),
predições por gravação via iara_results.py, src/iara/dataset_info/iara.csv.
Cache: data/iara_processed/cache_lofar_espectro_medio.pkl (espectro médio de cada gravação; criado se não existir).

Rodar na raiz do repositório:  python notebooks/src/generate_eda_lofar_notebook.py [LARGE|MEDIUM|SMALL]
LARGE usa build() (passos 0, (a)–(e)). MEDIUM e SMALL usam build_destino(): os erros dessas classes vão para mais de
uma classe, então as erradas são separadas por destino, e a banda 31–86 Hz das LARGE é testada fora da amostra.
LARGE usa build() (passos 0, (a)–(e)); MEDIUM e SMALL usam build_destino(): os erros dessas classes vão para mais
de uma classe, então as erradas são separadas por destino, e a banda 31–86 Hz das LARGE é testada fora da amostra.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")
NOTEBOOKS_DIR = os.path.abspath(os.path.join(HERE, ".."))


def br(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


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
from scipy.stats import mannwhitneyu, false_discovery_control
try:
    from IPython.display import display
except ImportError:
    display = print

for _p in ('.', 'src', 'notebooks', 'notebooks/src'):
    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))
import iara_results as R

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'figure.dpi': 120})
pd.set_option('display.precision', 2); pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)

CLASSE = '__CLASSE__'
CLS = dict(enumerate(R.CLASSES)); CID = {v: k for k, v in CLS.items()}
C = CID[CLASSE]
MODELO = 'cnn_lofar'                                    # CNN LOFAR do artigo (test_scripts/cnn_lofar.py)
OUTROS = {'MLP LOFAR': 'mlp_lofar', 'SVM LOFAR': 'svm_nystroem_6000_lofar_elasticnet_l1r0.15'}
LOFAR_DIR = os.path.join(R.ROOT, 'data', 'iara_processed', 'lofar_bc1559516c522e2b993895bf5cb2ac94')
CACHE = os.path.join(R.ROOT, 'data', 'iara_processed', 'cache_lofar_espectro_medio.pkl')

# Eixo de frequência do LOFAR do pipeline (src/iara/processing/analysis.py): fs = 48 kHz / decimação 3 = 16 kHz,
# nfft = 2 * n_pts = 2048, sem o bin DC -> 1024 faixas de 7,8125 Hz, de 7,8 Hz a 8 kHz
FS, NFFT = 16000, 2048
F = (np.arange(1024) + 1) * FS / NFFT

if os.path.exists(CACHE):
    S = pd.read_pickle(CACHE)
else:
    ids = sorted(int(f[:-4]) for f in os.listdir(LOFAR_DIR) if f.endswith('.pkl'))
    S = pd.DataFrame({i: pd.read_pickle(os.path.join(LOFAR_DIR, f'{i}.pkl'))['df'].values.mean(0) for i in ids}).T
    S.index.name = 'ID'; S.to_pickle(CACHE)

meta = pd.read_csv(os.path.join(R.ROOT, 'src', 'iara', 'dataset_info', 'iara.csv'), dtype={'Ship ID': str, 'ID': str})
meta['Ship ID'] = meta['Ship ID'].str.strip()
meta['Length'] = pd.to_numeric(meta['Length'], errors='coerce')
meta = meta.set_index(meta.ID.astype(int))
print(f'{len(S)} gravações com LOFAR; {len(F)} faixas de {F[0]:.4f} Hz')
'''

ROTULOS_CODE = r'''
def taxa_acerto(exp, c):
    # fração dos folds de teste em que a gravação da classe c foi acertada (voto majoritário por gravação)
    d = pd.concat(R.file_predictions(exp).values()); d = d[d.Target == c]
    t = d.groupby('File').apply(lambda x: (x.Prediction == x.Target).mean()); t.index = t.index.astype(int)
    return t

def grupo(t):
    return np.where(t == 1, 'acertada', np.where(t == 0, 'errada', 'oscila'))

T = pd.DataFrame({'CNN LOFAR': taxa_acerto(MODELO, C), **{k: taxa_acerto(v, C) for k, v in OUTROS.items()}})
T['grupo'] = grupo(T['CNN LOFAR'])
todos = T[['CNN LOFAR'] + list(OUTROS)]
T['consenso'] = np.where((todos == 1).all(1), 'acertada', np.where((todos == 0).all(1), 'errada', 'outro'))
T = T.join(meta[['Ship ID', 'Dataset', 'DETAILED TYPE', 'Length']])

# rótulo verdadeiro de todas as gravações (para os espectros de referência das outras classes)
LAB = pd.concat(R.file_predictions(MODELO).values()).groupby('File').Target.first(); LAB.index = LAB.index.astype(int)

# destino dos erros (todas as predições erradas da classe, gravação × fold)
_d = pd.concat(R.file_predictions(MODELO).values()); _d = _d[(_d.Target == C) & (_d.Prediction != C)]
DESTINO = (_d.Prediction.map(CLS).value_counts(normalize=True) * 100).round(1)

CONT = pd.DataFrame({'CNN LOFAR': T.grupo.value_counts(), 'consenso CNN+MLP+SVM': T.consenso.value_counts()}).fillna(0).astype(int)
POR_DS = pd.crosstab(T.Dataset, T.grupo, margins=True, margins_name='total')
POR_DS['% errada'] = (100 * POR_DS['errada'] / POR_DS['total']).round(1)
N_NAVIOS = T['Ship ID'].nunique()
_sh = T[T.grupo != 'oscila'].groupby('Ship ID').grupo.nunique()
N_NAVIOS_MISTOS = int((_sh > 1).sum())
print(f'{CLASSE}: {len(T)} gravações de {N_NAVIOS} navios'); display(CONT)
print('Destino dos erros da CNN LOFAR (%):'); display(DESTINO.to_frame('%'))
print('Grupos por subconjunto de gravação:'); display(POR_DS)
print('Comprimento (m) por grupo:'); display(T.groupby('grupo').Length.describe()[['count', '25%', '50%', '75%']].round(0))
'''

MEDIO_CODE = r'''
G = {g: S.loc[T.index[T.grupo == g]] for g in ('acertada', 'oscila', 'errada')}
REF = {CLS[c]: S.loc[LAB.index[LAB == c]] for c in range(4) if c != C}
cores = {'acertada': 'tab:green', 'oscila': 'tab:olive', 'errada': 'tab:red'}
cores_ref = dict(zip(REF, ['tab:blue', 'tab:purple', 'tab:gray']))

fig, ax = plt.subplots(2, 1, figsize=(13, 8))
for a, fmax in zip(ax, (8000, 500)):
    k = F <= fmax
    for g, X in G.items():
        m, se = X.mean().values, X.std().values / np.sqrt(len(X))
        a.plot(F[k], m[k], color=cores[g], lw=1.6, label=f'{CLASSE} {g} (n={len(X)})')
        a.fill_between(F[k], (m - 1.96 * se)[k], (m + 1.96 * se)[k], color=cores[g], alpha=0.2)
    for c, X in REF.items():
        a.plot(F[k], X.mean().values[k], color=cores_ref[c], lw=1, ls='--', label=f'{c} (todas, n={len(X)})')
    a.set_xlabel('frequência (Hz)'); a.set_ylabel('LOFAR médio (dB acima do fundo TPSW)')
    a.set_xlim(0, fmax)
ax[0].set_title(f'Espectro LOFAR médio — {CLASSE} acertadas, oscilantes e erradas pela CNN LOFAR (faixa = IC 95% da média)')
ax[1].set_title('Zoom: 0–500 Hz'); ax[0].legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.show()
'''

TESTE_CODE = r'''
def por_navio(ids):
    # média das gravações de cada navio: cada navio conta uma vez em cada grupo
    return S.loc[ids].groupby(T.loc[ids, 'Ship ID'].values).mean()

def teste_faixas(ids_a, ids_e):
    A, E = por_navio(ids_a), por_navio(ids_e)
    U = np.array([mannwhitneyu(E.iloc[:, k], A.iloc[:, k]).statistic for k in range(S.shape[1])])
    p = np.array([mannwhitneyu(E.iloc[:, k], A.iloc[:, k]).pvalue for k in range(S.shape[1])])
    q = false_discovery_control(p)
    return pd.DataFrame({'f (Hz)': F, 'acertada': A.mean().values, 'errada': E.mean().values,
                         'Δ (errada − acertada)': E.mean().values - A.mean().values,
                         'AUC': U / (len(A) * len(E)), 'q (FDR)': q}), len(A), len(E)

TF, NA, NE = teste_faixas(T.index[T.grupo == 'acertada'], T.index[T.grupo == 'errada'])
TF['sig'] = TF['q (FDR)'] < 0.05

BANDAS = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8001)]
def resumo_bandas(tf):
    rows = []
    for lo, hi in BANDAS:
        b = tf[(tf['f (Hz)'] >= lo) & (tf['f (Hz)'] < hi)]
        rows.append({'banda (Hz)': f'{lo}–{min(hi, 8000)}', 'faixas': len(b), 'sig. (q < 0,05)': int(b.sig.sum()),
                     'sig. com errada > acertada': int((b.sig & (b['Δ (errada − acertada)'] > 0)).sum()),
                     'sig. com errada < acertada': int((b.sig & (b['Δ (errada − acertada)'] < 0)).sum()),
                     'Δ médio (dB)': b['Δ (errada − acertada)'].mean(), 'maior |Δ| (dB)': b['Δ (errada − acertada)'].abs().max()})
    return pd.DataFrame(rows).set_index('banda (Hz)')
RB = resumo_bandas(TF)
print(f'Navios: {NA} com gravações acertadas, {NE} com erradas; {int(TF.sig.sum())} de {len(TF)} faixas com q < 0,05')
display(RB.round(2))
print('Faixas com maior |Δ| entre as significativas:')
TOP = TF[TF.sig].reindex(TF[TF.sig]['Δ (errada − acertada)'].abs().sort_values(ascending=False).index).head(12)
display(TOP.round(3))

fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
k = F <= 1000
ax[0].plot(F[k], TF['Δ (errada − acertada)'][k], color='k', lw=1)
ax[0].scatter(F[k & TF.sig.values], TF['Δ (errada − acertada)'][k & TF.sig.values], color='tab:red', s=10, label='q < 0,05')
ax[0].axhline(0, color='gray', lw=0.8); ax[0].set_ylabel('Δ LOFAR (dB)'); ax[0].legend()
ax[0].set_title(f'{CLASSE}: errada − acertada, por faixa (médias por navio), 0–1 kHz')
ax[1].plot(F[k], TF['AUC'][k], color='tab:blue', lw=1); ax[1].axhline(0.5, color='gray', lw=0.8)
ax[1].set_ylabel('AUC (errada > acertada)'); ax[1].set_xlabel('frequência (Hz)'); ax[1].set_ylim(0, 1)
plt.tight_layout(); plt.show()
'''

SEMELHANCA_CODE = r'''
# As erradas se parecem com a classe para onde vão? Correlação e distância RMS entre espectros médios (0–8 kHz e 0–500 Hz)
rows = []
for g in ('acertada', 'oscila', 'errada'):
    x = G[g].mean().values
    for c, X in REF.items():
        y = X.mean().values
        for nome, k in (('0–8 kHz', F <= 8000), ('0–500 Hz', F <= 500)):
            rows.append({'grupo': f'{CLASSE} {g}', 'referência': c, 'faixa': nome,
                         'r de Pearson': np.corrcoef(x[k], y[k])[0, 1], 'distância RMS (dB)': np.sqrt(((x[k] - y[k]) ** 2).mean())})
SEM = pd.DataFrame(rows)
display(SEM.pivot_table(index=['faixa', 'grupo'], columns='referência', values='distância RMS (dB)').round(2))
'''

ESTRATO_CODE = r'''
# O subconjunto de gravação (A–D: campanhas, locais e equipamentos diferentes) confunde a comparação?
FAIXAS = {'35–75 Hz': (F >= 35) & (F <= 75), '100–200 Hz': (F >= 100) & (F < 200), '4–8 kHz': F >= 4000}
B = pd.DataFrame({n: S.loc[:, k].mean(1) for n, k in FAIXAS.items()})
B['classe'] = LAB.map(CLS); B['Dataset'] = meta.Dataset
print('LOFAR médio por banda, subconjunto e classe (todas as gravações):')
display(B.groupby(['Dataset', 'classe'])[list(FAIXAS)].mean().round(2).unstack())

L = B.loc[T.index].join(T[['grupo', 'Ship ID']])
rows = []
for ds in sorted(L.Dataset.unique()):
    x = L[L.Dataset == ds]
    a = x[x.grupo == 'acertada'].groupby('Ship ID')[list(FAIXAS)].mean()
    e = x[x.grupo == 'errada'].groupby('Ship ID')[list(FAIXAS)].mean()
    r = {'Dataset': ds, 'navios acertada': len(a), 'navios errada': len(e)}
    for n in FAIXAS:
        r[f'{n}: acertada'] = a[n].mean() if len(a) else np.nan
        r[f'{n}: errada'] = e[n].mean() if len(e) else np.nan
        r[f'{n}: p'] = mannwhitneyu(e[n], a[n]).pvalue if min(len(a), len(e)) >= 5 else np.nan
    rows.append(r)
EST = pd.DataFrame(rows).set_index('Dataset')
print(f'{CLASSE} acertada × errada dentro de cada subconjunto (médias por navio; teste só com ≥ 5 navios por grupo):')
display(EST.round(3))
'''

ROBUSTO_CODE = r'''
# Mesmo teste com os grupos de consenso (os três modelos LOFAR sempre acertam / sempre erram)
TFc, NAc, NEc = teste_faixas(T.index[T.consenso == 'acertada'], T.index[T.consenso == 'errada'])
TFc['sig'] = TFc['q (FDR)'] < 0.05
RBc = resumo_bandas(TFc)
print(f'Consenso: {NAc} navios acertados, {NEc} errados; {int(TFc.sig.sum())} faixas com q < 0,05')
display(RBc.round(2))
R_DELTA = np.corrcoef(TF['Δ (errada − acertada)'], TFc['Δ (errada − acertada)'])[0, 1]
print(f'Correlação entre os perfis Δ(f) da CNN e do consenso: r = {R_DELTA:.3f}')
'''

PERS_CODE = r'''
from scipy.signal import find_peaks
from scipy.stats import fisher_exact, spearmanr

# Persistência: fração das janelas de cada gravação em que a faixa passa de um limiar (dB acima do fundo TPSW)
PCACHE = os.path.join(R.ROOT, 'data', 'iara_processed', 'cache_lofar_persistencia.pkl')
if os.path.exists(PCACHE):
    PERS = pd.read_pickle(PCACHE)
else:
    _P = {t: {} for t in (3, 6, 9, 12)}
    for i in S.index:
        x = pd.read_pickle(os.path.join(LOFAR_DIR, f'{i}.pkl'))['df'].values
        for t in _P:
            _P[t][i] = (x > t).mean(0)
    PERS = {t: pd.DataFrame(v).T for t, v in _P.items()}; pd.to_pickle(PERS, PCACHE)

TH, PMIN = 6, 0.5   # linha = máximo local da persistência com >= 50% das janelas acima de 6 dB
FRAC_6DB = 100 * float(PERS[6].values.mean())

def linhas(i, th=TH, pmin=PMIN):
    pk, _ = find_peaks(np.r_[0, PERS[th].loc[i].values, 0], height=pmin)
    return pk - 1

BANDAS_L = [0, 31, 86, 200, 500, 1000, 2000, 4000, 8001]
_rows = []
for g in ('acertada', 'oscila', 'errada'):
    ids = T.index[T.grupo == g]
    Ls = np.concatenate([F[linhas(i)] for i in ids])
    _rows.append((pd.cut(pd.Series(Ls), BANDAS_L, right=False).value_counts(sort=False) / len(ids)).rename(g))
NL = pd.DataFrame(_rows)
NL.columns = [f'{a}–{min(b, 8000)} Hz' for a, b in zip(BANDAS_L[:-1], BANDAS_L[1:])]
NL['total'] = NL.sum(1)
print(f'{FRAC_6DB:.1f}% das células (janela × faixa) de todas as gravações passam de 6 dB acima do fundo.')
print('Linhas por gravação, por banda (média do grupo):'); display(NL.round(2))

def tem_linha(i, th, pm, lo=31, hi=86):
    f = F[linhas(i, th, pm)]
    return bool(np.any((f >= lo) & (f < hi)))
SENS = pd.DataFrame([{'limiar (dB)': th, 'persistência mínima': pm,
                      **{f'% {g} com linha em 31–86 Hz': 100 * np.mean([tem_linha(i, th, pm) for i in T.index[T.grupo == g]])
                         for g in ('acertada', 'oscila', 'errada')}}
                     for th, pm in [(3, 0.8), (6, 0.5), (6, 0.8), (9, 0.5), (12, 0.5)]])
print('Sensibilidade ao critério de linha:'); display(SENS.round(1))
'''

OCUP_CODE = r'''
# Ocupação por faixa: fração de navios com persistência >= 50% naquela faixa (um navio conta se alguma gravação dele tem)
def ocupacao(ids):
    return (PERS[TH].loc[ids] >= PMIN).groupby(T.loc[ids, 'Ship ID'].values).max()
OA, OE = ocupacao(T.index[T.grupo == 'acertada']), ocupacao(T.index[T.grupo == 'errada'])
a, e = OA.sum().values, OE.sum().values
_p = np.array([fisher_exact([[e[k], len(OE) - e[k]], [a[k], len(OA) - a[k]]])[1] for k in range(len(F))])
OCU = pd.DataFrame({'f (Hz)': F, '% navios acertada': 100 * a / len(OA), '% navios errada': 100 * e / len(OE),
                    'q (FDR)': false_discovery_control(_p)})
OCU['sig'] = OCU['q (FDR)'] < 0.05
k = F <= 1000
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(F[k], OCU['% navios acertada'][k], color='tab:green', label=f'acertada ({len(OA)} navios)')
ax.plot(F[k], OCU['% navios errada'][k], color='tab:red', label=f'errada ({len(OE)} navios)')
s = k & OCU.sig.values
ax.scatter(F[s], np.full(s.sum(), -3), marker='|', color='k', label='q < 0,05 (Fisher + FDR)')
ax.set_xlabel('frequência (Hz)'); ax.set_ylabel('% de navios com a faixa ocupada')
ax.set_title(f'{CLASSE}: faixas ocupadas por linhas persistentes (>= {int(100 * PMIN)}% das janelas > {TH} dB)'); ax.legend()
plt.tight_layout(); plt.show()
OCU_SIG = OCU[OCU.sig & (OCU['f (Hz)'] < 1000)].copy()
OCU_SIG['Δ (errada − acertada, p.p.)'] = OCU_SIG['% navios errada'] - OCU_SIG['% navios acertada']
display(OCU_SIG.round(3))
'''

PICO_CODE = r'''
# Pico grave de cada gravação: faixa de maior LOFAR médio entre 20 e 250 Hz (exclui as 2 primeiras faixas, borda do TPSW)
KP = (F >= 20) & (F <= 250)
_ip = np.where(KP)[0][S.values[:, KP].argmax(1)]
PK = pd.DataFrame({'f pico (Hz)': F[_ip], 'altura (dB)': S.values[np.arange(len(S)), _ip], 'bin': _ip,
                   'classe': LAB.reindex(S.index).map(CLS).values}, index=S.index)
PK = PK.join(meta[['Ship ID', 'Length', 'Dataset']]).join(T[['grupo']])

def largura(i, k):
    # largura (Hz) do trecho contíguo em torno da faixa k com persistência >= PMIN
    p = PERS[TH].loc[i].values
    if p[k] < PMIN:
        return 0.0
    a_, b_ = k, k
    while a_ > 0 and p[a_ - 1] >= PMIN:
        a_ -= 1
    while b_ < len(p) - 1 and p[b_ + 1] >= PMIN:
        b_ += 1
    return (b_ - a_ + 1) * FS / NFFT
PK['largura (Hz)'] = [largura(i, k) for i, k in zip(PK.index, PK['bin'])]

FAIXAS_PICO = [20, 31, 86, 140, 251]
PK['faixa do pico'] = pd.cut(PK['f pico (Hz)'], FAIXAS_PICO, right=False,
                            labels=['20–31 Hz', '31–86 Hz', '86–140 Hz', '140–250 Hz'])
navios = PK[PK.classe != 'BACKGROUND']
DIST_CLASSE = pd.crosstab(navios['faixa do pico'], navios.classe, normalize='columns') * 100
_L = PK.loc[T.index]
DIST_GRUPO = (pd.crosstab(_L['faixa do pico'], _L.grupo, normalize='columns') * 100)[['acertada', 'oscila', 'errada']]
DIST = pd.concat({f'{CLASSE} por grupo': DIST_GRUPO, 'todas as gravações, por classe': DIST_CLASSE}, axis=1)
print('Onde cai o pico grave (% das gravações):'); display(DIST.round(1))
RESUMO_PICO = _L.groupby('grupo')[['f pico (Hz)', 'altura (dB)', 'largura (Hz)']].median()
print('Medianas por grupo:'); display(RESUMO_PICO.round(1))

fig, ax = plt.subplots(figsize=(11, 4))
edges = np.arange(20, 251, FS / NFFT) - FS / NFFT / 2
series = [(f'{CLASSE} acertada', _L[_L.grupo == 'acertada'], 'tab:green', '-'),
          (f'{CLASSE} errada', _L[_L.grupo == 'errada'], 'tab:red', '-')]
series += [(c, navios[navios.classe == c], cor, '--')
           for c, cor in (('SMALL', 'tab:blue'), ('MEDIUM', 'tab:purple'), ('LARGE', 'tab:orange')) if c != CLASSE]
for nome, x, cor, ls in series:
    ax.hist(x['f pico (Hz)'], bins=edges, density=True, histtype='step', lw=1.8, color=cor, ls=ls, label=f'{nome} (n={len(x)})')
ax.set_xlabel('frequência do pico grave (Hz)'); ax.set_ylabel('densidade'); ax.legend()
ax.set_title('Distribuição da frequência do pico grave (20–250 Hz) por gravação'); plt.tight_layout(); plt.show()

# frequência do pico × comprimento do navio, dentro da classe (um ponto por navio)
_n = _L.dropna(subset=['Length']).groupby('Ship ID')[['f pico (Hz)', 'Length']].median()
RHO_LEN = spearmanr(_n['f pico (Hz)'], _n['Length'])
print(f"{CLASSE}: rho de Spearman entre frequência do pico e comprimento (por navio, n = {len(_n)}): "
      f"{RHO_LEN[0]:.3f} (p = {RHO_LEN[1]:.1e})")
'''

HARM_CODE = r'''
# Harmônicos: nas gravações com pico grave em 31–86 Hz, há linha em 2x e 3x a frequência do pico?
# Tolerância de ±h faixas (a incerteza de ±1/2 faixa no pico cresce com o múltiplo). Taxa de base: chance de haver
# linha a <= h faixas de uma posição qualquer da faixa [h·31, h·86] Hz na mesma gravação.
_rows = []
for i in T.index[T.grupo.isin(['acertada', 'errada'])]:
    f0, k0 = PK.loc[i, 'f pico (Hz)'], int(PK.loc[i, 'bin'])
    if not (31 <= f0 < 86):
        continue
    ls = linhas(i)
    for h in (2, 3):
        tgt = h * (k0 + 1) - 1
        cand = np.arange(int(np.ceil(h * 31 / (FS / NFFT))) - 1, int(h * 86 / (FS / NFFT)))
        _rows.append({'grupo': T.grupo[i], 'múltiplo': h,
                      'linha no múltiplo (%)': 100 * any(abs(m - tgt) <= h for m in ls),
                      'taxa de base (%)': 100 * np.mean([any(abs(m - c) <= h for m in ls) for c in cand])})
HARM = pd.DataFrame(_rows).groupby(['grupo', 'múltiplo']).agg(
    gravações=('múltiplo', 'size'), linha_no_multiplo=('linha no múltiplo (%)', 'mean'),
    taxa_de_base=('taxa de base (%)', 'mean')).rename(columns={'linha_no_multiplo': 'linha no múltiplo (%)',
                                                               'taxa_de_base': 'taxa de base (%)'})
display(HARM.round(1))
'''

EXEMPLOS_CODE = r'''
# Espectrogramas LOFAR (0–500 Hz) de gravações sorteadas do subconjunto com mais gravações da classe (semente fixa)
_rng = np.random.default_rng(1)
_ds = T.Dataset.value_counts().index[0]
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
for r, g in enumerate(('acertada', 'errada')):
    pool = T.index[(T.grupo == g) & (T.Dataset == _ds)]
    for c, i in enumerate(_rng.choice(pool, min(4, len(pool)), replace=False)):
        x = pd.read_pickle(os.path.join(LOFAR_DIR, f'{i}.pkl'))['df'].values[:, F <= 500]
        ax[r, c].imshow(x, aspect='auto', origin='lower', extent=[F[0], 500, 0, len(x)], vmin=0, vmax=15, cmap='viridis')
        ax[r, c].set_title(f'{CLASSE} {g} — ID {i} (pico {PK.loc[i, "f pico (Hz)"]:.0f} Hz)', fontsize=9)
        ax[r, c].set_xlabel('Hz'); ax[r, c].set_ylabel('janela')
fig.suptitle(f'LOFAR 0–500 Hz (0 a 15 dB acima do fundo), subconjunto {_ds}'); plt.tight_layout(); plt.show()
EXEMPLO_DS = _ds
'''

TIPO_CODE = r'''
from scipy.stats import binomtest
# Tabela por gravação (só acertadas e erradas): banda = pico grave em 31–86 Hz
DC = T[T.grupo != 'oscila'].join(PK[['f pico (Hz)', 'altura (dB)']]).join(meta[['Rain state', 'Wind', 'Sea state', 'SHIPTYPE']])
DC['banda'] = (DC['f pico (Hz)'] >= 31) & (DC['f pico (Hz)'] < 86)
DC['errada'] = (DC.grupo == 'errada').astype(int)
for _c in ('Wind', 'Sea state'):
    DC[_c] = pd.to_numeric(DC[_c], errors='coerce')

TIPO = DC.groupby('DETAILED TYPE').agg(gravações=('errada', 'size'), navios=('Ship ID', 'nunique'),
                                       pct_errada=('errada', 'mean'), pct_banda=('banda', 'mean'),
                                       comprimento=('Length', 'median'))
TIPO[['pct_errada', 'pct_banda']] *= 100
TIPO = TIPO[TIPO.gravações >= 10].sort_values('pct_errada').rename(
    columns={'pct_errada': '% errada', 'pct_banda': '% com banda 31–86 Hz', 'comprimento': 'comprimento mediano (m)'})
display(TIPO.round(1))
RHO_TIPO = spearmanr(TIPO['% errada'], TIPO['% com banda 31–86 Hz'])
print(f'Entre os tipos: rho(% errada, % com banda) = {RHO_TIPO[0]:.2f} (p = {RHO_TIPO[1]:.3f}, {len(TIPO)} tipos)')
'''

MESMO_NAVIO_CODE = r'''
# Navios com gravações nos dois grupos: a banda acompanha o navio ou a gravação?
_m = DC.groupby('Ship ID').grupo.nunique()
MIX = DC[DC['Ship ID'].isin(_m[_m > 1].index)]
N_MIX = MIX['Ship ID'].nunique()
PAR = MIX.groupby(['Ship ID', 'grupo'])[['banda', 'Wind', 'Sea state', 'altura (dB)']].mean().unstack('grupo')
display(pd.crosstab(MIX.grupo, MIX.banda.map({True: 'com banda', False: 'sem banda'}), margins=True, margins_name='total'))
def sinal(v):
    # teste do sinal pareado por navio: acertada > errada × acertada < errada (empates fora)
    a, e = PAR[(v, 'acertada')], PAR[(v, 'errada')]
    ok = a.notna() & e.notna()
    mais, menos = int((a[ok] > e[ok]).sum()), int((a[ok] < e[ok]).sum())
    return {'variável': v, 'navios': int(ok.sum()), 'média acertada': a[ok].mean(), 'média errada': e[ok].mean(),
            'acertada > errada': mais, 'acertada < errada': menos, 'empates': int(ok.sum()) - mais - menos,
            'p (teste do sinal)': binomtest(mais, mais + menos).pvalue if mais + menos else np.nan}
PAREADO = pd.DataFrame([sinal(v) for v in ('banda', 'altura (dB)', 'Wind', 'Sea state')]).set_index('variável')
display(PAREADO.round(3))
'''

COND_CODE = r'''
# Condições de gravação e comprimento: diferem entre acertadas e erradas? E entre gravações com e sem banda?
_rows = []
for col in ('Length', 'Wind', 'Sea state'):
    x = DC.dropna(subset=[col])
    ac, er = x[x.grupo == 'acertada'][col], x[x.grupo == 'errada'][col]
    cb, sb = x[x.banda][col], x[~x.banda][col]
    _rows.append({'variável': col, 'mediana acertada': ac.median(), 'mediana errada': er.median(),
                  'p (acertada × errada)': mannwhitneyu(ac, er).pvalue,
                  'mediana com banda': cb.median(), 'mediana sem banda': sb.median(),
                  'p (com × sem banda)': mannwhitneyu(cb, sb).pvalue,
                  'rho com a altura do pico': spearmanr(x[col], x['altura (dB)'])[0]})
COND = pd.DataFrame(_rows).set_index('variável')
print('Por gravação (Mann-Whitney):'); display(COND.round(3))
print('Vento mediano (m/s) por subconjunto e grupo:')
display(DC.pivot_table(index='Dataset', columns='grupo', values='Wind', aggfunc='median').round(1))
print('Gravações por estado do mar e grupo:')
display(pd.crosstab(DC['Sea state'], DC.grupo).assign(**{'% errada': lambda t: (100 * t.errada / t.sum(1)).round(1)}))
print('Gravações por estado de chuva e grupo:'); display(pd.crosstab(DC['Rain state'].str.strip(), DC.grupo))
'''

LOGIT_CODE = r'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# Quanto do erro cada conjunto de variáveis explica? Regressão logística (errada = 1), validação cruzada em 5 partes
# agrupadas por navio (o mesmo navio nunca está no treino e no teste), repetida com 10 sementes; AUC no teste.
DC['banda (0/1)'] = DC.banda.astype(float)
CONJ = {'subconjunto': ([], ['Dataset']),
        'metadados (subconjunto, tipo, comprimento, vento, estado do mar)': (['Length', 'Wind', 'Sea state'], ['Dataset', 'SHIPTYPE']),
        'banda (0/1)': (['banda (0/1)'], []),
        'banda + altura do pico': (['banda (0/1)', 'altura (dB)'], []),
        'subconjunto + banda + altura': (['banda (0/1)', 'altura (dB)'], ['Dataset']),
        'metadados + banda + altura': (['Length', 'Wind', 'Sea state', 'banda (0/1)', 'altura (dB)'], ['Dataset', 'SHIPTYPE'])}
def auc_cv(num, cat, rep=10):
    pre = ColumnTransformer([('n', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num),
                             ('c', OneHotEncoder(handle_unknown='ignore'), cat)])
    m = make_pipeline(pre, LogisticRegression(max_iter=2000))
    a = [roc_auc_score(DC.errada, cross_val_predict(m, DC[num + cat], DC.errada, groups=DC['Ship ID'], method='predict_proba',
                                                    cv=StratifiedGroupKFold(5, shuffle=True, random_state=s))[:, 1])
         for s in range(rep)]
    return np.mean(a), np.std(a)
LOGIT = pd.DataFrame([(k, *auc_cv(*v)) for k, v in CONJ.items()], columns=['variáveis', 'AUC', 'desvio (sementes)']).set_index('variáveis')
display(LOGIT.round(3))
'''

L1HZ_CODE = r'''
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_widths
# lofar1hz (scripts/extrair_lofar1hz.py): 48 kHz -> 2 kHz, FFT de 2048 pontos (1,024 s), faixas de 0,977 Hz até 1 kHz,
# TPSW com a mesma largura em Hz do pipeline (±164 Hz). Resumo por gravação: espectro médio e persistência > 6 dB.
R1 = pd.read_pickle(os.path.join(R.ROOT, 'data', 'iara_processed', 'cache_lofar1hz_resumo.pkl'))
F1 = np.asarray(R1['freqs']); S1 = R1['lofar_medio'].astype(float); P1 = R1['pers6'].astype(float); DF1 = F1[0]
IA, IE = T.index[T.grupo == 'acertada'], T.index[T.grupo == 'errada']
def por_navio_1hz(X, ids):
    return X.loc[ids].groupby(T.loc[ids, 'Ship ID'].values).mean()
A1, E1 = por_navio_1hz(S1, IA), por_navio_1hz(S1, IE)
k3 = F1 <= 300
_U, _p = mannwhitneyu(E1.values[:, k3], A1.values[:, k3], axis=0)
TF1 = pd.DataFrame({'f (Hz)': F1[k3], 'acertada': A1.values[:, k3].mean(0), 'errada': E1.values[:, k3].mean(0),
                    'AUC': _U / (len(A1) * len(E1)), 'q (FDR)': false_discovery_control(_p)})
TF1['Δ (errada − acertada)'] = TF1.errada - TF1.acertada

fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
ax[0].plot(TF1['f (Hz)'], TF1.acertada, color='tab:green', lw=1, label=f'acertada ({len(A1)} navios), lofar1hz')
ax[0].plot(TF1['f (Hz)'], TF1.errada, color='tab:red', lw=1, label=f'errada ({len(E1)} navios), lofar1hz')
k7 = F <= 300
ax[0].plot(F[k7], G['acertada'].mean().values[k7], 'o', ms=3, color='darkgreen', label='acertada, LOFAR do pipeline (7,8 Hz)')
ax[0].plot(F[k7], G['errada'].mean().values[k7], 'o', ms=3, color='darkred', label='errada, LOFAR do pipeline (7,8 Hz)')
ax[0].set_ylabel('dB acima do fundo'); ax[0].legend(fontsize=8)
ax[0].set_title(f'{CLASSE}: espectro médio 0–300 Hz com faixas de 0,98 Hz (lofar1hz)')
ax[1].plot(TF1['f (Hz)'], TF1['Δ (errada − acertada)'], color='k', lw=0.8)
s_ = TF1['q (FDR)'] < 0.05
ax[1].scatter(TF1['f (Hz)'][s_], TF1['Δ (errada − acertada)'][s_], s=5, color='tab:red', label='q < 0,05')
ax[1].axhline(0, color='gray', lw=0.8); ax[1].set_ylabel('Δ (dB)'); ax[1].set_xlabel('frequência (Hz)'); ax[1].legend()
plt.tight_layout(); plt.show()
_i = TF1['Δ (errada − acertada)'].idxmin()
L1_MIN = TF1.loc[_i]; L1_PICO = TF1['f (Hz)'][TF1.acertada.idxmax()]
print(f"Maior Δ: {L1_MIN['Δ (errada − acertada)']:.2f} dB em {L1_MIN['f (Hz)']:.1f} Hz (AUC {L1_MIN['AUC']:.2f}); "
      f"pico do espectro médio das acertadas em {L1_PICO:.1f} Hz")
'''

DECOMP_CODE = r'''
# Banda larga ou raias? Em cada gravação, o espectro médio (lofar1hz) é separado em:
#   componente larga = mediana móvel de 15 faixas (~15 Hz): ignora estruturas mais estreitas que ~7 Hz;
#   raias = o que sobe acima dela; uma raia é um máximo local com proeminência >= 3 dB.
# Num espectro médio de ~580 linhas, a flutuação do ruído é de décimos de dB: 3 dB é uma raia estável, não ruído.
def decompoe(i, lo=31, hi=86, jan=15, prom=3.0):
    y = S1.loc[i].values
    base = median_filter(y, size=jan, mode='nearest')
    r = y - base
    k = (F1 >= lo) & (F1 <= hi)
    pk, _ = find_peaks(r, prominence=prom); pk = pk[k[pk]]
    return {'componente larga (dB)': base[k].mean(), 'energia das raias (dB)': np.clip(r[k], 0, None).mean(),
            'raias (n)': len(pk), 'altura das raias (dB)': r[pk].mean() if len(pk) else np.nan}

def compara(Dx):
    rows = []
    for c in ('componente larga (dB)', 'energia das raias (dB)', 'raias (n)', 'altura das raias (dB)'):
        sh = Dx.groupby(['Ship ID', 'grupo'])[c].mean().unstack()
        a, e = sh['acertada'].dropna(), sh['errada'].dropna()
        U, p = mannwhitneyu(e, a)
        rows.append({'medida': c, 'mediana acertada': a.median(), 'mediana errada': e.median(),
                     'AUC (errada > acertada)': U / (len(a) * len(e)), 'p': p})
    return pd.DataFrame(rows).set_index('medida')

_ids = list(IA) + list(IE)
def tabela(lo=31, hi=86, jan=15, prom=3.0):
    Dx = pd.DataFrame({i: decompoe(i, lo, hi, jan, prom) for i in _ids}).T.astype(float)
    return Dx.join(T[['grupo', 'Ship ID']])
DEC = tabela()
DEC_CMP = compara(DEC)
print('31–86 Hz, por navio (Mann-Whitney):'); display(DEC_CMP.round(3))
DEC_CTRL = tabela(150, 205)
print(f"Controle 150–205 Hz: {DEC_CTRL.loc[IA, 'raias (n)'].mean():.2f} raias por gravação nas acertadas e "
      f"{DEC_CTRL.loc[IE, 'raias (n)'].mean():.2f} nas erradas")

# sensibilidade: largura da mediana móvel e proeminência mínima
_rows = []
for jan in (11, 15, 21):
    for prom in (2.0, 3.0, 6.0):
        c = compara(tabela(jan=jan, prom=prom))['AUC (errada > acertada)']
        _rows.append({'mediana móvel (faixas)': jan, 'proeminência (dB)': prom, 'AUC componente larga': c['componente larga (dB)'],
                      'AUC energia das raias': c['energia das raias (dB)'], 'AUC nº de raias': c['raias (n)']})
DEC_SENS = pd.DataFrame(_rows)
print('Sensibilidade (AUC por navio; 0,5 = sem diferença):'); display(DEC_SENS.round(3))

# exemplo visual: a acertada e a errada mais típicas (mais perto da mediana da componente larga do seu grupo)
fig, ax = plt.subplots(1, 2, figsize=(15, 3.8), sharey=True)
for a_, g, cor in ((ax[0], 'acertada', 'tab:green'), (ax[1], 'errada', 'tab:red')):
    x = DEC[DEC.grupo == g]['componente larga (dB)']
    i = (x - x.median()).abs().idxmin()
    k = F1 <= 150
    y = S1.loc[i].values; base = median_filter(y, size=15, mode='nearest')
    a_.plot(F1[k], y[k], color=cor, lw=0.9, label='espectro médio (lofar1hz)')
    a_.plot(F1[k], base[k], color='k', lw=1.6, label='componente larga (mediana de ~15 Hz)')
    a_.axvspan(31, 86, color='0.9', zorder=0)
    a_.set_title(f'{CLASSE} {g} típica — ID {i}', fontsize=10); a_.set_xlabel('frequência (Hz)'); a_.legend(fontsize=8)
ax[0].set_ylabel('dB acima do fundo'); plt.tight_layout(); plt.show()
'''

BAIXAS_CODE = r'''
# Raias estreitas e persistentes: máximo local da persistência > 6 dB com >= 50% das janelas, que se destaca pelo menos
# 20 pontos percentuais dos vizinhos (proeminência >= 0,2; sem isso, ondulações sobre a banda larga contariam como raias)
# e tem largura a meia proeminência <= 3 Hz. Abaixo de 31 Hz, o fundo TPSW só tem vizinhos de um lado e inclui a banda
# grave: leia com cautela.
import warnings
warnings.filterwarnings('ignore', message='some peaks have')   # picos na borda do eixo têm largura 0; são descartados
def estreitas(i, lo, hi):
    y = P1.loc[i].values
    pk, _ = find_peaks(np.r_[0, y, 0], height=0.5, prominence=0.2); pk = pk - 1
    if len(pk) == 0:
        return np.array([])
    w = peak_widths(y, pk, rel_height=0.5)[0] * DF1
    f = F1[pk]
    return f[(w <= 3) & (f >= lo) & (f < hi)]
_rows = []
for lo, hi in ((5, 31), (31, 86), (86, 300)):
    L = {i: estreitas(i, lo, hi) for i in _ids}
    tem = pd.Series({i: len(v) > 0 for i, v in L.items()})
    na = tem[IA].groupby(T.loc[IA, 'Ship ID'].values).max(); ne = tem[IE].groupby(T.loc[IE, 'Ship ID'].values).max()
    _rows.append({'faixa': f'{lo}–{hi} Hz', '% gravações acertadas com raia': 100 * tem[IA].mean(),
                  '% gravações erradas com raia': 100 * tem[IE].mean(),
                  'raias por gravação (acertada)': np.mean([len(L[i]) for i in IA]),
                  'raias por gravação (errada)': np.mean([len(L[i]) for i in IE]),
                  'p (Fisher, por navio)': fisher_exact([[ne.sum(), len(ne) - ne.sum()], [na.sum(), len(na) - na.sum()]])[1]})
    if lo == 5:
        FB_A = np.concatenate([L[i] for i in IA]); FB_E = np.concatenate([L[i] for i in IE])
BAIXAS = pd.DataFrame(_rows).set_index('faixa')
display(BAIXAS.round(3))
fig, ax = plt.subplots(figsize=(10, 3))
bins = np.arange(5, 31.5, 1)
ax.hist(FB_A, bins=bins, histtype='step', lw=1.8, color='tab:green', weights=np.full(len(FB_A), 1 / len(IA)), label='acertada')
ax.hist(FB_E, bins=bins, histtype='step', lw=1.8, color='tab:red', weights=np.full(len(FB_E), 1 / len(IE)), label='errada')
ax.set_xlabel('frequência da raia (Hz)'); ax.set_ylabel('raias por gravação'); ax.legend()
ax.set_title('Raias estreitas persistentes abaixo de 31 Hz'); plt.tight_layout(); plt.show()
'''

# ---------------------------------------------------------------------------------------------------------------------
# Versão "por destino" (MEDIUM, SMALL): os erros dessas classes vão para mais de uma classe, com assinaturas opostas,
# então as erradas são separadas pela classe para onde vão.
# ---------------------------------------------------------------------------------------------------------------------
DEST_CODE = r'''
# destino de cada gravação errada: a classe mais prevista nos folds em que ela foi teste
_d = pd.concat(R.file_predictions(MODELO).values()); _d = _d[_d.Target == C].copy(); _d['File'] = _d.File.astype(int)
_dest = _d[_d.Prediction != C].groupby('File').Prediction.agg(lambda x: x.value_counts().index[0]).map(CLS)
T['destino'] = _dest.reindex(T.index)
T['subgrupo'] = np.where(T.grupo == 'errada', 'errada→' + T.destino.fillna('?'), T.grupo)
NAV_DEST = T[T.grupo == 'errada'].groupby('destino')['Ship ID'].nunique().sort_values(ascending=False)
DESTS = [d for d in NAV_DEST.index if NAV_DEST[d] >= 10]          # destinos com pelo menos 10 navios
SUBS = ['acertada'] + [f'errada→{d}' for d in DESTS]
RESUMO_SUB = T.groupby('subgrupo').agg(gravações=('grupo', 'size'), navios=('Ship ID', 'nunique'),
                                       comprimento_mediano=('Length', 'median'))
print('Subgrupos (errada→X = gravação sempre errada, prevista sobretudo como X):'); display(RESUMO_SUB)
print('Gravações por subconjunto:'); display(pd.crosstab(T.subgrupo, T.Dataset, margins=True, margins_name='total'))

# a classe é definida pelo comprimento: um navio não aparece em duas classes (os navios das outras classes são outros)
_cl = LAB.map(CLS).to_frame('classe').join(meta[['Ship ID']]).dropna()
_cl = _cl[_cl['Ship ID'].str.strip().isin(['', '-']) == False]
N_NAVIOS_COMUNS = int((_cl.groupby('Ship ID').classe.nunique() > 1).sum())
print(f'Navios presentes em mais de uma classe: {N_NAVIOS_COMUNS}')
'''

MEDIO_DEST_CODE = r'''
REF = {CLS[c]: S.loc[LAB.index[LAB == c]] for c in range(4) if c != C}
GS = {s: S.loc[T.index[T.subgrupo == s]] for s in SUBS}
cores_s = dict(zip(SUBS, ['tab:green', 'tab:red', 'tab:orange', 'tab:brown']))
cores_ref = {'SMALL': 'tab:blue', 'MEDIUM': 'tab:purple', 'LARGE': 'k', 'BACKGROUND': 'tab:gray'}
fig, ax = plt.subplots(2, 1, figsize=(13, 8))
for a, fmax in zip(ax, (8000, 500)):
    k = F <= fmax
    for s, X in GS.items():
        m, se = X.mean().values, X.std().values / np.sqrt(len(X))
        a.plot(F[k], m[k], color=cores_s[s], lw=1.6, label=f'{CLASSE} {s} (n={len(X)})')
        a.fill_between(F[k], (m - 1.96 * se)[k], (m + 1.96 * se)[k], color=cores_s[s], alpha=0.2)
    for c, X in REF.items():
        a.plot(F[k], X.mean().values[k], color=cores_ref[c], lw=1, ls='--', label=f'{c} (todas, n={len(X)})')
    a.set_xlabel('frequência (Hz)'); a.set_ylabel('LOFAR médio (dB acima do fundo)'); a.set_xlim(0, fmax)
ax[0].set_title(f'Espectro LOFAR médio — {CLASSE} acertadas e erradas por destino (faixa = IC 95% da média)')
ax[1].set_title('Zoom: 0–500 Hz'); ax[0].legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.show()
'''

TESTE_DEST_CODE = r'''
ia = T.index[T.subgrupo == 'acertada']
CONTRASTES = {'errada (todas)': T.index[T.grupo == 'errada'], **{f'errada→{d}': T.index[T.subgrupo == f'errada→{d}'] for d in DESTS}}
TFD, _rows = {}, []
FAIXAS_B = [0, 31, 86, 150, 250, 500, 1000, 2000, 4000, 8001]
for nome, ie in CONTRASTES.items():
    tf, na, ne = teste_faixas(ia, ie); tf['sig'] = tf['q (FDR)'] < 0.05; TFD[nome] = tf
    b = tf.groupby(pd.cut(tf['f (Hz)'], FAIXAS_B, right=False), observed=False)
    r = {'contraste': f'acertada × {nome}', 'navios acertada': na, 'navios errada': ne,
         'faixas com q < 0,05': int(tf.sig.sum()), 'abaixo de 1 kHz': int(tf[tf['f (Hz)'] < 1000].sig.sum()),
         'acima de 1 kHz': int(tf[tf['f (Hz)'] >= 1000].sig.sum())}
    r.update({f'Δ {iv.left}–{min(iv.right, 8000)} Hz': v for iv, v in b['Δ (errada − acertada)'].mean().items()})
    r.update({f'AUC {iv.left}–{min(iv.right, 8000)} Hz': v for iv, v in b['AUC'].mean().items()})
    _rows.append(r)
CONTR = pd.DataFrame(_rows).set_index('contraste')
print('Δ e AUC médios por banda (errada − acertada; médias por navio):'); display(CONTR.round(2).T)

fig, ax = plt.subplots(len(TFD), 1, figsize=(13, 2.6 * len(TFD)), sharex=True, sharey=True)
for a, (nome, tf) in zip(np.atleast_1d(ax), TFD.items()):
    k = tf['f (Hz)'] <= 1000
    a.plot(tf['f (Hz)'][k], tf['Δ (errada − acertada)'][k], color='k', lw=1)
    s_ = k & tf.sig
    a.scatter(tf['f (Hz)'][s_], tf['Δ (errada − acertada)'][s_], color='tab:red', s=10, label='q < 0,05')
    a.axhline(0, color='gray', lw=0.8); a.axvspan(31, 86, color='0.92', zorder=0)
    a.set_ylabel('Δ (dB)'); a.set_title(f'{CLASSE}: {nome} − acertada ({int(tf.sig.sum())} faixas com q < 0,05)', fontsize=10)
    if s_.any():
        a.legend(fontsize=8)
np.atleast_1d(ax)[-1].set_xlabel('frequência (Hz) — faixa cinza: banda 31–86 Hz das LARGE')
plt.tight_layout(); plt.show()
'''

APRIORI_CODE = r'''
# Hipótese fixada ANTES de olhar esta classe: a banda 31–86 Hz que separa as LARGE acertadas das erradas (NB35 LARGE),
# e a faixa 100–150 Hz, onde as LARGE erradas tinham mais conteúdo. Os navios desta classe não entraram naquela análise.
BL, BC = (F >= 31) & (F < 86), (F >= 100) & (F < 150)
_kp = (F >= 20) & (F <= 250)
_pico = pd.Series(F[np.where(_kp)[0][S.values[:, _kp].argmax(1)]], index=S.index)
X = pd.DataFrame({'banda 31–86 Hz (dB)': S.loc[T.index, BL].mean(1), '100–150 Hz (dB)': S.loc[T.index, BC].mean(1),
                  'pico grave em 31–86 Hz': _pico.loc[T.index].between(31, 86, inclusive='left')}).join(T[['subgrupo', 'Ship ID']])
NAV = X.groupby(['subgrupo', 'Ship ID']).mean(numeric_only=True).reset_index()
_a = NAV[NAV.subgrupo == 'acertada']
_rows = []
for s in SUBS:
    x = NAV[NAV.subgrupo == s]
    r = {'subgrupo': f'{CLASSE} {s}', 'navios': len(x)}
    for c in ('banda 31–86 Hz (dB)', '100–150 Hz (dB)'):
        r[f'{c}: mediana'] = x[c].median()
        if s != 'acertada':
            U, p = mannwhitneyu(x[c], _a[c]); r[f'{c}: AUC vs acertada'] = U / (len(x) * len(_a)); r[f'{c}: p'] = p
    r['% gravações com pico em 31–86 Hz'] = 100 * X[X.subgrupo == s]['pico grave em 31–86 Hz'].mean()
    _rows.append(r)
for c in REF:
    ids = LAB.index[LAB == CID[c]]
    _rows.append({'subgrupo': f'{c} (todas)', 'navios': meta.loc[ids, 'Ship ID'].nunique() if c != 'BACKGROUND' else np.nan,
                  'banda 31–86 Hz (dB): mediana': S.loc[ids, BL].mean(1).median(), '100–150 Hz (dB): mediana': S.loc[ids, BC].mean(1).median(),
                  '% gravações com pico em 31–86 Hz': 100 * _pico.loc[ids].between(31, 86, inclusive='left').mean()})
APRIORI = pd.DataFrame(_rows).set_index('subgrupo')
display(APRIORI.round(3))
'''

SEM_DEST_CODE = r'''
# distância RMS (dB) entre o espectro médio de cada subgrupo e o de cada classe, 0–500 Hz
k = F <= 500
REF_ALL = {CLS[c]: S.loc[LAB.index[LAB == c]].mean().values for c in range(4)}
SEMD = pd.DataFrame({f'{CLASSE} {s}': {c: np.sqrt(((GS[s].mean().values[k] - m[k]) ** 2).mean()) for c, m in REF_ALL.items()}
                     for s in SUBS}).T
display(SEMD.round(2))
'''

BANDA_PREV_CODE = r'''
# A banda como pista do modelo: em que fração dos folds a gravação é prevista como LARGE, sem e com o pico grave
# (maior LOFAR médio entre 20 e 250 Hz) em 31–86 Hz? Só gravações de navio (sem BACKGROUND).
_kp = (F >= 20) & (F <= 250)
_pk = pd.Series(F[np.where(_kp)[0][S.values[:, _kp].argmax(1)]], index=S.index)
_dd = pd.concat(R.file_predictions(MODELO).values()); _dd['File'] = _dd.File.astype(int)
_t = _dd.assign(L=_dd.Prediction == CID['LARGE']).groupby('File').agg(L=('L', 'mean'), Target=('Target', 'first'))
_t = _t[_t.Target != CID['BACKGROUND']]
_t['banda'] = _pk.reindex(_t.index).between(31, 86, inclusive='left')
_t['classe'] = _t.Target.map(CLS)
BANDA_PREV = (100 * _t.groupby(['classe', 'banda']).L.mean().unstack()).rename(
    columns={False: '% prevista LARGE, sem a banda', True: '% prevista LARGE, com a banda'})
BANDA_PREV.columns.name = None
BANDA_PREV['% das gravações com a banda'] = 100 * _t.groupby('classe').banda.mean()
BANDA_PREV = BANDA_PREV.loc[['SMALL', 'MEDIUM', 'LARGE']]
FRAC_LARGE_BANDA = 100 * (_t[_t.banda].classe == 'LARGE').mean()
display(BANDA_PREV.round(1))
print(f'Das gravações de navio com a banda, {FRAC_LARGE_BANDA:.0f}% são LARGE.')
'''


ACERT_CODE = r'''
# As acertadas têm um padrão próprio de frequência, ou o acerto vem de navios repetidos entre treino e teste?
# (1) acertadas × cada outra classe de navio, faixa a faixa por navio (Mann-Whitney + FDR)
_A = por_navio(T.index[T.subgrupo == 'acertada'])
def nav_classe(c):
    ids = LAB.index[LAB.map(CLS) == c]
    return S.loc[ids].groupby(meta.loc[ids, 'Ship ID'].values).mean()
_rows = []
for c in [x for x in ('SMALL', 'MEDIUM', 'LARGE') if x != CLASSE]:
    B = nav_classe(c)
    U, p = mannwhitneyu(_A.values, B.values, axis=0); q = false_discovery_control(p)
    d = _A.values.mean(0) - B.values.mean(0); auc = U / (len(_A) * len(B))
    r = {'comparação': f'{CLASSE} acertada × {c}', 'navios': f'{len(_A)} × {len(B)}', 'faixas com q < 0,05': int((q < 0.05).sum())}
    for lo, hi in ((31, 86), (86, 150), (150, 250)):
        k = (F >= lo) & (F < hi)
        r[f'Δ {lo}–{hi} Hz (dB)'] = d[k].mean(); r[f'AUC {lo}–{hi} Hz'] = auc[k].mean()
    _rows.append(r)
ACV = pd.DataFrame(_rows).set_index('comparação')
print('Acertadas contra as outras classes (médias por navio; Δ = acertada − outra classe):'); display(ACV.round(3))

# (2) onde cai o pico grave: acertadas × classes inteiras
_cut = lambda x: pd.cut(x, [20, 31, 86, 140, 251], right=False, labels=['20–31 Hz', '31–86 Hz', '86–140 Hz', '140–250 Hz'])
PICO_AC = pd.DataFrame({f'{CLASSE} acertada': _cut(_pico.loc[T.index[T.subgrupo == 'acertada']]).value_counts(normalize=True, sort=False),
                        **{c: _cut(_pico.loc[LAB.index[LAB.map(CLS) == c]]).value_counts(normalize=True, sort=False)
                           for c in ('SMALL', 'MEDIUM', 'LARGE')}}) * 100
print('Onde cai o pico grave (% das gravações):'); display(PICO_AC.round(1))

# (3) taxa de acerto conforme o número de gravações do navio (nesta classe e nas LARGE, para comparação)
def por_tamanho(Tx):
    n = Tx['Ship ID'].map(Tx.groupby('Ship ID').size())
    fx = pd.cut(n, [0, 1, 3, 10, 10000], labels=['1', '2–3', '4–10', 'mais de 10'])
    t = (pd.crosstab(fx, Tx.grupo, normalize='index') * 100)[['acertada', 'oscila', 'errada']]
    return t.join(Tx.groupby(fx, observed=True).agg(gravações=('grupo', 'size'), navios=('Ship ID', 'nunique')))
_tl = taxa_acerto(MODELO, CID['LARGE'])
_TL = pd.DataFrame({'grupo': grupo(_tl)}, index=_tl.index).join(meta[['Ship ID']])   # grupo() devolve array: manter o índice
TAM = pd.concat({CLASSE: por_tamanho(T), 'LARGE': por_tamanho(_TL)}, axis=0)
print('Grupo (% das gravações) conforme o número de gravações do navio:'); display(TAM.round(1))
_n = T['Ship ID'].map(T.groupby('Ship ID').size())
N_AC, N_AC_GRANDES = int((T.grupo == 'acertada').sum()), int(((T.grupo == 'acertada') & (_n > 10)).sum())
N_NAV_GRANDES = int(T.loc[_n > 10, 'Ship ID'].nunique())
TOP_AC = T[T.grupo == 'acertada']['Ship ID'].value_counts().head(5).rename('acertadas').to_frame()
TOP_AC = TOP_AC.join(meta.drop_duplicates('Ship ID').set_index('Ship ID')[['Name', 'DETAILED TYPE', 'Length']])
TOP_AC['gravações do navio'] = T.groupby('Ship ID').size().reindex(TOP_AC.index)
print('Navios com mais gravações acertadas:'); display(TOP_AC)

# (4) o navio mais frequente: quantas gravações dele estão no teste e quantas no treino, em cada fold?
_top = TOP_AC.index[0]
_ids = set(T.index[T['Ship ID'] == _top])
_fp = R.file_predictions(MODELO)
FOLDS_TOP = pd.DataFrame({f: {'no teste': len(_ids & set(v.File.astype(int))), 'fora do teste (treino)': len(_ids) - len(_ids & set(v.File.astype(int)))}
                          for f, v in _fp.items()}).T
FOLDS_TOP.index.name = 'fold'
print(f'Navio {_top}: {len(_ids)} gravações, divididas entre teste e treino em cada fold do split do artigo:'); display(FOLDS_TOP.T)
'''

META_DEST_CODE = r'''
# Comprimento, condições e subconjunto por subgrupo (por gravação; Mann-Whitney contra as acertadas)
M = T[['subgrupo', 'Length', 'Dataset']].join(meta[['Wind', 'Sea state']])
for c in ('Wind', 'Sea state'):
    M[c] = pd.to_numeric(M[c], errors='coerce')
_rows = []
for s in SUBS:
    x = M[M.subgrupo == s]; r = {'subgrupo': s, 'gravações': len(x)}
    for c in ('Length', 'Wind', 'Sea state'):
        r[f'{c}: mediana'] = x[c].median()
        if s != 'acertada':
            a, b = M[M.subgrupo == 'acertada'][c].dropna(), x[c].dropna()
            r[f'{c}: p'] = mannwhitneyu(b, a).pvalue
    _rows.append(r)
METAD = pd.DataFrame(_rows).set_index('subgrupo')
display(METAD.round(3))
print('% das gravações de cada subgrupo por subconjunto:')
display((pd.crosstab(M.subgrupo, M.Dataset, normalize='index') * 100).loc[SUBS].round(1))
'''

MISTO_DEST_CODE = r'''
from scipy.stats import binomtest
# Navios com gravações acertadas e erradas: a banda acompanha a gravação também aqui?
_x = X[X.subgrupo.isin(SUBS)]
_m = _x.groupby('Ship ID').subgrupo.agg(lambda s: set(s))
MIXD = {}
for s in SUBS[1:]:
    navs = _m[_m.apply(lambda v: 'acertada' in v and s in v)].index
    if len(navs) == 0:
        MIXD[s] = None; continue
    p_ = _x[_x['Ship ID'].isin(navs)].groupby(['Ship ID', 'subgrupo'])['banda 31–86 Hz (dB)'].mean().unstack()[['acertada', s]]
    mais, menos = int((p_[s] > p_['acertada']).sum()), int((p_[s] < p_['acertada']).sum())
    MIXD[s] = {'navios': len(navs), 'média acertada (dB)': p_['acertada'].mean(), 'média errada (dB)': p_[s].mean(),
               'errada > acertada': mais, 'errada < acertada': menos,
               'p (teste do sinal)': binomtest(mais, mais + menos).pvalue if mais + menos else np.nan}
MIXD_T = pd.DataFrame({k: v for k, v in MIXD.items() if v}).T
print('Banda 31–86 Hz nos navios com gravações acertadas e erradas (média por navio):'); display(MIXD_T.round(3))
'''

L1HZ_DEST_CODE = r'''
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
R1 = pd.read_pickle(os.path.join(R.ROOT, 'data', 'iara_processed', 'cache_lofar1hz_resumo.pkl'))
F1 = np.asarray(R1['freqs']); S1 = R1['lofar_medio'].astype(float)
def decompoe(i, lo=31, hi=86, jan=15, prom=3.0):
    y = S1.loc[i].values; base = median_filter(y, size=jan, mode='nearest'); r = y - base
    k = (F1 >= lo) & (F1 <= hi); pk, _ = find_peaks(r, prominence=prom); pk = pk[k[pk]]
    return {'componente larga (dB)': base[k].mean(), 'energia das raias (dB)': np.clip(r[k], 0, None).mean(), 'raias (n)': len(pk)}
DD = pd.DataFrame({i: decompoe(i) for i in T.index[T.subgrupo.isin(SUBS)]}).T.astype(float).join(T[['subgrupo', 'Ship ID']])
NV = DD.groupby(['subgrupo', 'Ship ID']).mean().reset_index(); _a = NV[NV.subgrupo == 'acertada']
_rows = []
for s in SUBS[1:]:
    x = NV[NV.subgrupo == s]
    for c in ('componente larga (dB)', 'energia das raias (dB)', 'raias (n)'):
        U, p = mannwhitneyu(x[c], _a[c])
        _rows.append({'contraste': f'{s} × acertada', 'medida': c, 'mediana acertada': _a[c].median(), 'mediana errada': x[c].median(),
                      'AUC (errada > acertada)': U / (len(x) * len(_a)), 'p': p})
L1D = pd.DataFrame(_rows).set_index(['contraste', 'medida'])
print('lofar1hz, 31–86 Hz, por navio:'); display(L1D.round(3))
'''

VALID_CODE = r'''
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import fisher_exact
# (d) Validação em metades de navios. Em cada divisão, os navios são sorteados em duas metades (estratificadas por ter ou não
# gravação errada). Na metade de DESCOBERTA, a banda é escolhida às cegas pelo mesmo procedimento das Seções 2 e 7:
# teste faixa a faixa por navio + FDR, e a banda é o trecho contíguo de faixas com q < 0,05 e Δ < 0 em torno do menor Δ
# entre 20 e 250 Hz. Na metade de VALIDAÇÃO, que não participou da escolha, medimos o efeito nessa banda.
# 100 sementes × 2 direções (cada metade faz os dois papéis) = 200 divisões.
def faixas_rapido(ids_a, ids_e):
    A, E = por_navio(ids_a), por_navio(ids_e)
    U, p = mannwhitneyu(E.values, A.values, axis=0)
    return E.mean().values - A.mean().values, U / (len(A) * len(E)), false_discovery_control(p)

def escolhe_banda(delta, q):
    cand = np.where((F >= 20) & (F <= 250))[0]
    k0 = cand[np.argmin(delta[cand])]
    sig = (q < 0.05) & (delta < 0)
    if not sig[k0]:
        return None
    a_, b_ = k0, k0
    while a_ > 0 and sig[a_ - 1] and F[a_ - 1] >= 20:
        a_ -= 1
    while b_ < len(F) - 1 and sig[b_ + 1]:
        b_ += 1
    return a_, b_

DV = DC.copy()
PRED = ['Length', 'Wind', 'Sea state']
def modelo(num, cat):
    pre = ColumnTransformer([('n', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num),
                             ('c', OneHotEncoder(handle_unknown='ignore'), cat)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000))

_navios = DV.groupby('Ship ID').errada.max()
_rows, _perfis = [], []
for seed in range(100):
    cv = StratifiedGroupKFold(2, shuffle=True, random_state=seed)
    halves = [_navios.index[te] for _, te in cv.split(_navios.index, _navios.values, groups=_navios.index)]
    for d in (0, 1):
        desc, val = DV[DV['Ship ID'].isin(halves[d])], DV[DV['Ship ID'].isin(halves[1 - d])]
        dl_d, _, q_d = faixas_rapido(desc.index[desc.grupo == 'acertada'], desc.index[desc.grupo == 'errada'])
        dl_v, auc_v, _ = faixas_rapido(val.index[val.grupo == 'acertada'], val.index[val.grupo == 'errada'])
        k = F <= 1000
        r = {'semente': seed, 'direção': d, 'r do perfil Δ (0–1 kHz)': np.corrcoef(dl_d[k], dl_v[k])[0, 1]}
        bb = escolhe_banda(dl_d, q_d)
        if bb is None:
            _rows.append({**r, 'banda achada': False}); continue
        lo, hi = F[bb[0]], F[bb[1]]
        kb = (F >= lo) & (F <= hi)
        # efeito na validação: LOFAR médio na banda, média por navio em cada grupo (navio misto entra nos dois)
        ba =S.loc[val.index[val.grupo == 'acertada'], kb].mean(1).groupby(val['Ship ID']).mean()
        be = S.loc[val.index[val.grupo == 'errada'], kb].mean(1).groupby(val['Ship ID']).mean()
        U, p = mannwhitneyu(be, ba)
        # pico grave dentro da banda escolhida
        dentro = lambda x: ((x['f pico (Hz)'] >= lo) & (x['f pico (Hz)'] <= hi))
        desc = desc.assign(banda_d=dentro(desc).astype(float)); val = val.assign(banda_d=dentro(val).astype(float))
        pa, pe = val[val.grupo == 'acertada'].banda_d, val[val.grupo == 'errada'].banda_d
        # regressão logística treinada na descoberta, avaliada na validação
        aucs = {}
        for nome, (num, cat) in {'banda + altura': (['banda_d', 'altura (dB)'], []),
                                 'metadados': (PRED, ['Dataset', 'SHIPTYPE']),
                                 'metadados + banda + altura': (PRED + ['banda_d', 'altura (dB)'], ['Dataset', 'SHIPTYPE'])}.items():
            m = modelo(num, cat).fit(desc[num + cat], desc.errada)
            aucs[f'AUC logística: {nome}'] = roc_auc_score(val.errada, m.predict_proba(val[num + cat])[:, 1])
        _rows.append({**r, 'banda achada': True, 'banda: início (Hz)': lo, 'banda: fim (Hz)': hi,
                      'Δ na banda (dB)': be.mean() - ba.mean(), 'AUC na banda': U / (len(ba) * len(be)), 'p na banda': p,
                      '% acertada com pico na banda': 100 * pa.mean(), '% errada com pico na banda': 100 * pe.mean(),
                      'p Fisher': fisher_exact([[pe.sum(), len(pe) - pe.sum()], [pa.sum(), len(pa) - pa.sum()]])[1], **aucs})
VAL = pd.DataFrame(_rows)
N_DIV, N_ACHOU = len(VAL), int(VAL['banda achada'].sum())
V = VAL[VAL['banda achada']]
print(f'{N_DIV} divisões; banda achada na descoberta em {N_ACHOU}')
RESUMO_VAL = V.drop(columns=['semente', 'direção', 'banda achada']).describe(percentiles=[0.05, 0.5, 0.95]).T[['5%', '50%', '95%']]
RESUMO_VAL.columns = ['percentil 5', 'mediana', 'percentil 95']
display(RESUMO_VAL.round(3))
FRAC_P = {'p na banda < 0,05': 100 * (V['p na banda'] < 0.05).mean(), 'p Fisher < 0,05': 100 * (V['p Fisher'] < 0.05).mean()}
print('% das divisões com efeito significativo na validação:', {k: round(v, 1) for k, v in FRAC_P.items()})

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].hist(V['banda: início (Hz)'], bins=np.arange(15, 120, FS / NFFT), alpha=0.7, label='início')
ax[0].hist(V['banda: fim (Hz)'], bins=np.arange(15, 260, FS / NFFT), alpha=0.7, label='fim')
ax[0].axvline(31, color='k', ls=':'); ax[0].axvline(86, color='k', ls=':')
ax[0].set_xlabel('Hz'); ax[0].set_title('Limites da banda escolhida na descoberta'); ax[0].legend()
ax[1].hist(V['AUC na banda'], bins=20, color='tab:blue'); ax[1].axvline(0.5, color='gray')
ax[1].set_title('Validação: AUC (errada > acertada) na banda'); ax[1].set_xlim(0, 1)
for c, cor in (('AUC logística: metadados', 'tab:gray'), ('AUC logística: banda + altura', 'tab:green'),
               ('AUC logística: metadados + banda + altura', 'tab:purple')):
    ax[2].hist(V[c], bins=20, alpha=0.6, color=cor, label=c.replace('AUC logística: ', ''))
ax[2].axvline(0.5, color='gray'); ax[2].set_title('Validação: AUC da logística'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
'''


def run(ns, src):
    exec(src, ns)


def build(classe="LARGE"):
    filename = os.path.join(NOTEBOOKS_DIR, f"35_eda_lofar_{classe.lower()}.ipynb")
    setup = SETUP_CODE.replace("__CLASSE__", classe)
    ns = {}
    for src in (setup, ROTULOS_CODE, MEDIO_CODE, TESTE_CODE, SEMELHANCA_CODE, ESTRATO_CODE, ROBUSTO_CODE,
                PERS_CODE, OCUP_CODE, PICO_CODE, HARM_CODE, EXEMPLOS_CODE, TIPO_CODE, MESMO_NAVIO_CODE, COND_CODE,
                LOGIT_CODE, VALID_CODE, L1HZ_CODE, DECOMP_CODE, BAIXAS_CODE, BANDA_PREV_CODE):
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    T, CONT, DESTINO, POR_DS = ns["T"], ns["CONT"], ns["DESTINO"], ns["POR_DS"]
    TF, RB, NA, NE, TOP = ns["TF"], ns["RB"], ns["NA"], ns["NE"], ns["TOP"]
    SEM, EST, RBc, NAc, NEc, R_DELTA = ns["SEM"], ns["EST"], ns["RBc"], ns["NAc"], ns["NEc"], ns["R_DELTA"]
    G, F, np_ = ns["G"], ns["F"], ns["np"]
    cnt = CONT["CNN LOFAR"]

    # pico da diferença negativa e positiva abaixo de 1 kHz
    low = TF[TF["f (Hz)"] < 1000]
    fneg = low.loc[low["Δ (errada − acertada)"].idxmin()]
    _mid = TF[(TF["f (Hz)"] >= 90) & (TF["f (Hz)"] < 1000)]   # acima da corcova; a 1ª faixa (7,8 Hz) é borda do TPSW
    fpos = _mid.loc[_mid["Δ (errada − acertada)"].idxmax()]
    _r2 = TF[(TF["f (Hz)"] >= 250) & (TF["f (Hz)"] < 500)]
    fneg2 = _r2.loc[_r2["Δ (errada − acertada)"].idxmin()]
    # região contígua de Δ negativo significativo em torno do mínimo
    sig_neg = TF.sig & (TF["Δ (errada − acertada)"] < 0)
    i0 = int(fneg.name); a_, b_ = i0, i0
    while a_ > 0 and sig_neg.iloc[a_ - 1]:
        a_ -= 1
    while b_ < len(TF) - 1 and sig_neg.iloc[b_ + 1]:
        b_ += 1
    reg_lo, reg_hi = TF["f (Hz)"].iloc[a_], TF["f (Hz)"].iloc[b_]
    pico_a = G["acertada"].mean().values; pico_e = G["errada"].mean().values
    ipk = int(np_.argmax(np_.where(F < 500, pico_a, -np_.inf)))

    sem500 = SEM[SEM.faixa == "0–500 Hz"].pivot_table(index="grupo", columns="referência", values="distância RMS (dB)")
    err_lbl, ac_lbl = f"{classe} errada", f"{classe} acertada"
    ref_prox_err = sem500.loc[err_lbl].idxmin()
    des = ", ".join(f"{c} {br(v, 1)}%" for c, v in DESTINO.items())
    ds_err = POR_DS.drop(index="total")["% errada"].sort_values(ascending=False)
    est_txt = "; ".join(
        f"{ds}: {br(r['35–75 Hz: acertada'], 1)} × {br(r['35–75 Hz: errada'], 1)} dB ({pv(r['35–75 Hz: p']) if r['35–75 Hz: p'] == r['35–75 Hz: p'] else 'poucos navios para teste'})"
        for ds, r in EST.iterrows())
    nsig = int(TF.sig.sum())
    _p = EST['35–75 Hz: p'].dropna()
    est_ok = len(_p) > 0 and bool((_p < 0.05).all())
    rob_ok = R_DELTA > 0.9
    rb_hi = RB.loc["4000–8000"]
    NL, SENS, OCU_SIG, DIST, RP = ns["NL"], ns["SENS"], ns["OCU_SIG"], ns["DIST"], ns["RESUMO_PICO"]
    RHO_LEN, HARM, FRAC_6DB, EX_DS = ns["RHO_LEN"], ns["HARM"], ns["FRAC_6DB"], ns["EXEMPLO_DS"]
    sens = SENS.set_index(["limiar (dB)", "persistência mínima"])
    s_ac, s_er = "% acertada com linha em 31–86 Hz", "% errada com linha em 31–86 Hz"
    sens_txt = "; ".join(f"{int(th)} dB/{int(100 * pm)}%: {br(r[s_ac], 0)}% × {br(r[s_er], 0)}%" for (th, pm), r in sens.iterrows())
    sens_ok = bool((sens[s_ac] > sens[s_er]).all())
    oc_neg = OCU_SIG[OCU_SIG["Δ (errada − acertada, p.p.)"] < 0]
    oc_pos = OCU_SIG[(OCU_SIG["Δ (errada − acertada, p.p.)"] > 0) & (OCU_SIG["f (Hz)"] > 20)]
    oc_max = oc_neg.loc[oc_neg["Δ (errada − acertada, p.p.)"].idxmin()] if len(oc_neg) else None
    dg = DIST[f"{classe} por grupo"]; dc = DIST["todas as gravações, por classe"]
    outras = [c for c in dc.columns if c != classe]
    outras_txt = " e ".join(f"{c}, {br(dc.loc['31–86 Hz', c], 0)}% em 31–86 Hz e {br(dc.loc['86–140 Hz', c], 0)}% em 86–140 Hz" for c in outras)
    harm_ok = bool((HARM["linha no múltiplo (%)"] > HARM["taxa de base (%)"]).any())
    harm_txt = "; ".join(f"{g}, {h}×: {br(r['linha no múltiplo (%)'], 1)}% contra base de {br(r['taxa de base (%)'], 1)}%" for (g, h), r in HARM.iterrows())
    larg_ac = RP.loc["acertada", "largura (Hz)"]
    oc_txt = (f"A maior diferença de ocupação está em {br(oc_max['f (Hz)'], 1)} Hz: {br(oc_max['% navios acertada'], 0)}% dos navios acertados "
              f"contra {br(oc_max['% navios errada'], 0)}% dos errados. As faixas mais ocupadas nas acertadas vão de "
              f"{br(oc_neg['f (Hz)'].min(), 0)} a {br(oc_neg['f (Hz)'].max(), 0)} Hz.") if oc_max is not None else "Nenhuma faixa tem menos ocupação nas erradas."
    if len(oc_pos):
        oc_txt += " Nas erradas, são mais ocupadas as faixas de " + ", ".join(br(f, 0) for f in oc_pos["f (Hz)"]) + " Hz."

    # passo (c)
    TIPO, RHO_TIPO, MIX, N_MIX, PAR = ns["TIPO"], ns["RHO_TIPO"], ns["MIX"], ns["N_MIX"], ns["PAREADO"]
    COND, LOGIT, DC = ns["COND"], ns["LOGIT"], ns["DC"]
    tipo_fac = TIPO.iloc[0]; tipo_dif = TIPO.iloc[-1]
    tipo_txt = (f"vai de {br(tipo_fac['% errada'], 0)}% ({TIPO.index[0]}, {br(tipo_fac['% com banda 31–86 Hz'], 0)}% com banda) "
                f"a {br(tipo_dif['% errada'], 0)}% ({TIPO.index[-1]}, {br(tipo_dif['% com banda 31–86 Hz'], 0)}% com banda)")
    pb = PAR.loc["banda"]; pw = PAR.loc["Wind"]; ps = PAR.loc["Sea state"]; ph = PAR.loc["altura (dB)"]
    mix_ok = pb["p (teste do sinal)"] < 0.05 and pb["média acertada"] > pb["média errada"]
    def lin(r, nome, un, nd=1):
        return (f"{nome}: média {br(r['média acertada'], nd)}{un} nas acertadas × {br(r['média errada'], nd)}{un} nas erradas; "
                f"{int(r['acertada > errada'])} navios com mais nas acertadas, {int(r['acertada < errada'])} com mais nas erradas, "
                f"{int(r['empates'])} {'empate' if int(r['empates']) == 1 else 'empates'} ({pv(r['p (teste do sinal)'])})")
    cw, cs, cl = COND.loc["Wind"], COND.loc["Sea state"], COND.loc["Length"]
    la = LOGIT["AUC"]
    k_meta = "metadados (subconjunto, tipo, comprimento, vento, estado do mar)"
    ganho_banda = la["metadados + banda + altura"] - la[k_meta]
    ganho_meta = la["metadados + banda + altura"] - la["banda + altura do pico"]

    # passo (d)
    V, RV, N_DIV, N_ACHOU, FRAC_P = ns["V"], ns["RESUMO_VAL"], ns["N_DIV"], ns["N_ACHOU"], ns["FRAC_P"]
    lim = V.groupby(["banda: início (Hz)", "banda: fim (Hz)"]).size().sort_values(ascending=False)
    lim_txt = "; ".join(f"{br(a, 1)}–{br(b, 1)} Hz em {n} divisões" for (a, b), n in lim.head(3).items())
    def faixa(c, nd=2, pct=False):
        r = RV.loc[c]; u = "%" if pct else ""
        return f"mediana {br(r['mediana'], nd)}{u} (percentis 5–95: {br(r['percentil 5'], nd)}{u} a {br(r['percentil 95'], nd)}{u})"
    val_ok = N_ACHOU == N_DIV and FRAC_P["p na banda < 0,05"] >= 95 and FRAC_P["p Fisher < 0,05"] >= 95
    rv = RV["mediana"]

    BP, FLB = ns["BANDA_PREV"], ns["FRAC_LARGE_BANDA"]

    # lofar1hz
    TF1, L1_MIN, L1_PICO, DCMP, DCTRL = ns["TF1"], ns["L1_MIN"], ns["L1_PICO"], ns["DEC_CMP"], ns["DEC_CTRL"]
    DSENS, BAIX, IA_, IE_ = ns["DEC_SENS"], ns["BAIXAS"], ns["IA"], ns["IE"]
    n1sig = int((TF1["q (FDR)"] < 0.05).sum())
    cl_, en_, nr_, al_ = (DCMP.loc[c] for c in ("componente larga (dB)", "energia das raias (dB)", "raias (n)", "altura das raias (dB)"))
    larga_ok = cl_["p"] < 0.05 and bool((DSENS["AUC componente larga"] < 0.3).all())
    raias_iguais = bool(((DSENS["AUC energia das raias"] - 0.5).abs() < 0.1).all() and ((DSENS["AUC nº de raias"] - 0.5).abs() < 0.1).all())
    ctrl_a, ctrl_e = DCTRL.loc[IA_, "raias (n)"].mean(), DCTRL.loc[IE_, "raias (n)"].mean()
    bx, bx2, bx3 = BAIX.loc["5–31 Hz"], BAIX.loc["31–86 Hz"], BAIX.loc["86–300 Hz"]
    def lin_d(r, nome):
        return (f"{nome}: mediana {br(r['mediana acertada'], 2)} nas acertadas × {br(r['mediana errada'], 2)} nas erradas "
                f"(AUC = {br(r['AUC (errada > acertada)'], 2)}, {pv(r['p'])})")

    cells = [md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 35: Análise Exploratória das Frequências LOFAR — {classe} Acertadas × Erradas

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

**Tarefa 1 (reunião de 26/10/2026):** comparar o espectro LOFAR das gravações que os modelos acertam e erram e procurar frequências comuns a cada grupo. Começamos pela classe {classe}, a mais fácil entre os navios. Este notebook cobre o **passo 0** (rótulos de acerto e erro), o **passo (a)** (espectro médio, faixa a faixa) o **passo (b)** (linhas persistentes de cada gravação, pico grave e harmônicos) e o **passo (c)** (a banda grave contra tipo de navio, comprimento, condições de gravação e o mesmo navio em gravações diferentes) e o **passo (d)** (validação em metades de navios: a banda é escolhida numa metade e medida na outra), e uma última seção com o LOFAR de faixas de 1 Hz (**lofar1hz**) para separar banda larga de raias.

**Dados.** LOFAR do pipeline `iara` sem normalização (`data/iara_processed/lofar_bc15…`): log-espectrograma (20·log10 da magnitude) menos o fundo estimado pelo TPSW, com valores abaixo de −0,2 zerados. Cada valor é **quanto a faixa sobe acima do fundo de banda larga local, em dB**. Faixas de 7,8125 Hz de 7,8 Hz a 8 kHz (fs = 16 kHz após decimação 3, nfft = 2048). Para cada gravação usamos o **espectro médio** (média de todas as janelas).

**Cuidados de leitura.**
- A CNN LOFAR recebeu este mesmo LOFAR com normalização L2 por janela (`lofar_c5de…`), que muda a escala de cada janela mas não a forma do espectro.
- O TPSW estima o fundo com uma janela de ±21 faixas (±164 Hz). Nas primeiras faixas ele só tem vizinhos de um lado e usa uma correção de borda; os valores absolutos abaixo de ~150 Hz devem ser lidos com cautela. A **comparação entre grupos** continua válida, porque todos passam pelo mesmo processamento.
- O espectro médio esconde linhas que mudam de frequência de navio para navio: uma "corcova" no espectro médio pode ser a soma de linhas estreitas em frequências diferentes. O passo (b) olha as linhas de cada gravação.
"""), code(setup),
        md(f"""
## 0. Rótulos de acerto e erro

Modelo de referência: a **CNN LOFAR do artigo** (`cnn_lofar`, SP 66,49 ± 1,28). Cada gravação é teste em ~5 dos 10 folds; a taxa de acerto é a fração desses folds em que o voto majoritário acertou. Grupos: **acertada** (acerta em todos), **errada** (erra em todos) e **oscila** (o resto, fica fora do contraste). Para robustez, o grupo de **consenso** exige o mesmo resultado na CNN, no MLP LOFAR e no SVM LOFAR.
"""), code(ROTULOS_CODE),
        md(f"""
**Leitura.**
- {len(T)} gravações {classe} de {T['Ship ID'].nunique()} navios: {cnt.get('acertada', 0)} acertadas, {cnt.get('oscila', 0)} oscilam e {cnt.get('errada', 0)} erradas. No consenso dos três modelos: {CONT['consenso CNN+MLP+SVM'].get('acertada', 0)} acertadas e {CONT['consenso CNN+MLP+SVM'].get('errada', 0)} erradas.
- Destino dos erros: {des}.
- **O erro não se distribui igualmente entre os subconjuntos:** a fração de erradas vai de {br(ds_err.min(), 1)}% ({ds_err.idxmin()}) a {br(ds_err.max(), 1)}% ({ds_err.idxmax()}). Por isso a Seção 4 repete a comparação dentro de cada subconjunto.
"""),
        md(f"""
## 1. Espectro LOFAR médio por grupo

Média dos espectros das gravações de cada grupo (faixa sombreada = IC 95% da média) e, tracejadas, as médias das outras classes:
"""), code(MEDIO_CODE),
        md(f"""
## 2. Diferença faixa a faixa

Em cada uma das 1024 faixas, Mann-Whitney entre erradas e acertadas. Para um navio com várias gravações não pesar mais, cada navio entra com a média das suas gravações em cada grupo ({NA} navios com gravações acertadas e {NE} com erradas; {ns['N_NAVIOS_MISTOS']} navios têm gravações nos dois grupos). Os p-valores são corrigidos pelo FDR de Benjamini-Hochberg. **AUC** = probabilidade de uma gravação errada ter valor maior que uma acertada naquela faixa (0,5 = sem diferença).
"""), code(TESTE_CODE),
        md(f"""
**Leitura.**
- {nsig} das 1024 faixas têm q < 0,05. Com {NA} e {NE} navios, diferenças pequenas já são significativas; o que importa é **onde** está o Δ e o seu tamanho.
- **A maior diferença está em {br(reg_lo, 1)}–{br(reg_hi, 1)} Hz:** nas {classe} acertadas o LOFAR médio tem uma corcova com pico em {br(F[ipk], 1)} Hz ({br(pico_a[ipk], 1)} dB acima do fundo), que quase desaparece nas erradas ({br(pico_e[ipk], 1)} dB). O maior Δ é {br(fneg['Δ (errada − acertada)'], 1)} dB em {br(fneg['f (Hz)'], 1)} Hz (AUC = {br(fneg['AUC'], 2)}).
- **Logo acima, o sinal se inverte:** as erradas têm mais conteúdo acima do fundo em ~100–250 Hz (maior Δ positivo entre 90 Hz e 1 kHz: {sg_(fpos['Δ (errada − acertada)'])} dB em {br(fpos['f (Hz)'], 1)} Hz, AUC = {br(fpos['AUC'], 2)}).
- **Segunda região, 250–500 Hz:** maior Δ negativo de {br(fneg2['Δ (errada − acertada)'], 2)} dB em {br(fneg2['f (Hz)'], 1)} Hz (AUC = {br(fneg2['AUC'], 2)}{', q < 0,05' if fneg2['sig'] else ', não significativo'}): as acertadas também têm um pouco mais de conteúdo acima do fundo nessa faixa, num efeito bem menor que o da corcova grave.
- **Acima de 4 kHz** quase todas as faixas são significativas ({int(rb_hi['sig. (q < 0,05)'])} de {int(rb_hi['faixas'])}), mas o Δ é pequeno (médio {br(rb_hi['Δ médio (dB)'], 2)} dB): as erradas têm um pouco menos de estrutura tonal em alta frequência.
"""),
        md("""
## 3. As erradas se parecem com outra classe?
"""), code(SEMELHANCA_CODE),
        md(f"""
**Leitura.** Em 0–500 Hz, o espectro médio das {classe} erradas está a {br(sem500.loc[err_lbl, ref_prox_err], 2)} dB (RMS) do de {ref_prox_err}, contra {br(sem500.loc[ac_lbl, ref_prox_err], 2)} dB das acertadas. As {classe} erradas **parecem, em média, navios de outra classe** na faixa grave, coerente com o destino dos erros.
"""),
        md("""
## 4. Estratificação por subconjunto de gravação

Os subconjuntos A–D do IARA foram gravados em campanhas e condições diferentes, e a fração de erros muda entre eles (Seção 0). Se a diferença de frequência fosse só efeito do subconjunto, ela sumiria dentro de cada um:
"""), code(ESTRATO_CODE),
        md(f"""
**Leitura.**
- Banda 35–75 Hz, acertada × errada dentro de cada subconjunto (médias por navio): {est_txt}.
- {('**A diferença aparece em todos os subconjuntos testados**: não é efeito do subconjunto. O subconjunto muda a **composição** (onde há mais erros, há mais ' + classe + ' sem a corcova grave).') if est_ok else '**A diferença não é significativa em todos os subconjuntos testados**; parte do efeito pode vir do subconjunto de gravação.'}
"""),
        md("""
## 5. Robustez: consenso dos três modelos LOFAR
"""), code(ROBUSTO_CODE),
        md(f"""
**Leitura.** Com os grupos de consenso ({NAc} navios acertados, {NEc} errados), a correlação do perfil Δ(f) com o da CNN é r = {br(R_DELTA, 3)}. {('O padrão é do **dado**, não de um modelo: os três modelos LOFAR erram as mesmas ' + classe + ', e elas têm o mesmo perfil de frequência.') if rob_ok else 'O perfil muda com o critério de grupo; o padrão pode depender do modelo.'}
"""),
        md(f"""
## 6. (b) Linhas persistentes em cada gravação

O espectro médio de um grupo pode juntar linhas de navios diferentes. Aqui olhamos cada gravação. **Persistência** de uma faixa = fração das janelas da gravação em que ela passa de 6 dB acima do fundo TPSW (só {br(FRAC_6DB, 1)}% das células janela × faixa passam disso). Uma **linha** é um máximo local da persistência com pelo menos 50% das janelas acima do limiar.
"""), code(PERS_CODE),
        md(f"""
**Leitura.**
- As gravações têm poucas linhas persistentes ({br(NL.loc['acertada', 'total'], 1)} por gravação nas acertadas e {br(NL.loc['errada', 'total'], 1)} nas erradas), quase todas abaixo de 1 kHz.
- **Nas acertadas, a linha típica está em 31–86 Hz** ({br(NL.loc['acertada', '31–86 Hz'], 2)} por gravação, contra {br(NL.loc['errada', '31–86 Hz'], 2)} nas erradas). Nas erradas, as linhas se espalham: abaixo de 31 Hz ({br(NL.loc['errada', '0–31 Hz'], 2)}) e em 86–500 Hz ({br(NL.loc['errada', '86–200 Hz'] + NL.loc['errada', '200–500 Hz'], 2)}).
- **Critério de linha** (acertada × errada com linha em 31–86 Hz): {sens_txt}. {'A diferença se mantém em todos os critérios; o resultado não depende do limiar escolhido.' if sens_ok else 'A diferença muda de sinal em algum critério; o resultado depende do limiar.'}
"""),
        md("""
### 6.1 Faixas ocupadas, por navio

Para cada faixa, a fração de navios em que ela está ocupada (persistência ≥ 50%) em ao menos uma gravação do grupo; teste exato de Fisher por faixa com FDR:
"""), code(OCUP_CODE),
        md(f"""
**Leitura.** {oc_txt}
"""),
        md("""
## 7. O pico grave: frequência, altura e largura

Para cada gravação, o **pico grave** é a faixa de maior LOFAR médio entre 20 e 250 Hz. A largura é o trecho contíguo em torno do pico com persistência ≥ 50%. Comparamos também com as outras classes de navio:
"""), code(PICO_CODE),
        md(f"""
**Leitura.**
- **Onde fica o pico:** nas {classe} acertadas, {br(dg.loc['31–86 Hz', 'acertada'], 0)}% dos picos estão em 31–86 Hz; nas erradas, só {br(dg.loc['31–86 Hz', 'errada'], 0)}%, e o resto vai para 20–31 Hz ({br(dg.loc['20–31 Hz', 'errada'], 0)}%), 86–140 Hz ({br(dg.loc['86–140 Hz', 'errada'], 0)}%) e 140–250 Hz ({br(dg.loc['140–250 Hz', 'errada'], 0)}%).
- **As erradas têm o pico onde as outras classes têm:** nas {outras_txt}. A distribuição das {classe} erradas é parecida com essas, e não com a das {classe} acertadas. O acúmulo na primeira faixa considerada (23,4 Hz) deve ser lido com cautela: ela está perto da borda inferior do espectro, onde o TPSW só tem vizinhos de um lado; mesmo sem ela, as erradas se concentram fora de 31–86 Hz (86–140 Hz).
- **Não é uma linha estreita:** a largura mediana do pico nas acertadas é {br(larg_ac, 0)} Hz (~{br(larg_ac / 7.8125, 0)} faixas), e a altura mediana é {br(RP.loc['acertada', 'altura (dB)'], 1)} dB ({br(RP.loc['errada', 'altura (dB)'], 1)} dB nas erradas). É uma **banda de baixa frequência**, estreita o bastante em relação à janela do TPSW (±164 Hz) para sobressair do fundo estimado, e não uma raia tonal de uma faixa. Com faixas de 7,8 Hz, raias estreitas muito próximas também apareceriam como uma banda; o LOFAR do pipeline não as separa.
- **Frequência do pico × tamanho:** dentro das {classe}, o pico fica mais grave quanto maior o navio (ρ de Spearman = {br(RHO_LEN[0], 2)}, {pv(RHO_LEN[1])}, um ponto por navio). A relação é fraca, mas tem o sentido esperado: navios maiores, componentes mais graves.
"""),
        md("""
### 7.1 A banda como pista do modelo

Para todas as gravações de navio: em que fração dos folds a CNN LOFAR prevê LARGE, conforme a gravação tenha ou não o pico grave em 31–86 Hz?
"""), code(BANDA_PREV_CODE),
        md(f"""
**Leitura.** {banda_txt(BP, FLB)}
"""),
        md("""
## 8. Harmônicos

Se o pico grave fosse a fundamental de uma máquina (eixo, motor, gerador), esperaríamos linhas nos múltiplos. A **taxa de base** é a chance de encontrar uma linha numa posição qualquer da mesma faixa de frequência, com a mesma tolerância:
"""), code(HARM_CODE),
        md(f"""
**Leitura.** {harm_txt}. {('Há mais linhas nos múltiplos do que o acaso em algum grupo; vale investigar com mais resolução.' if harm_ok else 'Em nenhum grupo há mais linhas nos múltiplos do que o esperado ao acaso: **não há estrutura harmônica** detectável a partir do pico grave, coerente com ele ser uma banda e não uma fundamental.')}
"""),
        md(f"""
## 9. Exemplos

Espectrogramas LOFAR de 0 a 500 Hz de gravações sorteadas do subconjunto {EX_DS}:
"""), code(EXEMPLOS_CODE),
        md(f"""
**Leitura.** Nas acertadas, a banda de ~30–80 Hz aparece como uma faixa vertical clara e estável ao longo da gravação. Nas erradas, a energia grave fica abaixo de ~30 Hz ou acima de ~90 Hz, ou a banda é fraca. As listras diagonais e curvas que aparecem nos dois grupos parecem padrões de interferência que mudam com a distância do navio durante a passagem (não verificado aqui); a olho, elas não separam os grupos.
"""),
        md(f"""
## 10. (c) A banda grave depende do navio ou da gravação?

Nesta seção, uma gravação **tem a banda** quando o seu pico grave (Seção 7) cai em 31–86 Hz. Cruzamos a banda e o grupo (acertada × errada) com o tipo de navio, o comprimento e as condições registradas no `iara.csv` (vento em m/s, estado do mar, chuva). O IARA não registra velocidade nem regime de máquinas, as variáveis físicas mais prováveis para a banda.

### 10.1 Tipo de navio

Tipos com pelo menos 10 gravações acertadas ou erradas, ordenados pela fração de erradas:
"""), code(TIPO_CODE),
        md(f"""
**Leitura.** A fração de erradas {tipo_txt}. Entre os {len(TIPO)} tipos, quanto menos gravações com banda, mais erros (ρ = {br(RHO_TIPO[0], 2)}, {pv(RHO_TIPO[1])}). O tipo de navio muda a chance de ter a banda, mas, como cada tipo tem poucos navios, essa tabela sozinha não separa o efeito do tipo do efeito do navio.
"""),
        md(f"""
### 10.2 O mesmo navio, gravações acertadas e erradas

O teste mais direto: {N_MIX} navios têm gravações nos dois grupos. Se a banda fosse uma assinatura fixa do navio, ela apareceria (ou não) nas duas. Para cada navio, comparamos a média nas gravações acertadas com a média nas erradas (teste do sinal):
"""), code(MESMO_NAVIO_CODE),
        md(f"""
**Leitura.**
- {lin(pb, 'Banda (fração das gravações)', '', 2)}.
- {lin(ph, 'Altura do pico grave', ' dB')}.
- {lin(pw, 'Vento', ' m/s')}.
- {lin(ps, 'Estado do mar', '')}.
- {('**No mesmo navio, a banda está nas gravações acertadas e falta nas erradas.** Ela não é uma assinatura fixa do navio: depende da gravação (regime de operação, distância, propagação, ruído de fundo). Quando falta, o navio é confundido com outra classe.' if mix_ok else 'No mesmo navio, a banda não difere de forma significativa entre as gravações acertadas e erradas; com poucos navios, o teste tem pouco poder.')} São só {N_MIX} navios, então esse resultado precisa ser confirmado em mais dados.
"""),
        md("""
### 10.3 Condições de gravação e comprimento
"""), code(COND_CODE),
        md(f"""
**Leitura.**
- **Vento:** mediana de {br(cw['mediana acertada'], 1)} m/s nas acertadas e {br(cw['mediana errada'], 1)} m/s nas erradas ({pv(cw['p (acertada × errada)'])}); {br(cw['mediana com banda'], 1)} m/s com banda e {br(cw['mediana sem banda'], 1)} m/s sem banda ({pv(cw['p (com × sem banda)'])}). A altura do pico cai um pouco com o vento (ρ = {br(cw['rho com a altura do pico'], 2)}).
- **Estado do mar:** mediana {br(cs['mediana acertada'], 0)} nas acertadas e {br(cs['mediana errada'], 0)} nas erradas ({pv(cs['p (acertada × errada)'])}); com × sem banda: {pv(cs['p (com × sem banda)'])}.
- **Comprimento:** mediana de {br(cl['mediana acertada'], 0)} m nas acertadas e {br(cl['mediana errada'], 0)} m nas erradas ({pv(cl['p (acertada × errada)'])}), mas o comprimento quase não muda entre gravações com e sem banda ({br(cl['mediana com banda'], 0)} × {br(cl['mediana sem banda'], 0)} m, {pv(cl['p (com × sem banda)'])}). {('As ' + classe + ' menores erram mais, e por outro caminho que não a presença da banda (a frequência do pico, Seção 7, é um candidato).') if cl['p (acertada × errada)'] < 0.05 and cl['p (com × sem banda)'] >= 0.05 else ''}
- O vento e o mar mais agitados aumentam o ruído de fundo em baixa frequência, que pode **mascarar** a banda. {('Mas, no mesmo navio (Seção 10.2), o vento e o estado do mar não diferem entre as gravações acertadas e erradas (' + pv(pw['p (teste do sinal)']) + ' e ' + pv(ps['p (teste do sinal)']) + '), enquanto a banda difere: o vento não explica sozinho a falta da banda. Parte da associação geral vem de o vento variar entre subconjuntos e campanhas.') if pw['p (teste do sinal)'] >= 0.05 and mix_ok else 'O efeito do vento sobre a banda precisa ser separado do efeito do navio; ver a Seção 10.2.'}
"""),
        md("""
### 10.4 Quanto do erro cada fator explica

Regressão logística para prever se a gravação é errada, com validação cruzada agrupada por navio (o mesmo navio nunca está no treino e no teste) e 10 sementes. **AUC** no teste: 0,5 = acaso, 1 = separação perfeita.
"""), code(LOGIT_CODE),
        md(f"""
**Leitura.**
- Só o subconjunto já dá AUC = {br(la['subconjunto'], 2)}: a fração de erros muda muito entre A, B, C e D (Seção 0). Os metadados juntos chegam a {br(la[k_meta], 2)}.
- Só a banda grave (presença e altura do pico) dá AUC = {br(la['banda + altura do pico'], 2)}, sem saber nada do navio nem do subconjunto.
- Juntos: {br(la['metadados + banda + altura'], 2)}. A banda acrescenta {sg_(ganho_banda)} aos metadados, e os metadados acrescentam {sg_(ganho_meta)} à banda: **as duas fontes são complementares**. A banda explica boa parte dos erros, mas o subconjunto, o tipo e o tamanho do navio também pesam por outros caminhos (por exemplo, outras faixas do espectro).
- **Cuidado:** os limites da banda (31–86 Hz) foram escolhidos olhando estes mesmos dados (Seção 2), então a AUC da banda é otimista. A Seção 11 escolhe a faixa numa metade dos navios e mede na outra.
"""),
        md(f"""
## 11. (d) Validação em metades de navios

Até aqui, a banda foi escolhida e medida nos mesmos dados, o que favorece o resultado. Agora os navios são sorteados em duas metades (o navio inteiro vai para uma só, e as metades têm proporções parecidas de navios com erradas):
- na metade de **descoberta**, a banda é escolhida **às cegas**, pelo mesmo procedimento das Seções 2 e 7: teste faixa a faixa por navio com FDR, e a banda é o trecho contíguo de faixas com q < 0,05 e Δ < 0 em torno do menor Δ entre 20 e 250 Hz;
- na metade de **validação**, que não participou da escolha, medimos o efeito nessa banda: Δ e AUC do LOFAR médio na banda (por navio), fração das gravações com o pico grave na banda e a AUC das regressões logísticas treinadas na descoberta.

São 100 sorteios × 2 direções (cada metade faz os dois papéis) = {N_DIV} divisões.
"""), code(VALID_CODE),
        md(f"""
**Leitura.**
- **A banda é encontrada em todas as divisões** ({N_ACHOU} de {N_DIV}) e quase sempre no mesmo lugar: {lim_txt}. Os limites das Seções 2 e 7 (31–86 Hz) são os mais frequentes.
- **O perfil de diferença se repete entre metades:** a correlação do Δ(f) de 0 a 1 kHz entre descoberta e validação tem {faixa('r do perfil Δ (0–1 kHz)')}.
- **Na validação, o efeito se mantém:** Δ na banda com {faixa('Δ na banda (dB)')} dB e AUC com {faixa('AUC na banda')}; p < 0,05 em {br(FRAC_P['p na banda < 0,05'], 0)}% das divisões. O Δ é um pouco menor que o da Seção 2 ({br(fneg['Δ (errada − acertada)'], 1)} dB na faixa de maior diferença), como esperado: lá era o máximo de uma faixa, aqui é a média da banda inteira.
- **Pico grave na banda:** {br(rv['% acertada com pico na banda'], 0)}% das acertadas contra {br(rv['% errada com pico na banda'], 0)}% das erradas (medianas; Fisher com p < 0,05 em {br(FRAC_P['p Fisher < 0,05'], 0)}% das divisões).
- **Logística treinada numa metade e testada na outra:** banda + altura com {faixa('AUC logística: banda + altura')}; metadados com {faixa('AUC logística: metadados')}; tudo junto com {faixa('AUC logística: metadados + banda + altura')}. Os valores batem com os da Seção 10.4, então o otimismo de escolher a banda nos mesmos dados era pequeno.
- {('**O achado se replica em navios novos.** A banda grave não é um artefato de ter procurado em 1024 faixas: ela é escolhida às cegas numa metade e confirmada na outra.' if val_ok else '**A replicação é parcial:** a banda nem sempre é encontrada ou nem sempre é significativa na validação.')}
"""),
        md(f"""
## 12. (e) Banda larga ou raias? LOFAR com faixas de 1 Hz (lofar1hz)

Com faixas de 7,8 Hz, uma banda larga e um conjunto de raias próximas ficam iguais. A extração **lofar1hz** (`scripts/extrair_lofar1hz.py`, `data/iara_processed/lofar1hz/`) refaz o LOFAR só para baixa frequência:
- decimação de 48 kHz para 2 kHz;
- FFT de 2048 pontos, ou seja, janelas de 1,024 s e **faixas de 0,98 Hz** (8 vezes mais finas);
- uma linha a cada 0,512 s (50% de sobreposição), o mesmo passo do LOFAR do pipeline;
- TPSW com **a mesma largura em Hz** do pipeline (±164 Hz), para a banda não virar "fundo".
Usamos 0–300 Hz; acima de ~900 Hz há aliasing.

### 12.1 O espectro médio com 1 Hz
"""), code(L1HZ_CODE),
        md(f"""
**Leitura.** O lofar1hz reproduz o LOFAR do pipeline (os pontos ficam sobre as curvas), agora com detalhe de 1 Hz. O pico médio das acertadas fica em {br(L1_PICO, 1)} Hz e a maior diferença é {br(L1_MIN['Δ (errada − acertada)'], 1)} dB em {br(L1_MIN['f (Hz)'], 1)} Hz (AUC = {br(L1_MIN['AUC'], 2)}); {n1sig} das {len(TF1)} faixas entre 0 e 300 Hz têm q < 0,05. A diferença continua sendo uma **corcova larga**, sem picos estreitos que se destaquem só num dos grupos.
"""),
        md("""
### 12.2 Separando a componente larga das raias

Em cada gravação, o espectro médio de 31–86 Hz é separado em uma **componente larga** (mediana móvel de ~15 Hz, que ignora estruturas mais estreitas que ~7 Hz) e **raias** (máximos locais que sobem pelo menos 3 dB acima dela). A comparação entre grupos é feita por navio:
"""), code(DECOMP_CODE),
        md(f"""
**Leitura.**
- {lin_d(cl_, 'Componente larga')}.
- {lin_d(en_, 'Energia das raias')}.
- {lin_d(nr_, 'Número de raias')}.
- {lin_d(al_, 'Altura média das raias')}.
- A altura das raias sobre a componente larga é um pouco maior nas erradas ({br(al_['mediana errada'], 1)} × {br(al_['mediana acertada'], 1)} dB): sem a banda forte por baixo, as mesmas raias se destacam um pouco mais. O efeito é pequeno perto do da componente larga.
- Na faixa de controle 150–205 Hz há {br(ctrl_a, 2)} raias por gravação nas acertadas e {br(ctrl_e, 2)} nas erradas: as raias de 31–86 Hz são de fato uma estrutura da região grave.
- **Sensibilidade:** com mediana móvel de 11, 15 ou 21 faixas e proeminência de 2, 3 ou 6 dB, a AUC da componente larga fica entre {br(DSENS['AUC componente larga'].min(), 2)} e {br(DSENS['AUC componente larga'].max(), 2)}; a da energia das raias, entre {br(DSENS['AUC energia das raias'].min(), 2)} e {br(DSENS['AUC energia das raias'].max(), 2)}; a do número de raias, entre {br(DSENS['AUC nº de raias'].min(), 2)} e {br(DSENS['AUC nº de raias'].max(), 2)}.
- {('**O que separa acertadas de erradas é a componente larga, não as raias.** As duas têm raias estreitas em 31–86 Hz em número e energia parecidos (no exemplo acertado acima, elas aparecem sobre a banda); nas acertadas, essas raias ficam sobre uma componente larga forte, que nas erradas é fraca. A diferença que os modelos aproveitam é, portanto, **principalmente de banda larga** de baixa frequência, e não um conjunto de raias que o LOFAR do pipeline não resolvia (fontes candidatas: hélice e escoamento em torno do casco; não verificado).' if larga_ok and raias_iguais else ('A componente larga separa os grupos, mas as raias também diferem em algum critério; a banda pode ter as duas naturezas.' if larga_ok else 'A componente larga não separa os grupos de forma robusta.'))}
- **Limite:** raias espaçadas de menos de ~2 Hz (por exemplo, harmônicos de um eixo a 60–120 rpm) ainda se fundiriam com faixas de 1 Hz e janela de Hanning. Um "pente" muito fino continuaria parecendo banda larga.
"""),
        md("""
### 12.3 Raias estreitas abaixo de 31 Hz

Nos espectrogramas com 1 Hz aparecem raias finas e estáveis abaixo de 30 Hz, que o LOFAR do pipeline não resolve. Elas diferem entre acertadas e erradas?
"""), code(BAIXAS_CODE),
        md(f"""
**Leitura.**
- Têm ao menos uma raia estreita persistente entre 5 e 31 Hz {br(bx['% gravações acertadas com raia'], 0)}% das acertadas e {br(bx['% gravações erradas com raia'], 0)}% das erradas (Fisher por navio, {pv(bx['p (Fisher, por navio)'])}). {('**As raias graves não distinguem os grupos**: aparecem nos dois, em frequências espalhadas.' if bx['p (Fisher, por navio)'] >= 0.05 else '**As raias graves diferem entre os grupos.**')}
- O mesmo critério em 31–86 Hz dá {br(bx2['% gravações acertadas com raia'], 0)}% × {br(bx2['% gravações erradas com raia'], 0)}% ({pv(bx2['p (Fisher, por navio)'])}) e em 86–300 Hz, {br(bx3['% gravações acertadas com raia'], 0)}% × {br(bx3['% gravações erradas com raia'], 0)}% ({pv(bx3['p (Fisher, por navio)'])}). As raias estreitas persistentes não favorecem as acertadas em nenhuma região, coerente com a Seção 12.2.
- Abaixo de 31 Hz o fundo TPSW é estimado só do lado de cima e inclui a banda grave, então esses valores são os menos confiáveis do espectro.
"""),
        md(f"""
## 13. Conclusões parciais e próximos passos

1. **Frequência comum às {classe} acertadas:** uma banda de baixa frequência em ~{br(reg_lo, 0)}–{br(reg_hi, 0)} Hz (pico médio em {br(F[ipk], 0)} Hz); {br(dg.loc['31–86 Hz', 'acertada'], 0)}% delas têm o pico grave em 31–86 Hz. As {classe} erradas têm o pico grave em outras frequências, com distribuição parecida com a de {' e '.join(outras)}, e no espectro médio se parecem com {ref_prox_err}.
2. **Subconjunto de gravação:** {'a diferença aparece dentro de cada subconjunto testado' if est_ok else 'a diferença não aparece em todos os subconjuntos'}. **Modelo:** {'o perfil se repete no consenso CNN + MLP + SVM' if rob_ok else 'o perfil muda no consenso dos três modelos'}.
3. **É uma banda, não uma raia:** largura mediana de {br(larg_ac, 0)} Hz e {'nenhuma estrutura harmônica acima do acaso' if not harm_ok else 'algum sinal de harmônicos'}. Com 7,8 Hz não dava para separar banda larga de raias próximas; o item 8 responde com faixas de 1 Hz.
4. **A banda é uma pista importante para o modelo dizer LARGE, mas fraca de tamanho real:** com a banda, a chance de ser chamada de LARGE sobe em todas as classes (SMALL de {br(BP.loc['SMALL', BP.columns[0]], 0)}% para {br(BP.loc['SMALL', BP.columns[1]], 0)}%, LARGE de {br(BP.loc['LARGE', BP.columns[0]], 0)}% para {br(BP.loc['LARGE', BP.columns[1]], 0)}%), mas ela aparece também em {br(BP.loc['SMALL', BP.columns[2]], 0)}% das SMALL e {br(BP.loc['MEDIUM', BP.columns[2]], 0)}% das MEDIUM (Seção 7.1). Dentro das {classe}, o pico fica um pouco mais grave nos navios maiores (ρ = {br(RHO_LEN[0], 2)}), mas a relação é fraca.
5. **A banda depende da gravação, não só do navio:** {('nos ' + str(N_MIX) + ' navios com gravações nos dois grupos, ela aparece em ' + br(100 * pb['média acertada'], 0) + '% das gravações acertadas e em ' + br(100 * pb['média errada'], 0) + '% das erradas (' + pv(pb['p (teste do sinal)']) + ')') if mix_ok else 'no mesmo navio, a diferença não foi significativa (poucos navios)'}. Vento e mar agitado reduzem um pouco a banda, mas não explicam sozinhos a sua falta. A explicação mais provável fica no que o IARA não registra: regime de operação (velocidade, carga), distância e propagação. Isso ainda é uma **hipótese**.
6. **Fatores complementares:** a banda sozinha prevê o erro com AUC = {br(la['banda + altura do pico'], 2)}, os metadados com {br(la[k_meta], 2)} e os dois juntos com {br(la['metadados + banda + altura'], 2)}.
7. **Validação:** {('escolhida às cegas numa metade dos navios, a banda é a mesma (' + lim_txt.split(';')[0] + ') e o efeito se confirma na outra metade em ' + br(FRAC_P['p na banda < 0,05'], 0) + '% das ' + str(N_DIV) + ' divisões (AUC mediana ' + br(rv['AUC na banda'], 2) + '; logística banda + altura ' + br(rv['AUC logística: banda + altura'], 2) + ' em navios não vistos).') if val_ok else 'a replicação em metades de navios é parcial (ver Seção 11).'}
8. **Banda larga, não raias:** {('com faixas de 1 Hz (lofar1hz), a diferença entre acertadas e erradas está na componente larga de 31–86 Hz (AUC = ' + br(cl_['AUC (errada > acertada)'], 2) + '); as raias estreitas dessa região aparecem nos dois grupos em número e energia parecidos.' if larga_ok and raias_iguais else 'ver a Seção 12: o resultado com faixas de 1 Hz é misto.')} Raias espaçadas de menos de ~2 Hz ainda não seriam resolvidas.

**Próximos passos:** repetir para MEDIUM e SMALL (o lofar1hz já está extraído para todas as gravações).
"""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "pygments_lexer": "ipython3"}},
          "nbformat": 4, "nbformat_minor": 4}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook gerado em '{filename}'")


def banda_txt(BP, frac_l):
    """Leitura da tabela BANDA_PREV: a banda é pista importante para o modelo, mas fraca de tamanho real."""
    c = BP.columns
    sem, com, pres = c[0], c[1], c[2]
    sobe = "; ".join(f"{k} de {br(BP.loc[k, sem], 0)}% para {br(BP.loc[k, com], 0)}%" for k in BP.index)
    return (f"**A banda é uma pista importante para o modelo dizer LARGE, mas uma pista fraca de tamanho real.** "
            f"Em qualquer classe, ter a banda aumenta a chance de a gravação ser chamada de LARGE ({sobe}). "
            f"Não é uma regra: mesmo com a banda, {br(100 - BP.loc['SMALL', com], 0)}% das SMALL e "
            f"{br(100 - BP.loc['MEDIUM', com], 0)}% das MEDIUM não são chamadas de LARGE, e as LARGE sem a banda ainda são "
            f"chamadas de LARGE em {br(BP.loc['LARGE', sem], 0)}% dos casos; o modelo usa outras pistas também. "
            f"O problema é que a banda separa mal as classes reais: aparece em {br(BP.loc['LARGE', pres], 0)}% das LARGE, mas "
            f"também em {br(BP.loc['SMALL', pres], 0)}% das SMALL e {br(BP.loc['MEDIUM', pres], 0)}% das MEDIUM, e das gravações "
            f"com a banda só {br(frac_l, 0)}% são LARGE. Usá-la não é irracional (ela aumenta a probabilidade de LARGE), mas é "
            f"fraca, e parte dos erros vem daí: as SMALL e MEDIUM chamadas de LARGE se concentram nas que têm a banda, e as "
            f"LARGE chamadas de outra classe, nas que não a têm.")


def sg_(x, nd=2):
    return ("+" if x >= 0.005 else "") + br(x, nd)


def build_destino(classe):
    """Versão para classes cujos erros se dividem entre destinos (MEDIUM, SMALL)."""
    filename = os.path.join(NOTEBOOKS_DIR, f"35_eda_lofar_{classe.lower()}.ipynb")
    setup = SETUP_CODE.replace("__CLASSE__", classe)
    func = TESTE_CODE.split("TF, NA, NE")[0].strip("\n")      # só por_navio e teste_faixas
    blocos = (setup, ROTULOS_CODE, func, DEST_CODE, MEDIO_DEST_CODE, TESTE_DEST_CODE, APRIORI_CODE, BANDA_PREV_CODE, MISTO_DEST_CODE,
              SEM_DEST_CODE, ACERT_CODE, META_DEST_CODE, L1HZ_DEST_CODE)
    ns = {}
    for src in blocos:
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    T, CONT, DESTINO, POR_DS = ns["T"], ns["CONT"], ns["DESTINO"], ns["POR_DS"]
    SUBS, DESTS, RS, CONTR, TFD = ns["SUBS"], ns["DESTS"], ns["RESUMO_SUB"], ns["CONTR"], ns["TFD"]
    AP, MIXD, SEMD, METAD, L1D = ns["APRIORI"], ns["MIXD"], ns["SEMD"], ns["METAD"], ns["L1D"]
    BP, FLB = ns["BANDA_PREV"], ns["FRAC_LARGE_BANDA"]
    cnt = CONT["CNN LOFAR"]
    des = ", ".join(f"{c} {br(v, 1)}%" for c, v in DESTINO.items())
    ds_err = POR_DS.drop(index="total")["% errada"]
    errs = [s for s in SUBS if s != "acertada"]
    sub_txt = "; ".join(f"**{s}**: {int(RS.loc[s, 'gravações'])} gravações de {int(RS.loc[s, 'navios'])} navios" for s in errs)
    outros_dest = [d for d in RS.index if d.startswith("errada→") and d not in errs]
    pool = CONTR.loc["acertada × errada (todas)"]

    BANDAS_T = [b[2:] for b in CONTR.columns if b.startswith("Δ ")]

    def marcantes(r):
        # bandas com AUC média longe de 0,5 (<= 0,35 ou >= 0,65)
        return [(b, r[f"Δ {b}"], r[f"AUC {b}"]) for b in BANDAS_T if abs(r[f"AUC {b}"] - 0.5) >= 0.15]

    def contr_txt(s):
        r = CONTR.loc[f"acertada × {s}"]
        out = (f"**{s}** ({int(r['navios errada'])} navios): {int(r['faixas com q < 0,05'])} faixas com q < 0,05 "
               f"({int(r['abaixo de 1 kHz'])} abaixo e {int(r['acima de 1 kHz'])} acima de 1 kHz)")
        m = marcantes(r)
        if m:
            out += "; bandas que mais diferem: " + ", ".join(
                f"{b} {sg_(d, 2 if abs(d) < 0.5 else 1)} dB (AUC {br(a, 2)})" for b, d, a in m)
        return out

    def ap_txt(s):
        r = AP.loc[f"{classe} {s}"]
        return (f"**{s}**: banda 31–86 Hz com mediana {br(r['banda 31–86 Hz (dB): mediana'], 1)} dB "
                f"(AUC {br(r['banda 31–86 Hz (dB): AUC vs acertada'], 2)}, {pv(r['banda 31–86 Hz (dB): p'])}), "
                f"100–150 Hz com {br(r['100–150 Hz (dB): mediana'], 1)} dB (AUC {br(r['100–150 Hz (dB): AUC vs acertada'], 2)}, "
                f"{pv(r['100–150 Hz (dB): p'])}); {br(r['% gravações com pico em 31–86 Hz'], 0)}% das gravações com o pico grave em 31–86 Hz")
    ra = AP.loc[f"{classe} acertada"]; rL = AP.loc["LARGE (todas)"] if "LARGE (todas)" in AP.index else None
    ref_txt = ", ".join(f"{c.replace(' (todas)', '')} {br(AP.loc[c, '% gravações com pico em 31–86 Hz'], 0)}%"
                        for c in AP.index if c.endswith("(todas)") and not c.startswith("BACKGROUND"))
    sig_ap = {s: AP.loc[f"{classe} {s}", "banda 31–86 Hz (dB): p"] < 0.05 for s in errs}
    para_large = [s for s in errs if s == "errada→LARGE"]
    large_ok = bool(para_large) and sig_ap["errada→LARGE"] and AP.loc[f"{classe} errada→LARGE", "banda 31–86 Hz (dB): AUC vs acertada"] > 0.5

    mix_txt = []
    for s in errs:
        m = MIXD.get(s)
        if m is None:
            mix_txt.append(f"**{s}**: nenhum navio tem gravações acertadas e {s}.")
        else:
            mix_txt.append(f"**{s}**: {int(m['navios'])} navios; banda de {br(m['média acertada (dB)'], 1)} dB nas gravações acertadas e "
                           f"{br(m['média errada (dB)'], 1)} dB nas erradas; {int(m['errada > acertada'])} navios com mais banda na errada, "
                           f"{int(m['errada < acertada'])} com menos ({pv(m['p (teste do sinal)'])})")
    def leitura_dest(s):
        r = CONTR.loc[f"acertada × {s}"]; ap = AP.loc[f"{classe} {s}"]
        partes = []
        if s == "errada→LARGE" and r["AUC 31–86 Hz"] >= 0.65:
            partes.append("têm a **assinatura das LARGE**: mais conteúdo em 31–86 Hz e menos em 86–150 Hz")
        elif r["faixas com q < 0,05"] == 0:
            partes.append("nenhuma faixa difere das acertadas no teste faixa a faixa")
        elif r["acima de 1 kHz"] > 0.8 * r["faixas com q < 0,05"]:
            hf = [f"AUC {b}" for b in ("1000–2000 Hz", "2000–4000 Hz", "4000–8000 Hz")]
            dhf = r["Δ 1000–2000 Hz"] + r["Δ 2000–4000 Hz"] + r["Δ 4000–8000 Hz"]
            partes.append(f"as faixas significativas estão quase todas acima de 1 kHz, com Δ de centésimos a décimos de dB "
                          f"(AUC média em 1–8 kHz: {br(r[hf].mean(), 2)}): {'um pouco menos' if dhf < 0 else 'um pouco mais'} "
                          f"de estrutura tonal em alta frequência")
        else:
            partes.append(f"{int(r['faixas com q < 0,05'])} faixas diferem das acertadas")
        if s != "errada→LARGE" and ap["banda 31–86 Hz (dB): p"] < 0.05:
            partes.append(f"a banda 31–86 Hz é {'mais forte' if ap['banda 31–86 Hz (dB): AUC vs acertada'] > 0.5 else 'mais fraca'} "
                          f"(mediana {br(ap['banda 31–86 Hz (dB): mediana'], 1)} × {br(ra['banda 31–86 Hz (dB): mediana'], 1)} dB, "
                          f"AUC {br(ap['banda 31–86 Hz (dB): AUC vs acertada'], 2)}, {pv(ap['banda 31–86 Hz (dB): p'])})")
        viz_s = SEMD.loc[f"{classe} {s}"].idxmin()
        partes.append(f"o espectro médio fica mais perto de {viz_s} ({br(SEMD.loc[f'{classe} {s}', viz_s], 2)} dB RMS)")
        if s in vento:
            partes.append("há um indício de vento mais forte")
        return "; ".join(partes)

    _ml = MIXD.get("errada→LARGE")
    if _ml and _ml["p (teste do sinal)"] < 0.05:
        mix_leitura = (f"Como nas LARGE, **a banda acompanha a gravação, não o navio**: no mesmo navio {classe}, as gravações "
                       "chamadas de LARGE são as que têm mais banda.")
    elif _ml and _ml["errada > acertada"] > _ml["errada < acertada"]:
        _n = int(_ml["errada > acertada"] + _ml["errada < acertada"])
        mix_leitura = (f"O sentido é o mesmo das LARGE ({int(_ml['errada > acertada'])} de {_n} navios com mais banda na gravação "
                       f"chamada de LARGE), mas com poucos navios mistos o teste não chega a p < 0,05 ({pv(_ml['p (teste do sinal)'])}).")
    else:
        mix_leitura = "Com poucos navios mistos, este teste tem pouco poder."
    pct = {c: AP.loc[f"{c} (todas)", "% gravações com pico em 31–86 Hz"] for c in ("SMALL", "MEDIUM", "LARGE") if f"{c} (todas)" in AP.index}
    pct[classe] = 100 * ns["X"]["pico grave em 31–86 Hz"].mean()
    pct_txt = ", ".join(f"{c} {br(pct[c], 0)}%" for c in ("SMALL", "MEDIUM", "LARGE"))
    sem_txt = "; ".join(f"{classe} {s}: mais perto de {SEMD.loc[f'{classe} {s}'].idxmin()} ({br(SEMD.loc[f'{classe} {s}'].min(), 2)} dB)"
                        for s in SUBS)
    meta_txt = []
    for s in errs:
        r = METAD.loc[s]
        meta_txt.append(f"**{s}**: comprimento {br(r['Length: mediana'], 0)} m ({pv(r['Length: p'])}), vento {br(r['Wind: mediana'], 1)} m/s "
                        f"({pv(r['Wind: p'])}), estado do mar {br(r['Sea state: mediana'], 0)} ({pv(r['Sea state: p'])})")
    rm = METAD.loc["acertada"]
    l1_txt = []
    for s in errs:
        c_ = L1D.loc[(f"{s} × acertada", "componente larga (dB)")]; e_ = L1D.loc[(f"{s} × acertada", "energia das raias (dB)")]
        n_ = L1D.loc[(f"{s} × acertada", "raias (n)")]
        l1_txt.append(f"**{s}**: componente larga {br(c_['mediana errada'], 1)} × {br(c_['mediana acertada'], 1)} dB "
                      f"(AUC {br(c_['AUC (errada > acertada)'], 2)}, {pv(c_['p'])}); energia das raias AUC {br(e_['AUC (errada > acertada)'], 2)} "
                      f"({pv(e_['p'])}); nº de raias AUC {br(n_['AUC (errada > acertada)'], 2)} ({pv(n_['p'])})")
    sem_sinal = [s for s in errs if CONTR.loc[f"acertada × {s}", "faixas com q < 0,05"] == 0]
    ACV, PAC, TAM, TOPA, FT = ns["ACV"], ns["PICO_AC"], ns["TAM"], ns["TOP_AC"], ns["FOLDS_TOP"]
    n_ac, n_acg, n_navg = ns["N_AC"], ns["N_AC_GRANDES"], ns["N_NAV_GRANDES"]
    iguais = [i.split(' × ')[1] for i, r in ACV.iterrows() if r['faixas com q < 0,05'] == 0]
    difs = [i.split(' × ')[1] for i, r in ACV.iterrows() if r['faixas com q < 0,05'] > 0]
    acv_txt = "; ".join(f"contra {i.split(' × ')[1]} ({r['navios']} navios): {int(r['faixas com q < 0,05'])} faixas com q < 0,05, "
                        f"Δ de {sg_(r['Δ 31–86 Hz (dB)'], 1)} dB em 31–86 Hz e {sg_(r['Δ 86–150 Hz (dB)'], 1)} dB em 86–150 Hz"
                        for i, r in ACV.iterrows())
    tc, tl = TAM.loc[classe], TAM.loc["LARGE"]
    top1 = TOPA.iloc[0]; top_id = TOPA.index[0]
    def tg(t, linha, col="acertada"):
        return t.loc[linha, col] if linha in t.index else float("nan")
    memoriza = bool(tg(tc, "1") < tg(tl, "1") - 15 and tg(tc, "mais de 10") > tg(tc, "1") + 15)
    tam_txt = "; ".join(f"navios com {lin} {'gravação' if lin == '1' else 'gravações'}: {br(tg(tc, lin), 0)}% acertadas e {br(tg(tc, lin, 'errada'), 0)}% erradas "
                        f"({int(tg(tc, lin, 'gravações'))} gravações)" for lin in tc.index)
    viz = SEMD.loc[f"{classe} acertada"].drop(classe).idxmin()
    vento = [s for s in sem_sinal if METAD.loc[s, "Wind: p"] < 0.1]
    vento_txt = ((" Nas erradas sem assinatura espectral (" + ", ".join(vento) + "), o vento é um pouco mais forte: candidato a "
                  "explicação (ruído de fundo mais alto), mas com poucos navios e p perto de 0,05 é só indício.") if vento else "")

    cells = [md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 35: Análise Exploratória das Frequências LOFAR — {classe} Acertadas × Erradas

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

**Tarefa 1 (reunião de 26/10/2026)**, segunda classe. Em `35_eda_lofar_large.ipynb`, as LARGE acertadas têm uma **banda larga em ~31–86 Hz**, e as LARGE sem essa banda são confundidas com SMALL e MEDIUM. Aqui repetimos a análise para {classe}.

**Diferença em relação ao notebook das LARGE.** Os erros das {classe} se dividem entre classes diferentes ({des}). Se juntarmos todas as erradas, assinaturas opostas podem se cancelar. Por isso as erradas são separadas pelo **destino**: a classe em que a gravação é prevista na maioria dos folds.

**Hipótese fixada antes de olhar esta classe:** as {classe} previstas como LARGE têm a banda de 31–86 Hz das LARGE, e menos conteúdo em 100–150 Hz. Os navios das {classe} são outros, não entraram na análise das LARGE (a classe é definida pelo comprimento), então este é um teste **fora da amostra**.

**Dados e cuidados:** os mesmos do notebook das LARGE: LOFAR do pipeline sem normalização (`lofar_bc15…`), espectro médio por gravação, testes por navio com FDR; e a extração **lofar1hz** (faixas de 0,98 Hz) para separar banda larga de raias. Valores abaixo de ~150 Hz devem ser lidos com cautela (borda do TPSW); a comparação entre grupos continua válida.
"""), code(setup),
        md("""
## 0. Rótulos de acerto e erro, e destino dos erros

Mesmo critério das LARGE: **acertada** = acerta em todos os folds de teste da CNN LOFAR do artigo; **errada** = erra em todos; **oscila** = o resto (fora do contraste).
"""), code(ROTULOS_CODE), code(func), code(DEST_CODE),
        md(f"""
**Leitura.**
- {len(T)} gravações {classe} de {T['Ship ID'].nunique()} navios: {cnt.get('acertada', 0)} acertadas, {cnt.get('oscila', 0)} oscilam e {cnt.get('errada', 0)} erradas. As {classe} oscilam muito mais que as LARGE: a classe é mais difícil.
- Destino dos erros: {des}. Subgrupos analisados (destinos com pelo menos 10 navios): {sub_txt}.{(' Ficam de fora, por terem poucos navios: ' + ', '.join(outros_dest) + '.') if outros_dest else ''}
- A fração de erradas por subconjunto vai de {br(ds_err.min(), 1)}% ({ds_err.idxmin()}) a {br(ds_err.max(), 1)}% ({ds_err.idxmax()}).
- Nenhum navio aparece em mais de uma classe ({ns['N_NAVIOS_COMUNS']}): os navios desta análise são independentes dos da análise das LARGE.
"""),
        md("""
## 1. Espectro LOFAR médio por subgrupo
"""), code(MEDIO_DEST_CODE),
        md("""
## 2. Diferença faixa a faixa, por destino

Mann-Whitney por faixa entre as erradas e as acertadas, com médias por navio e FDR, primeiro juntando todas as erradas e depois separando por destino:
"""), code(TESTE_DEST_CODE),
        md(f"""
**Leitura.**
- **Juntando todas as erradas**, {('nenhuma faixa é significativa' if pool['faixas com q < 0,05'] == 0 else 'só ' + str(int(pool['faixas com q < 0,05'])) + ' faixas são significativas (' + str(int(pool['abaixo de 1 kHz'])) + ' abaixo de 1 kHz)')}; {int(pool['navios acertada'])} navios acertados e {int(pool['navios errada'])} errados.
{chr(10).join('- ' + contr_txt(s) + '.' for s in errs)}
{chr(10).join('- **' + s + ':** ' + leitura_dest(s) + '.' for s in errs)}
- Juntas, as direções se misturam e o teste conjunto acha pouco: as erradas precisam ser vistas por destino.
"""),
        md("""
## 3. Teste fora da amostra: a banda das LARGE

A banda de 31–86 Hz e a faixa de 100–150 Hz foram escolhidas na análise das LARGE, com outros navios. Aqui elas são só **medidas**, sem nenhuma escolha a partir dos dados desta classe. Para comparação, a mesma medida nas classes inteiras:
"""), code(APRIORI_CODE),
        md(f"""
**Leitura.**
- {classe} acertadas: banda 31–86 Hz com mediana {br(ra['banda 31–86 Hz (dB): mediana'], 1)} dB e {br(ra['% gravações com pico em 31–86 Hz'], 0)}% das gravações com o pico grave em 31–86 Hz.
- {'; '.join(ap_txt(s) for s in errs)}.
- Nas classes inteiras, a fração de gravações com o pico grave em 31–86 Hz é: {ref_txt}.
- {(f"**A hipótese se confirma fora da amostra:** as {classe} previstas como LARGE têm a banda de 31–86 Hz no nível das LARGE ({br(AP.loc[f'{classe} errada→LARGE', 'banda 31–86 Hz (dB): mediana'], 1)} × {br(rL['banda 31–86 Hz (dB): mediana'], 1)} dB) e o pico grave na banda tão frequente quanto nas LARGE. Para o modelo, essa banda é uma pista de LARGE, e ela aparece também em navios {classe} (Seção 3.1)." if large_ok and rL is not None else 'A hipótese não se confirma nesta classe.')}
"""),
        md("""
### 3.1 A banda como pista do modelo

Para todas as gravações de navio: em que fração dos folds a CNN LOFAR prevê LARGE, conforme a gravação tenha ou não o pico grave em 31–86 Hz?
"""), code(BANDA_PREV_CODE),
        md(f"""
**Leitura.** {banda_txt(BP, FLB)}
"""),
        md("""
## 4. O mesmo navio, gravações acertadas e erradas
"""), code(MISTO_DEST_CODE),
        md(f"""
**Leitura.**
- {chr(10).join('- ' + t if i else t for i, t in enumerate(mix_txt))}
- {mix_leitura}
"""),
        md("""
## 5. Com que classe cada subgrupo se parece?

Distância RMS (dB) entre espectros médios em 0–500 Hz:
"""), code(SEM_DEST_CODE),
        md(f"""
**Leitura.** {sem_txt}. Sem contar a própria classe, as {classe} acertadas ficam mais perto de {viz} ({br(SEMD.loc[f'{classe} acertada', viz], 2)} dB) do que de {' e '.join(f'{c} ({br(SEMD.loc[f"{classe} acertada", c], 2)} dB)' for c in SEMD.columns if c not in (classe, viz))}: no espectro médio, {classe} e {viz} são parecidas, o que ajuda a explicar por que as duas classes se confundem.
"""),
        md(f"""
## 6. As acertadas: padrão de frequência ou navios conhecidos?

O que as {classe} acertadas têm em comum? Comparamos o espectro delas com o das outras classes, a posição do pico grave e a taxa de acerto conforme o número de gravações do navio. O split do artigo separa treino e teste **por gravação**, então gravações do mesmo navio podem estar nos dois lados.
"""), code(ACERT_CODE),
        md(f"""
**Leitura.**
- **Espectro:** {acv_txt}.{(' As ' + classe + ' acertadas são **indistinguíveis de ' + ' e '.join(iguais) + '** no LOFAR médio; o que as separa de ' + ' e '.join(difs) + ' é sobretudo não terem a banda de 31–86 Hz.') if iguais else ''}
- **Pico grave:** {br(PAC.loc['31–86 Hz', f'{classe} acertada'], 0)}% das {classe} acertadas têm o pico em 31–86 Hz, contra {' e '.join(br(PAC.loc['31–86 Hz', c], 0) + '% das ' + c for c in ('SMALL', 'MEDIUM', 'LARGE') if c != classe)}; {br(PAC.loc['86–140 Hz', f'{classe} acertada'], 0)}% o têm em 86–140 Hz.
- **Navios repetidos:** {tam_txt}. Nas LARGE, os navios gravados uma só vez são acertados em {br(tg(tl, '1'), 0)}% dos casos.
- {n_acg} das {n_ac} {classe} acertadas ({br(100 * n_acg / max(n_ac, 1), 0)}%) vêm de {n_navg} navios com mais de 10 gravações. O navio com mais acertos é o {top_id} ({top1['Name']}, {top1['DETAILED TYPE']}, {br(top1['Length'], 0) if top1['Length'] == top1['Length'] else '?'} m): {int(top1['acertadas'])} acertadas de {int(top1['gravações do navio'])} gravações. Em cada fold, {int(FT['no teste'].min())} a {int(FT['no teste'].max())} gravações dele estão no teste e o resto no treino: **o modelo é testado num navio que já viu**.
- {('**O acerto das ' + classe + ' depende mais de reconhecer navios repetidos do que de um padrão de frequência que generalize.** Em navios novos, a CNN erra mais do que acerta. Isso é coerente com o NB29: com split por navio, o recall de MEDIUM cai ~22 p.p.') if memoriza else 'A taxa de acerto não depende tanto do número de gravações do navio: o acerto não parece vir só de navios repetidos.'}
"""),
        md("""
## 7. Comprimento, condições e subconjunto
"""), code(META_DEST_CODE),
        md(f"""
**Leitura.**
- Acertadas: comprimento mediano {br(rm['Length: mediana'], 0)} m, vento {br(rm['Wind: mediana'], 1)} m/s, estado do mar {br(rm['Sea state: mediana'], 0)}.
- {chr(10).join('- ' + t if i else t for i, t in enumerate(meta_txt))}
- As diferenças de comprimento são pequenas dentro de uma classe definida por uma faixa estreita de comprimento.{vento_txt}
"""),
        md("""
## 8. Banda larga ou raias? (lofar1hz)

Mesma decomposição do notebook das LARGE (Seção 12.2): no espectro médio com faixas de 0,98 Hz, a **componente larga** é a mediana móvel de ~15 Hz e as **raias** são máximos locais com pelo menos 3 dB acima dela, em 31–86 Hz.
"""), code(L1HZ_DEST_CODE),
        md(f"""
**Leitura.**
- {chr(10).join('- ' + t if i else t for i, t in enumerate(l1_txt))}
- Como nas LARGE, o que muda é a **componente larga**; as raias estreitas não diferem entre os grupos.
"""),
        md(f"""
## 9. Conclusões parciais

1. **Os erros das {classe} vão para {len(errs)} destinos, com padrões diferentes.** Juntas, as erradas {('não diferem das acertadas em nenhuma faixa' if pool['faixas com q < 0,05'] == 0 else 'quase não diferem das acertadas')}; separadas por destino, aparecem padrões distintos.
2. **{classe} previstas como LARGE:** {('têm a banda larga de 31–86 Hz das LARGE e menos conteúdo em 100–150 Hz. A banda foi definida nas LARGE, com outros navios, e se confirma aqui fora da amostra' if large_ok else 'não confirmam a banda das LARGE')}{(f"; no mesmo navio, a banda aparece nas gravações chamadas de LARGE ({int(MIXD['errada→LARGE']['errada > acertada'])} de {int(MIXD['errada→LARGE']['navios'])} navios)" if MIXD.get('errada→LARGE') else '')}. Com faixas de 1 Hz, de novo é banda larga, não raias.
{chr(10).join('3' + chr(97 + i) + '. **' + classe + ' previstas como ' + s.split('→')[1] + ':** ' + leitura_dest(s) + '.' for i, s in enumerate([e for e in errs if e != 'errada→LARGE']))}
4. **{classe} acertadas:** {('no LOFAR médio são indistinguíveis de ' + ' e '.join(iguais) + '; ' if iguais else 'no LOFAR médio diferem pouco de ' + ' e '.join(difs) + ' fora da banda de 31–86 Hz; ')}{(str(br(100 * n_acg / max(n_ac, 1), 0)) + '% delas vêm de ' + str(n_navg) + ' navios gravados mais de 10 vezes, que aparecem no treino e no teste em todos os folds. Em navios gravados uma só vez, a CNN acerta ' + br(tg(tc, '1'), 0) + '% (LARGE: ' + br(tg(tl, '1'), 0) + '%). O acerto vem mais de reconhecer navios conhecidos do que de um padrão de frequência; com split por navio, ele cairia.') if memoriza else ('o acerto não depende tanto de navios repetidos: a CNN acerta ' + br(tg(tc, '1'), 0) + '% das gravações de navios gravados uma só vez (LARGE: ' + br(tg(tl, '1'), 0) + '%) e ' + br(tg(tc, 'mais de 10'), 0) + '% das de navios gravados mais de 10 vezes. A classe é difícil mesmo para navios conhecidos.')}
5. **Leitura conjunta com as LARGE: a banda é uma pista importante para o modelo, mas fraca de tamanho real.** A fração de gravações com o pico grave em 31–86 Hz é {pct_txt}. Ter a banda aumenta a chance de a gravação ser chamada de LARGE em qualquer classe ({'; '.join(k + ' de ' + br(BP.loc[k, BP.columns[0]], 0) + '% para ' + br(BP.loc[k, BP.columns[1]], 0) + '%' for k in BP.index)}), mas não é regra: a maioria das SMALL e MEDIUM com a banda não é chamada de LARGE. Como a banda separa mal as classes reais, parte dos erros vem dela: falta em parte das LARGE (que viram SMALL ou MEDIUM) e aparece em parte das {classe} (que viram LARGE). No mesmo navio, ela muda de uma gravação para outra, então parece depender mais da **condição da gravação** (regime de operação, distância, propagação; não registrados no IARA) do que do tamanho do navio.

**Próximos passos:** {('repetir para SMALL; ' if classe == 'MEDIUM' else 'síntese das três classes; ')}investigar os erros sem assinatura no LOFAR ({', '.join(s for s in errs if s != 'errada→LARGE')}) com outras representações (MEL, DEMON) ou com as condições de gravação; avaliar a CNN com split por navio.
"""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "pygments_lexer": "ipython3"}},
          "nbformat": 4, "nbformat_minor": 4}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook gerado em '{filename}'")


if __name__ == "__main__":
    _c = sys.argv[1].upper() if len(sys.argv) > 1 else "LARGE"
    build(_c) if _c == "LARGE" else build_destino(_c)
