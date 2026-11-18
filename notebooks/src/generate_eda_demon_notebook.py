"""
Gera notebooks/36_eda_demon_<classe>.ipynb: análise exploratória dos picos de modulação DEMON das gravações de uma
classe que os modelos DEMON acertam e erram (Tarefa 1 da reunião de 26/10/2026, segunda representação).

O NB35 fez o mesmo com o LOFAR. No DEMON, cada navio tem as suas linhas de modulação (eixo, pás) em frequências
próprias: a média faixa a faixa entre navios apaga essas linhas, então a análise se concentra no pico dominante de cada
gravação (frequência, altura e fundamental harmônica).

Seções: 0 rótulos e destinos; 1 espectro médio e comparação faixa a faixa; 2 pico dominante e comprimento;
3 acertadas × erradas por destino; 4 o pico forte como pista do modelo; 5 relação com a banda LOFAR de 31–86 Hz;
6 navios conhecidos e tipos de navio; 7 exemplos; 8 conclusões.

Fontes: data/iara_processed/demon_features/<ID>.pkl (DEMON do projeto: passa-faixa 1–2 kHz, envelope |x|, fs = 50 Hz,
FFT de 512 pontos = 10,24 s, passo de 5,12 s, 253 faixas de 0,098 Hz de 0,39 a 25 Hz, amplitude normalizada por
gravação); predições por gravação via iara_results.py; src/iara/dataset_info/iara.csv.
Cache: data/iara_processed/cache_demon_relativo.pkl (espectro médio em dB acima do fundo TPSW e persistência > 6 dB).

Rodar na raiz do repositório:  python notebooks/src/generate_eda_demon_notebook.py [LARGE|MEDIUM|SMALL]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")
from generate_eda_lofar_notebook import br, pv, md, code, run, sg_  # noqa: E402

NOTEBOOKS_DIR = os.path.abspath(os.path.join(HERE, ".."))

SETUP_CODE = r'''
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.stats import mannwhitneyu, false_discovery_control, spearmanr
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
MODELO = 'bigru_att_demon_h64_s15-5_e20_4c'               # melhor modelo DEMON do projeto (SP 55,5)
OUTROS = {'MLP DEMON': 'mlp_demon_h32-16_e30_4c', 'HCNN DEMON': 'hcnn_demon_tpsw_med5_e60p8_4c'}
DEMON_DIR = os.path.join(R.ROOT, 'data', 'iara_processed', 'demon_features')
CACHE = os.path.join(R.ROOT, 'data', 'iara_processed', 'cache_demon_relativo.pkl')

def fundo_tpsw(x, n=16, g=3, alpha=2.0):
    # mesmo TPSW do HCNN DEMON (scratch/eval_demon_harmonic_cnn_5x2cv.py): média de ±16 faixas (±1,56 Hz) sem as
    # ±3 centrais (±0,29 Hz); valores acima de 2x a primeira estimativa são cortados e a média é refeita
    k = np.ones(2 * n + 1); k[n - g:n + g + 1] = 0; k /= k.sum()
    b1 = ndi.convolve1d(x, k, axis=-1, mode='reflect')
    return ndi.convolve1d(np.where(x > alpha * b1, b1, x), k, axis=-1, mode='reflect')

if os.path.exists(CACHE):
    _c = pd.read_pickle(CACHE)
else:
    _med, _pers = {}, {}
    for f in os.listdir(DEMON_DIR):
        if not f.endswith('.pkl'):
            continue
        i = int(f[:-4]); d = pd.read_pickle(os.path.join(DEMON_DIR, f))
        avg, W = d['avg_spectrum'], d['df'].values
        _med[i] = 20 * np.log10(avg / (fundo_tpsw(avg) + 1e-12))                 # espectro médio, dB acima do fundo
        Wm = ndi.median_filter(W, size=(5, 1), mode='nearest')                   # mediana de 5 janelas (~25 s)
        _pers[i] = (20 * np.log10(Wm / (fundo_tpsw(Wm) + 1e-12)) > 6).mean(0)   # fração das janelas > 6 dB
    _c = {'freqs': d['frequencies'], 'medio': pd.DataFrame(_med).T.sort_index(), 'pers6': pd.DataFrame(_pers).T.sort_index()}
    pd.to_pickle(_c, CACHE)
FD = np.asarray(_c['freqs']); SD = _c['medio']; PD = _c['pers6']; DF = FD[1] - FD[0]

meta = pd.read_csv(os.path.join(R.ROOT, 'src', 'iara', 'dataset_info', 'iara.csv'), dtype={'Ship ID': str, 'ID': str})
meta['Ship ID'] = meta['Ship ID'].str.strip()
meta['Length'] = pd.to_numeric(meta['Length'], errors='coerce')
meta = meta.set_index(meta.ID.astype(int))
LAB = pd.concat(R.file_predictions(MODELO).values()).groupby('File').Target.first(); LAB.index = LAB.index.astype(int)
print(f'{len(SD)} gravações com DEMON; {len(FD)} faixas de {DF:.4f} Hz, de {FD[0]:.2f} a {FD[-1]:.0f} Hz')
'''

ROTULOS_CODE = r'''
def taxa_acerto(exp, c):
    # fração dos folds de teste em que a gravação da classe c foi acertada (voto majoritário por gravação)
    d = pd.concat(R.file_predictions(exp).values()); d = d[d.Target == c]
    t = d.groupby('File').apply(lambda x: (x.Prediction == x.Target).mean(), include_groups=False); t.index = t.index.astype(int)
    return t

def grupo(t):
    return pd.Series(np.where(t == 1, 'acertada', np.where(t == 0, 'errada', 'oscila')), index=t.index)

_t = taxa_acerto(MODELO, C)
T = pd.DataFrame({'BiGRU DEMON': _t, **{k: taxa_acerto(v, C) for k, v in OUTROS.items()}})
T['grupo'] = grupo(_t)
_todos = T[['BiGRU DEMON'] + list(OUTROS)]
T['consenso'] = np.where((_todos == 1).all(1), 'acertada', np.where((_todos == 0).all(1), 'errada', 'outro'))
_d = pd.concat(R.file_predictions(MODELO).values()); _d = _d[_d.Target == C].copy(); _d['File'] = _d.File.astype(int)
T['destino'] = _d[_d.Prediction != C].groupby('File').Prediction.agg(lambda x: x.value_counts().index[0]).map(CLS).reindex(T.index)
T['subgrupo'] = np.where(T.grupo == 'errada', 'errada→' + T.destino.fillna('?'), T.grupo)
T = T.join(meta[['Ship ID', 'Dataset', 'DETAILED TYPE', 'Length']])
DESTINO = (_d[_d.Prediction != C].Prediction.map(CLS).value_counts(normalize=True) * 100).round(1)
NAV_DEST = T[T.grupo == 'errada'].groupby('destino')['Ship ID'].nunique().sort_values(ascending=False)
DESTS = [x for x in NAV_DEST.index if NAV_DEST[x] >= 8]               # destinos com pelo menos 8 navios
SUBS = ['acertada'] + [f'errada→{x}' for x in DESTS]
CONT = pd.DataFrame({'BiGRU DEMON': T.grupo.value_counts(), 'consenso BiGRU+MLP+HCNN': pd.Series(T.consenso).value_counts()}).fillna(0).astype(int)
RESUMO_SUB = T.groupby('subgrupo').agg(gravações=('grupo', 'size'), navios=('Ship ID', 'nunique'))
print(f'{CLASSE}: {len(T)} gravações de {T["Ship ID"].nunique()} navios'); display(CONT)
print('Destino dos erros do BiGRU DEMON (%):'); display(DESTINO.to_frame('%'))
print('Subgrupos:'); display(RESUMO_SUB)
'''

MEDIO_CODE = r'''
# Espectro DEMON médio (dB acima do fundo TPSW) por subgrupo e por classe
fig, ax = plt.subplots(figsize=(13, 4.2))
cores = dict(zip(SUBS, ['tab:green', 'tab:red', 'tab:orange', 'tab:brown']))
for s in SUBS:
    X = SD.loc[T.index[T.subgrupo == s]]
    m, se = X.mean().values, X.std().values / np.sqrt(len(X))
    ax.plot(FD, m, color=cores[s], lw=1.4, label=f'{CLASSE} {s} (n={len(X)})')
    ax.fill_between(FD, m - 1.96 * se, m + 1.96 * se, color=cores[s], alpha=0.2)
for c, cor in (('SMALL', 'tab:blue'), ('MEDIUM', 'tab:purple'), ('LARGE', 'k'), ('BACKGROUND', 'tab:gray')):
    if c != CLASSE:
        ax.plot(FD, SD.loc[LAB.index[LAB == CID[c]]].mean().values, color=cor, lw=1, ls='--', label=f'{c} (todas)')
ax.axhline(0, color='gray', lw=0.6); ax.set_xlabel('frequência de modulação (Hz)'); ax.set_ylabel('dB acima do fundo')
ax.set_title(f'Espectro DEMON médio — {CLASSE} acertadas e erradas por destino (faixa = IC 95%)'); ax.legend(fontsize=8, ncol=3)
plt.tight_layout(); plt.show()

# Comparação faixa a faixa entre as classes (médias por navio, Mann-Whitney + FDR)
def nav(ids, X=SD):
    return X.loc[ids].groupby(meta.loc[ids, 'Ship ID'].values).mean()
def compara(ids_a, ids_b):
    A, B = nav(ids_a), nav(ids_b)
    U, p = mannwhitneyu(B.values, A.values, axis=0); q = false_discovery_control(p); auc = U / (len(A) * len(B))
    return len(A), len(B), int((q < 0.05).sum()), auc
_rows = []
for a, b in (('SMALL', 'MEDIUM'), ('MEDIUM', 'LARGE'), ('SMALL', 'LARGE')):
    na, nb, ns_, auc = compara(LAB.index[LAB == CID[a]], LAB.index[LAB == CID[b]])
    _rows.append({'comparação': f'{a} × {b}', 'navios': f'{na} × {nb}', 'faixas com q < 0,05': ns_,
                  'AUC mais longe de 0,5': auc[np.abs(auc - 0.5).argmax()], 'em (Hz)': FD[np.abs(auc - 0.5).argmax()]})
for s in SUBS[1:]:
    na, nb, ns_, auc = compara(T.index[T.subgrupo == 'acertada'], T.index[T.subgrupo == s])
    _rows.append({'comparação': f'{CLASSE} acertada × {s}', 'navios': f'{na} × {nb}', 'faixas com q < 0,05': ns_,
                  'AUC mais longe de 0,5': auc[np.abs(auc - 0.5).argmax()], 'em (Hz)': FD[np.abs(auc - 0.5).argmax()]})
CMP = pd.DataFrame(_rows).set_index('comparação')
print(f'Faixa a faixa ({len(FD)} faixas), por navio:'); display(CMP.round(3))
'''

PICO_CODE = r'''
# Pico dominante de cada gravação: faixa de maior DEMON médio (dB acima do fundo) entre 0,5 e 25 Hz.
# Fundamental harmônica: a frequência f (0,5–12,5 Hz) cujos múltiplos f, 2f, 3f, 4f somam mais energia
# (tolerância de ±h faixas no h-ésimo múltiplo). O pico dominante pode ser o eixo, a passagem das pás (BPF) ou um
# harmônico; a fundamental tende a apontar a rotação do eixo, mas sem saber o número de pás isso não é garantido.
_k = FD >= 0.5
_ip = np.where(_k)[0][SD.values[:, _k].argmax(1)]
def fundamental(y, fmin=0.5, fmax=12.5, K=4):
    best, bf = -np.inf, np.nan
    for i0 in np.where((FD >= fmin) & (FD <= fmax))[0]:
        s = []
        for h in range(1, K + 1):
            j = int(round((h * FD[i0] - FD[0]) / DF))
            if j >= len(FD):
                break
            s.append(y[max(j - h, 0):j + h + 1].max())
        if len(s) >= 2 and np.mean(s) > best:
            best, bf = np.mean(s), FD[i0]
    return bf
PK = pd.DataFrame({'f pico (Hz)': FD[_ip], 'altura (dB)': SD.values[np.arange(len(SD)), _ip],
                   'persistência do pico': PD.values[np.arange(len(PD)), _ip],
                   'fundamental (Hz)': [fundamental(SD.loc[i].values) for i in SD.index]}, index=SD.index)
PK['pico forte'] = PK['altura (dB)'] >= 10
PK = PK.join(LAB.map(CLS).rename('classe')).join(meta[['Ship ID', 'Length', 'DETAILED TYPE']])

POR_CLASSE = PK.groupby('classe').agg(**{'f pico mediana (Hz)': ('f pico (Hz)', 'median'),
                                         'fundamental mediana (Hz)': ('fundamental (Hz)', 'median'),
                                         'altura mediana (dB)': ('altura (dB)', 'median'),
                                         '% com pico forte (>= 10 dB)': ('pico forte', lambda x: 100 * x.mean())}).loc[['SMALL', 'MEDIUM', 'LARGE', 'BACKGROUND']]
print('Pico dominante por classe (todas as gravações):'); display(POR_CLASSE.round(2))
_bins = [0, 2, 4, 6, 8, 12, 16, 25.1]
DIST_F = pd.crosstab(pd.cut(PK['fundamental (Hz)'], _bins), PK.classe, normalize='columns')[['SMALL', 'MEDIUM', 'LARGE', 'BACKGROUND']] * 100
print('Onde cai a fundamental (% das gravações):'); display(DIST_F.round(1))

# fundamental × comprimento, um ponto por navio (só navios)
_n = PK[PK.classe.isin(['SMALL', 'MEDIUM', 'LARGE'])].dropna(subset=['Length']).groupby('Ship ID').agg(
    f=('fundamental (Hz)', 'median'), fp=('f pico (Hz)', 'median'), L=('Length', 'first'), forte=('pico forte', 'mean'))
RHO_F = spearmanr(_n.f, _n.L); RHO_FP = spearmanr(_n.fp, _n.L)
_nf = _n[_n.forte >= 0.5]
RHO_FF = spearmanr(_nf.f, _nf.L)
print(f'Spearman, por navio: fundamental × comprimento ρ = {RHO_F[0]:.3f} (p = {RHO_F[1]:.1e}, n = {len(_n)}); '
      f'pico × comprimento ρ = {RHO_FP[0]:.3f}; só navios com pico forte: ρ = {RHO_FF[0]:.3f} (n = {len(_nf)})')
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].scatter(_n.L, _n.f, s=10, alpha=0.4, c=np.where(_n.forte >= 0.5, 'tab:red', 'tab:gray'))
_lb = pd.cut(_n.L, [0, 25, 50, 75, 100, 150, 200, 250, 400])
LEN_F = _n.groupby(_lb, observed=True).agg(navios=('f', 'size'), **{'fundamental mediana (Hz)': ('f', 'median')})
LEN_F.index = [f'{int(iv.left)}–{int(iv.right)} m' for iv in LEN_F.index]
print('Fundamental mediana por faixa de comprimento (por navio):'); display(LEN_F.round(2).T)
ax[0].plot(_n.groupby(_lb, observed=True).L.median(), _n.groupby(_lb, observed=True).f.median(), 'k-o', lw=2, label='mediana por faixa de comprimento')
ax[0].set_xscale('log'); ax[0].set_xlabel('comprimento (m, escala log)'); ax[0].set_ylabel('fundamental DEMON (Hz)')
ax[0].set_title('Fundamental × comprimento (vermelho: navio com pico forte)'); ax[0].legend(fontsize=8)
for c, cor in (('SMALL', 'tab:blue'), ('MEDIUM', 'tab:purple'), ('LARGE', 'k')):
    ax[1].hist(PK[PK.classe == c]['fundamental (Hz)'], bins=np.arange(0.5, 12.6, 0.5), density=True, histtype='step', lw=1.8, color=cor, label=c)
ax[1].set_xlabel('fundamental DEMON (Hz)'); ax[1].set_ylabel('densidade'); ax[1].legend(); ax[1].set_title('Fundamental por classe')
plt.tight_layout(); plt.show()
'''

SUB_CODE = r'''
# Acertadas × erradas por destino: medianas e comparação por navio (Mann-Whitney contra as acertadas)
X = T.join(PK[['f pico (Hz)', 'fundamental (Hz)', 'altura (dB)', 'persistência do pico', 'pico forte']])
_med = X.groupby(['subgrupo', 'Ship ID'])[['f pico (Hz)', 'fundamental (Hz)', 'altura (dB)', 'persistência do pico', 'pico forte']].median().reset_index()
_a = _med[_med.subgrupo == 'acertada']
_rows = []
for s in SUBS:
    x = _med[_med.subgrupo == s]; g = X[X.subgrupo == s]
    r = {'subgrupo': s, 'gravações': len(g), 'navios': len(x), '% gravações com pico forte': 100 * g['pico forte'].mean()}
    for c in ('f pico (Hz)', 'fundamental (Hz)', 'altura (dB)', 'persistência do pico'):
        r[f'{c}: mediana'] = g[c].median()
        if s != 'acertada':
            U, p = mannwhitneyu(x[c], _a[c]); r[f'{c}: AUC'] = U / (len(x) * len(_a)); r[f'{c}: p'] = p
    _rows.append(r)
SUB = pd.DataFrame(_rows).set_index('subgrupo')
print('AUC = probabilidade de um navio do subgrupo ter valor maior que um navio acertado (0,5 = sem diferença):')
display(SUB.round(3).T)
'''

PISTA_CODE = r'''
# O pico forte como pista do modelo: em que fração dos folds a gravação é prevista como LARGE, sem e com pico forte?
_dd = pd.concat(R.file_predictions(MODELO).values()); _dd['File'] = _dd.File.astype(int)
_t = _dd.assign(L=_dd.Prediction == CID['LARGE']).groupby('File').agg(L=('L', 'mean'), Target=('Target', 'first'))
_t = _t[_t.Target != CID['BACKGROUND']].join(PK[['pico forte']]); _t['classe'] = _t.Target.map(CLS)
PISTA = (100 * _t.groupby(['classe', 'pico forte']).L.mean().unstack()).rename(
    columns={False: '% prevista LARGE, sem pico forte', True: '% prevista LARGE, com pico forte'})
PISTA.columns.name = None
PISTA['% das gravações com pico forte'] = 100 * _t.groupby('classe')['pico forte'].mean()
PISTA = PISTA.loc[['SMALL', 'MEDIUM', 'LARGE']]
FRAC_LARGE_FORTE = 100 * (_t[_t['pico forte']].classe == 'LARGE').mean()
display(PISTA.round(1))
print(f'Das gravações de navio com pico forte, {FRAC_LARGE_FORTE:.0f}% são LARGE.')
'''

LOFAR_CODE = r'''
# Relação com a banda LOFAR de 31–86 Hz (NB35): o pico forte do DEMON e a banda grave aparecem juntos?
_S = pd.read_pickle(os.path.join(R.ROOT, 'data', 'iara_processed', 'cache_lofar_espectro_medio.pkl'))
_F = (np.arange(1024) + 1) * 16000 / 2048
_kp = (_F >= 20) & (_F <= 250)
_pl = pd.Series(_F[np.where(_kp)[0][_S.values[:, _kp].argmax(1)]], index=_S.index)
Y = PK[PK.classe.isin(['SMALL', 'MEDIUM', 'LARGE'])].join(_pl.between(31, 86, inclusive='left').rename('banda LOFAR'))
JUNTOS = (100 * Y.groupby(['classe', 'banda LOFAR'])['pico forte'].mean()).unstack('banda LOFAR').reindex(columns=[False, True])
JUNTOS.columns = ['sem a banda LOFAR', 'com a banda LOFAR']
print('% das gravações com pico forte no DEMON:')
display(JUNTOS.loc[['SMALL', 'MEDIUM', 'LARGE']].round(1))
RHO_LOFAR = spearmanr(Y['fundamental (Hz)'], _S.loc[Y.index, (_F >= 31) & (_F < 86)].mean(1))
print(f'Spearman entre a fundamental DEMON e o nível da banda LOFAR 31–86 Hz (todas as gravações de navio): '
      f'ρ = {RHO_LOFAR[0]:.3f} (p = {RHO_LOFAR[1]:.1e})')
'''

SENS_CODE = r'''
# Sensibilidade ao limiar de "pico forte" (6, 8, 10, 12 e 15 dB acima do fundo)
_pl = _dd.assign(L=_dd.Prediction == CID['LARGE']).groupby('File').L.mean()
_lab = LAB.map(CLS)
def _grupos(c):
    x = pd.concat(R.file_predictions(MODELO).values()); x['File'] = x.File.astype(int); x = x[x.Target == CID[c]]
    t = x.groupby('File').apply(lambda g: (g.Prediction == g.Target).mean(), include_groups=False)
    dst = x[x.Prediction != CID[c]].groupby('File').Prediction.agg(lambda v: v.value_counts().index[0]).map(CLS)
    g = pd.Series(np.where(t == 1, 'acertada', np.where(t == 0, 'errada', 'oscila')), index=t.index)
    return pd.Series(np.where(g == 'errada', 'errada→' + dst.reindex(t.index).fillna('?'), g), index=t.index)
_G = {c: _grupos(c) for c in ('SMALL', 'MEDIUM', 'LARGE')}
_SUBS_S = {'LARGE': ['acertada', 'errada→SMALL', 'errada→MEDIUM'], 'MEDIUM': ['acertada', 'errada→LARGE', 'errada→SMALL'],
           'SMALL': ['acertada', 'errada→LARGE', 'errada→MEDIUM']}
_banda = Y['banda LOFAR']
_rows = []
for th in (6, 8, 10, 12, 15):
    fx = PK['altura (dB)'] >= th
    r = {'limiar (dB)': th}
    for c in ('SMALL', 'MEDIUM', 'LARGE', 'BACKGROUND'):
        r[f'% com pico forte: {c}'] = 100 * fx[_lab.index[_lab == c]].mean()
    for c in ('SMALL', 'MEDIUM', 'LARGE'):
        ids = _lab.index[_lab == c]
        r[f'{c}: % prevista LARGE, sem → com'] = f"{100 * _pl[ids][~fx[ids]].mean():.0f} → {100 * _pl[ids][fx[ids]].mean():.0f}"
    nv = _lab.index[_lab != 'BACKGROUND']
    r['% LARGE entre as gravações com pico forte'] = 100 * (_lab[nv][fx[nv]] == 'LARGE').mean()
    for c, sl in _SUBS_S.items():
        g = _G[c]
        r[f'{c} % com pico forte: ' + ' / '.join(x.replace('errada→', '→') for x in sl)] = ' / '.join(f"{100 * fx[g.index[g == x]].mean():.0f}" for x in sl)
    L = Y.index[Y.classe == 'LARGE']
    r['LARGE % com pico forte: sem / com banda LOFAR'] = f"{100 * fx[L][~_banda[L]].mean():.0f} / {100 * fx[L][_banda[L]].mean():.0f}"
    _rows.append(r)
SENS = pd.DataFrame(_rows).set_index('limiar (dB)').T
display(SENS)
'''

CONHECIDOS_CODE = r'''
# Taxa de acerto do BiGRU DEMON conforme o número de gravações do navio (esta classe e LARGE, para comparação)
def por_tamanho(Tx):
    n = Tx['Ship ID'].map(Tx.groupby('Ship ID').size())
    fx = pd.cut(n, [0, 1, 3, 10, 10000], labels=['1', '2–3', '4–10', 'mais de 10'])
    t = (pd.crosstab(fx, Tx.grupo, normalize='index') * 100).reindex(columns=['acertada', 'oscila', 'errada'], fill_value=0)
    return t.join(Tx.groupby(fx, observed=True).agg(gravações=('grupo', 'size'), navios=('Ship ID', 'nunique')))
_tl = taxa_acerto(MODELO, CID['LARGE'])
_TL = pd.DataFrame({'grupo': grupo(_tl)}).join(meta[['Ship ID']])
TAM = pd.concat({CLASSE: por_tamanho(T), 'LARGE': por_tamanho(_TL)} if CLASSE != 'LARGE' else {CLASSE: por_tamanho(T)}, axis=0)
display(TAM.round(1))

# Tipos de navio da classe (pelo menos 10 gravações): acerto e pico dominante
TIPOS = X.groupby('DETAILED TYPE').agg(gravações=('grupo', 'size'), navios=('Ship ID', 'nunique'),
                                       pct_acertada=('grupo', lambda g: 100 * (g == 'acertada').mean()),
                                       pct_errada=('grupo', lambda g: 100 * (g == 'errada').mean()),
                                       fundamental=('fundamental (Hz)', 'median'), altura=('altura (dB)', 'median'),
                                       forte=('pico forte', lambda g: 100 * g.mean()))
TIPOS = TIPOS[TIPOS.gravações >= 10].sort_values('gravações', ascending=False).rename(columns={
    'pct_acertada': '% acertada', 'pct_errada': '% errada', 'fundamental': 'fundamental mediana (Hz)',
    'altura': 'altura mediana (dB)', 'forte': '% com pico forte'})
print('Tipos de navio:'); display(TIPOS.round(1))
'''

OSS_CODE = r'''
# Offshore Supply Ships (MEDIUM) × tipos SMALL parecidos no LOFAR (NB35): o DEMON os separa?
_tipos = {'Offshore Supply Ship (MEDIUM)': (PK['DETAILED TYPE'] == 'Offshore Supply Ship') & (PK.Length >= 50),
          'Tug (SMALL)': (PK['DETAILED TYPE'] == 'Tug') & (PK.Length < 50),
          'Fishing Vessel (SMALL)': (PK['DETAILED TYPE'] == 'Fishing Vessel') & (PK.Length < 50),
          'Pilot Vessel (SMALL)': (PK['DETAILED TYPE'] == 'Pilot Vessel') & (PK.Length < 50),
          'Motor Hopper (MEDIUM)': PK['DETAILED TYPE'] == 'Motor Hopper'}
_nv = {k: PK[m].groupby('Ship ID')[['fundamental (Hz)', 'f pico (Hz)', 'altura (dB)', 'pico forte']].mean() for k, m in _tipos.items()}
_o = _nv['Offshore Supply Ship (MEDIUM)']
_rows = []
for k, v in _nv.items():
    r = {'tipo': k, 'navios': len(v), 'gravações': int(_tipos[k].sum()), 'fundamental mediana (Hz)': v['fundamental (Hz)'].median(),
         'altura mediana (dB)': v['altura (dB)'].median(), '% com pico forte': 100 * PK[_tipos[k]]['pico forte'].mean()}
    if k != 'Offshore Supply Ship (MEDIUM)' and len(v) >= 5:
        for c in ('fundamental (Hz)', 'altura (dB)'):
            U, p = mannwhitneyu(v[c], _o[c]); r[f'{c}: AUC vs OSS'] = U / (len(v) * len(_o)); r[f'{c}: p'] = p
        _, _, nsig, _ = compara(SD.index[_tipos['Offshore Supply Ship (MEDIUM)']], SD.index[_tipos[k]])
        r['faixas com q < 0,05 vs OSS'] = nsig
    _rows.append(r)
OSS = pd.DataFrame(_rows).set_index('tipo')
display(OSS.round(3))
'''

EXEMPLOS_CODE = r'''
# Espectrogramas DEMON (dB acima do fundo, por janela de 10,24 s) de gravações sorteadas: acertadas e erradas
_rng = np.random.default_rng(1)
_lin = [s for s in SUBS][:3]
fig, ax = plt.subplots(len(_lin), 3, figsize=(15, 3.2 * len(_lin)))
for r, s in enumerate(_lin):
    pool = T.index[T.subgrupo == s]
    for c, i in enumerate(_rng.choice(pool, min(3, len(pool)), replace=False)):
        W = pd.read_pickle(os.path.join(DEMON_DIR, f'{i}.pkl'))['df'].values
        Wm = ndi.median_filter(W, size=(5, 1), mode='nearest')
        Z = 20 * np.log10(Wm / (fundo_tpsw(Wm) + 1e-12))
        ax[r, c].imshow(Z.T, aspect='auto', origin='lower', extent=[0, len(Z) * 5.12, FD[0], FD[-1]], vmin=0, vmax=12, cmap='viridis')
        ax[r, c].set_title(f'{CLASSE} {s} — ID {i} ({meta.loc[i, "DETAILED TYPE"]}, {meta.loc[i, "Length"]:.0f} m); '
                           f'fund. {PK.loc[i, "fundamental (Hz)"]:.1f} Hz', fontsize=8)
        ax[r, c].set_xlabel('tempo (s)'); ax[r, c].set_ylabel('Hz')
fig.suptitle('DEMON (0 a 12 dB acima do fundo; mediana de 5 janelas)'); plt.tight_layout(); plt.show()
'''


def build(classe):
    filename = os.path.join(NOTEBOOKS_DIR, f"36_eda_demon_{classe.lower()}.ipynb")
    setup = SETUP_CODE.replace("__CLASSE__", classe)
    oss_on = classe in ("MEDIUM", "SMALL")
    blocos = [setup, ROTULOS_CODE, MEDIO_CODE, PICO_CODE, SUB_CODE, PISTA_CODE, LOFAR_CODE, SENS_CODE, CONHECIDOS_CODE]
    if oss_on:
        blocos.append(OSS_CODE)
    blocos.append(EXEMPLOS_CODE)
    ns = {}
    for src in blocos:
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    T, CONT, DESTINO, RS, SUBS = ns["T"], ns["CONT"], ns["DESTINO"], ns["RESUMO_SUB"], ns["SUBS"]
    CMP, PC, DISTF, SUB, PISTA, FLF = ns["CMP"], ns["POR_CLASSE"], ns["DIST_F"], ns["SUB"], ns["PISTA"], ns["FRAC_LARGE_FORTE"]
    JUNTOS, RHO_L, TAM, TIPOS = ns["JUNTOS"], ns["RHO_LOFAR"], ns["TAM"], ns["TIPOS"]
    SENS = ns["SENS"]
    pf = {c: SENS.loc[f"% com pico forte: {c}"] for c in ("SMALL", "MEDIUM", "LARGE", "BACKGROUND")}
    large_maior = bool(all(pf["LARGE"][t] > max(pf["SMALL"][t], pf["MEDIUM"][t]) for t in SENS.columns))
    sm_iguais = bool(all(abs(pf["SMALL"][t] - pf["MEDIUM"][t]) < 10 for t in SENS.columns))

    def _par(x):
        a_, b_ = x.split(" → ")
        return float(a_), float(b_)
    pista_ok = bool(all(_par(SENS.loc[f"{c}: % prevista LARGE, sem → com", t])[1] > _par(SENS.loc[f"{c}: % prevista LARGE, sem → com", t])[0]
                        for c in ("SMALL", "MEDIUM", "LARGE") for t in SENS.columns))

    def _trio(lin, t):
        return [float(v) for v in SENS.loc[lin, t].split(" / ")]
    med_lin = "MEDIUM % com pico forte: acertada / →LARGE / →SMALL"
    med_inv = [t for t in SENS.columns if _trio(med_lin, t)[1] <= _trio(med_lin, t)[0]]
    lf = SENS.loc["% LARGE entre as gravações com pico forte"]
    RHO_F, RHO_FP, RHO_FF = ns["RHO_F"], ns["RHO_FP"], ns["RHO_FF"]
    n_nav = len(ns["_n"]); n_navf = len(ns["_nf"])
    LEN_F = ns["LEN_F"]
    _lf = LEN_F["fundamental mediana (Hz)"]
    lf_meio = _lf[[i for i in _lf.index if int(i.split('–')[0]) >= 25 and int(i.split('–')[1].split()[0]) <= 200]]
    lf_alto = "; ".join(f"{i}: {br(v, 1)} Hz" for i, v in _lf.items() if int(i.split('–')[0]) >= 200)
    _pk = ns["PK"]
    n_mh = int((_pk['DETAILED TYPE'] == 'Motor Hopper').sum()); f_mh = _pk[_pk['DETAILED TYPE'] == 'Motor Hopper']['fundamental (Hz)'].median()
    cnt = CONT["BiGRU DEMON"]
    errs = [s for s in SUBS if s != "acertada"]
    des = ", ".join(f"{c} {br(v, 1)}%" for c, v in DESTINO.items())
    sub_txt = "; ".join(f"**{s}**: {int(RS.loc[s, 'gravações'])} gravações de {int(RS.loc[s, 'navios'])} navios" for s in errs)
    fora = [s for s in RS.index if s.startswith("errada→") and s not in errs]
    ra = SUB.loc["acertada"]

    cls_rows = [r for r in CMP.index if " × " in r and not r.startswith(classe + " acertada")]
    def nf(n):
        return f"{int(n)} {'faixa' if int(n) == 1 else 'faixas'}"
    cmp_txt = "; ".join(f"{r}: {nf(CMP.loc[r, 'faixas com q < 0,05'])}" for r in cls_rows)
    sub_cmp_txt = "; ".join(f"{s}: {nf(CMP.loc[f'{classe} acertada × {s}', 'faixas com q < 0,05'])}" for s in errs)

    def sub_txt_s(s):
        r = SUB.loc[s]
        return (f"**{s}** ({int(r['navios'])} navios): fundamental mediana {br(r['fundamental (Hz): mediana'], 1)} Hz "
                f"(AUC {br(r['fundamental (Hz): AUC'], 2)}, {pv(r['fundamental (Hz): p'])}), altura {br(r['altura (dB): mediana'], 1)} dB "
                f"(AUC {br(r['altura (dB): AUC'], 2)}, {pv(r['altura (dB): p'])}), {br(r['% gravações com pico forte'], 0)}% com pico forte")

    def leitura_s(s):
        r = SUB.loc[s]
        partes = []
        alt_dif = r["altura (dB): p"] < 0.05
        fun_dif = r["fundamental (Hz): p"] < 0.05
        if alt_dif:
            partes.append(f"o pico é {'mais forte' if r['altura (dB): AUC'] > 0.5 else 'mais fraco'} que nas acertadas")
        if fun_dif:
            partes.append(f"a fundamental é {'mais alta' if r['fundamental (Hz): AUC'] > 0.5 else 'mais baixa'}")
        if not partes:
            partes.append("nem a altura nem a fundamental do pico diferem das acertadas")
        return "; ".join(partes)

    pista_sobe = "; ".join(f"{k} de {br(PISTA.loc[k, PISTA.columns[0]], 0)}% para {br(PISTA.loc[k, PISTA.columns[1]], 0)}%" for k in PISTA.index)
    pres = {k: PISTA.loc[k, PISTA.columns[2]] for k in PISTA.index}
    junto_txt = "; ".join(f"{k}: {br(JUNTOS.loc[k, 'sem a banda LOFAR'], 0)}% sem a banda e {br(JUNTOS.loc[k, 'com a banda LOFAR'], 0)}% com a banda"
                          for k in ("SMALL", "MEDIUM", "LARGE"))
    tc = TAM.loc[classe]

    def tg(t, lin, col="acertada"):
        return t.loc[lin, col] if lin in t.index else float("nan")
    tam_txt = "; ".join(f"navios com {lin} {'gravação' if lin == '1' else 'gravações'}: {br(tg(tc, lin), 0)}% acertadas "
                        f"({int(tg(tc, lin, 'gravações'))} gravações)" for lin in tc.index)
    # só conta como "navios conhecidos" se houver pelo menos 5 navios com mais de 10 gravações
    memoriza = bool(tg(tc, "mais de 10", "navios") >= 5 and tg(tc, "mais de 10") > tg(tc, "1") + 15)
    poucos_grandes = tg(tc, "mais de 10", "navios") < 5
    tl_txt = ""
    if classe != "LARGE":
        tl = TAM.loc["LARGE"]
        tl_txt = f" Nas LARGE, navios gravados uma só vez são acertados em {br(tg(tl, '1'), 0)}%."
    tipos_txt = "; ".join(f"{t} ({int(r['navios'])} navios): {br(r['% acertada'], 0)}% acertadas, fundamental {br(r['fundamental mediana (Hz)'], 1)} Hz, "
                          f"{br(r['% com pico forte'], 0)}% com pico forte" for t, r in TIPOS.head(5).iterrows())

    oss_cells = []
    oss_concl = ""
    if oss_on:
        OSS = ns["OSS"]
        linhas = []
        for t, r in OSS.iterrows():
            if t.startswith("Offshore"):
                continue
            if "fundamental (Hz): AUC vs OSS" in r and r["fundamental (Hz): AUC vs OSS"] == r["fundamental (Hz): AUC vs OSS"]:
                linhas.append(f"{t} ({int(r['navios'])} navios): fundamental {br(r['fundamental mediana (Hz)'], 1)} Hz "
                              f"(AUC {br(r['fundamental (Hz): AUC vs OSS'], 2)}, {pv(r['fundamental (Hz): p'])}), altura AUC "
                              f"{br(r['altura (dB): AUC vs OSS'], 2)} ({pv(r['altura (dB): p'])}), "
                              f"{int(r['faixas com q < 0,05 vs OSS'])} {'faixa diferente' if int(r['faixas com q < 0,05 vs OSS']) == 1 else 'faixas diferentes'} no espectro médio (de {len(ns['FD'])})")
            else:
                linhas.append(f"{t} ({int(r['navios'])} navios, poucos para teste): fundamental {br(r['fundamental mediana (Hz)'], 1)} Hz")
        ro = OSS.loc["Offshore Supply Ship (MEDIUM)"]
        # separa = fundamental ou altura com p < 0,05, ou pelo menos 5% das faixas do espectro médio com q < 0,05
        sig_any = [t for t, r in OSS.iterrows() if not t.startswith("Offshore") and
                   ((r.get("fundamental (Hz): p", 1) < 0.05) or (r.get("altura (dB): p", 1) < 0.05) or
                    (r.get("faixas com q < 0,05 vs OSS", 0) >= 0.05 * len(ns["FD"])))]
        tug_sep = "Tug (SMALL)" in sig_any
        oss_cells = [
            md("""
## 7. Offshore Supply Ships × rebocadores e pesqueiros

No LOFAR médio, os Offshore Supply Ships (MEDIUM) não diferem dos rebocadores e pesqueiros (SMALL) em nenhuma faixa (NB35). O DEMON mede outra coisa: a modulação do ruído de cavitação pela rotação do eixo e pela passagem das pás. Se o porte muda a rotação, o DEMON poderia separá-los:
"""), code(OSS_CODE),
            md(f"""
**Leitura.**
- Offshore Supply Ships: {int(ro['navios'])} navios, fundamental mediana {br(ro['fundamental mediana (Hz)'], 1)} Hz, {br(ro['% com pico forte'], 0)}% das gravações com pico forte.
{chr(10).join('- ' + x + '.' for x in linhas)}
- {('**O DEMON separa os OSS dos rebocadores** em pelo menos uma medida; vale testar um classificador OSS × Tug com split por navio.' if tug_sep else '**O DEMON também não separa os OSS dos rebocadores**: nem a fundamental, nem a altura do pico, nem o espectro médio diferem de forma significativa. Pelas duas representações, LOFAR e DEMON, um supply de 50–90 m soa como um rebocador de 20–35 m.')}
""")]
        oss_concl = (f"\n6. **Offshore Supply Ships × rebocadores:** {'o DEMON separa os dois em alguma medida (Seção 7).' if tug_sep else 'o DEMON também não os separa (Seção 7). LOFAR e DEMON concordam: no IARA, um supply de 50–90 m soa como um rebocador.'}")

    n_ex = 8 if oss_on else 7
    cells = [md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 36: Análise Exploratória dos Picos DEMON — {classe} Acertadas × Erradas

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

**Tarefa 1 (reunião de 26/10/2026), com a segunda representação.** O NB35 comparou o LOFAR de gravações acertadas e erradas. Aqui fazemos o mesmo com o **DEMON**, que mede outra propriedade física: a **modulação** do ruído de cavitação da hélice pela rotação do eixo e pela passagem das pás. Para navios, essas linhas ficam tipicamente entre 1 e 25 Hz, e a sua frequência depende da rotação, que tende a ser menor em navios maiores.

**Dados.** DEMON do projeto (`data/iara_processed/demon_features`):
- passa-faixa de 1–2 kHz e envelope pelo valor absoluto;
- reamostragem para 50 Hz e FFT de 512 pontos (janelas de 10,24 s, passo de 5,12 s);
- 253 faixas de 0,098 Hz, de 0,39 a 25 Hz;
- amplitude normalizada por gravação, então não há nível absoluto.

**Pré-processamento.** Cada espectro é dividido pelo fundo estimado com o mesmo TPSW do HCNN DEMON (±1,56 Hz, sem as ±0,29 Hz centrais) e convertido em **dB acima do fundo**, como no LOFAR.

**Modelo de referência.** O BiGRU DEMON (`{ns['MODELO']}`), o melhor modelo DEMON do projeto (SP 55,5, contra ~66 dos modelos LOFAR). Os modelos DEMON são bem mais fracos que os LOFAR, então os rótulos de acerto e erro são mais ruidosos. Para robustez, o consenso exige o mesmo resultado no MLP e no HCNN DEMON.

**Diferença em relação ao NB35.** Cada navio tem as suas linhas de modulação em frequências próprias. Na média faixa a faixa entre navios, essas linhas se espalham e somem, então a análise se concentra no **pico dominante de cada gravação**.
"""), code(setup),
        md("""
## 0. Rótulos de acerto e erro, e destino dos erros

**Acertada** = acerta em todos os folds de teste do BiGRU DEMON; **errada** = erra em todos; **oscila** = o resto. As erradas são separadas pela classe para onde vão na maioria dos folds.
"""), code(ROTULOS_CODE),
        md(f"""
**Leitura.**
- {len(T)} gravações {classe} de {T['Ship ID'].nunique()} navios: {cnt.get('acertada', 0)} acertadas, {cnt.get('oscila', 0)} oscilam e {cnt.get('errada', 0)} erradas. No consenso dos três modelos DEMON: {CONT['consenso BiGRU+MLP+HCNN'].get('acertada', 0)} acertadas e {CONT['consenso BiGRU+MLP+HCNN'].get('errada', 0)} erradas.
- Destino dos erros: {des}. Subgrupos analisados (destinos com pelo menos 8 navios): {sub_txt}.{(' Ficam de fora, por terem poucos navios: ' + ', '.join(fora) + '.') if fora else ''}
"""),
        md("""
## 1. Espectro médio e comparação faixa a faixa
"""), code(MEDIO_CODE),
        md(f"""
**Leitura.**
- **Entre as classes**, quase nenhuma faixa difere na média por navio ({cmp_txt}). **Entre acertadas e erradas**: {sub_cmp_txt}.
- Isso não quer dizer que o DEMON seja igual entre as classes: cada navio tem as suas linhas em frequências próprias, e a média entre navios as apaga. As próximas seções olham o **pico de cada gravação**.
"""),
        md("""
## 2. O pico dominante de cada gravação

Para cada gravação: a **frequência do pico** (faixa de maior DEMON entre 0,5 e 25 Hz), a **altura** (dB acima do fundo), a **persistência** (fração das janelas com o pico mais de 6 dB acima do fundo) e a **fundamental harmônica** (a frequência cujos múltiplos f, 2f, 3f e 4f somam mais energia). Um **pico forte** tem pelo menos 10 dB.
"""), code(PICO_CODE),
        md(f"""
**Leitura.**
- **Por classe:** a fundamental mediana é {br(PC.loc['SMALL', 'fundamental mediana (Hz)'], 1)} Hz nas SMALL, {br(PC.loc['MEDIUM', 'fundamental mediana (Hz)'], 1)} Hz nas MEDIUM e {br(PC.loc['LARGE', 'fundamental mediana (Hz)'], 1)} Hz nas LARGE; o pico forte aparece em {br(PC.loc['SMALL', '% com pico forte (>= 10 dB)'], 0)}%, {br(PC.loc['MEDIUM', '% com pico forte (>= 10 dB)'], 0)}% e {br(PC.loc['LARGE', '% com pico forte (>= 10 dB)'], 0)}% das gravações, e em {br(PC.loc['BACKGROUND', '% com pico forte (>= 10 dB)'], 0)}% do BACKGROUND. {('**SMALL e MEDIUM são iguais no DEMON**, como no LOFAR: mesma fundamental e mesma fração de picos fortes. Só as LARGE se destacam, com modulação mais grave e mais forte.' if abs(PC.loc['SMALL', 'fundamental mediana (Hz)'] - PC.loc['MEDIUM', 'fundamental mediana (Hz)']) < 1 and abs(PC.loc['SMALL', '% com pico forte (>= 10 dB)'] - PC.loc['MEDIUM', '% com pico forte (>= 10 dB)']) < 10 else '')}
- **Fundamental × comprimento:** quanto maior o navio, mais baixa a fundamental (ρ = {br(RHO_F[0], 2)}, {pv(RHO_F[1])}, {n_nav} navios; só navios com pico forte: ρ = {br(RHO_FF[0], 2)}, {n_navf} navios). É o sentido esperado fisicamente (hélices maiores giram mais devagar), mas a relação é moderada: navios de mesmo tamanho têm fundamentais bem diferentes (velocidade, número de pás, tipo de propulsão).
- **Mas a relação não é gradual:** a fundamental mediana fica entre {br(lf_meio.min(), 1)} e {br(lf_meio.max(), 1)} Hz de 25 a 200 m e só cai acima de 200 m ({lf_alto}). O DEMON distingue os navios **muito grandes**, não os pequenos dos médios.
- {('O pico do histograma das MEDIUM em 6–7 Hz vem dos Motor Hoppers (' + str(int(n_mh)) + ' gravações de 3 navios, fundamental mediana ' + br(f_mh, 1) + ' Hz): o histograma por gravação pesa os navios repetidos.') if classe == 'MEDIUM' else ''}
- **Cautela:** sem saber o número de pás, não dá para dizer se a fundamental é o eixo ou a passagem das pás; o pico dominante muitas vezes é um harmônico.
"""),
        md("""
## 3. Acertadas × erradas, por destino
"""), code(SUB_CODE),
        md(f"""
**Leitura.**
- Acertadas: fundamental mediana {br(ra['fundamental (Hz): mediana'], 1)} Hz, altura {br(ra['altura (dB): mediana'], 1)} dB, {br(ra['% gravações com pico forte'], 0)}% com pico forte.
{chr(10).join('- ' + sub_txt_s(s) + ': ' + leitura_s(s) + '.' for s in errs)}
"""),
        md("""
## 4. O pico forte como pista do modelo

Para todas as gravações de navio: em que fração dos folds o BiGRU DEMON prevê LARGE, conforme a gravação tenha ou não um pico forte?
"""), code(PISTA_CODE),
        md(f"""
**Leitura.** Em qualquer classe, ter um pico forte aumenta a chance de a gravação ser chamada de LARGE ({pista_sobe}). Como a banda de 31–86 Hz no LOFAR, o pico forte é uma pista de LARGE para o modelo, mas fraca de tamanho real: aparece em {br(pres['LARGE'], 0)}% das LARGE, {br(pres['SMALL'], 0)}% das SMALL e {br(pres['MEDIUM'], 0)}% das MEDIUM, e das gravações com pico forte só {br(FLF, 0)}% são LARGE.
"""),
        md("""
## 5. Relação com a banda LOFAR de 31–86 Hz

Fração das gravações com pico forte no DEMON, conforme tenham ou não o pico grave do LOFAR na banda de 31–86 Hz (NB35):
"""), code(LOFAR_CODE),
        md(f"""
**Leitura.** {junto_txt}. A fundamental DEMON cai quando a banda LOFAR é mais forte (ρ = {br(RHO_L[0], 2)}, {pv(RHO_L[1])}). A associação é forte nas LARGE e fraca nas SMALL e MEDIUM (poucos pontos percentuais). {('As duas pistas aparecem juntas: gravações com a banda grave no LOFAR têm mais vezes um pico forte e mais grave no DEMON. Elas podem refletir o mesmo regime da hélice (cavitação mais intensa e rotação mais baixa), o que poderia explicar por que os modelos LOFAR e DEMON tendem a errar nas mesmas direções (não verificado aqui).' if all(JUNTOS.loc[k, 'com a banda LOFAR'] > JUNTOS.loc[k, 'sem a banda LOFAR'] for k in ('SMALL', 'MEDIUM', 'LARGE')) else 'A relação entre as duas pistas não é a mesma em todas as classes.')}
"""),
        md("""
### 5.1 Sensibilidade ao limiar de pico forte

O limiar de 10 dB é uma escolha. As mesmas medidas com 6, 8, 12 e 15 dB acima do fundo:
"""), code(SENS_CODE),
        md(f"""
**Leitura.**
- {('Em todos os limiares, as LARGE têm mais pico forte que SMALL e MEDIUM' if large_maior else 'As LARGE nem sempre têm mais pico forte')}{(', e SMALL e MEDIUM ficam sempre parecidas (diferença menor que 10 p.p.)' if sm_iguais else '')}. {('Em todos os limiares e nas três classes, ter pico forte aumenta a chance de a gravação ser chamada de LARGE.' if pista_ok else 'A pista de LARGE muda com o limiar.')}
- Quanto mais alto o limiar, mais específico de LARGE é o pico forte (das gravações com pico forte, {br(lf[6], 0)}% são LARGE com 6 dB e {br(lf[15], 0)}% com 15 dB), mas mais raro (com 15 dB, só {br(pf['LARGE'][15], 0)}% das LARGE o têm).
- {('A única exceção: com ' + ', '.join(str(int(t)) for t in med_inv) + ' dB, as MEDIUM erradas→LARGE não têm mais pico forte que as acertadas. Com um limiar tão baixo, a maioria das gravações passa (' + br(pf['MEDIUM'][med_inv[0]], 0) + '% das MEDIUM e ' + str(int(_trio(med_lin, med_inv[0])[0])) + '% das MEDIUM acertadas) e a medida deixa de discriminar; a partir de 8 dB a diferença aparece.') if med_inv else 'As diferenças entre acertadas e erradas mantêm o sentido em todos os limiares.'}
- **As conclusões não dependem do limiar de 10 dB**, que fica no meio da faixa testada.
"""),
        md("""
## 6. Navios conhecidos e tipos de navio
"""), code(CONHECIDOS_CODE),
        md(f"""
**Leitura.**
- {tam_txt}.{tl_txt}
- {('Como no LOFAR, **o acerto depende muito de o navio ter muitas gravações** (que aparecem no treino e no teste no split do artigo).' if memoriza else ('Só ' + str(int(tg(tc, 'mais de 10', 'navios'))) + ' navios têm mais de 10 gravações, poucos para concluir algo sobre navios conhecidos; nos demais, o acerto não cresce com o número de gravações.' if poucos_grandes else 'O acerto não depende tanto do número de gravações do navio.'))}
- Tipos com mais gravações: {tipos_txt}.
"""),
    ] + oss_cells + [
        md(f"""
## {n_ex}. Exemplos
"""), code(EXEMPLOS_CODE),
        md(f"""
## {n_ex + 1}. Conclusões parciais

1. **A média faixa a faixa não separa as classes no DEMON**: cada navio tem as suas linhas em frequências próprias. A informação está no **pico de cada gravação**.
2. **A fundamental cai com o comprimento** (ρ = {br(RHO_F[0], 2)}), no sentido físico esperado, mas só nos navios acima de ~200 m; de 25 a 200 m ela fica praticamente constante, e a dispersão entre navios de mesmo tamanho é grande.
3. **{classe} acertadas × erradas:** {'; '.join(s + ': ' + leitura_s(s) for s in errs)}.
4. **O pico forte é uma pista de LARGE para o modelo**, como a banda de 31–86 Hz no LOFAR, e também fraca de tamanho real (aparece em {br(pres['SMALL'], 0)}% das SMALL e {br(pres['MEDIUM'], 0)}% das MEDIUM). As duas pistas tendem a aparecer juntas.
5. **Navios conhecidos:** {('o acerto cresce com o número de gravações do navio, como no LOFAR' + (f" (navios gravados uma só vez: {br(tg(tc, '1'), 0)}% acertadas; mais de 10 vezes: {br(tg(tc, 'mais de 10'), 0)}%)") + '.' if memoriza else 'sem evidência de que o acerto venha de navios repetidos.')}{oss_concl}

**Ressalva:** os modelos DEMON são bem mais fracos que os LOFAR (SP ~55 contra ~66), então os rótulos de acerto e erro são mais ruidosos. A relação entre fundamental e comprimento, que não depende do modelo, é o resultado mais sólido deste notebook.
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
    build(sys.argv[1].upper() if len(sys.argv) > 1 else "LARGE")
