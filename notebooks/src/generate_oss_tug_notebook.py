"""
Gera notebooks/37_oss_tug_novas_representacoes.ipynb: Offshore Supply Ships (MEDIUM) × rebocadores (SMALL), e
MEDIUM × SMALL, em representações que os modelos não usam (Tarefa 1).

No LOFAR (NB35) e no DEMON (NB36), os Offshore Supply Ships não diferem dos rebocadores. Aqui testamos:
  1. psd24k (scripts/extrair_psd24k.py): forma do espectro de 0 a 24 kHz, sem decimação;
  2. demon_multibanda (scripts/extrair_demon_multibanda.py): DEMON em 0,5–1, 1–2, 2–4, 4–8 e 8–16 kHz;
  3. lofar1hz (scripts/extrair_lofar1hz.py): raias estreitas persistentes ("impressão digital" de máquinas);
  4. nível: absoluto (depende do ganho do equipamento, não calibrado) e medidas que não dependem do ganho
     (faixa dinâmica da passagem dentro da própria gravação).

Rodar na raiz do repositório:  python notebooks/src/generate_oss_tug_notebook.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")
from generate_eda_lofar_notebook import br, pv, md, code, run  # noqa: E402

NOTEBOOKS_DIR = os.path.abspath(os.path.join(HERE, ".."))

SETUP_CODE = r'''
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, false_discovery_control, fisher_exact, spearmanr
from scipy.signal import find_peaks
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
pd.set_option('display.precision', 3); pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
PROC = os.path.join(R.ROOT, 'data', 'iara_processed')

meta = pd.read_csv(os.path.join(R.ROOT, 'src', 'iara', 'dataset_info', 'iara.csv'), dtype={'Ship ID': str, 'ID': str})
meta['Ship ID'] = meta['Ship ID'].str.strip(); meta['Length'] = pd.to_numeric(meta['Length'], errors='coerce')
meta = meta.set_index(meta.ID.astype(int))
meta = meta[meta.Dataset.isin(list('ABCD'))]                          # gravações de navio do 5x2cv (UO)
meta['classe'] = np.select([meta.Length < 50, meta.Length < 100], ['SMALL', 'MEDIUM'], 'LARGE')
GRUPOS = {'OSS (MEDIUM)': meta.index[(meta['DETAILED TYPE'] == 'Offshore Supply Ship') & (meta.Length >= 50)],
          'Tug (SMALL)': meta.index[(meta['DETAILED TYPE'] == 'Tug') & (meta.Length < 50)],
          'MEDIUM': meta.index[meta.classe == 'MEDIUM'], 'SMALL': meta.index[meta.classe == 'SMALL'],
          'LARGE': meta.index[meta.classe == 'LARGE']}
PARES = [('OSS (MEDIUM)', 'Tug (SMALL)'), ('MEDIUM', 'SMALL'), ('LARGE', 'MEDIUM')]

def nav(X, ids):
    ids = [i for i in ids if i in X.index]
    return X.loc[ids].groupby(meta.loc[ids, 'Ship ID'].values).mean()

def faixa_a_faixa(X, a, b):
    # Mann-Whitney por faixa entre as médias por navio, FDR de Benjamini-Hochberg; AUC = P(a > b)
    A, B = nav(X, GRUPOS[a]), nav(X, GRUPOS[b])
    U, p = mannwhitneyu(A.values, B.values, axis=0)
    return A, B, false_discovery_control(p), U / (len(A) * len(B))

print('Gravações e navios por grupo:')
display(pd.DataFrame({g: {'gravações': len(ix), 'navios': meta.loc[ix, 'Ship ID'].nunique(),
                          'comprimento (m)': f"{meta.loc[ix, 'Length'].min():.0f}–{meta.loc[ix, 'Length'].max():.0f}"}
                      for g, ix in GRUPOS.items()}).T)
'''

PSD_CODE = r'''
# psd24k: forma do espectro (dB menos a média da própria gravação em 20 Hz–20 kHz): ganho e distância viram um
# deslocamento constante e somem; sobra a distribuição da energia entre as frequências
P = pd.read_pickle(os.path.join(PROC, 'cache_psd24k_resumo.pkl'))
FP = np.asarray(P['freqs']); kf = (FP >= 20) & (FP <= 23000)
ABS = P['psd_db'].astype(float).loc[:, kf]; ABS.columns = FP[kf]
FORMA = ABS.sub(ABS.loc[:, ABS.columns <= 20000].mean(1), axis=0)
BANDAS = [(20, 1000), (1000, 4000), (4000, 8000), (8000, 16000), (16000, 23001)]
_rows = []
for a, b in PARES:
    A, B, q, auc = faixa_a_faixa(FORMA, a, b)
    r = {'comparação': f'{a} × {b}', 'navios': f'{len(A)} × {len(B)}', 'faixas com q < 0,05': int((q < 0.05).sum()), 'de': len(q)}
    for lo, hi in BANDAS:
        k = (FP[kf] >= lo) & (FP[kf] < hi)
        r[f'Δ {lo / 1000:g}–{min(hi, 23000) / 1000:g} kHz (dB)'] = (A.values.mean(0) - B.values.mean(0))[k].mean()
    _rows.append(r)
FORMA_T = pd.DataFrame(_rows).set_index('comparação')
display(FORMA_T.round(2))

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
for g, cor in (('OSS (MEDIUM)', 'tab:purple'), ('Tug (SMALL)', 'tab:blue'), ('LARGE', 'k')):
    m = nav(FORMA, GRUPOS[g]).mean().values
    ax[0].semilogx(FP[kf], m, color=cor, lw=1, label=g)
ax[0].set_xlabel('frequência (Hz)'); ax[0].set_ylabel('dB relativo à média da gravação'); ax[0].legend()
ax[0].set_title('Forma do espectro, 20 Hz–23 kHz (média por navio)')
for a, b in PARES[:2]:
    _, _, q, auc = faixa_a_faixa(FORMA, a, b)
    ax[1].semilogx(FP[kf], auc, lw=0.8, label=f'{a} × {b}')
ax[1].axhline(0.5, color='gray'); ax[1].set_ylim(0, 1); ax[1].set_xlabel('frequência (Hz)'); ax[1].set_ylabel('AUC por faixa')
ax[1].set_title('AUC faixa a faixa (0,5 = sem diferença)'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
'''

DEMON_CODE = r'''
# demon_multibanda: espectro relativo (dB acima do fundo TPSW) faixa a faixa e pico dominante, por banda
D = pd.read_pickle(os.path.join(PROC, 'cache_demon_multibanda_resumo.pkl')); FD = np.asarray(D['freqs'])
_rows = []
for banda in D['bandas']:
    X = D['rel_db'][banda].astype(float)
    k = FD >= 0.5; ip = np.where(k)[0][X.values[:, k].argmax(1)]
    PKb = pd.DataFrame({'f pico (Hz)': FD[ip], 'altura (dB)': X.values[np.arange(len(X)), ip]}, index=X.index)
    PKb['pico forte'] = (PKb['altura (dB)'] >= 10).astype(float)
    for a, b in PARES[:2]:
        A, B, q, auc = faixa_a_faixa(X, a, b)
        r = {'banda': banda, 'comparação': f'{a} × {b}', 'navios': f'{len(A)} × {len(B)}', 'faixas com q < 0,05 (de 253)': int((q < 0.05).sum())}
        pa, pb = nav(PKb, GRUPOS[a]), nav(PKb, GRUPOS[b])
        for c in ('f pico (Hz)', 'altura (dB)', 'pico forte'):
            U, p = mannwhitneyu(pa[c], pb[c]); r[f'{c}: AUC'] = U / (len(pa) * len(pb)); r[f'{c}: p'] = p
        _rows.append(r)
DEMON_T = pd.DataFrame(_rows).set_index(['banda', 'comparação'])
display(DEMON_T.round(3))
'''

RAIAS_CODE = r'''
# lofar1hz: faixas (1 Hz, 5–900 Hz) com raia estreita persistente (máximo local da persistência > 6 dB com >= 50% das
# janelas e proeminência >= 0,2, tolerância de ±1 faixa); por navio, a faixa conta se alguma gravação a tem.
L1 = pd.read_pickle(os.path.join(PROC, 'cache_lofar1hz_resumo.pkl')); F1 = np.asarray(L1['freqs']); P1 = L1['pers6'].astype(float)
k1 = (F1 >= 5) & (F1 <= 900)
def ocupadas(i):
    pk, _ = find_peaks(np.r_[0, P1.loc[i].values, 0], height=0.5, prominence=0.2); pk = pk - 1
    o = np.zeros(len(F1), bool); o[pk] = True
    return o | np.r_[o[1:], False] | np.r_[False, o[:-1]]
_rows, RAIAS_TOP = [], {}
for a, b in PARES[:2]:
    ids = [i for i in GRUPOS[a].union(GRUPOS[b]) if i in P1.index]
    O = pd.DataFrame({i: ocupadas(i) for i in ids}).T
    A, B = nav(O, GRUPOS[a]) > 0, nav(O, GRUPOS[b]) > 0
    na, nb = A.sum().values, B.sum().values
    p = np.array([fisher_exact([[na[j], len(A) - na[j]], [nb[j], len(B) - nb[j]]])[1] for j in range(len(F1))])
    q = np.ones_like(p); q[k1] = false_discovery_control(p[k1])
    T = pd.DataFrame({'f (Hz)': F1, f'% navios {a}': 100 * na / len(A), f'% navios {b}': 100 * nb / len(B), 'p': p, 'q (FDR)': q})[k1]
    RAIAS_TOP[f'{a} × {b}'] = T.sort_values('p').head(6)
    _rows.append({'comparação': f'{a} × {b}', 'navios': f'{len(A)} × {len(B)}', 'faixas com q < 0,05': int((T['q (FDR)'] < 0.05).sum()),
                  'menor p (sem correção)': T.p.min(), 'em (Hz)': T.loc[T.p.idxmin(), 'f (Hz)'], 'q dessa faixa': T.loc[T.p.idxmin(), 'q (FDR)']})
RAIAS_T = pd.DataFrame(_rows).set_index('comparação')
display(RAIAS_T.round(4))
print('Faixas com menor p, OSS × Tug:'); display(RAIAS_TOP['OSS (MEDIUM) × Tug (SMALL)'].round(4))
'''

NIVEL_CODE = r'''
# Nível por gravação a partir dos blocos de 10 s do psd24k (1–8 kHz): médio, máximo (perto do CPA), mínimo (trecho
# mais silencioso) e faixa dinâmica (máximo − mínimo, p90 − p10). O nível absoluto está em unidades do arquivo (16 bits):
# o IARA não fornece a calibração nem qual hidrofone gravou cada arquivo, e as sensibilidades dos hidrofones do projeto
# vão de −200 a −150 dB re 1 V/µPa. As diferenças (faixa dinâmica) não dependem do ganho.
CB = os.path.join(PROC, 'cache_psd24k_niveis.pkl')
if os.path.exists(CB):
    NV = pd.read_pickle(CB)
else:
    _r = {}
    for f in os.listdir(os.path.join(PROC, 'psd24k')):
        if not f.endswith('.pkl'):
            continue
        d = pd.read_pickle(os.path.join(PROC, 'psd24k', f)); kk = (d['freqs'] >= 1000) & (d['freqs'] < 8000)
        lv = 10 * np.log10((10 ** (d['blocos_db'][:, kk] / 10)).mean(1))
        _r[int(f[:-4])] = {'nível médio': 10 * np.log10((10 ** (lv / 10)).mean()), 'nível máximo': lv.max(), 'nível mínimo': lv.min(),
                           'faixa dinâmica (máx − mín)': lv.max() - lv.min(), 'faixa dinâmica (p90 − p10)': np.percentile(lv, 90) - np.percentile(lv, 10)}
    NV = pd.DataFrame(_r).T; NV.to_pickle(CB)
MEDIDAS = list(NV.columns)
NVn = NV.join(meta[['Ship ID', 'Dataset']], how='inner')
def por_navio(ids):
    x = NVn.loc[[i for i in ids if i in NVn.index]]
    return x.groupby('Ship ID').agg(**{m: (m, 'median') for m in MEDIDAS}, ds=('Dataset', lambda s: s.mode()[0]))
_rows = []
for a, b in PARES:
    A, B = por_navio(GRUPOS[a]), por_navio(GRUPOS[b])
    for m in MEDIDAS:
        U, p = mannwhitneyu(A[m], B[m])
        r = {'comparação': f'{a} × {b}', 'medida': m, 'mediana a': A[m].median(), 'mediana b': B[m].median(),
             'AUC (a > b)': U / (len(A) * len(B)), 'p': p, 'depende do ganho?': 'não' if 'dinâmica' in m else 'sim'}
        for ds in 'ACD':                                               # dentro do subconjunto (mesmas condições de seleção)
            x, y = A[A.ds == ds][m], B[B.ds == ds][m]
            r[f'AUC em {ds}'] = (mannwhitneyu(x, y).statistic / (len(x) * len(y))) if min(len(x), len(y)) >= 5 else np.nan
        _rows.append(r)
NIVEL_T = pd.DataFrame(_rows).set_index(['comparação', 'medida'])
display(NIVEL_T.round(3))

_n = por_navio(GRUPOS['SMALL'].union(GRUPOS['MEDIUM']).union(GRUPOS['LARGE'])).join(meta.drop_duplicates('Ship ID').set_index('Ship ID')[['Length']])
RHO_DIN = spearmanr(_n['faixa dinâmica (máx − mín)'], _n.Length, nan_policy='omit')
RHO_MED = spearmanr(_n['nível médio'], _n.Length, nan_policy='omit')
print(f"Por navio: ρ(faixa dinâmica, comprimento) = {RHO_DIN[0]:.3f} (p = {RHO_DIN[1]:.1e}); ρ(nível médio, comprimento) = {RHO_MED[0]:.3f} (p = {RHO_MED[1]:.1e})")
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for j, m in enumerate(('nível médio', 'nível mínimo', 'faixa dinâmica (máx − mín)')):
    dados = [por_navio(GRUPOS[g])[m] for g in ('Tug (SMALL)', 'OSS (MEDIUM)', 'SMALL', 'MEDIUM', 'LARGE')]
    ax[j].boxplot(dados, showfliers=False); ax[j].set_xticks(range(1, 6)); ax[j].set_xticklabels(['Tug', 'OSS', 'SMALL', 'MEDIUM', 'LARGE'])
    ax[j].set_title(m + (' (não depende do ganho)' if 'dinâmica' in m else ' (depende do ganho)'), fontsize=9); ax[j].set_ylabel('dB')
plt.tight_layout(); plt.show()
'''

MEL_CODE = r'''
# MEL do pipeline (src/iara/processing/analysis.py: log_melgram): 256 bandas mel até 8 kHz (decimação 3), em dB com
# librosa.power_to_db(ref=np.max), ou seja, relativo ao máximo da própria gravação. Duas versões processadas:
# sem normalização (log_melgram_d505…) e com normalização L2 por janela (log_melgram_a68e…, a entrada dos modelos MEL).
CM = os.path.join(PROC, 'cache_mel_espectro_medio.pkl')
if os.path.exists(CM):
    _m = pd.read_pickle(CM)
else:
    _m = {}
    for nome, d in (('sem normalização', 'log_melgram_d50517371dbc579e9074fcd15aa81e4f'), ('L2 (modelos)', 'log_melgram_a68edc75124f6c2ae6f093281fb42a9a')):
        _m[nome] = pd.DataFrame({int(f[:-4]): pd.read_pickle(os.path.join(PROC, d, f))['df'].values.mean(0)
                                 for f in os.listdir(os.path.join(PROC, d)) if f.endswith('.pkl')}).T.sort_index()
    pd.to_pickle(_m, CM)
import librosa
FM = librosa.mel_frequencies(n_mels=258, fmin=0, fmax=8000)[1:-1]          # centro aproximado das 256 bandas
MEL_REPS = {'MEL sem normalização (relativo ao máximo da gravação)': _m['sem normalização'],
            'MEL forma (sem a média da gravação)': _m['sem normalização'].sub(_m['sem normalização'].mean(1), axis=0),
            'MEL L2 (entrada dos modelos)': _m['L2 (modelos)']}

def por_subconjunto(X, pares, ncols):
    # Mann-Whitney por faixa (médias por navio) + FDR, juntando os subconjuntos e dentro de A, C e D
    rows = []
    for nome, Xr in X.items():
        for a, b in pares:
            for ds in (None, 'A', 'C', 'D'):
                ia = [i for i in GRUPOS[a] if ds is None or meta.loc[i, 'Dataset'] == ds]
                ib = [i for i in GRUPOS[b] if ds is None or meta.loc[i, 'Dataset'] == ds]
                A, B = nav(Xr, ia), nav(Xr, ib)
                cols = A.columns[A.notna().all() & B.notna().all()]
                if min(len(A), len(B)) < 5:
                    continue
                U, p = mannwhitneyu(A[cols].values, B[cols].values, axis=0); q = false_discovery_control(p)
                rows.append({'representação': nome, 'comparação': f'{a} × {b}', 'subconjunto': ds or 'todos',
                             'navios': f'{len(A)} × {len(B)}', 'faixas com q < 0,05': int((q < 0.05).sum()), 'de': len(cols)})
    return pd.DataFrame(rows).set_index(['representação', 'comparação', 'subconjunto'])
MEL_T = por_subconjunto(MEL_REPS, PARES, 256)
display(MEL_T.unstack('subconjunto')['faixas com q < 0,05'][['todos', 'A', 'C', 'D']])

# onde ficam as faixas da forma do MEL que diferem entre OSS e Tug juntando os subconjuntos
_X = MEL_REPS['MEL forma (sem a média da gravação)']
_A, _B = nav(_X, GRUPOS['OSS (MEDIUM)']), nav(_X, GRUPOS['Tug (SMALL)'])
_U, _p = mannwhitneyu(_A.values, _B.values, axis=0); _q = false_discovery_control(_p)
MEL_SIG = pd.DataFrame({'frequência (Hz)': FM, 'Δ OSS − Tug (dB)': _A.values.mean(0) - _B.values.mean(0), 'q': _q})[_q < 0.05]
print('Forma do MEL, OSS × Tug, todos os subconjuntos juntos: faixas com q < 0,05'); display(MEL_SIG.round(3))
'''

HHT_CODE = r'''
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
# HHT do projeto: sinal decimado para 2 kHz, janelas de 2,56 s a cada 5,12 s, EMD com 8 IMFs e energia instantânea
# acumulada por frequência (0–1 kHz), normalizada pelo pico da janela.
#  - HHT 1D (hht_features): espectro marginal de Hilbert, 256 faixas;
#  - HHT 2D (hht_2d_features): o mesmo com o tempo dentro da janela (32 passos de 80 ms × 128 faixas).
# Do HHT 2D tiramos uma medida que os espectros médios descartam: a INTERMITÊNCIA de cada faixa, o coeficiente de
# variação da energia ao longo dos passos de 80 ms (sem 2 passos em cada borda, por efeito de extremidade).
# Como cada janela é normalizada pelo seu pico, nenhuma dessas medidas depende do ganho.
CH = os.path.join(PROC, 'cache_hht_resumo.pkl')
if os.path.exists(CH):
    _h = pd.read_pickle(CH)
else:
    m1, m2, cv2 = {}, {}, {}
    for f in os.listdir(os.path.join(PROC, 'hht_features')):
        if f.endswith('.pkl'):
            d = pd.read_pickle(os.path.join(PROC, 'hht_features', f)); m1[int(f[:-4])] = np.log10(d['df'].values.mean(0) + 1e-6)
    for f in os.listdir(os.path.join(PROC, 'hht_2d_features')):
        if f.endswith('.pkl'):
            S2 = pd.read_pickle(os.path.join(PROC, 'hht_2d_features', f))['spectrograms'][:, 2:-2, :]
            m2[int(f[:-4])] = S2.mean((0, 1))
            mu, sd = S2.mean(1), S2.std(1)
            cv2[int(f[:-4])] = np.nanmean(np.where(mu > 1e-3, sd / (mu + 1e-9), np.nan), axis=0)
    _h = {'hht1d_log': pd.DataFrame(m1).T.sort_index(), 'hht2d_medio': pd.DataFrame(m2).T.sort_index(), 'hht2d_cv': pd.DataFrame(cv2).T.sort_index()}
    pd.to_pickle(_h, CH)
F2 = np.linspace(0, 1000, 129)[:-1] + 1000 / 256                  # centro das 128 faixas do HHT 2D
HHT_REPS = {'HHT 1D (espectro marginal, log)': _h['hht1d_log'],
            'HHT 1D forma (sem a média)': _h['hht1d_log'].sub(_h['hht1d_log'].mean(1), axis=0),
            'HHT 2D espectro médio': _h['hht2d_medio'],
            'HHT 2D intermitência (CV em passos de 80 ms)': _h['hht2d_cv']}
HHT_T = por_subconjunto(HHT_REPS, PARES, 0)
display(HHT_T.unstack('subconjunto')['faixas com q < 0,05'][['todos', 'A', 'C', 'D']])

# Intermitência OSS × Tug, faixa a faixa (só faixas com energia em todas as gravações)
CV = _h['hht2d_cv']
_A, _B = nav(CV, GRUPOS['OSS (MEDIUM)']), nav(CV, GRUPOS['Tug (SMALL)'])
OKF = (_A.notna().all() & _B.notna().all()).values
_U, _p = mannwhitneyu(_A.loc[:, OKF].values, _B.loc[:, OKF].values, axis=0); _q = false_discovery_control(_p)
CV_SIG = pd.DataFrame({'frequência (Hz)': F2[OKF], 'CV OSS': _A.loc[:, OKF].mean().values, 'CV Tug': _B.loc[:, OKF].mean().values,
                       'AUC (OSS > Tug)': _U / (len(_A) * len(_B)), 'q': _q})[_q < 0.05]
print(f'Intermitência, OSS × Tug: {len(CV_SIG)} de {OKF.sum()} faixas com q < 0,05'); display(CV_SIG.round(3))

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
for g, cor in (('OSS (MEDIUM)', 'tab:purple'), ('Tug (SMALL)', 'tab:blue'), ('LARGE', 'k')):
    ax[0].plot(F2[OKF], nav(CV, GRUPOS[g]).loc[:, OKF].mean().values, color=cor, label=g)
ax[0].set_xlabel('frequência (Hz)'); ax[0].set_ylabel('intermitência (CV)'); ax[0].legend(); ax[0].set_title('Intermitência média por navio')
ax[1].plot(F2[OKF], _U / (len(_A) * len(_B)), color='k'); ax[1].axhline(0.5, color='gray')
ax[1].scatter(CV_SIG['frequência (Hz)'], CV_SIG['AUC (OSS > Tug)'], color='tab:red', s=12, label='q < 0,05')
ax[1].set_ylim(0, 1); ax[1].set_xlabel('frequência (Hz)'); ax[1].set_ylabel('AUC (OSS > Tug)'); ax[1].legend()
ax[1].set_title('OSS × Tug, faixa a faixa (abaixo de 0,5: rebocador mais intermitente)')
plt.tight_layout(); plt.show()

# Por subconjunto, nas duas regiões que apareceram (escolhidas olhando os dados: só servem para ver o sentido)
_rows = []
for lo, hi in ((100, 200), (720, 830)):
    b = CV.loc[:, (F2 >= lo) & (F2 < hi)].mean(1)
    for a_, b_ in PARES[:2]:
        r = {'região': f'{lo}–{hi} Hz', 'comparação': f'{a_} × {b_}'}
        for ds in (None, 'A', 'B', 'C', 'D'):
            ia = [i for i in GRUPOS[a_] if i in b.index and (ds is None or meta.loc[i, 'Dataset'] == ds)]
            ib = [i for i in GRUPOS[b_] if i in b.index and (ds is None or meta.loc[i, 'Dataset'] == ds)]
            x, y = b.loc[ia].groupby(meta.loc[ia, 'Ship ID'].values).mean(), b.loc[ib].groupby(meta.loc[ib, 'Ship ID'].values).mean()
            if min(len(x), len(y)) >= 5:
                U, p = mannwhitneyu(x, y); r[f'AUC {ds or "todos"}'] = U / (len(x) * len(y)); r[f'p {ds or "todos"}'] = p
        _rows.append(r)
CV_DS = pd.DataFrame(_rows).set_index(['região', 'comparação'])
display(CV_DS.round(3))

# Validação em metades de navios: na metade de descoberta escolhe-se a faixa de menor p; na outra metade, que não
# participou da escolha, mede-se essa faixa. 200 sorteios.
_rng = np.random.default_rng(0); _res = []
_so = np.array(sorted(set(meta.loc[GRUPOS['OSS (MEDIUM)'], 'Ship ID']))); _st = np.array(sorted(set(meta.loc[GRUPOS['Tug (SMALL)'], 'Ship ID'])))
def _sel(ids, navios):
    return [i for i in ids if i in CV.index and meta.loc[i, 'Ship ID'] in navios]
for _ in range(200):
    ho, ht = _rng.permutation(_so), _rng.permutation(_st)
    do, vo = set(ho[:len(ho) // 2]), set(ho[len(ho) // 2:]); dt, vt = set(ht[:len(ht) // 2]), set(ht[len(ht) // 2:])
    Ad, Bd = nav(CV, _sel(GRUPOS['OSS (MEDIUM)'], do)).loc[:, OKF], nav(CV, _sel(GRUPOS['Tug (SMALL)'], dt)).loc[:, OKF]
    Ud, pd_ = mannwhitneyu(Ad.values, Bd.values, axis=0); j = int(np.argmin(pd_)); sinal = Ud[j] / (len(Ad) * len(Bd)) > 0.5
    Av, Bv = nav(CV, _sel(GRUPOS['OSS (MEDIUM)'], vo)).loc[:, OKF], nav(CV, _sel(GRUPOS['Tug (SMALL)'], vt)).loc[:, OKF]
    Uv, pv_ = mannwhitneyu(Av.values[:, j], Bv.values[:, j]); av = Uv / (len(Av) * len(Bv))
    _res.append({'faixa escolhida (Hz)': F2[OKF][j], 'AUC na validação': av, 'p na validação': pv_, 'mesmo sentido': (av > 0.5) == sinal})
VAL_CV = pd.DataFrame(_res)
print(f"Validação em metades (200 sorteios): mesmo sentido em {100 * VAL_CV['mesmo sentido'].mean():.0f}%, "
      f"p < 0,05 em {100 * (VAL_CV['p na validação'] < 0.05).mean():.0f}%, AUC mediana {VAL_CV['AUC na validação'].median():.2f}")
display(VAL_CV['faixa escolhida (Hz)'].round(0).value_counts().head(5).rename('vezes escolhida').to_frame())
'''

ENERGIA_CODE = r'''
# A intermitência mede dinâmica no tempo ou só energia relativa? O HHT 2D normaliza cada janela pelo seu pico: uma faixa
# fraca em relação ao componente dominante tem média pequena, e o coeficiente de variação (desvio ÷ média) cresce por
# causa do denominador. Comparamos a intermitência com a energia relativa da mesma faixa e com a banda LOFAR 31–86 Hz.
from scipy.stats import spearmanr
_S = pd.read_pickle(os.path.join(PROC, 'cache_lofar_espectro_medio.pkl')); _F = (np.arange(1024) + 1) * 16000 / 2048
MU = _h['hht2d_medio']
_ids = [i for i in meta.index if i in CV.index and i in _S.index]
_k = (F2 >= 100) & (F2 < 200)
cv_b, en_b = CV.loc[_ids, _k].mean(1), np.log(MU.loc[_ids, _k].mean(1))
banda = _S.loc[_ids, (_F >= 31) & (_F < 86)].mean(1)
_coef = np.polyfit(en_b, cv_b, 1); resid = cv_b - np.polyval(_coef, en_b)
REL = {'ρ(intermitência, energia relativa)': spearmanr(cv_b, en_b)[0],
       'ρ(banda LOFAR, intermitência)': spearmanr(banda, cv_b)[0],
       'ρ(banda LOFAR, energia relativa)': spearmanr(banda, en_b)[0],
       'ρ(banda LOFAR, resíduo da intermitência)': spearmanr(banda, resid)[0]}
print('100–200 Hz, todas as gravações de navio (Spearman):'); display(pd.Series(REL).round(2).to_frame('ρ'))
_rows = []
for nome, v in (('intermitência', cv_b), ('energia relativa (log)', en_b), ('intermitência sem o efeito da energia relativa (resíduo)', resid)):
    a = v.loc[[i for i in GRUPOS['OSS (MEDIUM)'] if i in v.index]]; b = v.loc[[i for i in GRUPOS['Tug (SMALL)'] if i in v.index]]
    a = a.groupby(meta.loc[a.index, 'Ship ID'].values).mean(); b = b.groupby(meta.loc[b.index, 'Ship ID'].values).mean()
    U, p = mannwhitneyu(a, b); _rows.append({'medida (100–200 Hz)': nome, 'AUC (OSS > Tug)': U / (len(a) * len(b)), 'p': p})
RES_EN = pd.DataFrame(_rows).set_index('medida (100–200 Hz)')
print('OSS × Tug, por navio:'); display(RES_EN.round(3))
'''


SMALL_CODE = r'''
# Seção 7: pista POSITIVA de SMALL no psd24k? Roda scratch/small_assinatura/psd24k_small.py (escreve os CSVs) e lê o resultado.
import runpy, contextlib, io
_cwd = os.getcwd(); os.chdir(R.ROOT)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path('scratch/small_assinatura/psd24k_small.py', run_name='__main__')
finally:
    os.chdir(_cwd)
_rep = os.path.join(R.ROOT, 'reports', 'small_medium', 'psd24k_small')
SM_X = pd.read_csv(_rep + '_medidas.csv', index_col=0)
SM_T = pd.read_csv(_rep + '_testes.csv').set_index('medida')
SM_PT = pd.read_csv(_rep + '_por_tipo.csv').pivot(index='medida', columns='tipo', values='AUC vs LARGE').loc[SM_T.index]
SM_PS = pd.read_csv(_rep + '_por_subconjunto.csv').pivot(index='medida', columns='subconjunto', values='AUC SMALL×LARGE').loc[SM_T.index]
print('Por navio, centrado no subconjunto; AUC = P(1º grupo > 2º). BACKGROUND (subconjunto E) sem centrar.')
display(SM_T[['SMALL med', 'MEDIUM med', 'LARGE med', 'AUC SMALL×LARGE', 'q S×L', 'AUC MEDIUM×LARGE',
              'AUC SMALL×BG (bruto)', 'q S×BG', 'AUC LARGE×BG (bruto)']].round(3))
print('AUC (tipo de SMALL > LARGE), por navio:'); display(SM_PT.round(2))
print('AUC SMALL × LARGE dentro de cada subconjunto:'); display(SM_PS.round(2))

# figura: forma e variação na passagem, mediana entre navios
_cor = {'SMALL': '#2a78d6', 'MEDIUM': '#eb6834', 'LARGE': '#1baf7a', 'BACKGROUND': '#8a8a8a'}
_fo = [c for c in SM_X.columns if c.startswith('forma')]; _di = [c for c in SM_X.columns if c.startswith('dinâmica')]
_cent = np.r_[[np.sqrt(31.25 * 62.5) * 2 ** k for k in range(9)], np.sqrt(16000 * 24000)]
_nav = SM_X['Ship ID'].astype(str).where(SM_X.classe != 'BACKGROUND', 'bg' + SM_X.index.astype(str))
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for c in ['SMALL', 'MEDIUM', 'LARGE', 'BACKGROUND']:
    sel = SM_X.classe == c
    for ax, cols in zip(axs, [_fo, _di]):
        g = SM_X.loc[sel, cols].groupby(_nav[sel]).median()
        ax.plot(_cent, g.median().values, '-o', ms=3, color=_cor[c], label=c)
        ax.fill_between(_cent, g.quantile(.25).values, g.quantile(.75).values, color=_cor[c], alpha=.1, lw=0)
for ax, t in zip(axs, ['forma: dB da oitava − média de 125 Hz–16 kHz', 'variação na passagem: p90 − p10 dos blocos de 10 s']):
    ax.set_xscale('log'); ax.set_xlabel('frequência (Hz, centro da oitava)'); ax.set_ylabel('dB'); ax.set_title(t); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
'''


def build():
    filename = os.path.join(NOTEBOOKS_DIR, "37_oss_tug_novas_representacoes.ipynb")
    ns = {}
    for src in (SETUP_CODE, PSD_CODE, DEMON_CODE, RAIAS_CODE, NIVEL_CODE, MEL_CODE, HHT_CODE, ENERGIA_CODE, SMALL_CODE):
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    FT, DT, RT, NT = ns["FORMA_T"], ns["DEMON_T"], ns["RAIAS_T"], ns["NIVEL_T"]
    RHO_DIN, RHO_MED = ns["RHO_DIN"], ns["RHO_MED"]
    ot, ms, lm = "OSS (MEDIUM) × Tug (SMALL)", "MEDIUM × SMALL", "LARGE × MEDIUM"
    dmin_q = int(DT["faixas com q < 0,05 (de 253)"].max())
    dmin_p = DT[[c for c in DT.columns if c.endswith(": p")]].min().min()
    n_testes_demon = DT[[c for c in DT.columns if c.endswith(": p")]].size
    rt = RT.loc[ot]

    def nl(par, m):
        r = NT.loc[(par, m)]
        sub = ", ".join(f"{ds}: {br(r[f'AUC em {ds}'], 2)}" for ds in "ACD" if r[f'AUC em {ds}'] == r[f'AUC em {ds}'])
        return (f"{br(r['mediana a'], 1)} × {br(r['mediana b'], 1)} dB (AUC {br(r['AUC (a > b)'], 2)}, {pv(r['p'])}; por subconjunto: {sub})")
    din_ms, din_ot = NT.loc[(ms, "faixa dinâmica (máx − mín)")], NT.loc[(ot, "faixa dinâmica (máx − mín)")]
    din_lm = NT.loc[(lm, "faixa dinâmica (máx − mín)")]
    abs_ok = NT.loc[(ot, "nível médio"), "p"] < 0.05
    min_ok = NT.loc[(ot, "nível mínimo"), "p"] < 0.05
    din_nada = din_ms["p"] >= 0.05 and din_ot["p"] >= 0.05

    MT, MSIG, HT, CVS, CVD, VAL = ns["MEL_T"], ns["MEL_SIG"], ns["HHT_T"], ns["CV_SIG"], ns["CV_DS"], ns["VAL_CV"]
    REL, RES_EN = ns["REL"], ns["RES_EN"]
    rr = RES_EN.loc["intermitência sem o efeito da energia relativa (resíduo)"]; re_ = RES_EN.loc["energia relativa (log)"]
    def nsub(T, rep, par):
        x = T.loc[(rep, par)]["faixas com q < 0,05"]
        return f"{int(x['todos'])} juntando os subconjuntos; " + ", ".join(f"{int(x[d])} em {d}" for d in "ACD" if d in x.index)
    mel_dentro_zero = all(MT.loc[(r, ot)]["faixas com q < 0,05"].drop("todos").sum() == 0 for r in ns["MEL_REPS"])
    hht_espec_zero = all(HT.loc[(r, ot)]["faixas com q < 0,05"].drop("todos").sum() == 0 for r in list(ns["HHT_REPS"])[:3])
    cv_rep = "HHT 2D intermitência (CV em passos de 80 ms)"
    n_cv_sig = len(CVS); n_cv_tot = int(ns["OKF"].sum())
    reg_txt = ", ".join(f"{br(CVS['frequência (Hz)'].min(), 0)}–{br(CVS[CVS['frequência (Hz)'] < 400]['frequência (Hz)'].max(), 0)} Hz" for _ in [0]) if n_cv_sig else ""
    alto = CVS[CVS['frequência (Hz)'] >= 400]
    if len(alto):
        reg_txt += f" e {br(alto['frequência (Hz)'].min(), 0)}–{br(alto['frequência (Hz)'].max(), 0)} Hz"
    sent = VAL["mesmo sentido"].mean() * 100; vsig = (VAL["p na validação"] < 0.05).mean() * 100
    cvd_ot = CVD.loc[("100–200 Hz", ot)]
    aucs_ds = {d: cvd_ot.get(f"AUC {d}") for d in "ABCD"}
    todos_abaixo = all(v < 0.5 for v in aucs_ds.values() if v == v and v is not None)
    cvd_ms = CVD.loc[("720–830 Hz", ms)]
    mel_sig_txt = ""
    if len(MSIG):
        lo_ = MSIG[MSIG['frequência (Hz)'] < 6400]; hi_ = MSIG[MSIG['frequência (Hz)'] >= 6400]
        mel_sig_txt = (f" As faixas da forma que diferem juntando os subconjuntos ficam em {', '.join(br(f, 0) for f in lo_['frequência (Hz)'])} Hz"
                       f"{(' e em ' + br(hi_['frequência (Hz)'].min(), 0) + '–' + br(hi_['frequência (Hz)'].max(), 0) + ' Hz, acima do corte do filtro de decimação (6,4 kHz), onde o sinal chega muito atenuado e perto do piso de −80 dB do MEL') if len(hi_) else ''}.")

    SM_T, SM_PT, SM_PS = ns["SM_T"], ns["SM_PT"], ns["SM_PS"]
    dsl = (SM_T["AUC SMALL×LARGE"] - SM_T["AUC MEDIUM×LARGE"]).abs()
    n_sig_sl = int((SM_T["q S×L"] < 0.05).sum()); n_med = len(SM_T)
    sl_pos = SM_T[(SM_T["AUC SMALL×LARGE"] > 0.5) & (SM_T["q S×L"] < 0.05)]
    sl_neg = SM_T[(SM_T["AUC SMALL×LARGE"] < 0.5) & (SM_T["q S×L"] < 0.05)]
    inc, f48, g31, d12 = ("inclinação 125 Hz–8 kHz (dB/oit)", "forma 4000–8000 Hz", "forma 31.25–62 Hz", "dinâmica 1000–2000 Hz")
    sub_ok = lambda v: ", ".join(d for d in "ABCD" if abs(SM_PS.loc[v, d] - 0.5) >= 0.08)  # noqa: E731
    tipo_txt = lambda v: "; ".join(f"{t} {br(SM_PT.loc[v, t], 2)}" for t in ["Pilot Vessel", "Tug", "Fishing Vessel"])  # noqa: E731
    bg_min, bg_max = SM_T["AUC SMALL×BG (bruto)"].min(), SM_T["AUC SMALL×BG (bruto)"].max()
    entre = ((SM_T["AUC SMALL×BG (bruto)"] - 0.5) * (SM_T["AUC LARGE×BG (bruto)"] - 0.5) >= 0).mean() * 100

    cells = [md("""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 37: Offshore Supply Ships × Rebocadores em Representações Novas

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

**Pergunta.** No IARA, a fronteira SMALL/MEDIUM é a mais difícil: com split por navio, nenhum modelo reconhece MEDIUM acima do acaso. O tipo mais comum de MEDIUM, o **Offshore Supply Ship (OSS)**, não difere dos **rebocadores** (SMALL) nem no LOFAR (NB35) nem no DEMON (NB36). Alguma representação que os modelos **não** usam os separa?

**Representações testadas** (extrações novas, com nomes próprios):
- **psd24k**: espectro de Welch do áudio original de 48 kHz, de 0 a 24 kHz, sem decimação (o LOFAR para em 8 kHz). Usamos a **forma** do espectro (dB menos a média da própria gravação).
- **demon_multibanda**: DEMON em cinco bandas de cavitação (0,5–1, 1–2, 2–4, 4–8 e 8–16 kHz); o DEMON do projeto usa só 1–2 kHz.
- **lofar1hz**: posição das raias estreitas persistentes (faixas de 1 Hz), uma "impressão digital" das máquinas.
- **Nível**, a partir dos blocos de 10 s do psd24k: absoluto e medidas que não dependem do ganho do equipamento.
- **MEL** do pipeline (a entrada dos modelos MEL) e **HHT** do projeto (espectro marginal e, do HHT 2D, a intermitência da energia em passos de 80 ms).

**Unidade de análise:** o navio (média das suas gravações), para que navios com muitas gravações não pesem mais. Testes de Mann-Whitney com FDR de Benjamini-Hochberg quando há muitas faixas. Para comparação, MEDIUM × SMALL (classes inteiras) e LARGE × MEDIUM.
"""), code(SETUP_CODE),
        md("""
## 1. Forma do espectro de 0 a 24 kHz (psd24k)
"""), code(PSD_CODE),
        md(f"""
**Leitura.** OSS × Tug: {int(FT.loc[ot, 'faixas com q < 0,05'])} de {int(FT.loc[ot, 'de'])} faixas com q < 0,05; MEDIUM × SMALL: {int(FT.loc[ms, 'faixas com q < 0,05'])}; LARGE × MEDIUM: {int(FT.loc[lm, 'faixas com q < 0,05'])}. {('**Nem a alta frequência (8–24 kHz), que nenhuma representação do projeto usa, separa os OSS dos rebocadores ou MEDIUM de SMALL.**' if FT.loc[ot, 'faixas com q < 0,05'] == 0 and FT.loc[ms, 'faixas com q < 0,05'] == 0 else 'Há faixas com diferença; ver a tabela.')}
"""),
        md("""
## 2. DEMON em cinco bandas (demon_multibanda)

Em cada banda: o espectro DEMON relativo faixa a faixa, e o pico dominante de cada gravação (frequência, altura e fração de picos fortes, com pelo menos 10 dB acima do fundo), por navio.
"""), code(DEMON_CODE),
        md(f"""
**Leitura.** No máximo {dmin_q} {'faixa' if dmin_q == 1 else 'faixas'} com q < 0,05 em qualquer banda e comparação. Nas medidas do pico, o menor p é {br(dmin_p, 3)} entre {n_testes_demon} testes, o que se espera ao acaso. **O DEMON em outras bandas também não separa os grupos.**
"""),
        md("""
## 3. Raias estreitas persistentes (lofar1hz)
"""), code(RAIAS_CODE),
        md(f"""
**Leitura.** OSS × Tug: {int(rt['faixas com q < 0,05'])} faixas com q < 0,05. A faixa mais diferente é {br(rt['em (Hz)'], 1)} Hz (p = {br(rt['menor p (sem correção)'], 4)} sem correção, q = {br(rt['q dessa faixa'], 2)}): os rebocadores têm raias persistentes em ~64–80 Hz com mais frequência que os OSS. Com ~900 faixas testadas, isso pode ser acaso; fica como **indício a confirmar** em outros navios, não como resultado.
"""),
        md("""
## 4. Nível: absoluto e independente do ganho

O IARA grava em 16 bits **sem calibração**: o artigo do dataset informa sensibilidades de hidrofone de −200 a −150 dB re 1 V/µPa no projeto, e nem o `iara.csv` nem o `iara.xlsx` dizem qual hidrofone gravou cada arquivo. A distância só é conhecida por subconjunto (região de inclusão até 250 m em A e B, e de 250 a 500 m em C e D). Por isso separamos:
- medidas que **dependem do ganho** (nível médio, máximo e mínimo, em unidades do arquivo);
- medidas que **não dependem** (faixa dinâmica da passagem, diferença em dB dentro da mesma gravação).
"""), code(NIVEL_CODE),
        md(f"""
**Leitura.**
- **Nível médio** (depende do ganho): OSS × Tug {nl(ot, 'nível médio')}; MEDIUM × SMALL {nl(ms, 'nível médio')}.
- **Nível mínimo**, o trecho mais silencioso da gravação (depende do ganho e do ruído ambiente local): OSS × Tug {nl(ot, 'nível mínimo')}; MEDIUM × SMALL {nl(ms, 'nível mínimo')}.
- **Faixa dinâmica da passagem** (não depende do ganho): OSS × Tug {nl(ot, 'faixa dinâmica (máx − mín)')}; MEDIUM × SMALL {nl(ms, 'faixa dinâmica (máx − mín)')}; LARGE × MEDIUM {nl(lm, 'faixa dinâmica (máx − mín)')}.
- **Leitura conjunta:** {('os OSS e as MEDIUM chegam ao arquivo mais altos que os rebocadores e as SMALL, mas a diferença aparece igual no **trecho mais silencioso** da gravação, que reflete sobretudo o equipamento e o ruído do local, não o navio; e não se sustenta na faixa dinâmica, que não depende do ganho (não significativa e com sentido diferente entre subconjuntos). A explicação mais provável é **instrumental ou de local**: esses tipos de navio teriam sido gravados mais vezes com outro hidrofone, ganho ou ponto de instalação. Sem a calibração e o hidrofone de cada gravação, o nível absoluto não pode ser usado como pista de porte.' if abs_ok and min_ok and din_nada else 'ver a tabela: o padrão de nível não é o esperado para um efeito puramente instrumental.')}
- **A faixa dinâmica separa as LARGE** ({br(din_lm['mediana a'], 1)} × {br(din_lm['mediana b'], 1)} dB contra as MEDIUM, AUC {br(din_lm['AUC (a > b)'], 2)}) e cresce com o comprimento (ρ = {br(RHO_DIN[0], 2)}, {pv(RHO_DIN[1])}): navios grandes variam mais de nível durante a passagem. O nível médio acompanha o comprimento só fracamente (ρ = {br(RHO_MED[0], 2)}).
"""),
        md("""
## 5. MEL do pipeline

O MEL é o mesmo espectro de potência agrupado em bandas mel (mais finas no grave e mais largas no agudo). Não pode conter mais informação que o psd24k, mas é a representação que os modelos MEL usam. Número de faixas com q < 0,05, juntando os subconjuntos e dentro de A, C e D:
"""), code(MEL_CODE),
        md(f"""
**Leitura.**
- OSS × Tug: MEL sem normalização, {nsub(MT, 'MEL sem normalização (relativo ao máximo da gravação)', ot)}; forma, {nsub(MT, 'MEL forma (sem a média da gravação)', ot)}; **L2 (entrada dos modelos), {nsub(MT, 'MEL L2 (entrada dos modelos)', ot)}**.
- O MEL "sem normalização" do pipeline não é nível absoluto: é convertido em dB com referência ao máximo da própria gravação (`power_to_db(ref=np.max)`).{mel_sig_txt}
- {('**Dentro de cada subconjunto, nenhuma versão do MEL separa OSS de rebocadores.** O que aparece juntando os subconjuntos vem da mistura de condições de gravação, como no nível (Seção 4).' if mel_dentro_zero else 'Há diferenças dentro de algum subconjunto; ver a tabela.')}
"""),
        md("""
## 6. HHT do projeto: espectro e intermitência

A HHT decompõe o sinal em modos intrínsecos (EMD) e mede a energia instantânea em cada frequência. Além do espectro marginal, o HHT 2D guarda como a energia **varia dentro da janela**, em passos de 80 ms. Daí tiramos a **intermitência** de cada faixa (coeficiente de variação ao longo do tempo), uma informação que os espectros médios descartam e que não depende do ganho.
"""), code(HHT_CODE),
        md(f"""
**Leitura.**
- **Espectro marginal (1D, inclusive a forma) e espectro médio do HHT 2D:** {('as diferenças OSS × Tug aparecem só juntando os subconjuntos e somem dentro de cada um (0 faixas em A, C e D), como no MEL.' if hht_espec_zero else 'ver a tabela.')}
- **Intermitência:** {n_cv_sig} de {n_cv_tot} faixas com q < 0,05 entre OSS e Tug, em {reg_txt}: os rebocadores têm intermitência maior. A Seção 6.1 mostra o que isso significa de fato.
- **Por subconjunto** (região de 100–200 Hz): AUC (OSS > Tug) de {', '.join(f'{br(v, 2)} em {d}' for d, v in aucs_ds.items() if v == v and v is not None)}. {('O sentido é o mesmo nos quatro subconjuntos, mais forte em B e C.' if todos_abaixo else 'O sentido muda entre subconjuntos.')} As regiões foram escolhidas olhando os dados, então esta tabela só mostra a consistência do sentido.
- **Validação em metades de navios** (faixa escolhida numa metade, medida na outra): mesmo sentido em {br(sent, 0)}% dos 200 sorteios, mas p < 0,05 em só {br(vsig, 0)}% (AUC mediana {br(VAL['AUC na validação'].median(), 2)}). O sentido se confirma; a força, com ~23 OSS e ~17 rebocadores por metade, é limitada.
- **MEDIUM × SMALL:** mesmo sentido, mais fraco (720–830 Hz: AUC {br(cvd_ms['AUC todos'], 2)}, {pv(cvd_ms['p todos'])}).
"""),
        md("""
### 6.1 Intermitência ou energia relativa?

O HHT 2D normaliza cada janela pelo seu pico. Numa faixa fraca em relação ao componente dominante, a média é pequena e o coeficiente de variação cresce só por causa do denominador. Se for isso, a intermitência mede **quanto a faixa é fraca em relação ao componente mais forte**, e não quanto a energia oscila no tempo:
"""), code(ENERGIA_CODE),
        md(f"""
**Leitura.**
- Em 100–200 Hz, a intermitência e a energia relativa da mesma faixa têm ρ = {br(REL['ρ(intermitência, energia relativa)'], 2)}: **a intermitência é, quase toda, a energia relativa com o sinal trocado**. A banda LOFAR de 31–86 Hz se relaciona com as duas (ρ = {br(REL['ρ(banda LOFAR, intermitência)'], 2)} com a intermitência e {br(REL['ρ(banda LOFAR, energia relativa)'], 2)} com a energia relativa): quando a banda grave domina, as outras faixas ficam relativamente fracas e a "intermitência" sobe.
- **OSS × Tug:** a energia relativa em 100–200 Hz separa tanto quanto a intermitência (AUC {br(re_['AUC (OSS > Tug)'], 2)}, {pv(re_['p'])}); o que sobra da intermitência depois de descontar a energia relativa é fraco (AUC {br(rr['AUC (OSS > Tug)'], 2)}, {pv(rr['p'])}, um teste entre muitos).
- **Correção da leitura anterior:** a diferença não é que os rebocadores "oscilem mais no tempo". É que os OSS têm **relativamente mais energia em 100–200 Hz** em relação ao componente dominante; coerente com o pico dominante deles cair mais vezes em 86–150 Hz. É uma diferença de **balanço espectral**, fraca e exploratória (testamos muitas representações até achá-la).
"""),
        md("""
## 7. Existe uma pista positiva de SMALL? SMALL × LARGE e SMALL × BACKGROUND (psd24k)

As seções anteriores compararam SMALL com MEDIUM. A pergunta agora é outra: há algo no espectro cru que **só o SMALL tem**, em comparação com LARGE e BACKGROUND? O LOFAR usa o TPSW, que achata ruído largo e contínuo (por exemplo, o chiado de hélice rápida); o psd24k não tem esse achatamento.

**Medidas** (todas independentes do ganho, já que o nível do IARA não é calibrado), em `scratch/small_assinatura/psd24k_small.py`:
- **forma:** média em dB de cada oitava (31 Hz–24 kHz) menos a média da gravação em 125 Hz–16 kHz;
- **inclinação** (dB por oitava) em 125 Hz–8 kHz e em 1–16 kHz;
- **variação na passagem:** p90 − p10 dos blocos de 10 s, por oitava.

**Unidade:** o navio. Cada medida é centrada na mediana do seu subconjunto (A–D, remove a resposta do hidrofone/local) e resumida pela mediana do navio; Mann-Whitney com FDR. O BACKGROUND está todo no subconjunto E: a comparação com ele usa valores **não centrados** e mistura classe com equipamento.
"""), code(SMALL_CODE),
        md(f"""
**Leitura.**
- **SMALL e MEDIUM são iguais contra LARGE em todas as {n_med} medidas:** a AUC SMALL × LARGE e a AUC MEDIUM × LARGE diferem no máximo {br(dsl.max(), 2)}. O que separa o SMALL do LARGE é o mesmo que separa o MEDIUM: é diferença de "navio < 100 m", não de SMALL.
- **SMALL × LARGE:** {n_sig_sl} de {n_med} medidas com q < 0,05, {len(sl_neg)} no sentido "o LARGE tem mais" e {len(sl_pos)} no sentido "o SMALL tem mais". O LARGE tem mais grave relativo (31–62 Hz: AUC {br(SM_T.loc[g31, 'AUC SMALL×LARGE'], 2)}), espectro mais inclinado e muito mais variação ao longo da passagem (1–2 kHz: AUC {br(SM_T.loc[d12, 'AUC SMALL×LARGE'], 2)}).
- **A única pista no sentido "SMALL tem mais"** é um espectro mais plano, com relativamente mais energia acima de 4 kHz (inclinação: AUC {br(SM_T.loc[inc, 'AUC SMALL×LARGE'], 2)}; 4–8 kHz: AUC {br(SM_T.loc[f48, 'AUC SMALL×LARGE'], 2)}). É fraca e não se repete em todos os subconjuntos (afastada de 0,5 em {sub_ok(inc)}) nem em todos os tipos (inclinação, por tipo: {tipo_txt(inc)}).
- **SMALL × BACKGROUND:** AUCs de {br(bg_min, 2)} a {br(bg_max, 2)}; em {br(entre, 0)}% das medidas o SMALL fica do mesmo lado do BACKGROUND que o LARGE, só que mais perto: o SMALL é "um navio, só que mais fraco". A variação na passagem separa navio de fundo (AUC ~0,95), o que só diz que há uma fonte passando.
- **Leitura conjunta:** no espectro cru, tudo se resume a um eixo: **o quanto a embarcação se destaca do fundo**. O SMALL não tem uma "cor" espectral própria; tem menos do que o LARGE tem. Sem a distância de cada gravação, "se destaca pouco" pode ser barco pequeno ou barco longe.
"""),
        md(f"""
## 8. Conclusões

1. **Nas representações espectrais, nada separa Offshore Supply Ships de rebocadores**, nem MEDIUM de SMALL: a forma do espectro até 24 kHz, o DEMON em cinco bandas, a posição das raias estreitas, o MEL e o espectro do HHT dão 0 faixas significativas dentro de cada subconjunto. O indício de raias de rebocador em ~64–80 Hz não sobrevive à correção para múltiplos testes.
2. **O nível absoluto difere, mas não serve:** a diferença aparece também no trecho mais silencioso de cada gravação e não se sustenta na medida que não depende do ganho. Sem calibração nem identificação do hidrofone, ela é mais provavelmente instrumental ou de local do que acústica.
3. **A "intermitência" do HHT 2D separa os dois tipos, mas mede sobretudo energia relativa** (Seção 6.1): os OSS têm relativamente mais energia em 100–200 Hz em relação ao componente dominante. O sentido se repete em todos os subconjuntos e nas metades de navios, mas o efeito é moderado, e o que sobra de dinâmica temporal é marginal. O NB38 testa se essa pista ajuda um classificador a generalizar para navios novos.
4. **Somando LOFAR (NB35), DEMON (NB36) e este notebook:** no conteúdo espectral, um supply de 50–90 m soa como um rebocador de 20–35 m. A fronteira SMALL/MEDIUM do IARA, definida por comprimento, quase não tem correlato acústico nestes dados, coerente com a física (propulsão parecida) e com o fato de nenhum modelo a generalizar para navios novos.
5. **O que distingue as LARGE** aparece em várias representações independentes: banda de 31–86 Hz (LOFAR), modulação forte e grave (DEMON) e maior faixa dinâmica na passagem (psd24k).
6. **Não há pista positiva de SMALL no espectro cru** (Seção 7): contra LARGE, o SMALL só tem "menos do que o LARGE tem" (menos grave relativo, espectro menos inclinado, menos variação na passagem), exatamente como o MEDIUM. A única pista própria, um espectro um pouco mais plano acima de 4 kHz, é fraca e não se repete em todos os subconjuntos e tipos.

**Ressalvas:** 46 OSS e 34 rebocadores dão poder estatístico para diferenças moderadas, não para diferenças pequenas; e "não detectável nestes dados" não é o mesmo que "não existe". Calibração e distância de cada gravação permitiriam usar o nível de fonte, que é a medida física mais direta de porte.
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
    build()
