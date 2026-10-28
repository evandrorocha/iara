"""
Gera notebooks/32_estudo_small.ipynb: estudo da classe SMALL (navios < 50 m).

Perguntas:
  1. Quantos dos 16 modelos (RF, SVM, MLP, BiGRU × MEL, LOFAR, HHT, DEMON) erram cada gravação SMALL?
  2. Quais gravações quase todos erram (núcleo)? São gravações isoladas ou navios inteiros?
  3. Ter o mesmo navio no treino muda o erro (split oficial por gravação)? E no split por navio (NB29)?
  4. O núcleo difere em metadados (subconjunto, tipo, comprimento, CPA, clima) ou em medidas acústicas
     (ganho no CPA, faixa dinâmica, pico fora do CPA, bandas)?
  5. Para que classe vão os erros?

Fontes: predições por gravação (iara_results.py), src/iara/dataset_info/iara.csv, splits do 5x2cv oficial
(<exp>/eval/fold_k/<exp>_split.csv), reports/small/features_acusticas.csv (scratch/small_acoustic_features.py)
e os experimentos split_{id,navio}_* do NB29.
Como nos outros geradores, o mesmo código das células roda aqui para calcular os números citados no markdown.
Rodar na raiz do repositório:  python notebooks/src/generate_estudo_small_notebook.py
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
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, spearmanr
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
CLS = dict(enumerate(R.CLASSES))
SMALL = 0

# 16 modelos: 4 famílias × 4 entradas (os mesmos dos Notebooks 30 e 31)
FAM = {
    'RF': {'MEL': 'rf_mel5s_n100_balanced_4c', 'LOFAR': 'rf_lofar5s_n100_balanced_4c',
           'HHT': 'rf_hht_n100_balanced_4c', 'DEMON': 'rf_demon_n100_balanced_4c'},
    'SVM': {'MEL': 'svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0',
            'LOFAR': 'svm_nystroem_6000_lofar_elasticnet_l1r0.15',
            'HHT': 'svm_nystroem_6000_hht_C2.0_elasticnet_l1r0.15_4c',
            'DEMON': 'svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000_4c'},
    'MLP': {'MEL': 'mlp', 'LOFAR': 'mlp_lofar', 'HHT': 'mlp_hht_h32-16_e30_4c', 'DEMON': 'mlp_demon_h32-16_e30_4c'},
    'BiGRU': {'MEL': 'bigru_att_mel5s_h64_s15-5_e20_4c', 'LOFAR': 'bigru_att_lofar5s_h128_s15-5_e15_4c',
              'HHT': 'bigru_att_hht_h64_s15-5_e20_4c', 'DEMON': 'bigru_att_demon_h64_s15-5_e20_4c'},
}
MOD = {f'{f} {r}': e for f, reps in FAM.items() for r, e in reps.items()}
NM = len(MOD)

meta = pd.read_csv(os.path.join(R.ROOT, 'src', 'iara', 'dataset_info', 'iara.csv'), dtype={'Ship ID': str, 'ID': str})
meta['Ship ID'] = meta['Ship ID'].str.strip()
for c in ('Length', 'CPA time', 'Recording time', 'Rain', 'Wind', 'Sea state'):
    meta[c] = pd.to_numeric(meta[c], errors='coerce')
meta['Arquivo'] = meta.Dataset + '-' + meta.ID + '.wav'
meta = meta.set_index(meta.ID.astype(int))

# Predições alinhadas por fold (só gravações presentes nos 16 modelos) + navio no treino do fold
AL = R.aligned(list(MOD.values()))
rows = []
for k, d in AL.items():
    sp = pd.read_csv(os.path.join(R.exp_dir(FAM['RF']['LOFAR']), 'eval', f'fold_{k}', f"{FAM['RF']['LOFAR']}_split.csv"))
    trn_ships = set(meta.loc[sp[sp.Subset == 'trn'].File.astype(int), 'Ship ID'])
    P = np.column_stack([d[e].values for e in MOD.values()])
    E = P != d.Target.values[:, None]
    x = pd.DataFrame(E.astype(int), columns=list(MOD))
    x['fold'], x['File'], x['Target'], x['n_err'] = k, d.File.values, d.Target.values, E.sum(1)
    for c in range(4):
        x[f'pred {CLS[c]}'] = (P == c).sum(1)
    x['navio no treino'] = [s in trn_ships for s in meta.loc[d.File, 'Ship ID']]
    rows.append(x)
I = pd.concat(rows, ignore_index=True)

# Uma linha por gravação: média sobre os folds em que ela foi teste (5, uma vez por repetição)
G = I.groupby('File').agg(Target=('Target', 'first'), n_err=('n_err', 'mean'), vezes=('fold', 'size'),
                          **{f'pred {CLS[c]}': (f'pred {CLS[c]}', 'mean') for c in range(4)},
                          **{m: (m, 'mean') for m in MOD})
G['Classe'] = G.Target.map(CLS)
G['grupo'] = np.select([G.n_err >= 15, G.n_err <= 4], ['núcleo', 'fácil'], 'meio')
G = G.join(meta[['Arquivo', 'Dataset', 'Ship ID', 'Name', 'DETAILED TYPE', 'Length', 'CPA time', 'Recording time',
                 'Rain', 'Wind', 'Sea state']], on='File')

FEAT_CSV = os.path.join(R.ROOT, 'reports', 'small', 'features_acusticas.csv')
FEAT = pd.read_csv(FEAT_CSV).set_index('File') if os.path.exists(FEAT_CSV) else None
if FEAT is not None:
    G = G.join(FEAT, on='File')
S = G[G.Target == SMALL].copy()
print(f'{len(G)} gravações de teste; {len(S)} SMALL; medidas acústicas: {FEAT is not None}')
'''

DIST_CODE = r'''
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
bins = np.arange(-0.5, NM + 1.5, 1)
for c in range(4):
    g = G[G.Target == c]
    ax[0].hist(g.n_err, bins=bins, histtype='step', lw=2, density=True, label=f'{CLS[c]} (n = {len(g)})')
ax[0].set_xlabel(f'nº médio de modelos que erram (de {NM})'); ax[0].set_ylabel('fração das gravações'); ax[0].legend()
ax[0].set_title('Quantos modelos erram cada gravação')
gr = G.groupby(['Classe', 'grupo']).size().unstack().reindex(R.CLASSES)[['fácil', 'meio', 'núcleo']]
(gr.div(gr.sum(1), axis=0) * 100).plot.barh(stacked=True, ax=ax[1], color=['tab:green', 'tab:gray', 'tab:red'])
ax[1].set_xlabel('% das gravações da classe'); ax[1].set_title(f'fácil: ≤ 4/{NM} erram · núcleo: ≥ 15/{NM} erram')
plt.tight_layout(); plt.show()
GR = gr.assign(total=gr.sum(1))
display(GR)

# Recall SMALL por modelo (por gravação, média dos folds)
REC = pd.DataFrame({f: {r: 100 * (1 - S[f'{f} {r}'].mean()) for r in FAM[f]} for f in FAM}).T
REC['oráculo das 4 entradas'] = [100 * np.mean([1 - (d[list(FAM[f].values())].values != 0).all(1).mean()
                                               for d in [x[x.Target == SMALL] for x in AL.values()]]) for f in FAM]
print('Recall SMALL (%) por modelo:'); display(REC.round(1))
'''

NUCLEO_CODE = r'''
N = S[S.grupo == 'núcleo'].sort_values('n_err', ascending=False)
cols = ['Arquivo', 'Dataset', 'Ship ID', 'Name', 'DETAILED TYPE', 'Length', 'n_err', 'vezes',
        'pred SMALL', 'pred MEDIUM', 'pred LARGE', 'pred BACKGROUND']
LISTA = N[cols].rename(columns={'n_err': f'erram (de {NM})', 'Length': 'comprimento (m)', 'DETAILED TYPE': 'tipo',
                                'vezes': 'folds como teste'})
print(f'{len(N)} gravações SMALL no núcleo (≥ 15 dos {NM} modelos erram, média dos folds):')
display(LISTA.round(1))
LISTA.to_csv(os.path.join(R.ROOT, 'reports', 'small', 'nucleo_small.csv'))

# Gravação ou navio? Para cada navio com gravação no núcleo: as outras gravações dele também falham?
nav = S.groupby('Ship ID').agg(nome=('Name', 'first'), tipo=('DETAILED TYPE', 'first'), comprimento=('Length', 'first'),
                               gravações=('n_err', 'size'), no_núcleo=('grupo', lambda s: (s == 'núcleo').sum()),
                               erram_média=('n_err', 'mean'), erram_min=('n_err', 'min'), erram_max=('n_err', 'max'))
NAV = nav[nav.no_núcleo > 0].sort_values(['no_núcleo', 'erram_média'], ascending=False)
print('Navios com gravação no núcleo:')
display(NAV.round(1))
'''

VISTO_CODE = r'''
# Split oficial (por gravação): o navio da gravação de teste tem gravação no treino do mesmo fold?
Is = I[I.Target == SMALL]
VIS = pd.DataFrame({m: Is.groupby('navio no treino')[m].mean() * 100 for m in MOD}).T
VIS.columns = ['erro (%) navio NÃO visto', 'erro (%) navio visto']
VIS['Δ (visto − não visto)'] = VIS['erro (%) navio visto'] - VIS['erro (%) navio NÃO visto']
n_vis = Is.groupby('navio no treino').size()
print(f'Gravações SMALL de teste (somando os 10 folds): navio visto {n_vis.get(True, 0)}, não visto {n_vis.get(False, 0)}')
display(VIS.round(1))
NUC_VIS = 100 * Is.assign(nuc=Is.n_err >= 15).groupby('navio no treino').nuc.mean()
print('% das gravações SMALL de teste com ≥ 15 modelos errando:', NUC_VIS.round(1).to_dict())

# Split por navio (NB29): recall SMALL com controle por ID × por navio
SPL = []
for lab, tag in [('RF', 'rf'), ('SVM', 'svm2000'), ('MLP', 'mlp')]:
    for r in ('mel', 'lofar', 'hht'):
        try:
            a, b = R.fold_metrics(f'split_id_{tag}_{r}_4c'), R.fold_metrics(f'split_navio_{tag}_{r}_4c')
        except FileNotFoundError:
            continue
        SPL.append({'modelo': f'{lab} {r.upper()}', 'recall SMALL por ID': a.R_SMALL.mean(),
                    'recall SMALL por navio': b.R_SMALL.mean(), 'Δ': b.R_SMALL.mean() - a.R_SMALL.mean()})
SPL = pd.DataFrame(SPL).set_index('modelo')
display(SPL.round(1))
'''

META_CODE = r'''
ORD = ['fácil', 'meio', 'núcleo']
num = ['Length', 'CPA time', 'Recording time', 'Wind', 'Sea state', 'Rain']
MT = S.groupby('grupo')[num].median().reindex(ORD)
LEN_MEAN = S.groupby('grupo').Length.mean().reindex(ORD)
MT.loc['p (Kruskal-Wallis)'] = [kruskal(*[S.loc[S.grupo == g, c].dropna() for g in ORD]).pvalue for c in num]
display(MT.round(3))

DS = pd.crosstab(S.grupo, S.Dataset).reindex(ORD)
p_ds = chi2_contingency(DS.values)[1]
print(f'Subconjunto × grupo (χ² {"p < 0,001" if p_ds < 0.001 else f"p = {p_ds:.3f}"}):'); display(DS)
DSp = S.groupby('Dataset').agg(gravações=('n_err', 'size'), erram_média=('n_err', 'mean'),
                               núcleo=('grupo', lambda s: (s == 'núcleo').sum()))
DSp['% no núcleo'] = 100 * DSp.núcleo / DSp.gravações
display(DSp.round(1))

TP = S.groupby('DETAILED TYPE').agg(gravações=('n_err', 'size'), navios=('Ship ID', 'nunique'),
                                    erram_média=('n_err', 'mean'), núcleo=('grupo', lambda s: (s == 'núcleo').sum()))
display(TP[TP.gravações >= 5].sort_values('gravações', ascending=False).round(1))
'''

ACUST_CODE = r'''
FCOLS = ['ganho no CPA (dB)', 'faixa dinâmica (dB)', '|pico − CPA| (s)', 'nível p50 (dBFS)', 'centroide (Hz)',
         'planura 0–5 kHz', 'energia 0-200 Hz (%)', 'energia 200-1000 Hz (%)', 'energia 1000-5000 Hz (%)',
         'energia 5000-24000 Hz (%)']
if FEAT is None:
    print('Rode scratch/small_acoustic_features.py para gerar', FEAT_CSV)
    AC = AC_CLS = None
else:
    AC = S.groupby('grupo')[FCOLS].median().reindex(ORD)
    AC.loc['p (Kruskal-Wallis)'] = [kruskal(*[S.loc[S.grupo == g, c].dropna() for g in ORD]).pvalue for c in FCOLS]
    AC.loc['ρ Spearman com nº de erros'] = [spearmanr(S[c], S.n_err, nan_policy='omit')[0] for c in FCOLS]
    AC.loc['p (Spearman)'] = [spearmanr(S[c], S.n_err, nan_policy='omit')[1] for c in FCOLS]
    print('SMALL: medianas por grupo'); display(AC.round(3))
    AC_CLS = G.groupby('Classe')[FCOLS].median().reindex(R.CLASSES)
    print('Todas as classes: medianas'); display(AC_CLS.round(2))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, c in zip(ax, ['ganho no CPA (dB)', 'faixa dinâmica (dB)', '|pico − CPA| (s)']):
        data = [S.loc[S.grupo == g, c].dropna() for g in ORD] + [G.loc[G.Target == k, c].dropna() for k in (1, 2, 3)]
        a.boxplot(data, showfliers=False)
        a.set_xticks(range(1, 7), ['SMALL\nfácil', 'SMALL\nmeio', 'SMALL\nnúcleo', 'MEDIUM', 'LARGE', 'BACKG.'])
        a.set_title(c)
    plt.tight_layout(); plt.show()
'''

CONF_CODE = r'''
# Para onde vão os erros do SMALL: fração das predições (16 modelos) por classe, por grupo
CF = S.groupby('grupo')[[f'pred {CLS[c]}' for c in range(4)]].mean().reindex(ORD) / NM * 100
CF.columns = R.CLASSES
print('% das predições dos 16 modelos para cada classe (gravações SMALL):'); display(CF.round(1))
CFF = pd.DataFrame({f: {CLS[c]: 100 * np.mean(np.concatenate([d.loc[d.Target == SMALL, list(FAM[f].values())].values.ravel() == c
                                                               for d in AL.values()])) for c in range(4)} for f in FAM}).T
print('Por família (todas as gravações SMALL, 4 entradas):'); display(CFF.round(1))
'''


COMP_CODE = r'''
import itertools
REP = ['MEL', 'LOFAR', 'HHT', 'DEMON']
M = S[S.grupo == 'meio']
Im = I[I.File.isin(M.index)]

# (a) Acerto SMALL por família × entrada nas gravações do meio
ACM = pd.DataFrame({f: {r: 100 * (1 - M[f'{f} {r}'].mean()) for r in REP} for f in FAM}).T
ACM.loc['média'] = ACM.mean()
print(f'Acerto SMALL (%) nas {len(M)} gravações do meio:'); display(ACM.round(1))

# (b) Por gravação: acerto de cada entrada = média dos 4 modelos (famílias) × folds
A = pd.DataFrame({r: 1 - M[[f'{f} {r}' for f in FAM]].mean(axis=1) for r in REP})
EXC = pd.DataFrame({'acerta (≥ 50%)': (A >= 0.5).sum(),
                    'só ela acerta': [int(((A[r] >= 0.5) & (A.drop(columns=r) < 0.5).all(axis=1)).sum()) for r in REP],
                    'maior acerto da gravação': A.idxmax(axis=1).value_counts().reindex(REP).fillna(0).astype(int)})
N_NENHUMA = int((A < 0.5).all(axis=1).sum())
print(f'Gravações do meio por entrada (nenhuma entrada com ≥ 50%: {N_NENHUMA}):'); display(EXC)

# (c) Q de Yule dos erros entre entradas, só nas gravações do meio (por família e fold)
def q_yule(a, b):
    n11, n00, n10, n01 = (a & b).sum(), (~a & ~b).sum(), (a & ~b).sum(), (~a & b).sum()
    return (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01)
QM = pd.DataFrame({f'{a} × {b}': {f: np.nanmean([q_yule(g[f'{f} {a}'].values == 1, g[f'{f} {b}'].values == 1)
                                                for _, g in Im.groupby('fold')]) for f in FAM}
                   for a, b in itertools.combinations(REP, 2)})
QM.loc['média'] = QM.mean()
print('Q de Yule dos erros nas gravações do meio:'); display(QM.round(2))

# (d) Cobertura: % das gravações do meio (por fold) em que ao menos uma das entradas acerta
rows = []
for k in range(1, 5):
    for comb in itertools.combinations(REP, k):
        r = {'entradas': ' + '.join(comb), 'nº': k}
        for f in FAM:
            r[f] = 100 * (Im[[f'{f} {x}' for x in comb]] == 0).any(axis=1).mean()
        rows.append(r)
COB = pd.DataFrame(rows).set_index('entradas')
COB['média'] = COB[list(FAM)].mean(axis=1)
COB = COB.sort_values(['nº', 'média'], ascending=[True, False])
print('Cobertura (oráculo) nas gravações do meio (%):'); display(COB.round(1))

# (e) Custo: quanto cada entrada chama de SMALL o que não é SMALL (todas as gravações de teste)
fa, pr = {}, {}
for f, reps in FAM.items():
    fa[f], pr[f] = {}, {}
    for r, e in reps.items():
        y = np.concatenate([d.Target.values for d in AL.values()]); p = np.concatenate([d[e].values for d in AL.values()])
        fa[f][r] = 100 * np.mean(p[y != SMALL] == SMALL)
        pr[f][r] = 100 * np.mean(y[p == SMALL] == SMALL)
FA, PR = pd.DataFrame(fa).T[REP], pd.DataFrame(pr).T[REP]
FA.loc['média'], PR.loc['média'] = FA.mean(), PR.mean()
print('Falso alarme: % das gravações NÃO SMALL preditas como SMALL'); display(FA.round(1))
print('Precisão SMALL (%)'); display(PR.round(1))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ACM.loc['média'].plot.bar(ax=ax[0], color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'], rot=0)
ax[0].set_ylabel('acerto SMALL no meio (%)'); ax[0].set_title('Acerto médio por entrada (4 famílias)')
top = COB[COB['nº'] == 2]['média'].sort_values()
ax[1].barh(top.index, top.values, color='tab:gray'); ax[1].set_xlabel('cobertura média (%)')
ax[1].set_title('Pares de entradas: ao menos uma acerta'); plt.tight_layout(); plt.show()
'''


INSEG_CODE = r'''
from sklearn.metrics import roc_auc_score
from scipy.stats import gmean

PARES_MD = {  # família: (MEL, DEMON)
    'RF': (FAM['RF']['MEL'], FAM['RF']['DEMON']), 'SVM': (FAM['SVM']['MEL'], FAM['SVM']['DEMON']),
    'MLP': (FAM['MLP']['MEL'], FAM['MLP']['DEMON']), 'BiGRU': (FAM['BiGRU']['MEL'], FAM['BiGRU']['DEMON']),
    'CNN': ('cnn_mel_artigo', 'hcnn_demon_tpsw_med5_e60p8_4c'),
}

def prob_medias(name):
    # {fold: DataFrame(File, p0..p3)}: média por gravação das saídas por janela (normalizadas para somar 1);
    # None se o experimento não guardou probabilidades (SVM)
    d = R.exp_dir(name); out = {}
    for k in range(10):
        df = pd.read_csv(os.path.join(d, 'eval', f'fold_{k}', f'{name}_multiclass_test.csv'))
        cols = [f'Prediction_{c}' for c in range(4)]
        if not set(cols) <= set(df.columns):
            return None
        P = df[cols].clip(lower=0).values
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
        g = pd.DataFrame(P, columns=['p0', 'p1', 'p2', 'p3']); g['File'] = df.File.values
        out[k] = g.groupby('File').mean().reset_index()
    return out

rows = []
for fam, (mel, dem) in PARES_MD.items():
    fp, fd, sc = R.file_predictions(mel), R.file_predictions(dem), prob_medias(mel)
    for k in range(10):
        a = fp[k].merge(fd[k][['File', 'Prediction']].rename(columns={'Prediction': 'pred_demon'}), on='File')
        if sc is not None:
            a = a.merge(sc[k], on='File')
        a['fam'], a['fold'] = fam, k
        rows.append(a)
D = pd.concat(rows, ignore_index=True)
D['acertou'] = D.Prediction == D.Target
Pm = D[['p0', 'p1', 'p2', 'p3']].values
D['margem'] = np.sort(Pm, 1)[:, -1] - np.sort(Pm, 1)[:, -2]
D['rank SMALL'] = (Pm > Pm[:, [0]]).sum(1) + 1
D['gap SMALL'] = D[['p0', 'p1', 'p2', 'p3']].max(axis=1) - D.p0
DS = D[D.Target == SMALL]

# (a) confiança nos acertos × erros do SMALL
cf = []
for fam, g in DS.groupby('fam', sort=False):
    ok, er = g[g.acertou], g[~g.acertou]
    r = {'família': fam, 'erro SMALL (%)': 100 * len(er) / len(g),
         'concordância: acerto': ok.agreement.median(), 'concordância: erro': er.agreement.median(),
         'AUC concordância': roc_auc_score(g.acertou, g.agreement)}
    if g.margem.notna().all():
        on = D[(D.fam == fam) & (D.Target != SMALL) & D.acertou]
        r.update({'margem: acerto': ok.margem.median(), 'margem: erro': er.margem.median(),
                  'AUC margem': roc_auc_score(g.acertou, g.margem),
                  'SMALL 2ª opção nos erros SMALL (%)': 100 * (er['rank SMALL'] == 2).mean(),
                  'SMALL 2ª opção nos acertos de outras classes (%)': 100 * (on['rank SMALL'] == 2).mean()})
    cf.append(r)
CONF = pd.DataFrame(cf).set_index('família')
print('MEL nas gravações SMALL: confiança quando acerta × quando erra (medianas; AUC = 0,5 é não separar)')
display(CONF.round(3))
ERR_OUT = D[~D.acertou].groupby(['fam', D.Target.map(CLS)]).agreement.median().unstack()[R.CLASSES]
print('Concordância mediana das janelas nos ERROS, por classe verdadeira:'); display(ERR_OUT.round(3))

# (b) gating: limiar de concordância 0,6 -> o que ele separa e se o DEMON acerta nos erros inseguros
THR = 0.6
gt = []
for fam, g in DS.groupby('fam', sort=False):
    er, ok = g[~g.acertou], g[g.acertou]
    ins = er.agreement < THR
    gt.append({'família': fam, 'erros marcados como inseguros (%)': 100 * ins.mean(),
               'acertos marcados como inseguros (%)': 100 * (ok.agreement < THR).mean(),
               'DEMON acerta nos erros inseguros (%)': 100 * (er[ins].pred_demon == SMALL).mean(),
               'DEMON acerta nos erros seguros (%)': 100 * (er[~ins].pred_demon == SMALL).mean()})
GATE = pd.DataFrame(gt).set_index('família')
print(f'Gating pela confiança do MEL (concordância < {THR}):'); display(GATE.round(1))

# (c) deslocar o limiar do SMALL: somar δ a p(SMALL) (predição pela média das probabilidades)
sh = []
for fam, g in D[D.margem.notna()].groupby('fam', sort=False):
    for delta in (0, 0.05, 0.10, 0.20):
        rec, sps = [], []
        for _, h in g.groupby('fold'):
            P = h[['p0', 'p1', 'p2', 'p3']].values.copy(); P[:, 0] += delta
            pr, y = P.argmax(1), h.Target.values
            rc = np.array([np.mean(pr[y == c] == c) for c in range(4)])
            rec.append(rc); sps.append(np.sqrt(rc.mean() * gmean(rc)) * 100 if (rc > 0).all() else 0)
        sh.append({'família': fam, 'δ': delta, 'SP': np.mean(sps),
                   **{f'recall {CLS[c]}': v for c, v in enumerate(np.mean(rec, 0) * 100)}})
SHIFT = pd.DataFrame(sh).set_index(['família', 'δ'])
print('Somar δ à probabilidade do SMALL (escolhido no próprio teste, portanto otimista):'); display(SHIFT.round(1))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for fam, g in DS[DS.margem.notna()].groupby('fam', sort=False):
    ax[0].hist(g[~g.acertou].margem, bins=np.linspace(0, 1, 26), histtype='step', lw=1.8, density=True, label=f'{fam} erro')
ax[0].hist(DS[DS.acertou & DS.margem.notna()].margem, bins=np.linspace(0, 1, 26), histtype='stepfilled', alpha=0.25,
           color='gray', density=True, label='acertos (todas)')
ax[0].set_xlabel('margem (p top1 − p top2)'); ax[0].set_title('MEL em SMALL: margem nos erros × acertos'); ax[0].legend(fontsize=8)
for fam in SHIFT.index.get_level_values(0).unique():
    t = SHIFT.loc[fam]
    ax[1].plot(t['recall SMALL'], t['recall MEDIUM'], marker='o', label=fam)
ax[1].set_xlabel('recall SMALL (%)'); ax[1].set_ylabel('recall MEDIUM (%)')
ax[1].set_title('Somar δ a p(SMALL): troca SMALL por MEDIUM'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
'''


def run(ns, src):
    exec(src, ns)


def build(filename=os.path.join(NOTEBOOKS_DIR, "32_estudo_small.ipynb")):
    ns = {"display": lambda *a, **k: None}
    os.makedirs(os.path.join(NOTEBOOKS_DIR, "..", "reports", "small"), exist_ok=True)
    for src in (SETUP_CODE, DIST_CODE, NUCLEO_CODE, VISTO_CODE, META_CODE, ACUST_CODE, CONF_CODE, COMP_CODE, INSEG_CODE):
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    G, S, GR, REC, N, NAV = ns["G"], ns["S"], ns["GR"], ns["REC"], ns["N"], ns["NAV"]
    VIS, NUC_VIS, SPL, MT, DS, DSp, p_ds, TP, LEN_MEAN, n_vis = (ns[k] for k in ("VIS", "NUC_VIS", "SPL", "MT", "DS", "DSp", "p_ds", "TP", "LEN_MEAN", "n_vis"))
    AC, AC_CLS, CF, CFF, NM, R = ns["AC"], ns["AC_CLS"], ns["CF"], ns["CFF"], ns["NM"], ns["R"]
    ACM, EXC, QM, COB, FA, PR, N_NENHUMA, M = (ns[k] for k in ("ACM", "EXC", "QM", "COB", "FA", "PR", "N_NENHUMA", "M"))
    am = ACM.loc["média"]
    best_fam = {f: ACM.loc[f].idxmax() for f in ns["FAM"]}
    exc_top = EXC["só ela acerta"].idxmax()
    qm = QM.loc["média"]
    q_dem = qm[[c for c in qm.index if "DEMON" in c]]
    cob2 = COB[COB["nº"] == 2]["média"]
    cob3 = COB[COB["nº"] == 3]["média"]
    cob4 = COB[COB["nº"] == 4]["média"].iloc[0]
    sem_dem3 = cob3[[i for i in cob3.index if "DEMON" not in i]].iloc[0]
    melhor_par_f = {f: COB[COB["nº"] == 2][f].idxmax() for f in ns["FAM"]}
    CONF, GATE, SHIFT, ERR_OUT, THR = ns["CONF"], ns["GATE"], ns["SHIFT"], ns["ERR_OUT"], ns["THR"]
    auc_c = CONF["AUC concordância"]
    auc_m = CONF["AUC margem"].dropna()
    s2_err = CONF["SMALL 2ª opção nos erros SMALL (%)"].dropna()
    s2_ok = CONF["SMALL 2ª opção nos acertos de outras classes (%)"].dropna()
    sh0 = SHIFT.xs(0, level="δ"); sh2 = SHIFT.xs(0.2, level="δ")
    dsp = sh2.SP - sh0.SP
    dsm = sh2["recall SMALL"] - sh0["recall SMALL"]; dme = sh2["recall MEDIUM"] - sh0["recall MEDIUM"]
    dem_ins_maior = int((GATE["DEMON acerta nos erros inseguros (%)"] > GATE["DEMON acerta nos erros seguros (%)"]).sum())

    gs = GR.loc["SMALL"]
    n_nuc, n_nav_nuc = len(N), N["Ship ID"].nunique()
    nav_multi = NAV[NAV.gravações > 1]
    nav_todas = int((nav_multi.no_núcleo == nav_multi.gravações).sum())
    n_rec_isol = int(NAV.loc[NAV.gravações == 1, "no_núcleo"].sum())
    vis_d = VIS["Δ (visto − não visto)"]
    pior_cls = CF.loc["núcleo"].drop("SMALL").idxmax()
    pior_frac = CF.loc["núcleo", pior_cls]
    p_meta = MT.loc["p (Kruskal-Wallis)"]
    meta_sig = [c for c, p in p_meta.items() if p < 0.05]
    ds_top = DSp["% no núcleo"].idxmax()
    rec_med = REC.drop(columns="oráculo das 4 entradas").stack()
    tipo_top = TP.sort_values("gravações", ascending=False).index[0]

    if AC is not None:
        p_ac = AC.loc["p (Kruskal-Wallis)"]
        ac_sig = [c for c, p in p_ac.items() if p < 0.05]
        rho = AC.loc["ρ Spearman com nº de erros"]
        ac_txt = (f"Com p < 0,05 entre fácil, meio e núcleo: {', '.join(f'**{c}** ({pv(p_ac[c])}, ρ = {br(rho[c])})' for c in ac_sig)}."
                  if ac_sig else "Nenhuma medida acústica separa os grupos com p < 0,05.")
        ganho = AC.loc[["fácil", "meio", "núcleo"], "ganho no CPA (dB)"]
        ganho_cls = AC_CLS["ganho no CPA (dB)"]
        pico = AC.loc[["fácil", "meio", "núcleo"], "|pico − CPA| (s)"]
        e200 = AC.loc[["fácil", "meio", "núcleo"], "energia 0-200 Hz (%)"]
        cen = AC.loc[["fácil", "meio", "núcleo"], "centroide (Hz)"]
        e200_cls, cen_cls = AC_CLS["energia 0-200 Hz (%)"], AC_CLS["centroide (Hz)"]
        espectro_txt = (f"**O núcleo tem o espectro concentrado em baixa frequência:** {br(e200['núcleo'], 0)}% da energia abaixo de 200 Hz "
                        f"(fácil {br(e200['fácil'], 0)}%, meio {br(e200['meio'], 0)}%) e centroide de {br(cen['núcleo'], 0)} Hz (fácil {br(cen['fácil'], 0)} Hz). "
                        f"Na mediana das classes, LARGE tem {br(e200_cls['LARGE'], 0)}% abaixo de 200 Hz e centroide de {br(cen_cls['LARGE'], 0)} Hz; "
                        f"SMALL tem {br(e200_cls['SMALL'], 0)}% e {br(cen_cls['SMALL'], 0)} Hz. Acusticamente, o núcleo se parece mais com LARGE do que com o SMALL típico, "
                        f"o que é coerente com {br(pior_frac, 0)}% das predições irem para {pior_cls}.")
        ac_resumo = (f"- **Medidas acústicas:** {ac_txt} Ganho mediano no CPA: fácil {br(ganho['fácil'], 1)} dB, meio {br(ganho['meio'], 1)} dB, núcleo {br(ganho['núcleo'], 1)} dB "
                     f"(classes: " + ", ".join(f"{c} {br(v, 1)} dB" for c, v in ganho_cls.items()) + f"). Distância mediana do pico de nível ao CPA anotado: fácil {br(pico['fácil'], 0)} s, meio {br(pico['meio'], 0)} s, núcleo {br(pico['núcleo'], 0)} s.")
    else:
        ac_resumo = "- **Medidas acústicas:** não calculadas (falta `reports/small/features_acusticas.csv`)."

    cells = [md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 32: Estudo da Classe SMALL

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

A classe SMALL (navios com menos de 50 m) tem o pior recall em quase todos os modelos. Os Notebooks 30 e 31 mostraram que as quatro entradas erram, em boa parte, as mesmas gravações. Aqui estudamos o SMALL diretamente com 16 modelos (RF, SVM, MLP e BiGRU × MEL, LOFAR, HHT e DEMON):

1. Quantos dos 16 modelos erram cada gravação SMALL?
2. Quais gravações quase todos erram (o **núcleo**)? São gravações isoladas ou navios inteiros?
3. Ter o mesmo navio no treino muda o erro?
4. O núcleo difere em metadados ou em medidas acústicas simples?
5. Para que classe vão os erros?

**Resumo:**
- Das {int(gs.total)} gravações SMALL de teste, só **{int(gs['núcleo'])} ({br(100 * gs['núcleo'] / gs.total, 1)}%) formam o núcleo** (≥ 15 dos {NM} modelos erram), de **{n_nav_nuc} navios**. Outras {int(gs['fácil'])} ({br(100 * gs['fácil'] / gs.total, 1)}%) são fáceis (≤ 4 erram); **{int(gs['meio'])} ({br(100 * gs['meio'] / gs.total, 1)}%) ficam no meio**: o acerto depende do modelo e da entrada.
- Recall SMALL dos 16 modelos: de {br(rec_med.min(), 1)}% a {br(rec_med.max(), 1)}%.
- No núcleo, {br(pior_frac, 0)}% das predições vão para **{pior_cls}**.
- No split oficial, ter o navio no treino {'reduz' if vis_d.mean() < 0 else 'não reduz'} o erro no SMALL em média {br(abs(vis_d.mean()), 1)} pontos (de {br(vis_d.min(), 1)} a {br(vis_d.max(), 1)} entre os 16 modelos).
{ac_resumo}

Predição por gravação = voto majoritário das janelas; "erram" = média sobre os 5 folds em que a gravação é teste. Todas as tabelas são calculadas na execução (`notebooks/src/iara_results.py`).
"""), code(SETUP_CODE),
        md("""
## 1. Quantos modelos erram cada gravação?

Para cada gravação de teste contamos quantos dos 16 modelos erram (média dos 5 folds em que ela é teste). Grupos: **fácil** (≤ 4 erram), **meio** (5 a 14), **núcleo** (≥ 15).
"""), code(DIST_CODE),
        md(f"""
**Leitura.**
- SMALL: {int(gs['fácil'])} fáceis, {int(gs['meio'])} no meio, {int(gs['núcleo'])} no núcleo. Nas outras classes, o núcleo é de """ + ", ".join(f"{c} {int(GR.loc[c, 'núcleo'])}/{int(GR.loc[c, 'total'])}" for c in R.CLASSES if c != "SMALL") + f""".
- O que separa o SMALL das outras classes **não é o núcleo**, que é pequeno em todas, mas o **meio**: a maioria das gravações SMALL é acertada por alguns modelos e errada por outros.
- O oráculo das 4 entradas (algum modelo acerta) é otimista: escolhe o modelo certo depois de saber a resposta. Ele mostra que o acerto não é impossível, não que um combinador chegue lá.
"""),
        md("""
## 2. O núcleo: gravações ou navios?

A lista é gravada em `reports/small/nucleo_small.csv`.
"""), code(NUCLEO_CODE),
        md(f"""
**Leitura.**
- As {n_nuc} gravações do núcleo vêm de **{n_nav_nuc} navios**. {n_rec_isol} delas são de navios com uma só gravação SMALL de teste, então nesses casos não dá para separar "gravação ruim" de "navio difícil".
- Entre os {len(nav_multi)} navios do núcleo com mais de uma gravação, {'nenhum tem' if nav_todas == 0 else str(nav_todas) + ' têm'} **todas** as gravações no núcleo; {'em todos' if nav_todas == 0 else 'nos outros ' + str(len(nav_multi) - nav_todas)}, outras gravações do mesmo navio são acertadas por parte dos modelos (ex.: PILOT 14 tem {int(NAV.loc['456', 'gravações']) if '456' in NAV.index else '?'} gravações SMALL de teste e só {int(NAV.loc['456', 'no_núcleo']) if '456' in NAV.index else '?'} no núcleo). Quando o mesmo navio é acertado em uma gravação e errado em outra, o problema está mais na gravação que no navio.
"""),
        md("""
## 3. Navio visto no treino

No 5×2cv oficial a separação é por gravação: o mesmo navio pode ter gravações no treino e no teste. Comparamos o erro das gravações SMALL de teste cujo navio **tem** gravação no treino do fold com as que **não têm**. Depois, o recall SMALL nos experimentos com split por navio do Notebook 29.
"""), code(VISTO_CODE),
        md(f"""
**Leitura.**
- Só {br(100 * n_vis.get(False, 0) / n_vis.sum(), 1)}% das gravações SMALL de teste ({n_vis.get(False, 0)} de {n_vis.sum()}, somando os 10 folds) são de navios **sem** gravação no treino: no protocolo oficial, quase todo teste SMALL é de um navio já visto.
- Erro no SMALL com o navio visto × não visto: Δ médio de {br(vis_d.mean(), 1)} pontos ({(vis_d < 0).sum()} dos {len(vis_d)} modelos erram menos quando o navio foi visto).
- Fração de gravações com ≥ 15 modelos errando: {br(NUC_VIS.get(True, float('nan')), 1)}% com o navio visto, {br(NUC_VIS.get(False, float('nan')), 1)}% sem.
""" + (f"- **Split por navio (NB29):** recall SMALL cai em média {br(-SPL['Δ'].mean(), 1)} pontos (Δ entre {br(SPL['Δ'].min(), 1)} e {'+' if SPL['Δ'].max() > 0 else ''}{br(SPL['Δ'].max(), 1)} entre os {len(SPL)} modelos). Parte do recall SMALL do protocolo oficial vem de reconhecer o mesmo navio." if len(SPL) else "")),
        md("""
## 4. Metadados: o que distingue o núcleo?

Comparação entre fácil, meio e núcleo (SMALL): comprimento, CPA, duração, vento, estado do mar, chuva (Kruskal-Wallis), subconjunto do IARA (χ²) e tipo de navio.
"""), code(META_CODE),
        md(f"""
**Leitura.**
- {'Nenhum metadado numérico separa os grupos com p < 0,05.' if not meta_sig else 'Separam os grupos com p < 0,05: ' + ', '.join(f'{c} ({pv(p_meta[c])})' for c in meta_sig) + '; CPA, duração, vento, estado do mar e chuva não.'} Comprimento: mediana {br(MT.loc['núcleo', 'Length'], 0)} m no núcleo e {br(MT.loc['fácil', 'Length'], 0)} m nas fáceis; média {br(LEN_MEAN['fácil'], 1)} / {br(LEN_MEAN['meio'], 1)} / {br(LEN_MEAN['núcleo'], 1)} m (fácil / meio / núcleo). O núcleo **não** está concentrado perto do limiar de 50 m.
- **Subconjunto:** {pv(p_ds)} no χ²; o maior percentual de núcleo está em **{ds_top}** ({br(DSp.loc[ds_top, '% no núcleo'], 1)}%). Os subconjuntos do IARA diferem em local, período e equipamento de gravação, então essa diferença pode vir de qualquer um deles.
- **Tipo:** o SMALL é dominado por {tipo_top} ({int(TP.loc[tipo_top, 'gravações'])} gravações de apenas {int(TP.loc[tipo_top, 'navios'])} navios, ~{br(TP.loc[tipo_top, 'gravações'] / TP.loc[tipo_top, 'navios'], 0)} por navio). Muitas gravações do mesmo barco facilitam memorizar o navio em vez de aprender a classe.
"""),
        md("""
## 5. Medidas acústicas simples

Sem calibração dos hidrofones, as medidas são relativas à própria gravação (`scratch/small_acoustic_features.py`):
- **ganho no CPA:** nível mediano em ±30 s do CPA anotado menos o nível mediano dos 30 s iniciais e finais. Aproxima o quanto o navio se destaca do fundo; perto de 0 dB, o navio mal aparece.
- **faixa dinâmica:** p95 − p5 do nível em janelas de 1 s.
- **|pico − CPA|:** distância entre o instante mais alto (média de 10 s) e o CPA anotado. Se o pico está longe do CPA, outra fonte (outro navio, ruído) pode dominar a gravação.
- **bandas, centroide e planura:** como a energia se distribui na frequência.
"""), code(ACUST_CODE),
        md("""
**Leitura.**
""" + (ac_resumo.replace("- **Medidas acústicas:** ", "- ") + "\n- " + espectro_txt + """
- O ganho no CPA (quanto o navio se destaca do fundo) **não** separa os grupos: o barco não está "ausente" das gravações do núcleo. A diferença está em **qual faixa de frequência domina**. A curadoria do IARA (Seção IV-A do artigo) garante um único navio **reportado no AIS** dentro de 2 km, com a região de inclusão de 250 m (A, B) ou 250–500 m (C, D), e verifica que a energia sobe com a aproximação do navio, o que é coerente com o ganho no CPA normal do núcleo. O próprio artigo ressalva que isso **não garante uma gravação pura** do navio: cita barcos sem AIS, falhas no AIS, navios fora da região de exclusão insuficientemente atenuados, navios atracados e outros ruídos antrópicos. Baixa frequência é justamente a faixa que menos se atenua com a distância, então tráfego além de 2 km é uma candidata natural. Outra possibilidade é um regime de operação do próprio barco diferente do habitual. Nenhuma das duas foi testada aqui.
- Essas medidas são aproximações; confirmar que o áudio do núcleo é atípico exige ouvir as gravações e, idealmente, conferir o AIS de outros navios no mesmo horário.""" if AC is not None else "- Medidas não calculadas.")),
        md("""
## 6. Para onde vão os erros?
"""), code(CONF_CODE),
        md(f"""
**Leitura.**
- No núcleo, as predições vão para: """ + ", ".join(f"{c} {br(v, 1)}%" for c, v in CF.loc["núcleo"].items()) + """.
- No meio: """ + ", ".join(f"{c} {br(v, 1)}%" for c, v in CF.loc["meio"].items()) + """.
"""),
        md(f"""
## 7. Complementaridade das entradas nas gravações do meio

As {len(M)} gravações SMALL do meio são as que alguns modelos acertam e outros erram. Aqui perguntamos: **qual entrada acerta mais essas gravações, e quais entradas se complementam?**
- **(a)** acerto SMALL de cada modelo nessas gravações;
- **(b)** por gravação, quantas entradas acertam (≥ 50% dos seus 4 modelos × folds) e quantas gravações **só uma** entrada acerta;
- **(c)** Q de Yule dos erros entre entradas, restrito ao meio;
- **(d)** cobertura: % das gravações em que **ao menos uma** das entradas acerta (oráculo);
- **(e)** o custo: quanto cada entrada chama de SMALL o que não é SMALL.
"""), code(COMP_CODE),
        md(f"""
**Leitura.**
- **Quem mais acerta:** em média, {', '.join(f'{r} {br(v, 1)}%' for r, v in am.sort_values(ascending=False).items())}. A melhor entrada muda com a família: """ + ", ".join(f"{f} → {r}" for f, r in best_fam.items()) + f""".
- **Quem acerta sozinha:** """ + ", ".join(f"{r} {int(v)}" for r, v in EXC["só ela acerta"].items()) + f""" gravações; em {N_NENHUMA} nenhuma entrada passa de 50%. **{exc_top}** é a que mais acerta gravações que as outras erram.
- **Erram juntas?** Q de Yule no meio: """ + ", ".join(f"{p} {br(v, 2)}" for p, v in qm.items()) + f""". MEL × LOFAR continua o par mais redundante; os pares com DEMON ficam em torno de zero ({br(q_dem.min(), 2)} a {br(q_dem.max(), 2)}): nessas gravações, o DEMON erra de forma praticamente independente das outras entradas. Restringir ao meio (tirar as gravações que todos acertam ou todos erram) reduz o Q de todos os pares; o que vale é a comparação entre pares, não o valor absoluto.
- **Cobertura (oráculo):** melhor par {cob2.idxmax()} ({br(cob2.max(), 1)}%), e MEL + LOFAR {br(cob2['MEL + LOFAR'], 1)}%; melhor trio {cob3.idxmax()} ({br(cob3.max(), 1)}%) contra {br(sem_dem3, 1)}% do trio sem DEMON; as quatro, {br(cob4, 1)}%. Melhor par por família: """ + ", ".join(f"{f} {p}" for f, p in melhor_par_f.items()) + f""".
- **O custo:** o DEMON chama de SMALL {br(FA.loc['média', 'DEMON'], 1)}% das gravações que não são SMALL em média, perto das outras entradas ({br(FA.loc['média'].drop('DEMON').min(), 1)}–{br(FA.loc['média'].drop('DEMON').max(), 1)}%), mas {br(FA.loc['RF', 'DEMON'], 1)}% no RF, onde ele mais acerta SMALL. Tem a menor precisão SMALL ({br(PR.loc['média', 'DEMON'], 1)}%, contra {br(PR.loc['média', 'MEL'], 1)}% do MEL). Uma regra simples como "aceitar SMALL se o DEMON disser SMALL" ganharia recall SMALL e perderia em MEDIUM e LARGE.

**Como combinar MEL e DEMON?** A cobertura é um teto: supõe saber, em cada gravação, qual entrada seguir. Caminhos, do mais simples ao mais elaborado:
1. **Média ponderada das probabilidades** (fusão tardia): p = w·p_MEL + (1 − w)·p_DEMON, com w escolhido na validação. Simples, mas trata o DEMON igual em todas as classes.
2. **Stacking:** um classificador pequeno (regressão logística) recebe as 8 probabilidades (4 classes × 2 entradas) e aprende quando confiar em cada uma, por exemplo que um "SMALL" do DEMON vale mais quando o MEL hesita entre SMALL e LARGE e menos quando o MEL está confiante em MEDIUM.
3. **Fusão de características** (concatenar MEL e DEMON na entrada de um único modelo), já testada no Notebook 30 com o MLP.
4. **Gating por confiança:** usar o DEMON só quando a confiança do MEL (margem entre as duas maiores probabilidades) for baixa.

As fusões com DEMON testadas nos Notebooks 30 e 31 não deram ganho significativo no SP; o stacking (2) é o candidato que modela explicitamente o falso alarme do DEMON.
"""),
        md("""
## 8. O MEL estava inseguro quando errou?

Se os erros do MEL em SMALL fossem indecisão, daria para usar a confiança dele: chamar o DEMON só quando o MEL estiver inseguro (gating), ou favorecer o SMALL quando ele for a 2ª opção. Medimos a confiança de cinco modelos MEL (RF, SVM, MLP, BiGRU, CNN) em cada gravação SMALL:
- **concordância das janelas:** fração das janelas que votam na classe vencedora (existe para todos);
- **margem:** diferença entre as duas maiores probabilidades médias da gravação (o SVM não guarda probabilidades);
- **AUC:** quanto a medida separa acertos de erros (0,5 = não separa; 1 = separa perfeitamente).

Depois testamos as duas ideias: (b) gating com concordância < 0,6 e (c) somar δ à probabilidade do SMALL.
"""), code(INSEG_CODE),
        md(f"""
**Leitura.**
- **Não estava inseguro.** A AUC da concordância vai de {br(auc_c.min(), 2)} a {br(auc_c.max(), 2)} e a da margem de {br(auc_m.min(), 2)} a {br(auc_m.max(), 2)}: a confiança do MEL praticamente não distingue os erros dos acertos em SMALL. A concordância mediana nos erros em SMALL é parecida com a dos erros em MEDIUM e LARGE.
- **Gating pela confiança não isola os erros:** com concordância < {br(THR, 1)}, """ + "; ".join(f"{f} marca {br(r['erros marcados como inseguros (%)'], 0)}% dos erros e {br(r['acertos marcados como inseguros (%)'], 0)}% dos acertos" for f, r in GATE.iterrows()) + f""". E o DEMON acerta mais nos erros inseguros do que nos seguros em só {dem_ins_maior} das {len(GATE)} famílias.
- **SMALL como 2ª opção não é específico:** o SMALL é a 2ª opção em {br(s2_err.min(), 0)}–{br(s2_err.max(), 0)}% dos erros em SMALL, mas também em {br(s2_ok.min(), 0)}–{br(s2_ok.max(), 0)}% das gravações de outras classes que o MEL acerta.
- **Favorecer o SMALL troca SMALL por MEDIUM:** com δ = 0,2, o recall SMALL sobe {br(dsm.min(), 1)}–{br(dsm.max(), 1)} pontos e o MEDIUM cai {br(-dme.max(), 1)}–{br(-dme.min(), 1)}; o SP varia de {br(dsp.min(), 1)} a {'+' if dsp.max() >= 0.05 else ''}{br(dsp.max(), 1)} pontos. Mesmo escolhendo δ no próprio teste (otimista), não há ganho.
- **Leitura física:** o MEL erra o SMALL com a mesma convicção com que acerta as outras classes; para ele, essas gravações **parecem** MEDIUM ou LARGE (coerente com a Seção 5). Combinações que dependem de o MEL "hesitar" (gating, limiar) não têm onde atuar; resta o stacking, que aprende combinações das probabilidades das duas entradas sem depender da confiança do MEL.
"""),
        md(f"""
## 9. Conclusões

1. **O núcleo é pequeno:** {int(gs['núcleo'])} das {int(gs.total)} gravações SMALL ({br(100 * gs['núcleo'] / gs.total, 1)}%), de {n_nav_nuc} navios, são erradas por quase todos os modelos. Ele não explica, sozinho, um recall SMALL de ~{br(rec_med.median(), 0)}%.
2. **O problema principal está no meio:** {br(100 * gs['meio'] / gs.total, 0)}% das gravações SMALL são acertadas por alguns modelos e erradas por outros. Não dá para afirmar que o recall SMALL não pode melhorar "independentemente da entrada".
3. **Generalização para navios novos pesa:** {'ter o navio no treino reduz o erro e o split por navio derruba o recall SMALL' if vis_d.mean() < 0 else 'o efeito de ter o navio no treino é pequeno'}; aumentar o número de navios SMALL (e não de gravações do mesmo navio) é provavelmente mais útil que trocar a entrada.
4. **O núcleo é acusticamente atípico, não necessariamente defeituoso:** {('as gravações têm energia concentrada abaixo de 200 Hz, como as de navios grandes, e os modelos as chamam de ' + pior_cls + '. ') if AC is not None else ''}Os metadados (CPA, clima, subconjunto) não explicam isso. A lista (`reports/small/nucleo_small.csv`) é o ponto de partida para escuta e conferência manual (AIS de outros navios no horário).
5. **No meio, o DEMON é a entrada complementar:** acerta sozinho mais gravações que as outras entradas ({int(EXC.loc['DEMON', 'só ela acerta'])}) e erra de forma quase independente delas (Q médio ≈ {br(q_dem.mean(), 2)}), mas tem a menor precisão SMALL ({br(PR.loc['média', 'DEMON'], 1)}%) e, no RF, um falso alarme de SMALL de {br(FA.loc['RF', 'DEMON'], 1)}%. MEL + DEMON é o par de maior cobertura; o ganho só aparece com um combinador que aprenda quando confiar no DEMON.
6. **Os erros do MEL em SMALL não são indecisão:** a confiança dele não separa erros de acertos (AUC {br(min(auc_c.min(), auc_m.min()), 2)}–{br(max(auc_c.max(), auc_m.max()), 2)}); gating pela confiança e favorecer o SMALL não melhoram o SP.
7. **O efeito vai além do núcleo:** a correlação entre energia abaixo de 200 Hz e o número de modelos que erram vale para todo o SMALL{(' (ρ de Spearman = ' + br(AC.loc['ρ Spearman com nº de erros', 'energia 0-200 Hz (%)']) + ')') if AC is not None else ''}: quanto mais grave o espectro de uma gravação SMALL, mais modelos a erram.

**Limites:**
- 16 modelos de 4 famílias; a CNN só entra na Seção 8 (CNN MEL), porque não tem as 4 entradas avaliadas no mesmo protocolo.
- "Erram" é a média de 5 folds; as gravações se repetem entre as repetições do 5×2cv.
- As medidas acústicas não são calibradas e misturam subconjuntos com equipamentos diferentes.
- Navios com uma só gravação não permitem separar efeito do navio e da gravação.
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
