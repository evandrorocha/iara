"""
Gera notebooks/34_cnn_lofar_artigo_erros.ipynb: reprodução da CNN LOFAR do artigo e análise dos erros em SMALL e MEDIUM.

Perguntas:
  1. A CNN LOFAR (test_scripts/cnn_lofar.py, configuração do grid search do artigo) reproduz a Tabela 7?
  2. Como ela se compara aos outros modelos do repositório (F-test 5x2cv, como na Tabela 8)?
  3. Matriz de confusão: para onde vão os erros?
  4. Erros em SMALL: são sempre as mesmas gravações? Que navios? Ter o navio no treino ajuda? Coincidem com o núcleo
     do Notebook 32? O espectro (energia < 200 Hz) explica?
  5. Erros em MEDIUM: idem, e o comprimento (limiares de 50 m e 100 m) e o tipo de navio.
  6. A CNN estava insegura quando errou?
  7. Ela erra as mesmas gravações que os outros modelos? Resgata gravações que eles erram?

Fontes: results/trainings/tests/cnn_lofar/eval/fold_k (predições e saídas por janela), demais modelos via iara_results.py,
src/iara/dataset_info/iara.csv, splits do 5x2cv (<exp>_split.csv do RF LOFAR, mesmos folds), reports/small/features_acusticas.csv.
Como nos outros geradores, o mesmo código das células roda aqui para calcular os números citados no markdown.
Rodar na raiz do repositório:  python notebooks/src/generate_cnn_lofar_erros_notebook.py
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


def sg(x, nd=2):
    return ("+" if x >= 0.005 else "") + br(x, nd)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


SETUP_CODE = r'''
import os, sys, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score
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
SMALL, MEDIUM, LARGE, BG = 0, 1, 2, 3
CNN = 'cnn_lofar'

# Tabela 7 do artigo (IARA UO, por gravação)
ARTIGO = {'CNN LOFAR': {'SP': (66.05, 1.90), 'ACC': (67.02, 1.78), 'F1': (66.29, 2.13)},
          'CNN MEL': {'SP': (63.52, 2.26), 'ACC': (64.99, 2.09), 'F1': (63.04, 2.02)},
          'MLP LOFAR': {'SP': (66.51, 1.39)}, 'MLP MEL': {'SP': (63.38, 1.81)},
          'RF LOFAR': {'SP': (56.92, 1.69)}, 'RF MEL': {'SP': (62.22, 1.86)}}
# Tabela 8 do artigo: p do F-test 5x2cv (SP) da CNN LOFAR contra cada modelo
ARTIGO_P = {'RF MEL': 0.12, 'MLP MEL': 0.20, 'CNN MEL': 0.36, 'RF LOFAR': 0.00, 'MLP LOFAR': 0.76}

# Modelos do repositório (não necessariamente idênticos aos do artigo; ver a coluna 'igual ao artigo?')
OUTROS = {
    'MLP LOFAR': ('mlp_lofar', 'próximo: arquitetura do artigo, mas com dropout 0,2 (artigo: sem dropout) e paciência 16 (grid: 8)'),
    'MLP MEL': ('mlp', 'próximo: arquitetura do artigo, paciência 16 (grid: 8)'),
    'RF LOFAR': ('rf_lofar5s_n100_balanced_4c', 'não: janelas de 5 s, 100 árvores, balanceado'),
    'RF MEL': ('rf_mel5s_n100_balanced_4c', 'não: janelas de 5 s, 100 árvores, balanceado'),
    'CNN MEL': ('cnn', 'não: test_scripts/cnn.py (128/64, lr 1e-5, sem sobreposição)'),
    'SVM MEL': ('svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0', 'não está no artigo'),
    'SVM LOFAR': ('svm_nystroem_6000_lofar_elasticnet_l1r0.15', 'não está no artigo'),
    'BiGRU MEL': ('bigru_att_mel5s_h64_s15-5_e20_4c', 'não está no artigo'),
    'BiGRU LOFAR': ('bigru_att_lofar5s_h128_s15-5_e15_4c', 'não está no artigo'),
}

meta = pd.read_csv(os.path.join(R.ROOT, 'src', 'iara', 'dataset_info', 'iara.csv'), dtype={'Ship ID': str, 'ID': str})
meta['Ship ID'] = meta['Ship ID'].str.strip()
for c in ('Length', 'CPA time', 'Wind', 'Sea state', 'Rain'):
    meta[c] = pd.to_numeric(meta[c], errors='coerce')
meta['Arquivo'] = meta.Dataset + '-' + meta.ID + '.wav'
meta = meta.set_index(meta.ID.astype(int))

# Predições por gravação (voto das janelas) e probabilidades médias por gravação
FP = R.file_predictions(CNN)
rows = []
for k in range(10):
    w = pd.read_csv(os.path.join(R.exp_dir(CNN), 'eval', f'fold_{k}', f'{CNN}_multiclass_test.csv'))
    P = w[[f'Prediction_{c}' for c in range(4)]].clip(lower=0).values
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    pm = pd.DataFrame(P, columns=['p0', 'p1', 'p2', 'p3']).assign(File=w.File.values).groupby('File').mean()
    sp = pd.read_csv(os.path.join(R.exp_dir(OUTROS['RF LOFAR'][0]), 'eval', f'fold_{k}', f"{OUTROS['RF LOFAR'][0]}_split.csv"))
    trn = meta.loc[sp[sp.Subset == 'trn'].File.astype(int), 'Ship ID'].value_counts()
    d = FP[k].join(pm, on='File')
    d['fold'] = k
    d['navio no treino'] = [trn.get(s, 0) > 0 for s in meta.loc[d.File, 'Ship ID']]
    rows.append(d)
D = pd.concat(rows, ignore_index=True)
D['acertou'] = D.Prediction == D.Target
Pm = D[['p0', 'p1', 'p2', 'p3']].values
D['margem'] = np.sort(Pm, 1)[:, -1] - np.sort(Pm, 1)[:, -2]
D['Classe'] = D.Target.map(CLS); D['Predita'] = D.Prediction.map(CLS)

# Uma linha por gravação: taxa de erro da CNN nos ~5 folds em que a gravação é teste
G = D.groupby('File').agg(Target=('Target', 'first'), erro_cnn=('acertou', lambda s: 1 - s.mean()), vezes=('fold', 'size'),
                          **{f'→ {CLS[c]}': ('Prediction', lambda s, c=c: (s == c).mean()) for c in range(4)})
G['Classe'] = G.Target.map(CLS)
G['CNN'] = np.select([G.erro_cnn == 0, G.erro_cnn == 1], ['sempre acerta', 'sempre erra'], 'às vezes')
G = G.join(meta[['Arquivo', 'Dataset', 'Ship ID', 'Name', 'DETAILED TYPE', 'Length', 'CPA time']], on='File')

# Grupos do Notebook 32 (16 modelos: RF, SVM, MLP, BiGRU × MEL, LOFAR, HHT, DEMON)
FAM16 = {
    'RF': ['rf_mel5s_n100_balanced_4c', 'rf_lofar5s_n100_balanced_4c', 'rf_hht_n100_balanced_4c', 'rf_demon_n100_balanced_4c'],
    'SVM': ['svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0', 'svm_nystroem_6000_lofar_elasticnet_l1r0.15',
            'svm_nystroem_6000_hht_C2.0_elasticnet_l1r0.15_4c', 'svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000_4c'],
    'MLP': ['mlp', 'mlp_lofar', 'mlp_hht_h32-16_e30_4c', 'mlp_demon_h32-16_e30_4c'],
    'BiGRU': ['bigru_att_mel5s_h64_s15-5_e20_4c', 'bigru_att_lofar5s_h128_s15-5_e15_4c', 'bigru_att_hht_h64_s15-5_e20_4c',
              'bigru_att_demon_h64_s15-5_e20_4c'],
}
M16 = sum(FAM16.values(), [])
nerr = []
for k, d in R.aligned(M16).items():
    nerr.append(pd.DataFrame({'File': d.File.values,
                              'n_err16': (np.column_stack([d[e].values for e in M16]) != d.Target.values[:, None]).sum(1)}))
N16 = pd.concat(nerr).groupby('File').n_err16.mean()
G['erram (de 16)'] = N16
G['grupo NB32'] = np.select([G['erram (de 16)'] >= 15, G['erram (de 16)'] <= 4], ['núcleo', 'fácil'], 'meio')

FEAT_CSV = os.path.join(R.ROOT, 'reports', 'small', 'features_acusticas.csv')
FEAT = pd.read_csv(FEAT_CSV).set_index('File') if os.path.exists(FEAT_CSV) else None
if FEAT is not None:
    G = G.join(FEAT, on='File')
print(f'{len(G)} gravações; {len(D)} predições (gravação × fold); medidas acústicas: {FEAT is not None}')
'''

REPRO_CODE = r'''
FM = R.fold_metrics(CNN)
display(FM[['SP', 'ACC', 'F1'] + [f'R_{c}' for c in R.CLASSES]].round(2))
S = R.summary(CNN)
REPRO = pd.DataFrame({m: {'nosso (10 folds)': f'{S[m][0]:.2f} ± {S[m][1]:.2f}',
                          'artigo (Tabela 7)': f'{ARTIGO["CNN LOFAR"][m][0]:.2f} ± {ARTIGO["CNN LOFAR"][m][1]:.2f}',
                          'Δ': round(S[m][0] - ARTIGO['CNN LOFAR'][m][0], 2),
                          'Δ / desvio do artigo': round((S[m][0] - ARTIGO['CNN LOFAR'][m][0]) / ARTIGO['CNN LOFAR'][m][1], 2)}
                      for m in ('SP', 'ACC', 'F1')}).T
display(REPRO)

rows = []
for nome, (exp, igual) in OUTROS.items():
    s = R.summary(exp)['SP']; F_, p_ = R.f_test_5x2(CNN, exp)
    rows.append({'modelo': nome, 'igual ao artigo?': igual, 'SP (nosso)': f'{s[0]:.2f} ± {s[1]:.2f}',
                 'SP (artigo)': (f"{ARTIGO[nome]['SP'][0]:.2f} ± {ARTIGO[nome]['SP'][1]:.2f}" if nome in ARTIGO else '—'),
                 'Δ CNN LOFAR − modelo': S['SP'][0] - s[0], 'F': F_, 'p (nosso)': p_, 'SP nosso (média)': s[0],
                 'p (artigo, Tab. 8)': ARTIGO_P.get(nome, np.nan)})
FT = pd.DataFrame(rows).set_index('modelo')
print('CNN LOFAR × outros modelos (F-test 5×2cv sobre o SP):'); display(FT.round(3))
'''

CONF_CODE = r'''
CMf = []
for k, d in D.groupby('fold'):
    cm = pd.crosstab(d.Classe, d.Predita, normalize='index').reindex(index=R.CLASSES, columns=R.CLASSES).fillna(0) * 100
    CMf.append(cm.values)
CMf = np.array(CMf)
CM = pd.DataFrame(CMf.mean(0), index=R.CLASSES, columns=R.CLASSES)
CMs = pd.DataFrame(CMf.std(0), index=R.CLASSES, columns=R.CLASSES)
print('Matriz de confusão (% por linha; média ± desvio dos 10 folds):')
display(CM.round(1).astype(str) + ' ± ' + CMs.round(1).astype(str))

fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(CM.values, cmap='Blues', vmin=0, vmax=100)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{CM.values[i, j]:.1f}\n± {CMs.values[i, j]:.1f}', ha='center', va='center',
                color='white' if CM.values[i, j] > 55 else 'black', fontsize=9)
ax.set_xticks(range(4), R.CLASSES); ax.set_yticks(range(4), R.CLASSES)
ax.set_xlabel('predita'); ax.set_ylabel('verdadeira'); ax.set_title('CNN LOFAR (artigo): matriz de confusão')
plt.colorbar(im, ax=ax, fraction=0.046); plt.tight_layout(); plt.show()
'''

SMALL_CODE = r'''
def estudo_classe(c):
    g = G[G.Target == c]
    dd = D[D.Target == c]
    out = {}
    out['consistência'] = g.CNN.value_counts().reindex(['sempre acerta', 'às vezes', 'sempre erra']).fillna(0).astype(int)
    out['destino'] = dd[~dd.acertou].Predita.value_counts(normalize=True).mul(100).reindex(R.CLASSES).fillna(0)
    nv = dd.groupby('navio no treino').acertou.agg(['mean', 'size']); nv['mean'] *= 100
    out['navio'] = nv.rename(columns={'mean': 'acerto (%)', 'size': 'predições'})
    out['nb32'] = g.groupby('grupo NB32').agg(gravações=('erro_cnn', 'size'), acerto_cnn=('erro_cnn', lambda s: 100 * (1 - s.mean())),
                                             erram_de_16=('erram (de 16)', 'mean')).reindex(['fácil', 'meio', 'núcleo'])
    nav = g.groupby('Ship ID').agg(nome=('Name', 'first'), tipo=('DETAILED TYPE', 'first'), comprimento=('Length', 'first'),
                                   gravações=('erro_cnn', 'size'), erro_cnn=('erro_cnn', 'mean'))
    out['navios'] = nav
    return out

ES = estudo_classe(SMALL)
print('SMALL — consistência da CNN em cada gravação (nos ~5 folds em que ela é teste):'); display(ES['consistência'].to_frame('gravações'))
print('SMALL — para onde vão os erros (%):'); display(ES['destino'].round(1).to_frame('%'))
print('SMALL — acerto com o navio visto × não visto no treino do fold:'); display(ES['navio'].round(1))
print('SMALL — acerto da CNN LOFAR nos grupos do Notebook 32 (16 modelos):'); display(ES['nb32'].round(1))
nvS = ES['navios']
print(f"SMALL — navios: {len(nvS)}; navios com erro médio ≥ 50%: {(nvS.erro_cnn >= 0.5).sum()}")
display(nvS.sort_values(['erro_cnn', 'gravações'], ascending=False).head(12).round(2))
print('SMALL por tipo de navio (≥ 5 gravações):')
TPs = G[G.Target == SMALL].groupby('DETAILED TYPE').agg(gravações=('erro_cnn', 'size'), navios=('Ship ID', 'nunique'),
                                                      acerto_cnn=('erro_cnn', lambda s: 100 * (1 - s.mean())))
display(TPs[TPs.gravações >= 5].sort_values('gravações', ascending=False).round(1))

if FEAT is not None:
    gS = G[G.Target == SMALL]
    FCOLS = ['energia 0-200 Hz (%)', 'energia 1000-5000 Hz (%)', 'centroide (Hz)', 'planura 0–5 kHz', 'ganho no CPA (dB)']
    AS = pd.DataFrame({c: {'sempre acerta (mediana)': gS.loc[gS.CNN == 'sempre acerta', c].median(),
                           'sempre erra (mediana)': gS.loc[gS.CNN == 'sempre erra', c].median(),
                           'ρ Spearman com o erro da CNN': spearmanr(gS[c], gS.erro_cnn, nan_policy='omit')[0],
                           'p': spearmanr(gS[c], gS.erro_cnn, nan_policy='omit')[1]} for c in FCOLS}).T
    print('SMALL — espectro × erro da CNN:'); display(AS.round(3))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(gS['energia 0-200 Hz (%)'], gS.erro_cnn + np.random.default_rng(0).uniform(-0.03, 0.03, len(gS)), s=10, alpha=0.5)
    ax.set_xlabel('energia abaixo de 200 Hz (%)'); ax.set_ylabel('taxa de erro da CNN na gravação')
    ax.set_title('SMALL: espectro grave × erro da CNN LOFAR'); plt.tight_layout(); plt.show()
'''

MEDIUM_CODE = r'''
EM = estudo_classe(MEDIUM)
print('MEDIUM — consistência:'); display(EM['consistência'].to_frame('gravações'))
print('MEDIUM — para onde vão os erros (%):'); display(EM['destino'].round(1).to_frame('%'))
print('MEDIUM — navio visto × não visto:'); display(EM['navio'].round(1))
print('MEDIUM — grupos do Notebook 32:'); display(EM['nb32'].round(1))

dM = D[D.Target == MEDIUM].join(meta[['Length', 'DETAILED TYPE']], on='File')
bins = [50, 60, 70, 80, 90, 100]
LEN = dM.groupby(pd.cut(dM.Length, bins, right=False), observed=False).agg(predições=('acertou', 'size'), acerto=('acertou', 'mean'),
                                                          vai_SMALL=('Prediction', lambda s: (s == SMALL).mean()),
                                                          vai_LARGE=('Prediction', lambda s: (s == LARGE).mean()))
LEN[['acerto', 'vai_SMALL', 'vai_LARGE']] *= 100
print('MEDIUM por comprimento (m): acerto e destino dos erros (%):'); display(LEN.round(1))
TPm = dM.groupby('DETAILED TYPE').agg(predições=('acertou', 'size'), navios=('File', lambda s: meta.loc[s, 'Ship ID'].nunique()),
                                      acerto=('acertou', lambda s: 100 * s.mean()))
print('MEDIUM por tipo de navio (≥ 20 predições):'); display(TPm[TPm.predições >= 20].sort_values('predições', ascending=False).round(1))
nvM = EM['navios']
print(f"MEDIUM — navios: {len(nvM)}; navios com erro médio ≥ 50%: {(nvM.erro_cnn >= 0.5).sum()}")
display(nvM.sort_values(['erro_cnn', 'gravações'], ascending=False).head(12).round(2))
rho_len = spearmanr(dM.Length, dM.Prediction == SMALL)[0]
'''

INSEG_CODE = r'''
rows = []
for c in (SMALL, MEDIUM, LARGE, BG):
    g = D[D.Target == c]
    rows.append({'classe': CLS[c], 'erro (%)': 100 * (1 - g.acertou.mean()),
                 'concordância: acerto': g[g.acertou].agreement.median(), 'concordância: erro': g[~g.acertou].agreement.median(),
                 'AUC concordância': roc_auc_score(g.acertou, g.agreement),
                 'margem: acerto': g[g.acertou].margem.median(), 'margem: erro': g[~g.acertou].margem.median(),
                 'AUC margem': roc_auc_score(g.acertou, g.margem)})
CONF = pd.DataFrame(rows).set_index('classe')
print('Confiança da CNN LOFAR quando acerta × quando erra (AUC = 0,5: não separa):'); display(CONF.round(3))
'''

OUTROS_CODE = r'''
def q_yule(a, b):
    n11, n00, n10, n01 = (a & b).sum(), (~a & ~b).sum(), (a & ~b).sum(), (~a & b).sum()
    return (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01)

rows = []
for nome, (exp, _) in OUTROS.items():
    qs, qss, resg = [], [], []
    for k in range(10):
        a = FP[k][['File', 'Target', 'Prediction']].merge(R.file_predictions(exp)[k][['File', 'Prediction']], on='File', suffixes=('_cnn', '_o'))
        ec, eo = a.Prediction_cnn != a.Target, a.Prediction_o != a.Target
        qs.append(q_yule(ec.values, eo.values))
        m = a.Target == SMALL
        qss.append(q_yule(ec[m].values, eo[m].values))
        resg.append(100 * (~ec[m] & eo[m]).sum() / max(eo[m].sum(), 1))
    rows.append({'modelo': nome, 'Q dos erros (todas)': np.nanmean(qs), 'Q dos erros (SMALL)': np.nanmean(qss),
                 'erros SMALL do modelo que a CNN acerta (%)': np.mean(resg)})
QO = pd.DataFrame(rows).set_index('modelo')
print('CNN LOFAR × outros modelos: erram as mesmas gravações?'); display(QO.round(2))

gS = G[G.Target == SMALL]
RESG = gS.groupby('grupo NB32').agg(gravações=('erro_cnn', 'size'),
                                    CNN_sempre_acerta=('CNN', lambda s: (s == 'sempre acerta').sum()),
                                    acerto_medio_cnn=('erro_cnn', lambda s: 100 * (1 - s.mean()))).reindex(['fácil', 'meio', 'núcleo'])
print('SMALL: a CNN LOFAR resgata gravações que os 16 modelos erram?'); display(RESG.round(1))
NUC_OK = gS[(gS['grupo NB32'] == 'núcleo') & (gS.erro_cnn < 0.5)]
print(f'Gravações do núcleo do NB32 que a CNN acerta na maioria dos folds: {len(NUC_OK)}')
if len(NUC_OK):
    display(NUC_OK[['Arquivo', 'Name', 'DETAILED TYPE', 'Length', 'erram (de 16)', 'erro_cnn']].round(2))
'''


def run(ns, src):
    exec(src, ns)


def build(filename=os.path.join(NOTEBOOKS_DIR, "34_cnn_lofar_artigo_erros.ipynb")):
    ns = {}
    for src in (SETUP_CODE, REPRO_CODE, CONF_CODE, SMALL_CODE, MEDIUM_CODE, INSEG_CODE, OUTROS_CODE):
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    S, REPRO, FT, FM, CM, CMs = ns["S"], ns["REPRO"], ns["FT"], ns["FM"], ns["CM"], ns["CMs"]
    ES, EM, LEN, CONF, QO, RESG, NUC_OK = ns["ES"], ns["EM"], ns["LEN"], ns["CONF"], ns["QO"], ns["RESG"], ns["NUC_OK"]
    AS, FEAT, TPs, TPm = ns.get("AS"), ns["FEAT"], ns["TPs"], ns["TPm"]
    ART = ns["ARTIGO"]

    cs, cm_ = ES["consistência"], EM["consistência"]
    dS, dM = ES["destino"], EM["destino"]
    nvS, nvM = ES["navio"], EM["navio"]
    def vis(nv):
        a = nv.loc[True, "acerto (%)"] if True in nv.index else float("nan")
        b = nv.loc[False, "acerto (%)"] if False in nv.index else float("nan")
        return a, b
    sv, snv = vis(nvS); mv, mnv = vis(nvM)
    same = FT[FT["igual ao artigo?"].str.startswith("próximo")]
    qmax = QO['Q dos erros (todas)'].idxmax()
    sig_nosso = [m for m in FT.index if FT.loc[m, "p (nosso)"] < 0.05]
    nb32 = ES["nb32"]
    rmin, rmax = FM.SP.min(), FM.SP.max()
    e200_txt = ""
    if AS is not None:
        e200_txt = (f"Nas gravações SMALL que a CNN **sempre erra**, a energia abaixo de 200 Hz tem mediana de {br(AS.loc['energia 0-200 Hz (%)', 'sempre erra (mediana)'], 0)}%, "
                    f"contra {br(AS.loc['energia 0-200 Hz (%)', 'sempre acerta (mediana)'], 0)}% nas que ela sempre acerta (ρ de Spearman com o erro = {br(AS.loc['energia 0-200 Hz (%)', 'ρ Spearman com o erro da CNN'])}, {pv(AS.loc['energia 0-200 Hz (%)', 'p'])}).")
    len_lo = LEN.iloc[0]; len_hi = LEN.iloc[-1]

    cells = [md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 34: CNN LOFAR do Artigo — Reprodução e Erros em SMALL e MEDIUM

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

A CNN LOFAR é o melhor modelo da Tabela 7 do artigo do IARA (Silva et al., 2025). Até aqui o repositório não a tinha: a "CNN LOFAR" usada antes colapsava (SP 35,4). Ela foi implementada em `test_scripts/cnn_lofar.py` com a configuração do grid search do artigo (`training_scripts/grid_search.py`): conv 512/128, ReLU, max pooling 4×2, dropout 0,4, BatchNorm, kernel 5, camada densa de 128, saída Sigmoid, Adam (lr 1e-6, weight decay 1e-3), batch 32, paciência 32, imagens de 32 janelas LOFAR com 50% de sobreposição. Treinada nos 10 folds do 5×2cv oficial (~4–7 h por fold numa RTX 5060 Ti).

Também foi corrigida a avaliação das CNNs no pipeline (`src/iara/ml/models/trainer.py`): desde o commit f04bb3b ela passava janelas soltas ao modelo em vez de imagens.

**Perguntas:** (1) reproduzimos a Tabela 7? (2) como ela se compara aos outros modelos? (3–5) como são os erros em SMALL e MEDIUM? (6) ela estava insegura quando errou? (7) erra as mesmas gravações que os outros modelos?

**Resumo:**
- **Reprodução:** SP {br(S['SP'][0])} ± {br(S['SP'][1])} (artigo {br(ART['CNN LOFAR']['SP'][0])} ± {br(ART['CNN LOFAR']['SP'][1])}), ACC {br(S['ACC'][0])} ± {br(S['ACC'][1])} (artigo {br(ART['CNN LOFAR']['ACC'][0])}), F1 {br(S['F1'][0])} ± {br(S['F1'][1])} (artigo {br(ART['CNN LOFAR']['F1'][0])}). Diferenças de {br(REPRO['Δ'].abs().min())} a {br(REPRO['Δ'].abs().max())} ponto, dentro do desvio do artigo.
- **SMALL é a pior classe também aqui:** recall {br(CM.loc['SMALL', 'SMALL'], 1)}%; os erros vão para MEDIUM ({br(dS['MEDIUM'], 0)}%), LARGE ({br(dS['LARGE'], 0)}%) e BACKGROUND ({br(dS['BACKGROUND'], 0)}%).
- **MEDIUM:** recall {br(CM.loc['MEDIUM', 'MEDIUM'], 1)}%; {br(dM['SMALL'], 0)}% dos erros vão para SMALL.
- **Contra os outros modelos:** a CNN LOFAR não difere dos dois modelos mais próximos dos do artigo, os MLPs (""" + ", ".join(f"{m} {pv(same.loc[m, 'p (nosso)'])}" for m in same.index) + f"""), como no artigo.
- **Navio no treino:** o acerto sobe {sg(sv - snv, 1)} pontos no SMALL e {sg(mv - mnv, 1)} no MEDIUM quando o navio da gravação de teste tem gravações no treino.
- **Confiança:** a CNN erra o SMALL tão segura quanto acerta (AUC da margem {br(CONF.loc['SMALL', 'AUC margem'])}); no MEDIUM a confiança separa parcialmente ({br(CONF.loc['MEDIUM', 'AUC margem'])}).
- **Núcleo do NB32:** das {int(nb32.loc['núcleo', 'gravações'])} gravações SMALL que ≥ 15 de 16 modelos erram, a CNN acerta em média {br(nb32.loc['núcleo', 'acerto_cnn'], 1)}%; ela não resolve o núcleo.

Predição por gravação = voto majoritário das janelas (como no artigo); todas as tabelas são calculadas na execução.
"""), code(SETUP_CODE),
        md("""
## 1. Reprodução da Tabela 7 e comparação com os outros modelos

O F-test 5×2cv (Alpaydin, 1999) compara o SP fold a fold, como na Tabela 8 do artigo. A coluna **igual ao artigo?** indica o quanto o modelo do repositório se aproxima do artigo; nenhum é idêntico, e quanto mais diferente, menos o p é comparável ao da Tabela 8.
"""), code(REPRO_CODE),
        md(f"""
**Leitura.**
- **A CNN LOFAR do artigo foi reproduzida:** SP {br(S['SP'][0])} ± {br(S['SP'][1])} contra {br(ART['CNN LOFAR']['SP'][0])} ± {br(ART['CNN LOFAR']['SP'][1])} (Δ {sg(REPRO.loc['SP', 'Δ'])}, {br(abs(REPRO.loc['SP', 'Δ / desvio do artigo']), 2)} desvio do artigo). O SP por fold vai de {br(rmin)} a {br(rmax)}.
- **Contra os modelos mais próximos dos do artigo (os MLPs), as conclusões se repetem:** """ + "; ".join(f"{m}: p = {br(same.loc[m, 'p (nosso)'], 2)} (artigo {br(same.loc[m, 'p (artigo, Tab. 8)'], 2)}; SP {br(same.loc[m, 'SP nosso (média)'])} contra {br(ART[m]['SP'][0])} no artigo)" for m in same.index) + f""" — a CNN LOFAR não é significativamente melhor que os MLPs.
- **Onde os modelos diferem**, os p também diferem: o nosso RF LOFAR é mais forte que o do artigo ({br(FT.loc['RF LOFAR', 'SP nosso (média)'])} contra {br(ART['RF LOFAR']['SP'][0])}), e a nossa CNN MEL não tem a arquitetura do artigo (a CNN MEL do artigo está sendo treinada em `test_scripts/cnn_mel_artigo.py`).
- Com p < 0,05 no nosso F-test: {', '.join(f'{m} ({pv(FT.loc[m, "p (nosso)"])})' for m in sig_nosso) if sig_nosso else 'nenhum'}.
"""),
        md("""
## 2. Matriz de confusão
"""), code(CONF_CODE),
        md(f"""
**Leitura.**
- **SMALL:** {br(CM.loc['SMALL', 'SMALL'], 1)} ± {br(CMs.loc['SMALL', 'SMALL'], 1)}% de acerto; vai para MEDIUM {br(CM.loc['SMALL', 'MEDIUM'], 1)}% e LARGE {br(CM.loc['SMALL', 'LARGE'], 1)}% quase na mesma proporção.
- **MEDIUM:** {br(CM.loc['MEDIUM', 'MEDIUM'], 1)} ± {br(CMs.loc['MEDIUM', 'MEDIUM'], 1)}%; a confusão principal é com SMALL ({br(CM.loc['MEDIUM', 'SMALL'], 1)}%), depois LARGE ({br(CM.loc['MEDIUM', 'LARGE'], 1)}%).
- **LARGE** {br(CM.loc['LARGE', 'LARGE'], 1)}% e **BACKGROUND** {br(CM.loc['BACKGROUND', 'BACKGROUND'], 1)}%. Os três tamanhos de navio se confundem entre si; o ruído de fundo é separado com mais facilidade.
"""),
        md("""
## 3. Erros em SMALL

Cada gravação é teste em ~5 folds (uma vez por repetição do 5×2cv). "Sempre erra" = a CNN errou em todos eles.
"""), code(SMALL_CODE),
        md(f"""
**Leitura.**
- **Consistência:** das {int(cs.sum())} gravações SMALL, a CNN sempre acerta {int(cs['sempre acerta'])}, sempre erra {int(cs['sempre erra'])} e oscila em {int(cs['às vezes'])} ({br(100 * cs['às vezes'] / cs.sum(), 0)}%). Como nos outros modelos (NB32), boa parte do SMALL está perto da fronteira de decisão.
- **Destino dos erros:** MEDIUM {br(dS['MEDIUM'], 1)}%, LARGE {br(dS['LARGE'], 1)}%, BACKGROUND {br(dS['BACKGROUND'], 1)}%.
- **Navio no treino:** acerto {br(sv, 1)}% com o navio visto contra {br(snv, 1)}% sem ({sg(sv - snv, 1)} pontos).
- **Grupos do NB32:** acerto da CNN """ + ", ".join(f"{g} {br(nb32.loc[g, 'acerto_cnn'], 1)}%" for g in nb32.index) + f""". A CNN segue o mesmo padrão dos 16 modelos: acerta as fáceis, oscila no meio e quase não acerta o núcleo.
- **Espectro:** {e200_txt if e200_txt else 'medidas acústicas não disponíveis.'} O espectro grave, que explicava os erros dos outros modelos (NB32), também explica os da CNN LOFAR.
"""),
        md("""
## 4. Erros em MEDIUM

MEDIUM são navios de 50 a 100 m. Os erros para SMALL se concentram perto de 50 m? Para LARGE, perto de 100 m?
"""), code(MEDIUM_CODE),
        md(f"""
**Leitura.**
- **Consistência:** sempre acerta {int(cm_['sempre acerta'])}, sempre erra {int(cm_['sempre erra'])}, oscila {int(cm_['às vezes'])} de {int(cm_.sum())} gravações.
- **Destino dos erros:** SMALL {br(dM['SMALL'], 1)}%, LARGE {br(dM['LARGE'], 1)}%, BACKGROUND {br(dM['BACKGROUND'], 1)}%.
- **Comprimento:** na faixa de {LEN.index[0]} m, {br(len_lo['vai_SMALL'], 1)}% das predições vão para SMALL; na faixa de {LEN.index[-1]} m, {br(len_hi['vai_LARGE'], 1)}% vão para LARGE (veja a tabela por faixa). """ + ("Os erros acompanham o comprimento: os MEDIUM menores são confundidos com SMALL e os maiores com LARGE." if (LEN['vai_SMALL'].iloc[0] > LEN['vai_SMALL'].iloc[-1] and LEN['vai_LARGE'].iloc[-1] > LEN['vai_LARGE'].iloc[0]) else "A relação com o comprimento não é monotônica (a faixa de 80–90 m erra mais que a de 70–80 m, dominada por Motor Hoppers bem reconhecidos): o erro não se explica só pela proximidade dos limiares.") + f"""
- **Navio no treino:** acerto {br(mv, 1)}% com o navio visto contra {br(mnv, 1)}% sem ({sg(mv - mnv, 1)} pontos).
- **Tipo:** o tipo mais frequente é {TPm.sort_values('predições', ascending=False).index[0]} (acerto {br(TPm.sort_values('predições', ascending=False).iloc[0]['acerto'], 1)}%). Como no NB32, o tipo de navio pesa: tipos com muitas gravações de poucos navios são mais fáceis.
"""),
        md("""
## 5. A CNN estava insegura quando errou?

Concordância das janelas (fração que vota na classe vencedora) e margem (diferença entre as duas maiores probabilidades médias), por classe verdadeira.
"""), code(INSEG_CODE),
        md(f"""
**Leitura.**
- AUC da margem: """ + ", ".join(f"{c} {br(CONF.loc[c, 'AUC margem'])}" for c in CONF.index) + f"""; da concordância: """ + ", ".join(f"{c} {br(CONF.loc[c, 'AUC concordância'])}" for c in CONF.index) + f""".
- **SMALL:** como o MEL no NB32, a CNN LOFAR erra com a mesma convicção com que acerta (AUC ≈ 0,5): a confiança não serve para identificar esses erros.
- **MEDIUM, LARGE e BACKGROUND:** a confiança separa cada vez melhor (AUC da margem de {br(CONF.loc['MEDIUM', 'AUC margem'])} a {br(CONF.loc['BACKGROUND', 'AUC margem'])}). O SMALL é a única classe em que os erros são "convictos".
"""),
        md("""
## 6. A CNN LOFAR erra as mesmas gravações que os outros modelos?
"""), code(OUTROS_CODE),
        md(f"""
**Leitura.**
- Q de Yule dos erros (todas as classes) com os outros modelos: """ + ", ".join(f"{m} {br(QO.loc[m, 'Q dos erros (todas)'])}" for m in QO.index) + f""". Os erros da CNN LOFAR são fortemente associados aos de todos os outros modelos; o mais associado é o {qmax} ({br(QO.loc[qmax, 'Q dos erros (todas)'])}).
- Dos erros SMALL de cada modelo, a CNN acerta de {br(QO['erros SMALL do modelo que a CNN acerta (%)'].min(), 0)}% a {br(QO['erros SMALL do modelo que a CNN acerta (%)'].max(), 0)}%.
- **Núcleo do NB32:** {len(NUC_OK)} das {int(RESG.loc['núcleo', 'gravações'])} gravações que ≥ 15 de 16 modelos erram são acertadas pela CNN na maioria dos folds. A CNN LOFAR não muda o diagnóstico do NB32.
"""),
        md(f"""
## 7. Conclusões

1. **A CNN LOFAR do artigo foi reproduzida:** SP {br(S['SP'][0])} ± {br(S['SP'][1])}, contra {br(ART['CNN LOFAR']['SP'][0])} ± {br(ART['CNN LOFAR']['SP'][1])} no artigo. A configuração importa: a versão inicial (sem sobreposição, paciência 16) dava 62,8 no fold 0, contra 68,5 da versão do artigo.
2. **Contra os MLPs (os modelos mais próximos dos do artigo), a Tabela 8 se repete:** a CNN LOFAR não é significativamente melhor que eles.
3. **Ter o navio no treino pesa:** {sg(sv - snv, 1)} pontos de acerto no SMALL e {sg(mv - mnv, 1)} no MEDIUM (o split por gravação do artigo põe o mesmo navio em treino e teste).
4. **O SMALL continua sendo o problema** ({br(CM.loc['SMALL', 'SMALL'], 1)}% de recall), com erros divididos entre MEDIUM e LARGE; o **MEDIUM** se confunde sobretudo com SMALL.
5. **Os erros da CNN seguem o mesmo padrão dos outros modelos:** gravações na fronteira que oscilam entre folds, erros convictos, espectro grave nas gravações SMALL que ela sempre erra, e o núcleo do NB32 continua sem solução. O melhor modelo do artigo não tem acesso a uma informação que os outros não tenham.

**Limites:**
- Split por gravação (`ID`), como no artigo; o NB29 mostrou que ele infla os resultados (navio no treino e no teste).
- O modelo salvo é o da última época, não o de menor perda de validação (comportamento de `trainer.py` preservado para reproduzir o artigo).
- O texto do artigo diz padding 1; o código usado pelos autores dá padding 2 (mantido o do código).
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
