"""
Gera notebooks/29_concordancia_erros_comuns_navios.ipynb: concordância entre modelos, erros comuns e o efeito
da sobreposição de navios entre treino e teste.

Perguntas:
  1. Modelos diferentes (RF, SVM, MLP, CNN) sobre representações diferentes (MEL, LOFAR, HHT) erram nas mesmas gravações?
  2. As gravações que todos erram se concentram em poucos navios? Que fatores as explicam?
  3. Quanto o 5x2cv oficial (split por gravação, `ID`) é inflado por ter o mesmo navio no treino e no teste?
     (scratch/eval_ship_split_5x2cv.py para RF/SVM e scratch/eval_ship_split_nn_5x2cv.py para MLP/BiGRU:
      split por `Ship ID` × controle por `ID`, mesmo código e sementes)

Fontes: predições por janela em results/trainings/{multirep,tests}/<exp>/eval/fold_k (via iara_results.py),
src/iara/dataset_info/iara.csv (Ship ID, Length, Dataset) e reports/split_navio/folds.csv.
Como nos outros geradores, o mesmo código das células roda aqui para calcular os números citados no markdown.
Rodar na raiz do repositório:  python notebooks/src/generate_erros_comuns_navios_notebook.py
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
import os, sys, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
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
pd.set_option('display.precision', 2)
CLS = dict(enumerate(R.CLASSES))

# 10 modelos de 4 classes que não colapsaram (rótulo = "FAMÍLIA REPRESENTAÇÃO [variante]")
MODELOS = {
    'RF MEL': 'rf_mel5s_n100_balanced_4c', 'RF LOFAR': 'rf_lofar5s_n100_balanced_4c', 'RF HHT': 'rf_hht_n100_balanced_4c',
    'SVM MEL': 'svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0', 'SVM LOFAR': 'svm_nystroem_6000_lofar_elasticnet_l1r0.15',
    'MLP MEL': 'mlp', 'MLP LOFAR': 'mlp_lofar', 'CNN MEL (artigo)': 'cnn_mel_artigo',
    'CNN HHT 1D': 'speccnn_hht_ctx0_e60p8_4c', 'CNN HHT 2D': 'hht_2d_cnn_e40p7_4c',
}
NM = len(MODELOS)

meta = pd.read_csv(os.path.join(R.ROOT, 'src', 'iara', 'dataset_info', 'iara.csv'), dtype={'Ship ID': str})
meta['Ship ID'] = meta['Ship ID'].str.strip()
meta['Length'] = pd.to_numeric(meta['Length'], errors='coerce')
meta = meta.set_index(meta.ID.astype(int))
NGRAV = meta[meta.Dataset.isin(list('ABCDE'))].groupby('Ship ID').size()
'''

DESEMP_CODE = r'''
desemp = R.table([(k, v) for k, v in MODELOS.items()], metrics=('SP', 'ACC'))
display(desemp)
'''

CONC_CODE = r'''
AL = R.aligned(list(MODELOS.values()))
N_INST = sum(len(d) for d in AL.values())


def familia_rep(lab):
    t = lab.split()
    return t[0], t[1]


def tipo_par(a, b):
    (fa, ra), (fb, rb) = familia_rep(a), familia_rep(b)
    if fa == fb and ra == rb:
        return 'mesma família e rep. (variantes)'
    if ra == rb:
        return 'mesma rep., família diferente'
    if fa == fb:
        return 'mesma família, rep. diferente'
    return 'família e rep. diferentes'


def medidas_par(a, b):
    per = []
    for k, df in AL.items():
        y, pa, pb = df.Target.values, df[MODELOS[a]].values, df[MODELOS[b]].values
        ea, eb = pa != y, pb != y
        n11, n00, n10, n01 = (ea & eb).sum(), (~ea & ~eb).sum(), (ea & ~eb).sum(), (~ea & eb).sum()
        per.append({'Concordância (%)': np.mean(pa == pb) * 100, 'κ': cohen_kappa_score(pa, pb),
                    'Ambos erram (%)': n11 / len(y) * 100, 'Ambos erram se indep. (%)': ea.mean() * eb.mean() * 100,
                    'Mesmo erro | ambos erram (%)': np.mean(pa[ea & eb] == pb[ea & eb]) * 100,
                    'Q de Yule': (n00 * n11 - n01 * n10) / (n00 * n11 + n01 * n10)})
    return pd.DataFrame(per).mean()


pares = pd.DataFrame([{'Par': f'{a} × {b}', 'Tipo': tipo_par(a, b), **medidas_par(a, b)}
                      for a, b in itertools.combinations(MODELOS, 2)]).set_index('Par')
por_tipo = pares.groupby('Tipo')[['Concordância (%)', 'κ', 'Q de Yule', 'Mesmo erro | ambos erram (%)']].agg(['mean', 'min', 'max'])
por_tipo.insert(0, ('n pares', ''), pares.groupby('Tipo').size())
display(por_tipo.round(2))
display(pares.sort_values('Q de Yule', ascending=False).round(2))

Q = pd.DataFrame(np.nan, index=list(MODELOS), columns=list(MODELOS))
for p, r in pares.iterrows():
    a, b = p.split(' × ')
    Q.loc[a, b] = Q.loc[b, a] = r['Q de Yule']
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(Q.values, vmin=0, vmax=1, cmap='viridis')
ax.set_xticks(range(NM), Q.columns, rotation=60, ha='right'); ax.set_yticks(range(NM), Q.index)
for i in range(NM):
    for j in range(NM):
        if i != j:
            ax.text(j, i, f'{Q.iat[i, j]:.2f}', ha='center', va='center', fontsize=7, color='w' if Q.iat[i, j] < 0.6 else 'k')
fig.colorbar(im, ax=ax, label='Q de Yule dos erros (0 = independentes, 1 = sempre juntos)')
ax.set_title('Correlação dos erros por gravação (4 classes, 10 folds)')
plt.tight_layout(); plt.show()
'''

ERROS_CODE = r'''
rows = []
for k, df in AL.items():
    sp = pd.read_csv(os.path.join(R.exp_dir(MODELOS['RF LOFAR']), 'eval', f'fold_{k}', f"{MODELOS['RF LOFAR']}_split.csv"))
    trn_ships = meta.loc[sp[sp.Subset == 'trn'].File.astype(int), 'Ship ID'].value_counts()
    W = np.column_stack([df[e].values != df.Target.values for e in MODELOS.values()])
    P = np.column_stack([df[e].values for e in MODELOS.values()])
    d = pd.DataFrame({'fold': k, 'File': df.File.values, 'Target': df.Target.values, 'n_err': W.sum(1), 'erro_comum': W.all(1),
                      'pred_maioria': [np.bincount(r, minlength=4).argmax() for r in P]})
    d['ship'] = meta.loc[d.File, 'Ship ID'].values
    d['navio_no_treino'] = d.ship.map(trn_ships).fillna(0).astype(int).values > 0
    rows.append(d)
I = pd.concat(rows, ignore_index=True).join(meta[['Length', 'Dataset', 'Name']], on='File')
I['Classe'] = I.Target.map(CLS)
N_COMUM = int(I.erro_comum.sum())
por_classe = I.groupby('Classe', sort=False).agg(instancias=('erro_comum', 'size'), erros_comuns=('erro_comum', 'sum'),
                                                 pct=('erro_comum', lambda x: 100 * x.mean()))
display(por_classe.round(2))
destino = pd.crosstab(I[I.erro_comum].Classe, I[I.erro_comum].pred_maioria.map(CLS)).reindex(index=R.CLASSES, columns=R.CLASSES, fill_value=0)
print('Erros comuns: classe real (linhas) × classe prevista pela maioria dos 10 modelos (colunas)')
display(destino)
REC = I.groupby('File').agg(Classe=('Classe', 'first'), inst=('erro_comum', 'size'), err=('erro_comum', 'sum'))
REC['sempre'], REC['nunca'] = REC.err == REC.inst, REC.err == 0
consist = REC.groupby('Classe', sort=False).agg(**{'gravações': ('inst', 'size'),
                                                  'erro comum em todas as aparições': ('sempre', 'sum'),
                                                  'em nenhuma': ('nunca', 'sum')})
display(consist)
'''

NAVIO_CODE = r'''
rng = np.random.default_rng(0)
N_PERM = 2000
navios = I[I.Target < 3].copy()          # BACKGROUND não tem navio
conc, tops = [], {}
for c in range(3):
    S = navios[navios.Target == c]
    g = S.groupby('ship').agg(instancias=('erro_comum', 'size'), erros_comuns=('erro_comum', 'sum'), gravacoes=('File', 'nunique'),
                              comprimento=('Length', 'first'), nome=('Name', 'first')).sort_values('erros_comuns', ascending=False)
    g['taxa (%)'] = 100 * g.erros_comuns / g.instancias
    cum_err = g.erros_comuns.cumsum() / g.erros_comuns.sum()
    cum_inst = g.instancias.sort_values(ascending=False).cumsum() / len(S)
    rec = S.groupby('File').agg(ship=('ship', 'first'), inst=('erro_comum', 'size'), err=('erro_comum', 'sum'))
    p0 = rec.err.sum() / rec.inst.sum()
    n_ship = rec.inst.groupby(rec.ship).sum()

    def chi2(err_rec):
        e = pd.Series(err_rec, index=rec.index).groupby(rec.ship).sum()
        E = n_ship * p0
        return float((((e - E) ** 2) / (E * (1 - p0))).sum())
    # nulo: a taxa de erro comum de cada GRAVAÇÃO é embaralhada entre as gravações da classe (mantém a correlação
    # entre folds da mesma gravação e o nº de aparições); a contagem é reamostrada (binomial)
    taxa = (rec.err / rec.inst).values
    obs = chi2(rec.err.values)
    nulo = np.array([chi2(rng.binomial(rec.inst.values, rng.permutation(taxa))) for _ in range(N_PERM)])
    conc.append({'Classe': CLS[c], 'navios': len(g), 'gravações': len(rec), 'erros comuns': int(g.erros_comuns.sum()),
                 'navios com erro comum': int((g.erros_comuns > 0).sum()),
                 'navios com 50% dos erros': int((cum_err < 0.5).sum() + 1),
                 'navios com 50% das instâncias': int((cum_inst < 0.5).sum() + 1),
                 'χ² entre navios': obs, 'χ² nulo (média)': nulo.mean(),
                 'χ² nulo (IC 95%)': f'{np.percentile(nulo, 2.5):.0f}–{np.percentile(nulo, 97.5):.0f}',
                 'p (permutação)': (np.sum(nulo >= obs) + 1) / (N_PERM + 1)})
    tops[CLS[c]] = g.head(10)
CONC = pd.DataFrame(conc).set_index('Classe')
display(CONC.round(3))
for c, t in tops.items():
    print(f'\n{c}: 10 navios com mais erros comuns')
    display(t.round(1))
'''

FATORES_CODE = r'''
navios['Navio no treino do fold'] = np.where(navios.navio_no_treino, 'sim', 'não')
visto = navios.groupby(['Classe', 'Navio no treino do fold'], sort=False).agg(
    instancias=('erro_comum', 'size'), erro_comum_pct=('erro_comum', lambda x: 100 * x.mean()),
    erro_medio_pct=('n_err', lambda x: 100 * x.mean() / NM))
display(visto.round(2))

navios['Gravações do navio'] = pd.cut(navios.ship.map(NGRAV), [0, 1, 2, 5, 10, 30, 100])
ngrav = navios.groupby(['Classe', 'Gravações do navio'], sort=False, observed=True).agg(
    instancias=('erro_comum', 'size'), erro_comum_pct=('erro_comum', lambda x: 100 * x.mean()))
display(ngrav.round(2))

navios['Comprimento (m)'] = pd.cut(navios.Length, [0, 20, 40, 50, 60, 80, 100, 120, 150, 200, 400], right=False)
compr = navios.groupby('Comprimento (m)', observed=True).agg(
    instancias=('erro_comum', 'size'), navios=('ship', 'nunique'), erro_comum_pct=('erro_comum', lambda x: 100 * x.mean()),
    erro_medio_pct=('n_err', lambda x: 100 * x.mean() / NM))
display(compr.round(2))

dataset = I.groupby(['Classe', 'Dataset'], sort=False).agg(instancias=('erro_comum', 'size'),
                                                           erro_comum_pct=('erro_comum', lambda x: 100 * x.mean()))
display(dataset.round(2))
'''

SPLIT_CODE = r'''
folds_navio = pd.read_csv(os.path.join(R.ROOT, 'reports', 'split_navio', 'folds.csv'), dtype={'Ship ID': str})
sobrep = {}
for cond, d in folds_navio.groupby('condicao'):
    r = []
    for k, dk in d.groupby('fold'):
        tr = set(dk[dk.Subset == 'trn']['Ship ID']) - {'-'}
        te = dk[(dk.Subset == 'test') & (dk.Target < 3)]
        r.append(100 * te['Ship ID'].isin(tr).mean())
    sobrep[cond] = np.mean(r)
balanco = folds_navio[folds_navio.Subset == 'test'].groupby(['condicao', 'fold']).Target.value_counts().unstack().rename(columns=CLS)
print('Gravações de teste por classe em cada fold:')
display(balanco.groupby('condicao').agg(['min', 'max']))

NOME_FAM = {'rf': 'RF', 'svm2000': 'SVM m=2000', 'mlp': 'MLP', 'bigru': 'BiGRU'}
SPLIT_MODELOS = [(f'{NOME_FAM[m]} {r.upper()}', f'{m}_{r}') for m in ('rf', 'svm2000', 'mlp') for r in ('mel', 'lofar', 'hht')]
SPLIT_MODELOS.append(('BiGRU LOFAR', 'bigru_lofar'))
OFICIAL = {'rf_mel': 'rf_mel5s_n100_balanced_4c', 'rf_lofar': 'rf_lofar5s_n100_balanced_4c', 'rf_hht': 'rf_hht_n100_balanced_4c'}
linhas, fm = [], {}
for lab, tag in SPLIT_MODELOS:
    a, b = f'split_id_{tag}_4c', f'split_navio_{tag}_4c'
    fm[(tag, 'id')], fm[(tag, 'navio')] = R.fold_metrics(a), R.fold_metrics(b)
    sa, sb = R.summary(a), R.summary(b)
    linha = {'Modelo': lab,
             'SP oficial (ID)': R.fmt(R.summary(OFICIAL[tag])['SP']) if tag in OFICIAL else '—',
             'SP split por ID (controle)': R.fmt(sa['SP']), 'SP split por navio': R.fmt(sb['SP']),
             'Δ SP (p.p.)': round(sb['SP'][0] - sa['SP'][0], 2),
             'ACC split por ID': R.fmt(sa['ACC']), 'ACC split por navio': R.fmt(sb['ACC']),
             'Δ ACC (p.p.)': round(sb['ACC'][0] - sa['ACC'][0], 2),
             'folds com SP = 0 (navio)': int((fm[(tag, 'navio')].SP == 0).sum())}
    for c in R.CLASSES:
        linha[f'Δ {c} (p.p.)'] = round(sb['R_' + c][0] - sa['R_' + c][0], 2)
    linhas.append(linha)
SPLIT = pd.DataFrame(linhas).set_index('Modelo')
display(SPLIT)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(SPLIT_MODELOS))
for off, cond, cor in ((-0.2, 'id', '#4C72B0'), (0.2, 'navio', '#C44E52')):
    m = [fm[(t, cond)].SP.mean() for _, t in SPLIT_MODELOS]
    s = [fm[(t, cond)].SP.std(ddof=0) for _, t in SPLIT_MODELOS]
    axes[0].bar(x + off, m, 0.4, yerr=s, capsize=3, color=cor, label='split por ' + ('ID (controle)' if cond == 'id' else 'navio'))
axes[0].set_xticks(x, [l for l, _ in SPLIT_MODELOS], rotation=20); axes[0].set_ylabel('SP (%)'); axes[0].legend()
axes[0].set_title('SP: mesmo modelo, mesmo código, só o agrupamento muda')
dr = np.array([[fm[(t, 'navio')]['R_' + c].mean() - fm[(t, 'id')]['R_' + c].mean() for c in R.CLASSES] for _, t in SPLIT_MODELOS])
im = axes[1].imshow(dr, cmap='RdBu', vmin=-30, vmax=30)
axes[1].set_xticks(range(4), R.CLASSES); axes[1].set_yticks(x, [l for l, _ in SPLIT_MODELOS])
for i in range(dr.shape[0]):
    for j in range(4):
        axes[1].text(j, i, f'{dr[i, j]:+.1f}', ha='center', va='center', fontsize=8)
fig.colorbar(im, ax=axes[1], label='Δ recall navio − ID (p.p.)')
axes[1].set_title('Variação do recall por classe')
plt.tight_layout(); plt.show()
'''


def run(ns, src):
    exec(src, ns)


def build(filename=os.path.join(NOTEBOOKS_DIR, "29_concordancia_erros_comuns_navios.ipynb")):
    ns = {"display": lambda *a, **k: None}
    import matplotlib.pyplot as plt  # noqa: F401
    for src in (SETUP_CODE, DESEMP_CODE, CONC_CODE, ERROS_CODE, NAVIO_CODE, FATORES_CODE, SPLIT_CODE):
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    R, np = ns["R"], ns["np"]
    pares, por_tipo, por_classe, destino, consist = ns["pares"], ns["por_tipo"], ns["por_classe"], ns["destino"], ns["consist"]
    CONC, visto, compr, dataset, SPLIT, sobrep = ns["CONC"], ns["visto"], ns["compr"], ns["dataset"], ns["SPLIT"], ns["sobrep"]
    fm, SPLIT_MODELOS = ns["fm"], ns["SPLIT_MODELOS"]

    qt = pares.groupby("Tipo")["Q de Yule"]
    ct = pares.groupby("Tipo")["Concordância (%)"].mean()
    qmin, qmax = pares["Q de Yule"].min(), pares["Q de Yule"].max()
    ratio = pares["Ambos erram (%)"] / pares["Ambos erram se indep. (%)"]
    top_par = pares["Q de Yule"].idxmax()
    N_COMUM = int(ns["N_COMUM"])
    T_SR, T_SF, T_DD ="mesma rep., família diferente", "mesma família, rep. diferente", "família e rep. diferentes"
    rep_mais = qt.mean()[T_SR] > qt.mean()[T_SF]

    def vis(c, s):
        return visto.loc[(c, s), "erro_comum_pct"]

    def cb(lo):
        return compr.loc[[iv for iv in compr.index if iv.left == lo][0]]

    d_sp, d_acc = SPLIT["Δ SP (p.p.)"], SPLIT["Δ ACC (p.p.)"]
    sp0 = SPLIT["folds com SP = 0 (navio)"]
    sp0_txt = ", ".join(f"{m} ({int(n)} folds)" for m, n in sp0.items() if n > 0)
    cls_pior = min(R.CLASSES, key=lambda c: SPLIT[f"Δ {c} (p.p.)"].mean())
    pior = d_sp.idxmin()
    d_cls = {c: SPLIT[f"Δ {c} (p.p.)"] for c in R.CLASSES}
    std_id = {t: fm[(t, "id")].SP.std(ddof=0) for _, t in SPLIT_MODELOS}
    std_nv = {t: fm[(t, "navio")].SP.std(ddof=0) for _, t in SPLIT_MODELOS}
    sep = sum(fm[(t, "navio")].SP.max() < fm[(t, "id")].SP.min() for _, t in SPLIT_MODELOS)
    fam_d = {f: d_acc[[m for m in d_acc.index if m.startswith(f)]] for f in ("RF", "SVM", "MLP", "BiGRU")}
    ctrl_d = {r: float(SPLIT.loc[f"RF {r}", "SP split por ID (controle)"].split(" ± ")[0])
                 - float(SPLIT.loc[f"RF {r}", "SP oficial (ID)"].split(" ± ")[0]) for r in ("MEL", "LOFAR", "HHT")}

    cells = [md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 29: Concordância entre Modelos, Erros Comuns e o Efeito do Navio

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

Três perguntas, nesta ordem:

1. **Modelos diferentes erram nas mesmas gravações?** Dez modelos de 4 classes que não colapsaram — RF, SVM, MLP e CNN sobre MEL, LOFAR e HHT — comparados par a par por gravação.
2. **As gravações que todos erram se concentram em poucos navios?** E que fatores as explicam (navio visto no treino, nº de gravações do navio, comprimento, campanha de gravação)?
3. **Quanto o protocolo oficial é inflado pela sobreposição de navios?** O 5×2cv do artigo separa por gravação (`ID`): o mesmo navio (`Ship ID`) pode ter gravações no treino e no teste. Comparamos um split por navio com um controle por `ID`, com o mesmo código e as mesmas sementes (`scratch/eval_ship_split_5x2cv.py` para RF/SVM e `scratch/eval_ship_split_nn_5x2cv.py` para MLP/BiGRU).

**Resumo:**
- Os erros são fortemente correlacionados em todos os pares (Q de Yule de {br(qmin)} a {br(qmax)}; os dois modelos erram juntos {br(ratio.min(), 1)}–{br(ratio.max(), 1)}× mais do que se os erros fossem independentes). {'Trocar a representação traz mais diversidade do que trocar o classificador.' if rep_mais else 'Trocar o classificador traz mais diversidade do que trocar a representação.'}
- {br(N_COMUM, 0)} das {br(ns['N_INST'], 0)} instâncias arquivo × fold ({br(100 * N_COMUM / ns['N_INST'], 1)}%) são erradas pelos 10 modelos. Elas não se concentram em "poucos navios ruins" além do que a distribuição das gravações já impõe, exceto em `MEDIUM`; o fator mais forte é o navio **não** aparecer no treino do fold.
- Separando por navio, a ACC balanceada cai **{br(-d_acc.max(), 1)} a {br(-d_acc.min(), 1)} p.p.** nos {len(SPLIT)} modelos testados (SP: {br(-d_sp.max(), 1)} a {br(-d_sp.min(), 1)} p.p.), com a maior perda em `{cls_pior}`. O 5×2cv por gravação superestima o desempenho em navios novos.

Todas as tabelas são calculadas na execução a partir das predições por janela (`notebooks/src/iara_results.py`); predição por gravação = voto majoritário das janelas; SP e ACC balanceada como no artigo.
"""), code(SETUP_CODE),
        md("""
## 1. Os dez modelos

Todos de 4 classes, avaliados nos 10 folds oficiais (split por `ID`). MLP MEL, MLP LOFAR, CNN MEL (do artigo: `test_scripts/cnn_mel_artigo.py`) e os SVM MEL/LOFAR são do pipeline `iara` (features de 0,512 s); os RF usam caches com média em 5 s (`mel5s`/`lofar5s`); os modelos HHT usam o HHT re-extraído (Notebook 28). A CNN LOFAR disponível (`cnn2d_lofar5s_w32_b64_e15_4c`, SP ~35) colapsou e ficou de fora: concordar pouco com um modelo que colapsou não é complementaridade.
"""), code(DESEMP_CODE),
        md(f"""
## 2. Concordância por gravação

Para cada par de modelos e cada fold, nas {br(ns['N_INST'], 0)} instâncias arquivo × fold em comum:
- **Concordância** e **κ de Cohen** entre as predições (κ desconta a concordância esperada pelas frequências das classes);
- **ambos erram** observado e o esperado se os erros fossem independentes;
- **mesmo erro**: quando os dois erram, com que frequência escolhem a mesma classe errada;
- **Q de Yule** dos erros: 0 = independentes, 1 = sempre erram juntos. É a medida que responde "eles se complementam?"; a concordância bruta sobe só porque os dois acertam muito.
"""), code(CONC_CODE),
        md(f"""
**Leitura.**
- **Os erros andam juntos em todos os pares:** Q de Yule de {br(qmin)} a {br(qmax)}. O par mais parecido é **{top_par}** (Q {br(pares.loc[top_par, 'Q de Yule'])}, concordância {br(pares.loc[top_par, 'Concordância (%)'], 1)}%).
- **Por tipo de par (Q médio):** mesma representação, família diferente = {br(qt.mean()[T_SR])}; mesma família, representação diferente = {br(qt.mean()[T_SF])}; família e representação diferentes = {br(qt.mean()[T_DD])}. {'Com a mesma entrada, RF, SVM, MLP e CNN tendem a errar nas mesmas gravações; a representação pesa mais que o classificador.' if rep_mais else 'O classificador pesa mais que a representação.'}
- **Concordância média por tipo:** {', '.join(f'{t}: {br(v, 1)}%' for t, v in ct.items())}.
"""),
        md(f"""
## 3. Erros comuns: as gravações que os 10 modelos erram
"""), code(ERROS_CODE),
        md(f"""
**Leitura.**
- `SMALL` tem a maior taxa de erro comum ({br(por_classe.loc['SMALL', 'pct'], 1)}%) e `BACKGROUND` a menor ({br(por_classe.loc['BACKGROUND', 'pct'], 1)}%).
- O destino mais frequente do erro comum de `SMALL` é **{destino.loc['SMALL'].idxmax()}** ({int(destino.loc['SMALL'].max())} de {int(destino.loc['SMALL'].sum())}). *Hipótese, não testada:* barcos de prático trabalham junto a navios grandes, e a gravação pode conter o ruído do navio atendido; os metadados não registram outras embarcações próximas.
- O erro comum é uma propriedade estável da gravação: em `SMALL`, {int(consist.loc['SMALL', 'erro comum em todas as aparições'])} gravações são erradas pelos 10 modelos em todos os folds em que aparecem no teste, e {int(consist.loc['SMALL', 'em nenhuma'])} de {int(consist.loc['SMALL', 'gravações'])} nunca.
"""),
        md(f"""
## 4. Os erros comuns se concentram em poucos navios?

Contar "quantos navios têm metade dos erros" engana quando poucos navios têm muitas gravações (em `SMALL`, os barcos de prático PILOT 08–14 somam centenas de gravações). Por isso, comparamos com quantos navios têm metade das **instâncias** e fazemos um teste de permutação:

- estatística: χ² de Pearson das contagens de erro comum por navio contra a taxa da classe;
- nulo: a taxa de erro comum de cada **gravação** é embaralhada entre as gravações da mesma classe (preservando a correlação entre folds da mesma gravação e o nº de aparições), com reamostragem binomial; {br(2000, 0)} permutações;
- p pequeno = o navio explica o erro comum além do que a gravação sozinha explicaria. Três testes (um por classe), sem correção para comparações múltiplas.
"""), code(NAVIO_CODE),
        md(f"""
**Leitura.**
""" + "\n".join(
            f"- **{c}:** 50% dos erros comuns em {int(CONC.loc[c, 'navios com 50% dos erros'])} navios, contra {int(CONC.loc[c, 'navios com 50% das instâncias'])} navios para 50% das instâncias; "
            f"efeito de navio {('**forte**' if CONC.loc[c, 'p (permutação)'] < 0.001 else 'fraco' if CONC.loc[c, 'p (permutação)'] < 0.05 else 'não detectado')} "
            f"(χ² {br(CONC.loc[c, 'χ² entre navios'], 0)} vs nulo {br(CONC.loc[c, 'χ² nulo (média)'], 0)}; {pv(CONC.loc[c, 'p (permutação)'])})."
            for c in ("SMALL", "MEDIUM", "LARGE")) + """
- Em `SMALL`, a aparente concentração acompanha a distribuição das gravações (os barcos de prático têm a maioria delas) e o efeito de navio é pequeno. Em `MEDIUM`, ele é real: navios com muitas gravações quase não entram no erro comum, enquanto alguns com poucas gravações concentram erros. Em `LARGE`, a maioria dos navios tem 1–2 gravações e o erro comum se espalha.
"""),
        md("""
## 5. Fatores associados ao erro comum

São associações observacionais (os fatores se confundem entre si: navios com muitas gravações quase sempre aparecem no treino); a Seção 6 testa o primeiro deles diretamente.
"""), code(FATORES_CODE),
        md(f"""
**Leitura.**
- **Navio visto no treino do fold:** em `MEDIUM`, erro comum de {br(vis('MEDIUM', 'não'), 1)}% quando o navio não aparece no treino, contra {br(vis('MEDIUM', 'sim'), 1)}% quando aparece; em `SMALL`, {br(vis('SMALL', 'não'), 1)}% × {br(vis('SMALL', 'sim'), 1)}%; em `LARGE`, {br(vis('LARGE', 'não'), 1)}% × {br(vis('LARGE', 'sim'), 1)}%.
- **Comprimento perto dos limiares de classe (50 m e 100 m):** erro comum de {br(cb(50)['erro_comum_pct'], 1)}% em 50–60 m ({int(cb(50)['instancias'])} instâncias, {int(cb(50)['navios'])} navios) e {br(cb(100)['erro_comum_pct'], 1)}% em 100–120 m ({int(cb(100)['instancias'])} instâncias, {int(cb(100)['navios'])} navios), contra {br(cb(60)['erro_comum_pct'], 1)}% em 60–80 m e {br(cb(200)['erro_comum_pct'], 1)}% acima de 200 m. As classes são definidas por limiares de comprimento sem fronteira acústica; navios logo acima e logo abaixo se parecem. Poucos navios nessas faixas: indicativo.
- **Campanha de gravação (`Dataset`):** em `LARGE`, {br(dataset.loc[('LARGE', 'B'), 'erro_comum_pct'], 1)}% em B e {br(dataset.loc[('LARGE', 'D'), 'erro_comum_pct'], 1)}% em D, contra {br(dataset.loc[('LARGE', 'A'), 'erro_comum_pct'], 1)}% em A e {br(dataset.loc[('LARGE', 'C'), 'erro_comum_pct'], 1)}% em C. Os metadados não dizem se a diferença vem do local, do equipamento ou do tráfego.
"""),
        md(f"""
## 6. Teste direto: split por navio × split por gravação

`scratch/eval_ship_split_5x2cv.py` gera (e grava em `reports/split_navio/folds.csv`) dois 5×2cv com o **mesmo código e as mesmas sementes** (`StratifiedGroupKFold(2)`, 5 repetições), mudando só o agrupamento:
- **split por ID (controle):** grupos = gravação, como o oficial; {br(sobrep['split_id'], 1)}% das gravações de navio no teste têm o navio no treino;
- **split por navio:** grupos = `Ship ID`; {br(sobrep['split_navio'], 1)}% (nenhum navio em treino e teste). `BACKGROUND` não tem navio: cada gravação é o seu próprio grupo nas duas condições.

Treino = uma metade (~853 gravações, como o oficial), teste = a outra metade inteira; sem early stopping, então não há validação. Modelos, com a arquitetura e os hiperparâmetros dos scripts de origem, sobre os caches de 5 s (MEL, LOFAR) e o HHT:
- RF (100 árvores) e SVM Nyström (m = 2000): `scratch/eval_ship_split_5x2cv.py`;
- MLP h32-16 (35 épocas, como `mlp_mel5s_h32-16_e35`) e BiGRU com atenção sobre sequências de 15 janelas LOFAR (15 épocas, como `bigru_att_lofar5s_h128`): `scratch/eval_ship_split_nn_5x2cv.py`, na GPU, semente = fold nas duas condições. Os folds das duas condições são diferentes, então o F-test 5×2cv pareado **não** se aplica entre elas; a comparação é a diferença de médias, com o desvio entre folds.

Agrupar por navio desequilibra os folds (o PILOT 14 tem 87 gravações e o HANG BO 2003, 79): a tabela abaixo mostra o mínimo e o máximo de gravações de teste por classe.
"""), code(SPLIT_CODE),
        md(f"""
**Leitura.**
- **O controle é comparável ao oficial:** nos RF, o SP do split por ID difere do 5×2cv oficial em {', '.join(f'{r} {br(d, 2)}' for r, d in ctrl_d.items())} p.p.; os folds são outros e o teste aqui é a metade inteira (sem separar validação). A menor queda de ACC balanceada no split por navio ({br(-d_acc.max(), 1)} p.p.) {'é maior' if -d_acc.max() > max(abs(v) for v in ctrl_d.values()) else 'NÃO é maior'} que a maior dessas diferenças ({br(max(abs(v) for v in ctrl_d.values()), 2)} p.p.).
- **Separando por navio, o desempenho cai em todos os modelos:** SP de {br(-d_sp.max(), 1)} a {br(-d_sp.min(), 1)} p.p. e ACC balanceada de {br(-d_acc.max(), 1)} a {br(-d_acc.min(), 1)} p.p. {('Folds com SP = 0 (alguma classe com recall 0, o que zera a média geométrica): ' + sp0_txt + '; nesses modelos a queda de SP é exagerada por esse efeito e a ACC balanceada é a medida mais estável. ') if sp0_txt else ''}Em {sep} dos {len(SPLIT)} modelos, o melhor fold do split por navio fica abaixo do pior fold do controle.
- **Por família (Δ ACC balanceada média):** {', '.join(f'{f} {br(v.mean(), 1)} p.p. ({len(v)} modelo{"s" if len(v) > 1 else ""})' for f, v in fam_d.items())}. As redes neurais {'também caem' if max(fam_d['MLP'].max(), fam_d['BiGRU'].max()) < 0 else 'nem sempre caem'}: o efeito não é específico de RF/SVM.
- **A queda se concentra em `{cls_pior}`:** variação média do recall de {', '.join(f'`{c}` {br(d_cls[c].mean(), 1)} p.p.' for c in R.CLASSES)}; `BACKGROUND` é agrupado igual nas duas condições, então serve de referência.
- **Variabilidade:** o desvio do SP entre folds sobe de {br(np.mean(list(std_id.values())), 2)} para {br(np.mean(list(std_nv.values())), 2)} p.p. em média; parte disso vem do desequilíbrio dos folds, parte de quais navios caem no teste.
"""),
        md(f"""
## 7. Conclusões

1. **Os modelos erram juntos.** RF, SVM, MLP e CNN sobre MEL, LOFAR e HHT têm erros fortemente correlacionados; combinar famílias e representações (votos, médias, gating) redistribui acertos entre classes, mas o teto é limitado pelos erros comuns.
2. **O erro comum não é "culpa" de poucos navios**, exceto em `MEDIUM`. Ele se associa a três fatores: navio ausente do treino, comprimento perto dos limiares de 50 m e 100 m e campanha de gravação.
3. **O protocolo do artigo superestima a generalização para navios novos.** Com o mesmo código, em RF, SVM, MLP e BiGRU, só trocar o agrupamento de `ID` para `Ship ID` derruba a ACC balanceada em {br(-d_acc.max(), 1)}–{br(-d_acc.min(), 1)} p.p. (SP: {br(-d_sp.max(), 1)}–{br(-d_sp.min(), 1)} p.p.) Parte do desempenho reportado vem de reconhecer o navio específico, não a sua classe de porte. Isso vale para todos os números deste projeto e do artigo, que usam o split por `ID`.

**Recomendação para o artigo:** reportar também um split por `Ship ID` (ao menos para os modelos principais) e descrever o split por `ID` como "gravações inéditas de navios possivelmente vistos", não como "navios inéditos".

**Limites:**
- No split por navio foram avaliados RF, SVM, MLP e BiGRU; as CNN não (devem cair também, mas não foi medido). MLP e BiGRU usam os caches de 5 s, não o pipeline `iara` de 0,512 s com early stopping dos modelos oficiais.
- O split por navio desequilibra as classes entre folds; o controle por ID não tem esse desequilíbrio, e parte da diferença pode vir dele (o SP é balanceado por classe, o que atenua o efeito).
- `BACKGROUND` foi agrupado por gravação nas duas condições; gravações de fundo da mesma sessão podem ser correlacionadas.
- A definição de erro comum depende do conjunto de 10 modelos escolhido.
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
