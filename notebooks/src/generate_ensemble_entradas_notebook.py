"""
Gera notebooks/33_ensemble_entradas.ipynb: combinar as entradas (MEL, LOFAR, HHT, DEMON) de um mesmo classificador.

Perguntas:
  1. Qual família tem as entradas menos correlacionadas? Por que medidas sobre os erros (Q, Y, κ de Fleiss) e
     sobre as classes preditas (V de Cramér, κ de Cohen) apontam famílias diferentes?
  2. As entradas de cada família têm desempenho equilibrado?
  3. O voto majoritário das entradas supera cada entrada isolada (F-test 5x2cv)? Em qual família?
  4. O melhor voto (BiGRU × MLP) difere?
  5. Uma regra feita à mão (MEL = MEDIUM e LOFAR/HHT = LARGE -> SMALL) funciona?
  6. Uma tabela "combinação de predições -> classe" (BKS) aprendida na outra metade do fold funciona?
  7. Diversidade, equilíbrio e ganho: o que explica onde o voto ajuda?

Fontes: predições por gravação (voto majoritário das janelas) em results/trainings/{multirep,tests}/<exp>/eval/fold_k,
via iara_results.py. Mesmas 16 configurações dos Notebooks 30–32 (a CNN não tem as 4 entradas no mesmo protocolo).
Como nos outros geradores, o mesmo código das células roda aqui para calcular os números citados no markdown.
Rodar na raiz do repositório:  python notebooks/src/generate_ensemble_entradas_notebook.py
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
from scipy.stats import chi2_contingency
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
pd.set_option('display.precision', 2); pd.set_option('display.width', 200)
CLS = R.CLASSES
SMALL, MEDIUM, LARGE = 0, 1, 2
REP = ['MEL', 'LOFAR', 'HHT', 'DEMON']

# 4 famílias × 4 entradas (os mesmos modelos dos Notebooks 30–32)
FAM = {
    'RF': ['rf_mel5s_n100_balanced_4c', 'rf_lofar5s_n100_balanced_4c', 'rf_hht_n100_balanced_4c', 'rf_demon_n100_balanced_4c'],
    'SVM': ['svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0', 'svm_nystroem_6000_lofar_elasticnet_l1r0.15',
            'svm_nystroem_6000_hht_C2.0_elasticnet_l1r0.15_4c', 'svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000_4c'],
    'MLP': ['mlp', 'mlp_lofar', 'mlp_hht_h32-16_e30_4c', 'mlp_demon_h32-16_e30_4c'],
    'BiGRU': ['bigru_att_mel5s_h64_s15-5_e20_4c', 'bigru_att_lofar5s_h128_s15-5_e15_4c', 'bigru_att_hht_h64_s15-5_e20_4c',
              'bigru_att_demon_h64_s15-5_e20_4c'],
}
AL = {f: R.aligned(ex) for f, ex in FAM.items()}  # {família: {fold: DataFrame(File, Target, <exp>...)}}


def metr(y, p):
    """SP, ACC balanceada e recall por classe (mesma definição de iara_results)."""
    return R._metrics(np.asarray(y), np.asarray(p))


def voto(P, desempate):
    """Voto de pluralidade por linha de P (gravações × entradas); empate -> classe de 'desempate' se empatada,
    senão a menor classe empatada."""
    out = np.empty(len(P), dtype=int)
    for i, (row, d) in enumerate(zip(P, desempate)):
        c = np.bincount(row, minlength=4); top = np.flatnonzero(c == c.max())
        out[i] = d if d in top else top[0]
    return out


def por_fold(f, fn):
    """Aplica fn(df, ex) -> predição em cada fold da família f; devolve DataFrame de métricas por fold."""
    rows = []
    for k, d in AL[f].items():
        rows.append({'fold': k, **metr(d.Target.values, fn(d, FAM[f]))})
    return pd.DataFrame(rows).set_index('fold')


SOLO = {f: {r: por_fold(f, lambda d, ex, i=i: d[ex[i]].values) for i, r in enumerate(REP)} for f in FAM}
print('Gravações alinhadas por fold:', {f: int(np.mean([len(d) for d in AL[f].values()])) for f in FAM})
'''

DIV_CODE = r'''
def yule(ea, eb):
    n11, n00 = float((ea & eb).sum()), float((~ea & ~eb).sum())
    n10, n01 = float((ea & ~eb).sum()), float((~ea & eb).sum())
    ad, bc = n11 * n00, n10 * n01
    return (ad - bc) / (ad + bc), (np.sqrt(ad) - np.sqrt(bc)) / (np.sqrt(ad) + np.sqrt(bc))

def cramer_v(pa, pb):
    t = pd.crosstab(pa, pb).values
    return np.sqrt(chi2_contingency(t, correction=False)[0] / (t.sum() * (min(t.shape) - 1)))

def fleiss(M, n_cat):
    N, L = M.shape
    n = np.stack([(M == c).sum(1) for c in range(n_cat)], 1)
    pj = n.sum(0) / (N * L); Pi = (n * (n - 1)).sum(1) / (L * (L - 1)); Pe = np.sum(pj ** 2)
    return (Pi.mean() - Pe) / (1 - Pe)

rows, rows_f = [], []
for f, ex in FAM.items():
    for k, d in AL[f].items():
        y = d.Target.values
        P = np.stack([d[e].values for e in ex], 1); E = P != y[:, None]
        for i, j in itertools.combinations(range(4), 2):
            q, yy = yule(E[:, i], E[:, j])
            ambos = E[:, i] & E[:, j]
            rows.append({'Família': f, 'Par': f'{REP[i]} × {REP[j]}', 'fold': k, 'Q (erros)': q, 'Y (erros)': yy,
                         'V de Cramér (classes)': cramer_v(P[:, i], P[:, j]),
                         'κ de Cohen (classes)': cohen_kappa_score(P[:, i], P[:, j]),
                         'erram juntas com classes diferentes (%)': 100 * np.mean(P[ambos, i] != P[ambos, j]) if ambos.any() else np.nan})
        rows_f.append({'Família': f, 'fold': k, 'κ de Fleiss (acerto/erro)': fleiss((~E).astype(int), 2),
                       'κ de Fleiss (classes)': fleiss(P, 4)})
PAR = pd.DataFrame(rows).groupby(['Família', 'Par']).mean(numeric_only=True).drop(columns='fold')
DIV = PAR.groupby('Família').mean().reindex(list(FAM))
DIV = DIV.join(pd.DataFrame(rows_f).groupby('Família').mean().drop(columns='fold'))
print('Média dos 6 pares de entradas e κ de Fleiss das 4, por família:'); display(DIV.round(3))
QP = PAR['Q (erros)'].unstack('Família')[list(FAM)]
print('Q de Yule por par:'); display(QP.round(2))
'''

EQUIL_CODE = r'''
EQ = pd.DataFrame({f: {r: SOLO[f][r].SP.mean() for r in REP} for f in FAM}).T
EQ['melhor'] = EQ[REP].max(axis=1); EQ['melhor entrada'] = EQ[REP].idxmax(axis=1)
EQ['pior'] = EQ[REP].min(axis=1); EQ['amplitude (melhor − pior)'] = EQ.melhor - EQ.pior
EQ['desvio entre entradas'] = EQ[REP].std(axis=1, ddof=0)
print('SP (%) de cada entrada sozinha (gravações alinhadas, média dos 10 folds):'); display(EQ.round(2))
'''

VOTO_CODE = r'''
VOTO = {f: por_fold(f, lambda d, ex: voto(np.stack([d[e].values for e in ex], 1), d[ex[0]].values)) for f in FAM}
VOTO_SORT = {f: R.majority_vote(FAM[f]) for f in FAM}  # empate sorteado (sensibilidade)
rows = []
for f in FAM:
    r = {'Família': f, 'voto (SP)': VOTO[f].SP.mean(), 'voto ± (folds)': VOTO[f].SP.std(ddof=0),
         'voto, empate sorteado (SP)': VOTO_SORT[f].SP.mean()}
    for e in REP:
        F_, p_ = R.f_test_5x2(VOTO[f].SP.values, SOLO[f][e].SP.values)
        r[f'Δ vs {e}'] = VOTO[f].SP.mean() - SOLO[f][e].SP.mean(); r[f'p vs {e}'] = p_
    rows.append(r)
VT = pd.DataFrame(rows).set_index('Família')
VT['Δ vs melhor entrada'] = VT['voto (SP)'] - EQ.melhor
VT['p vs melhor entrada'] = [VT.loc[f, f'p vs {EQ.loc[f, "melhor entrada"]}'] for f in FAM]
# Correção para testes múltiplos (16 comparações: 4 famílias × 4 entradas): Bonferroni e Holm
PV = pd.Series({(f, e): VT.loc[f, f'p vs {e}'] for f in FAM for e in REP}).sort_values()
m = len(PV)
holm, ok = {}, True
for i, (key, p_) in enumerate(PV.items()):
    ok = ok and p_ <= 0.05 / (m - i)
    holm[key] = ok
MULT = pd.DataFrame({'p': PV, 'Bonferroni (p ≤ 0,05/16)': PV <= 0.05 / m, 'Holm': pd.Series(holm)})
print('Voto majoritário das 4 entradas (empate -> MEL) × cada entrada sozinha (F-test 5×2cv):')
display(VT.round(3))
REC = pd.DataFrame({f: {c: VOTO[f][f'R_{c}'].mean() - SOLO[f][EQ.loc[f, 'melhor entrada']][f'R_{c}'].mean() for c in CLS}
                    for f in FAM}).T
print('Δ recall por classe: voto − melhor entrada da família'); display(REC.round(1))
print('Testes múltiplos (significativo após a correção = True):'); display(MULT.round(4))

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(FAM))
for i, r in enumerate(REP):
    ax.bar(x + (i - 2) * 0.16, [SOLO[f][r].SP.mean() for f in FAM], 0.16, label=r, alpha=0.75)
ax.bar(x + 2 * 0.16, VT['voto (SP)'], 0.16, label='voto das 4', color='black')
ax.set_xticks(x + 0.08, list(FAM)); ax.set_ylim(40, 72); ax.set_ylabel('SP (%)'); ax.legend(ncol=5, fontsize=8)
ax.set_title('Cada entrada sozinha × voto das quatro'); plt.tight_layout(); plt.show()
'''

BG_CODE = r'''
# BiGRU × MLP: os dois melhores votos, nas gravações comuns às 8 configurações
allx = FAM['BiGRU'] + FAM['MLP']
vb, vm = [], []
for k, d in R.aligned(allx).items():
    y = d.Target.values
    for lst, ex in ((vb, FAM['BiGRU']), (vm, FAM['MLP'])):
        lst.append(metr(y, voto(np.stack([d[e].values for e in ex], 1), d[ex[0]].values))['SP'])
F_BM, P_BM = R.f_test_5x2(np.array(vb), np.array(vm))
print(f'voto BiGRU {np.mean(vb):.2f} ± {np.std(vb):.2f} | voto MLP {np.mean(vm):.2f} ± {np.std(vm):.2f} | '
      f'Δ {np.mean(vb) - np.mean(vm):+.2f}, F = {F_BM:.2f}, p = {P_BM:.3f}')
VB_M, VM_M = np.mean(vb), np.mean(vm)
'''

REGRA_CODE = r'''
REGRAS = {
    'MEL=MEDIUM e LOFAR=LARGE e HHT=LARGE': lambda m, l, h: (m == MEDIUM) & (l == LARGE) & (h == LARGE),
    'MEL=MEDIUM e LOFAR=LARGE': lambda m, l, h: (m == MEDIUM) & (l == LARGE),
    'MEL=MEDIUM e (LOFAR=LARGE ou HHT=LARGE)': lambda m, l, h: (m == MEDIUM) & ((l == LARGE) | (h == LARGE)),
}
rows = []
for f, ex in FAM.items():
    for nome, fn in REGRAS.items():
        cnt = np.zeros(4)
        for d in AL[f].values():
            m = fn(d[ex[0]].values, d[ex[1]].values, d[ex[2]].values)
            cnt += np.bincount(d.Target.values[m], minlength=4)
        rows.append({'Família': f, 'regra': nome, 'vezes (10 folds)': int(cnt.sum()),
                     **{f'% {c}': 100 * cnt[i] / max(cnt.sum(), 1) for i, c in enumerate(CLS)}})
PADRAO = pd.DataFrame(rows).set_index(['Família', 'regra'])
print('Classe verdadeira das gravações em que o padrão aparece (todas as gravações de teste):'); display(PADRAO.round(1))

def aplica(d, ex):
    p = d[ex[0]].values.copy()
    p[REGRAS['MEL=MEDIUM e LOFAR=LARGE e HHT=LARGE'](d[ex[0]].values, d[ex[1]].values, d[ex[2]].values)] = SMALL
    return p
rows = []
for f in FAM:
    rg = por_fold(f, aplica); F_, p_ = R.f_test_5x2(rg.SP.values, SOLO[f]['MEL'].SP.values)
    rows.append({'Família': f, 'SP MEL': SOLO[f]['MEL'].SP.mean(), 'SP com a regra': rg.SP.mean(),
                 'Δ SP': rg.SP.mean() - SOLO[f]['MEL'].SP.mean(), 'p': p_,
                 **{f'Δ recall {c}': rg[f'R_{c}'].mean() - SOLO[f]['MEL'][f'R_{c}'].mean() for c in CLS}})
REGRA = pd.DataFrame(rows).set_index('Família')
print('Trocar a predição do MEL por SMALL quando o padrão aparece:'); display(REGRA.round(3))
'''

BKS_CODE = r'''
def bks(f, usar):
    """Tabela 'combinação de predições -> classe' aprendida no fold k^1 (a outra metade da mesma repetição)
    e aplicada no fold k; pesos de classe balanceados; combinação não vista -> predição do MEL."""
    ex = FAM[f]; rows = []
    for k in range(10):
        tr, te = AL[f][k ^ 1], AL[f][k]
        Ptr = np.stack([tr[ex[i]].values for i in usar], 1); ytr = tr.Target.values
        w = {c: 1 / np.mean(ytr == c) for c in range(4)}
        tab = {}
        for row, yy in zip(map(tuple, Ptr), ytr):
            tab.setdefault(row, np.zeros(4))[yy] += w[yy]
        Pte = np.stack([te[ex[i]].values for i in usar], 1); mel = te[ex[0]].values
        pred = np.array([tab[tuple(r)].argmax() if tuple(r) in tab else m for r, m in zip(Pte, mel)])
        rows.append({'fold': k, **metr(te.Target.values, pred)})
    return pd.DataFrame(rows).set_index('fold')

rows = []
for usar in ([0, 1, 2], [0, 1, 2, 3]):
    nome = ' + '.join(REP[i] for i in usar)
    for f in FAM:
        b = bks(f, usar); F_, p_ = R.f_test_5x2(b.SP.values, SOLO[f]['MEL'].SP.values)
        rows.append({'entradas': nome, 'Família': f, 'combinações possíveis': 4 ** len(usar),
                     'SP MEL': SOLO[f]['MEL'].SP.mean(), 'SP BKS': b.SP.mean(), 'Δ vs MEL': b.SP.mean() - SOLO[f]['MEL'].SP.mean(), 'p': p_,
                     'Δ recall SMALL': b.R_SMALL.mean() - SOLO[f]['MEL'].R_SMALL.mean()})
BKS = pd.DataFrame(rows).set_index(['entradas', 'Família'])
display(BKS.round(3))
'''

DIVGANHO_CODE = r'''
RES = pd.DataFrame({'Q médio (erros)': DIV['Q (erros)'], 'κ de Fleiss (acerto/erro)': DIV['κ de Fleiss (acerto/erro)'],
                    'κ de Cohen (classes)': DIV['κ de Cohen (classes)'], 'amplitude das entradas': EQ['amplitude (melhor − pior)'],
                    'pior entrada (SP)': EQ.pior, 'melhor entrada (SP)': EQ.melhor, 'voto (SP)': VT['voto (SP)'],
                    'ganho do voto sobre a melhor': VT['Δ vs melhor entrada'], 'p': VT['p vs melhor entrada']})
display(RES.round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, col in zip(ax, ['κ de Fleiss (acerto/erro)', 'amplitude das entradas']):
    a.scatter(RES[col], RES['ganho do voto sobre a melhor'], s=80)
    for f in RES.index:
        a.annotate(f, (RES.loc[f, col], RES.loc[f, 'ganho do voto sobre a melhor']), xytext=(5, 5), textcoords='offset points')
    a.axhline(0, color='gray', lw=0.8); a.set_xlabel(col); a.set_ylabel('ganho do voto sobre a melhor entrada (SP)')
ax[0].set_title('Diversidade dos erros × ganho'); ax[1].set_title('Desequilíbrio entre entradas × ganho')
plt.tight_layout(); plt.show()
'''


def run(ns, src):
    exec(src, ns)


def build(filename=os.path.join(NOTEBOOKS_DIR, "33_ensemble_entradas.ipynb")):
    ns = {}
    for src in (SETUP_CODE, DIV_CODE, EQUIL_CODE, VOTO_CODE, BG_CODE, REGRA_CODE, BKS_CODE, DIVGANHO_CODE):
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    DIV, QP, EQ, VT, REC = ns["DIV"], ns["QP"], ns["EQ"], ns["VT"], ns["REC"]
    PADRAO, REGRA, BKS, RES, MULT = ns["PADRAO"], ns["REGRA"], ns["BKS"], ns["RES"], ns["MULT"]
    F_BM, P_BM, VB_M, VM_M = ns["F_BM"], ns["P_BM"], ns["VB_M"], ns["VM_M"]
    FAMS = list(ns["FAM"])
    REP = ns["REP"]
    holm_bg = [e for e in REP if MULT.loc[("BiGRU", e), "Holm"]]
    bonf_bg = [e for e in REP if MULT.loc[("BiGRU", e), "Bonferroni (p ≤ 0,05/16)"]]
    holm_ok = [f"{f} vs {e}" for (f, e), v in MULT["Holm"].items() if v]

    q, fl, kc, vc = DIV["Q (erros)"], DIV["κ de Fleiss (acerto/erro)"], DIV["κ de Cohen (classes)"], DIV["V de Cramér (classes)"]
    dif = DIV["erram juntas com classes diferentes (%)"]
    sig = [f for f in FAMS if VT.loc[f, "p vs melhor entrada"] < 0.05]
    sig_all = [f for f in FAMS if all(VT.loc[f, f"p vs {e}"] < 0.05 for e in REP)]
    best_v = VT["voto (SP)"].idxmax()
    top_g = VT["Δ vs melhor entrada"].idxmax()
    pmax_top = max(VT.loc[top_g, f"p vs {e}"] for e in REP)
    n_testes = 4 * len(FAMS)
    bonf = 0.05 / n_testes
    padrao3 = PADRAO.xs("MEL=MEDIUM e LOFAR=LARGE e HHT=LARGE", level="regra")
    bks_d = BKS["Δ vs MEL"]
    bks_sig = int((BKS["p"] < 0.05).sum())

    cells = [md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 33: Ensemble de Entradas — Diversidade, Equilíbrio e Voto

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

Os Notebooks 30–32 mostraram que MEL, LOFAR, HHT e DEMON erram, em parte, gravações diferentes. Aqui perguntamos se essa diversidade vira acerto quando as entradas de um mesmo classificador são combinadas, e por que isso acontece em uma família e não em outras.

1. Qual família tem as entradas menos correlacionadas? Por que as medidas **sobre os erros** e **sobre as classes** apontam famílias diferentes?
2. As entradas de cada família têm desempenho equilibrado?
3. O **voto majoritário** das quatro entradas supera cada entrada isolada?
4. O melhor voto: BiGRU × MLP.
5. Uma **regra feita à mão** (MEL = MEDIUM e LOFAR/HHT = LARGE → SMALL) funciona?
6. Uma **tabela de combinações** (BKS) aprendida nos dados funciona?
7. Diversidade, equilíbrio e ganho.

**Resumo:**
- **Erros (Q, Y, κ de Fleiss):** a família com as entradas menos correlacionadas é a **{q.idxmin()}** (Q {br(q.min())}, κ de Fleiss {br(fl[q.idxmin()])}); a mais correlacionada é a {q.idxmax()} (Q {br(q.max())}). As diferenças entre famílias são pequenas.
- **Classes (V, κ de Cohen):** o menor valor é do **{kc.idxmin()}**, porque nele as entradas que erram juntas dizem classes diferentes com mais frequência ({br(dif[kc.idxmin()], 0)}% dos erros em comum, contra {br(dif.drop(kc.idxmin()).min(), 0)}–{br(dif.drop(kc.idxmin()).max(), 0)}% nas outras famílias).
- **Voto das 4 entradas:** {'só no ' + ', '.join(sig) + ' o ganho sobre a melhor entrada é significativo' if sig else 'em nenhuma família o ganho sobre a melhor entrada é significativo'} ({top_g}: {sg(VT.loc[top_g, 'Δ vs melhor entrada'])} pontos; {pv(VT.loc[top_g, 'p vs melhor entrada'])}). {('No ' + ', '.join(sig_all) + ', o voto supera cada uma das quatro entradas sem correção (maior p = ' + br(pmax_top, 3) + '); com a correção de Holm para as 16 comparações, continua superando ' + (', '.join(holm_bg) if holm_bg else 'nenhuma') + '.') if sig_all else ''}
- **Melhor SP final:** voto {best_v} ({br(VT.loc[best_v, 'voto (SP)'])}); BiGRU × MLP: Δ = {sg(VB_M - VM_M)}, {pv(P_BM)} (empate).
- **Regra MEDIUM/LARGE → SMALL:** quando o padrão aparece, a gravação é SMALL em só {br(padrao3['% SMALL'].min(), 0)}–{br(padrao3['% SMALL'].max(), 0)}% dos casos (o mais comum é LARGE). Aplicá-la muda o SP em {sg(REGRA['Δ SP'].min())} a {sg(REGRA['Δ SP'].max())} pontos, sem significância.
- **BKS:** Δ de {sg(bks_d.min())} a {sg(bks_d.max())} pontos contra o MEL; {bks_sig} de {len(bks_d)} comparações significativas.
- **O que explica o ganho do voto:** diversidade dos erros **e** equilíbrio entre as entradas. A família em que o voto ganha é a que tem as duas coisas.

Predição por gravação = voto majoritário das janelas; todas as tabelas são calculadas na execução (`notebooks/src/iara_results.py`).
"""), code(SETUP_CODE),
        md("""
## 1. Diversidade por família: erros × classes

Duas perguntas diferentes:

| Medida | Sobre | Pergunta |
|---|---|---|
| Q e Y de Yule, κ de Fleiss (acerto/erro) | acerto/erro | as entradas erram **as mesmas gravações**? |
| V de Cramér, κ de Cohen, κ de Fleiss (classes) | classe predita | as entradas dizem **a mesma classe**? |

Elas divergem num caso: duas entradas erram a mesma gravação SMALL, uma dizendo MEDIUM e a outra LARGE. Para o Q é **erro em comum**; para o κ de Cohen e o V é **discordância**. A coluna "erram juntas com classes diferentes" mede quantas vezes isso acontece.

Para ensembles, a pergunta relevante é a primeira (quando uma erra, a outra acerta?); é a que Kuncheva e Whitaker (2003) usam para medir diversidade.
"""), code(DIV_CODE),
        md(f"""
**Leitura.**
- **Pelos erros**, a ordem do menos ao mais correlacionado é: """ + ", ".join(f"{f} (Q {br(q[f])}, Fleiss {br(fl[f])})" for f in q.sort_values().index) + f""". A diferença entre {q.idxmin()} e as demais é pequena (o κ de Fleiss varia ~0,02 entre folds).
- **Pelas classes**, o menor κ de Cohen é do {kc.idxmin()} ({br(kc.min())}) e o menor V é do {vc.idxmin()} ({br(vc.min())}). No {kc.idxmin()}, {br(dif[kc.idxmin()], 0)}% das vezes em que duas entradas erram juntas elas dizem classes diferentes (média das outras famílias: {br(dif.drop(kc.idxmin()).mean(), 0)}%). Discordar na classe errando não é complementaridade.
- **O par que mais distingue o BiGRU é LOFAR × HHT:** Q """ + ", ".join(f"{f} {br(QP.loc['LOFAR × HHT', f])}" for f in FAMS) + """. No RF e no MLP, LOFAR e HHT erram quase sempre juntas; no BiGRU, bem menos (coerente com a fusão BiGRU LOFAR + HHT, a única significativa do Notebook 30).
"""),
        md("""
## 2. Equilíbrio: as entradas têm desempenho parecido?
"""), code(EQUIL_CODE),
        md(f"""
**Leitura.**
- Amplitude entre a melhor e a pior entrada: """ + ", ".join(f"{f} {br(EQ.loc[f, 'amplitude (melhor − pior)'], 1)} pontos" for f in EQ['amplitude (melhor − pior)'].sort_values().index) + f""".
- A pior entrada é o DEMON em todas as famílias; no RF e no SVM ele fica perto de {br(EQ.loc[['RF', 'SVM'], 'pior'].mean(), 0)}%.
- Num voto, uma entrada fraca "vota errado" com frequência; ela só ajuda se as outras errarem gravações diferentes **e** ela acertar parte delas.
"""),
        md("""
## 3. Voto majoritário das quatro entradas

Para cada gravação, a classe final é a mais votada entre as quatro entradas da mesma família (empate → a classe do MEL, se empatada). Não há treino novo. Comparação com cada entrada isolada pelo F-test 5×2cv (Alpaydin, 1999). Como sensibilidade, o voto com empate sorteado (`iara_results.majority_vote`).
"""), code(VOTO_CODE),
        md(f"""
**Leitura.**
- **Ganho sobre a melhor entrada da família:** """ + "; ".join(f"{f} {sg(VT.loc[f, 'Δ vs melhor entrada'])} ({pv(VT.loc[f, 'p vs melhor entrada'])}, contra {EQ.loc[f, 'melhor entrada']})" for f in FAMS) + f""".
- {('**No ' + ', '.join(sig_all) + ', o voto supera cada uma das quatro entradas** (maior p = ' + br(pmax_top, 3) + ') e melhora todas as classes: ' + ', '.join(f'{c} {sg(REC.loc[sig_all[0], c], 1)}' for c in REC.columns) + '.') if sig_all else 'Em nenhuma família o voto supera as quatro entradas com significância.'}
- **Testes múltiplos:** são {n_testes} comparações (4 famílias × 4 entradas). Com Bonferroni (α = 0,05/{n_testes} = {br(bonf, 4)}), o voto do BiGRU continua superando: {', '.join(bonf_bg) if bonf_bg else 'nenhuma entrada'}; com Holm (menos conservador, controla o mesmo erro): {', '.join(holm_bg) if holm_bg else 'nenhuma entrada'}. Todas as comparações que sobrevivem ao Holm: {', '.join(holm_ok)}. Ou seja, com Holm o voto do BiGRU supera {', '.join(holm_bg) if holm_bg else 'nenhuma entrada'}{('; contra ' + ', '.join(f"{e} ({pv(MULT.loc[('BiGRU', e), 'p'])})" for e in REP if e not in holm_bg) + ' a diferença não resiste à correção') if len(holm_bg) < len(REP) else ''}.
- O desempate sorteado dá SP {', '.join(f'{f} {br(VT.loc[f, "voto, empate sorteado (SP)"])}' for f in FAMS)}: a conclusão não depende da regra de desempate.
"""),
        md("""
## 4. O melhor voto: BiGRU × MLP

O BiGRU é a família que mais ganha com o voto; o MLP tem a melhor entrada isolada (LOFAR). Qual chega mais alto?
"""), code(BG_CODE),
        md(f"""
**Leitura.**
- Voto BiGRU {br(VB_M)} × voto MLP {br(VM_M)}: Δ = {sg(VB_M - VM_M)}, {pv(P_BM)}. **Empatam.**
- O BiGRU ganha mais com o voto porque parte de entradas mais fracas e mais parecidas entre si (havia mais o que corrigir); o MLP já tem o LOFAR forte. **Ganhar mais com o ensemble não é o mesmo que chegar mais alto.**
"""),
        md("""
## 5. Uma regra feita à mão: MEL = MEDIUM e LOFAR/HHT = LARGE → SMALL?

Nas gravações SMALL que o MEL erra, ele costuma dizer MEDIUM e LOFAR/HHT costumam dizer LARGE (Notebook 32). A regra "quando aparecer esse padrão, é SMALL" funciona? Olhamos **todas** as gravações de teste em que o padrão aparece, não só as SMALL.
"""), code(REGRA_CODE),
        md(f"""
**Leitura.**
- Com o padrão completo (MEL = MEDIUM, LOFAR = LARGE, HHT = LARGE), a gravação é SMALL em """ + ", ".join(f"{f} {br(padrao3.loc[f, '% SMALL'], 0)}%" for f in FAMS) + " e LARGE em " + ", ".join(f"{f} {br(padrao3.loc[f, '% LARGE'], 0)}%" for f in FAMS) + f""". **O mais comum é a gravação ser LARGE**: são navios grandes que o MEL confunde com MEDIUM e que LOFAR/HHT acertam.
- **Armadilha da taxa-base:** um padrão comum entre os SMALL errados é ainda mais comum em outras classes.
- Aplicar a regra: recall SMALL {sg(REGRA['Δ recall SMALL'].min(), 1)} a {sg(REGRA['Δ recall SMALL'].max(), 1)}, MEDIUM {sg(REGRA['Δ recall MEDIUM'].min(), 1)} a {sg(REGRA['Δ recall MEDIUM'].max(), 1)}; SP {sg(REGRA['Δ SP'].min())} a {sg(REGRA['Δ SP'].max())} (p de {br(REGRA['p'].min(), 2)} a {br(REGRA['p'].max(), 2)}).
"""),
        md("""
## 6. Generalizando a regra: tabela de combinações (BKS)

O *Behavior Knowledge Space* (Huang e Suen, 1995) aprende, para **cada combinação** de predições das entradas, a classe mais provável. A tabela é aprendida no outro fold da mesma repetição (a outra metade dos dados) e aplicada no fold de teste, com pesos de classe balanceados; combinações nunca vistas usam a predição do MEL.
"""), code(BKS_CODE),
        md(f"""
**Leitura.**
- Δ contra o MEL de {sg(bks_d.min())} a {sg(bks_d.max())} pontos; {bks_sig} de {len(bks_d)} comparações com p < 0,05.
- Com 3 entradas há 64 combinações possíveis, com 4 há 256, para ~640 gravações de teste por fold: muitas combinações têm poucas gravações, e a tabela decora ruído. O voto, que não aprende nada, é mais robusto.
"""),
        md("""
## 7. Diversidade, equilíbrio e ganho
"""), code(DIVGANHO_CODE),
        md(f"""
**Leitura.**
- O voto ganha onde as duas condições aparecem juntas: **erros pouco correlacionados** e **entradas equilibradas**. """ + "; ".join(f"{f}: Fleiss {br(RES.loc[f, 'κ de Fleiss (acerto/erro)'])}, amplitude {br(RES.loc[f, 'amplitude das entradas'], 1)}, ganho {sg(RES.loc[f, 'ganho do voto sobre a melhor'])}" for f in FAMS) + f""".
- No RF e no SVM a diversidade é parecida com a do BiGRU, mas o DEMON fraco (~{br(EQ.loc[['RF', 'SVM'], 'pior'].mean(), 0)}%) atrapalha o voto; no MLP as entradas são as mais correlacionadas e o LOFAR já domina.
- É o que Kuncheva e Whitaker (2003) descrevem: as medidas de diversidade se relacionam de forma fraca com o ganho do ensemble; o que importa é o compromisso entre acurácia individual e diversidade. Com 4 famílias, este é um padrão descritivo, não um teste.
"""),
        md(f"""
## 8. Conclusões

1. **A família com as entradas menos correlacionadas nos erros é o {q.idxmin()}**, por pouco. As medidas sobre classes preditas (V, κ de Cohen) apontam o {kc.idxmin()}, porque nele as entradas erram juntas para classes diferentes — o que não é complementaridade.
2. **O voto majoritário das quatro entradas {'só dá ganho significativo no ' + ', '.join(sig) if sig else 'não dá ganho significativo em nenhuma família'}** ({top_g}: {sg(VT.loc[top_g, 'Δ vs melhor entrada'])} pontos sobre a melhor entrada, {pv(VT.loc[top_g, 'p vs melhor entrada'])} sem correção). Com a correção de Holm para as 16 comparações, o ganho sobre {', '.join(holm_bg) if holm_bg else 'nenhuma entrada'} se mantém{'' if 'MEL' in holm_bg else ', mas o ganho sobre o MEL fica no limite'}. É a combinação de entradas com evidência mais forte até aqui, mas pede confirmação (por exemplo, no split por navio).
3. **Ganhar mais não é chegar mais alto:** o voto do BiGRU empata com o do MLP ({br(VB_M)} × {br(VM_M)}, {pv(P_BM)}).
4. **Diversidade sozinha não basta:** o voto precisa de entradas equilibradas. Para RF e SVM, melhorar o DEMON vem antes de combinar.
5. **Regras feitas à mão e tabelas de combinações não funcionaram:** a regra MEDIUM/LARGE cai na armadilha da taxa-base; o BKS decora ruído.

**Limites:**
- Split por gravação (`ID`), como no artigo; o Notebook 29 mostrou que ele infla os resultados em 6–10 pontos de ACC. O ganho do voto precisa ser confirmado no split por navio.
- Vários testes nesta análise; os resultados além do BiGRU são exploratórios.
- No MLP, MEL/LOFAR (0,512 s) e HHT/DEMON (5 s) vêm de pipelines diferentes.
- A CNN fica de fora: não tem as 4 entradas avaliadas no mesmo protocolo (a CNN LOFAR do artigo ainda está treinando; a CNN MEL do repositório não tem a arquitetura do artigo).
- Combinar famílias diferentes (ex.: MLP LOFAR + BiGRU MEL) não foi testado; exigiria escolher os membros dentro de cada fold para não inflar o ganho.
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
