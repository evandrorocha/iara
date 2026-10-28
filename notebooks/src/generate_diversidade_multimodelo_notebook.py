"""
Gera notebooks/31_diversidade_multimodelo_entradas.ipynb: concordância e diversidade entre as entradas (MEL, LOFAR,
HHT, DEMON) para o mesmo classificador, par a par e com as quatro entradas ao mesmo tempo.

Medidas:
  * par a par: Q e Y de Yule (erros, tabela 2x2), V de Cramér e κ de Cohen (classes preditas, tabela 4x4);
  * as quatro de uma vez (medidas não pareadas, Kuncheva e Whitaker, 2003): κ de Fleiss sobre acerto/erro e sobre
    as classes preditas; distribuição do número de entradas que erram cada gravação contra a esperada se os erros
    fossem independentes (binomial de Poisson); efeito de retirar cada entrada (κ com 3 entradas).

Fontes: predições por janela em results/trainings/{multirep,tests}/<exp>/eval/fold_k (via iara_results.py).
Mesmos modelos do Notebook 30; a CNN entra com as entradas disponíveis (a CNN LOFAR do artigo, 'cnn_lofar',
entra automaticamente quando os 10 folds existirem).
Como nos outros geradores, o mesmo código das células roda aqui para calcular os números citados no markdown.
Rodar na raiz do repositório:  python notebooks/src/generate_diversidade_multimodelo_notebook.py
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
from scipy.stats import chi2_contingency, chisquare
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
REPS = ['MEL', 'LOFAR', 'HHT', 'DEMON']


def disponivel(exp):
    try:
        R.file_predictions(exp)
        return True
    except FileNotFoundError:
        return False


# Mesmos modelos do Notebook 30: um por (família, entrada), 4 classes, 10 folds oficiais (split por ID)
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
    'CNN': {'MEL': 'cnn', 'LOFAR': 'cnn_lofar', 'HHT': 'hht_2d_cnn_e40p7_4c', 'DEMON': 'hcnn_demon_tpsw_med5_e60p8_4c'},
}
for f in FAM:
    FAM[f] = {r: e for r, e in FAM[f].items() if disponivel(e)}
COMPLETAS = [f for f in FAM if len(FAM[f]) == 4]
print({f: list(r) for f, r in FAM.items()})
print('Famílias com as 4 entradas:', COMPLETAS)
'''

MEDIDAS_CODE = r'''
def yule(ea, eb):
    """Q e Y de Yule dos erros (a = ambos erram, d = ambos acertam, b/c = só um erra)."""
    n11, n00 = float((ea & eb).sum()), float((~ea & ~eb).sum())
    n10, n01 = float((ea & ~eb).sum()), float((~ea & eb).sum())
    ad, bc = n11 * n00, n10 * n01
    return (ad - bc) / (ad + bc), (np.sqrt(ad) - np.sqrt(bc)) / (np.sqrt(ad) + np.sqrt(bc))


def cramer_v(pa, pb):
    t = pd.crosstab(pa, pb).values
    chi2 = chi2_contingency(t, correction=False)[0]
    return np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1)))


def fleiss(M, n_cat):
    """κ de Fleiss: M = (gravações × avaliadores) com rótulos 0..n_cat-1."""
    N, L = M.shape
    n = np.stack([(M == c).sum(1) for c in range(n_cat)], 1)
    pj = n.sum(0) / (N * L)
    Pi = (n * (n - 1)).sum(1) / (L * (L - 1))
    Pe = np.sum(pj ** 2)
    return (Pi.mean() - Pe) / (1 - Pe)


def kappa_indep(y, pa, pb, n_cat=4):
    """κ de Cohen esperado se A e B errassem de forma independente dada a classe verdadeira:
    P(A = i, B = j) = Σ_c π_c · P_A(i | c) · P_B(j | c), com as matrizes de confusão observadas."""
    J = np.zeros((n_cat, n_cat))
    for c in np.unique(y):
        m = y == c
        J += m.mean() * np.outer(np.bincount(pa[m], minlength=n_cat) / m.sum(),
                                 np.bincount(pb[m], minlength=n_cat) / m.sum())
    po, pe = np.trace(J), J.sum(1) @ J.sum(0)
    return (po - pe) / (1 - pe)


def poisson_binomial(ps):
    d = np.array([1.0])
    for p in ps:
        d = np.convolve(d, [1 - p, p])
    return d


AL = {f: R.aligned(list(reps.values())) for f, reps in FAM.items()}
'''

PARES_CODE = r'''
rows = []
for f, reps in FAM.items():
    for ra, rb in itertools.combinations([r for r in REPS if r in reps], 2):
        a, b = reps[ra], reps[rb]
        per = []
        for d in AL[f].values():
            y = d.Target.values
            q, yy = yule(d[a].values != y, d[b].values != y)
            k = cohen_kappa_score(d[a].values, d[b].values)
            ki = kappa_indep(y.astype(int), d[a].values.astype(int), d[b].values.astype(int))
            per.append({'Q (erros)': q, 'Y (erros)': yy, 'V de Cramér (classes)': cramer_v(d[a].values, d[b].values),
                        'κ de Cohen (classes)': k, 'κ se erros indep.': ki, 'κ − κ indep.': k - ki,
                        'concordância (%)': 100 * np.mean(d[a].values == d[b].values)})
        rows.append({'Família': f, 'Par': f'{ra} × {rb}', **pd.DataFrame(per).mean()})
PARES = pd.DataFrame(rows)
display(PARES.set_index(['Família', 'Par']).round(2))

ORDEM = [f'{a} × {b}' for a, b in itertools.combinations(REPS, 2)]
MED = PARES.groupby('Par')[['Q (erros)', 'Y (erros)', 'V de Cramér (classes)', 'κ de Cohen (classes)',
                            'κ se erros indep.', 'κ − κ indep.']].mean()
MED = MED.reindex([p for p in ORDEM if p in MED.index]).sort_values('Q (erros)', ascending=False)
print('Média das famílias, por par de entradas:')
display(MED.round(2))

fig, ax = plt.subplots(figsize=(7, 4.5))
for col, mk in (('Q (erros)', 'o'), ('Y (erros)', 's'), ('V de Cramér (classes)', '^'), ('κ de Cohen (classes)', 'D')):
    ax.plot(range(len(MED)), MED[col], marker=mk, label=col)
ax.plot(range(len(MED)), MED['κ se erros indep.'], ls='--', color='gray', marker='D', mfc='none',
        label='κ de Cohen se erros indep.')
ax.set_xticks(range(len(MED)), MED.index, rotation=25, ha='right'); ax.set_ylim(0, 1)
ax.set_ylabel('valor (média das famílias)'); ax.legend(); ax.set_title('Pares de entradas: quatro medidas, mesma ordem')
plt.tight_layout(); plt.show()
'''

MULTI_CODE = r'''
def multi(f, reps_sel):
    ex = [FAM[f][r] for r in reps_sel]
    L = len(ex)
    ks, kc, obs, exp, pchi = [], [], np.zeros(L + 1), np.zeros(L + 1), []
    for d in AL[f].values():
        y = d.Target.values
        P = np.stack([d[e].values for e in ex], 1)
        C = (P == y[:, None]).astype(int)
        ks.append(fleiss(C, 2))
        kc.append(fleiss(P, 4))
        o = np.bincount(L - C.sum(1), minlength=L + 1)
        e = poisson_binomial(1 - C.mean(0)) * len(y)
        obs += o; exp += e
        pchi.append(chisquare(o, e)[1])
    return {'κ Fleiss acerto/erro': np.mean(ks), 'κ desvio (folds)': np.std(ks),
            'κ Fleiss classes': np.mean(kc), 'obs': obs / 10, 'exp': exp / 10, 'p χ² máx (folds)': max(pchi)}


MULTI, DIST = [], []
for f in FAM:
    reps_f = [r for r in REPS if r in FAM[f]]
    m = multi(f, reps_f)
    L = len(reps_f)
    MULTI.append({'Família': f, 'Entradas': ' + '.join(reps_f), 'κ Fleiss acerto/erro': m['κ Fleiss acerto/erro'],
                  'κ desvio (folds)': m['κ desvio (folds)'], 'κ Fleiss classes': m['κ Fleiss classes'],
                  'todas erram: obs.': m['obs'][L], 'todas erram: indep.': m['exp'][L],
                  'todas erram: obs./indep.': m['obs'][L] / m['exp'][L],
                  'todas acertam: obs.': m['obs'][0], 'todas acertam: indep.': m['exp'][0],
                  'p χ² (pior fold)': m['p χ² máx (folds)']})
    for k in range(L + 1):
        DIST.append({'Família': f, 'nº de entradas que erram': k, 'observado': m['obs'][k], 'independência': m['exp'][k]})
MULTI = pd.DataFrame(MULTI).set_index('Família')
DIST = pd.DataFrame(DIST)
display(MULTI.round(3))

fams = list(FAM)
fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.4), sharey=True)
for ax, f in zip(axes, fams):
    d = DIST[DIST['Família'] == f]
    x = d['nº de entradas que erram'].values
    ax.bar(x - 0.2, d.observado, 0.4, label='observado', color='#C44E52')
    ax.bar(x + 0.2, d['independência'], 0.4, label='se independentes', color='#4C72B0')
    ax.set_xticks(x); ax.set_xlabel('nº de entradas que erram'); ax.set_title(f'{f} ({len(x) - 1} entradas)')
axes[0].set_ylabel('gravações por fold (média)'); axes[0].legend(fontsize=8)
plt.suptitle('Quantas entradas erram cada gravação: observado × esperado se os erros fossem independentes', y=1.02)
plt.tight_layout(); plt.show()
'''

LOO_CODE = r'''
# Retirar uma entrada por vez (só famílias com as 4): κ maior sem a entrada = ela trazia diversidade
loo = []
for f in COMPLETAS:
    k4 = MULTI.loc[f, 'κ Fleiss acerto/erro']
    linha = {'Família': f, 'κ com as 4': k4}
    for r in REPS:
        linha[f'sem {r}'] = multi(f, [x for x in REPS if x != r])['κ Fleiss acerto/erro'] - k4
    loo.append(linha)
LOO = pd.DataFrame(loo).set_index('Família')
print('Δκ de Fleiss (acerto/erro) ao retirar cada entrada; positivo = a entrada retirada trazia diversidade')
display(LOO.round(3))
'''


def run(ns, src):
    exec(src, ns)


def build(filename=os.path.join(NOTEBOOKS_DIR, "31_diversidade_multimodelo_entradas.ipynb")):
    ns = {}
    for src in (SETUP_CODE, MEDIDAS_CODE, PARES_CODE, MULTI_CODE, LOO_CODE):
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    MED, PARES, MULTI, LOO, FAM = ns["MED"], ns["PARES"], ns["MULTI"], ns["LOO"], ns["FAM"]
    COMPLETAS = ns["COMPLETAS"]
    cnn_ok = "LOFAR" in FAM["CNN"]

    comp = MULTI.loc[COMPLETAS]
    rk = comp["todas erram: obs./indep."]
    kmin, kmax = comp["κ Fleiss acerto/erro"].min(), comp["κ Fleiss acerto/erro"].max()
    kcmin, kcmax = comp["κ Fleiss classes"].min(), comp["κ Fleiss classes"].max()
    top_loo = LOO.drop(columns="κ com as 4").mean().idxmax()
    loo_dem = LOO["sem DEMON"]
    ordem_q = list(MED.index)
    ordem_v = list(MED.sort_values("V de Cramér (classes)", ascending=False).index)
    ordem_k = list(MED.sort_values("κ de Cohen (classes)", ascending=False).index)
    vmin, vmax = PARES["V de Cramér (classes)"].min(), PARES["V de Cramér (classes)"].max()
    pchi = MULTI["p χ² (pior fold)"].max()
    k_txt = "; ".join(f"{p} {br(MED.loc[p, 'κ de Cohen (classes)'])} vs {br(MED.loc[p, 'κ se erros indep.'])}"
                      for p in MED.index)
    exc = MED["κ − κ indep."]
    exc_d, exc_o = exc[[p for p in exc.index if "DEMON" in p]], exc[[p for p in exc.index if "DEMON" not in p]]
    exc_dem = ("os três pares com DEMON têm o menor excesso" if exc_d.max() < exc_o.min()
               else f"excesso médio de {br(exc_d.mean())} nos pares com DEMON contra {br(exc_o.mean())} nos demais")

    cells = [md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 31: Diversidade entre Entradas — Par a Par e as Quatro ao Mesmo Tempo

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

O Notebook 30 mediu, par a par, se MEL, LOFAR, HHT e DEMON erram juntas para o mesmo classificador (Q e Y de Yule); os modelos são os mesmos (na CNN, o MEL é a `cnn` do repositório, `test_scripts/cnn.py`, com conv 128/64 e lr 1e-5 — não é a arquitetura do artigo). Aqui:

1. **Par a par, com quatro medidas:** Q e Y de Yule (sobre os erros) e V de Cramér e κ de Cohen (sobre as classes preditas). Elas ordenam os pares do mesmo jeito?
2. **As quatro entradas ao mesmo tempo:** κ de Fleiss e a distribuição de quantas entradas erram cada gravação, contra o esperado se os erros fossem independentes.
3. **Qual entrada traz mais diversidade:** κ ao retirar uma entrada por vez.

**Resumo:**
- Com as quatro entradas, o κ de Fleiss sobre acerto/erro fica entre {br(kmin)} e {br(kmax)} nas {len(COMPLETAS)} famílias completas: os erros estão longe de independentes (κ = 0), mas também longe de idênticos (κ = 1).
- As quatro entradas erram juntas a mesma gravação {br(rk.min(), 1)}–{br(rk.max(), 1)}× mais do que se fossem independentes; o teste χ² rejeita a independência em todos os folds ({'pior p = ' + br(pchi, 3) if pchi >= 0.001 else 'p < 0,001 em todos'}).
- A entrada que mais traz diversidade é **{top_loo.replace('sem ', '')}**: retirá-la aumenta o κ de Fleiss em média {br(LOO[top_loo].mean(), 3)} (de {br(LOO[top_loo].min(), 3)} a {br(LOO[top_loo].max(), 3)} entre as famílias completas).
- As medidas par a par concordam na ordem: MEL × LOFAR é o par mais associado e os pares com DEMON os menos associados.

Predição por gravação = voto majoritário das janelas; todas as tabelas são calculadas na execução (`notebooks/src/iara_results.py`).
"""), code(SETUP_CODE),
        md(f"""
## 1. As medidas

**Par a par.**

| Medida | Sobre | Tabela | Faixa | O que mede |
|---|---|---|---|---|
| Q de Yule | erros | 2×2 (acerta/erra × acerta/erra) | −1 a +1 | associação dos erros; Q = (OR − 1)/(OR + 1) |
| Y de Yule (colligação) | erros | 2×2 | −1 a +1 | idem, comprimida; Y = (√OR − 1)/(√OR + 1) |
| V de Cramér | classes preditas | 4×4 (classe de A × classe de B) | 0 a 1 | qualquer associação entre as predições: V = √(χ² / (N·(k − 1))) |
| κ de Cohen | classes preditas | 4×4 | −1 a +1 | concordância além do acaso (só a diagonal conta) |

O V de Cramér também é uma medida de **duas** variáveis; a diferença para o Q é aceitar mais de duas categorias. Ele não mede concordância: se A dissesse sempre `LARGE` quando B diz `SMALL`, V seria 1 e κ seria negativo. Por isso vem acompanhado do κ de Cohen.

**Quanto precisa dar para indicar concordância?** Não há ponto de corte formal, só escalas de convenção:

| κ de Cohen (Landis e Koch, 1977) | Leitura | | V de Cramér (Rea e Parker) | Leitura |
|---|---|---|---|---|
| < 0 | pior que o acaso | | < 0,10 | desprezível |
| 0,00–0,20 | leve | | 0,10–0,20 | fraca |
| 0,21–0,40 | razoável | | 0,20–0,40 | moderada |
| 0,41–0,60 | moderada | | 0,40–0,60 | relativamente forte |
| 0,61–0,80 | substancial | | 0,60–0,80 | forte |
| 0,81–1,00 | quase perfeita | | > 0,80 | muito forte |

McHugh (2012) é mais exigente (κ < 0,60 já é concordância fraca). Para o V, Cohen (1988) usa limiares que dependem dos graus de liberdade: com 4 classes (df* = 3), 0,06 / 0,17 / 0,29 = efeito pequeno / médio / grande. Com centenas de gravações por fold, qualquer κ ou V acima de ~0,1 é significativo (p < 0,001); o que importa é o tamanho.

**A referência não é κ = 0.** Dois classificadores que acertam 60–70% concordam muito só por acertarem as mesmas classes, mesmo que errem de forma independente. A referência justa para "erram juntos?" é o **κ esperado se os erros fossem independentes dada a classe verdadeira**: cada entrada mantém a sua matriz de confusão observada, mas as duas erram independentemente, P(A = i, B = j) = Σ_c π_c · P_A(i | c) · P_B(j | c). A coluna **κ − κ indep.** é a concordância que sobra além do que as duas acurácias já explicam.

**As quatro de uma vez** (medidas não pareadas, Kuncheva e Whitaker, 2003):
- **κ de Fleiss sobre acerto/erro:** trata as entradas como avaliadores que dão a cada gravação o rótulo "acertou" ou "errou". κ = 0: erros independentes; κ = 1: todas erram e acertam sempre juntas; κ < 0: mais diversas que o acaso. Para um ensemble, **quanto menor, melhor**. É a medida de concordância entre avaliadores de Kuncheva e Whitaker.
- **κ de Fleiss sobre as classes preditas:** a concordância entre as quatro nas 4 classes.
- **Número de entradas que erram cada gravação:** se os erros fossem independentes, esse número seguiria uma binomial de Poisson com as taxas de erro de cada entrada. O excesso nas pontas (todas erram, todas acertam) mostra a dependência; o teste χ² de aderência compara as duas distribuições em cada fold.

Todas as medidas são calculadas em cada fold (639 gravações de teste) e depois tirada a média dos 10 folds.
"""), code(MEDIDAS_CODE),
        md("""
## 2. Par a par: quatro medidas
"""), code(PARES_CODE),
        md(f"""
**Leitura.**
- **Ordem pelo Q:** {', '.join(ordem_q)}. **Pelo V de Cramér:** {', '.join(ordem_v)}. **Pelo κ de Cohen:** {', '.join(ordem_k)}.
- {'As três ordens coincidem' if ordem_q == ordem_v == ordem_k else 'As ordens são parecidas, mas não idênticas'}: os pares com DEMON ficam no fim em todas.
- **κ observado × κ esperado com erros independentes:** {k_txt}. Parte do κ vem só de as entradas acertarem bastante; o excesso (κ − κ indep., de {br(MED['κ − κ indep.'].min())} a {br(MED['κ − κ indep.'].max())}) é a concordância produzida pelos erros compartilhados; {exc_dem}.
- O V de Cramér vai de {br(vmin)} a {br(vmax)}. V e κ de Cohen ficam próximos porque as associações entre as predições estão quase todas na diagonal (as entradas concordam na classe, não trocam uma classe por outra de forma sistemática).
"""),
        md("""
## 3. As quatro entradas ao mesmo tempo
"""), code(MULTI_CODE),
        md(f"""
**Leitura.**
- **κ de Fleiss (acerto/erro) de {br(kmin)} a {br(kmax)}** nas famílias com as quatro entradas: dependência clara, longe de independência e de redundância total. Sobre as classes preditas, κ de {br(kcmin)} a {br(kcmax)}.
- **Todas erram juntas:** """ + "; ".join(f"{f} {br(comp.loc[f, 'todas erram: obs.'], 0)} gravações por fold, contra {br(comp.loc[f, 'todas erram: indep.'], 0)} esperadas" for f in COMPLETAS) + f""". É o núcleo de gravações difíceis do Notebook 29 (navio fora do treino, comprimento perto dos limiares de 50 m e 100 m).
- **Todas acertam juntas** também acima do esperado, e os casos intermediários (1–2 entradas erram) abaixo: a distribuição observada é mais "polarizada" que a independente. As gravações tendem a ser fáceis para todas ou difíceis para todas.
- {'A CNN tem as quatro entradas.' if cnn_ok else 'A CNN entra com três entradas (sem LOFAR); o κ dela não é diretamente comparável ao das famílias com quatro.'}
"""),
        md("""
## 4. Qual entrada traz mais diversidade?

Retiramos uma entrada por vez e recalculamos o κ de Fleiss (acerto/erro) com as três restantes. Se o κ **sobe** sem a entrada, ela era a que mais discordava das outras.
"""), code(LOO_CODE),
        md(f"""
**Leitura.**
- **Média do Δκ por entrada retirada:** """ + "; ".join(f"{c.replace('sem ', '')} {br(v, 3)}" for c, v in LOO.drop(columns='κ com as 4').mean().items()) + f""".
- **O DEMON é a entrada que mais traz diversidade** {'em todas as famílias completas' if (LOO.drop(columns='κ com as 4').idxmax(axis=1) == 'sem DEMON').all() else 'na maioria das famílias'}: sem ele, as três restantes (MEL, LOFAR, HHT) concordam bem mais entre si.
- Diversidade não é ganho: o DEMON é também a entrada mais fraca sozinha (Notebook 30, Seções 5 e 6). Um combinador só aproveita essa diversidade se souber quando confiar nele.
"""),
        md(f"""
## 5. Conclusões

1. **As quatro medidas par a par contam a mesma história:** MEL × LOFAR é o par mais redundante e os pares com DEMON os mais complementares. O V de Cramér é uma medida de duas variáveis, como o Q; para comparar as quatro entradas de uma vez, a medida adequada é o κ de Fleiss (ou a distribuição do número de entradas que erram).
2. **Com as quatro entradas ao mesmo tempo, os erros não são independentes:** existe um núcleo de gravações que todas erram, várias vezes maior do que o acaso produziria.
3. **O DEMON é quem traz diversidade ao conjunto;** MEL, LOFAR e HHT são parecidas entre si. Melhorar o DEMON sozinho é o caminho para que essa diversidade vire ganho num combinador.

**Limites:**
- No MLP, MEL/LOFAR (0,512 s) e HHT/DEMON (5 s) vêm de pipelines diferentes.
- {'A CNN tem as quatro entradas.' if cnn_ok else 'A CNN não tem LOFAR até a CNN LOFAR do artigo terminar; reexecutar este gerador a inclui.'}
- κ de Fleiss sobre acerto/erro depende das taxas de erro (como todo κ); comparar famílias com acurácias muito diferentes exige cuidado.
- O teste χ² de aderência é feito fold a fold; as gravações se repetem entre as 5 repetições, então os folds não são independentes entre si.
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
