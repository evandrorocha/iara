"""
Gera notebooks/30_correlacao_erros_entradas.ipynb: correlação dos erros entre as entradas (MEL, LOFAR, HHT, DEMON)
para o mesmo classificador, medida pelo Q de Yule e pelo Y de Yule (coeficiente de colligação).

Perguntas:
  1. Para a mesma família de modelo (RF, SVM, MLP, BiGRU, CNN), entradas diferentes erram nas mesmas gravações?
  2. Qual par de entradas é o mais complementar, e quanto o DEMON poderia resgatar dos erros do LOFAR?
  3. O que as combinações LOFAR/MEL + DEMON já testadas no projeto entregaram?

Fontes: predições por janela em results/trainings/{multirep,tests}/<exp>/eval/fold_k (via iara_results.py).
A CNN LOFAR do artigo (test_scripts/cnn_lofar.py, experimento 'cnn_lofar') entra automaticamente quando os 10 folds
existirem; a CNN LOFAR de 5 s (cnn2d_lofar5s_w32_b64_e15_4c) colapsou e só aparece como ilustração.
Como nos outros geradores, o mesmo código das células roda aqui para calcular os números citados no markdown.
Rodar na raiz do repositório:  python notebooks/src/generate_correlacao_entradas_notebook.py
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
import os, sys, itertools
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
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'figure.dpi': 120})
pd.set_option('display.precision', 2)
CLS = dict(enumerate(R.CLASSES))
REPS = ['MEL', 'LOFAR', 'HHT', 'DEMON']


def disponivel(exp):
    try:
        R.file_predictions(exp)
        return True
    except FileNotFoundError:
        return False


# Um modelo por (família, entrada), todos de 4 classes nos 10 folds oficiais (split por ID)
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
CNN_LOFAR_OK = 'LOFAR' in FAM['CNN']
print('CNN LOFAR do artigo (cnn_lofar):', 'incluída' if CNN_LOFAR_OK else 'ainda não disponível (10 folds) - fora das tabelas')
'''

DESEMP_CODE = r'''
linhas = [(f'{f} {r}', e) for f, reps in FAM.items() for r, e in reps.items()]
desemp = R.table(linhas, metrics=('SP', 'ACC'))
display(desemp)
'''

QY_CODE = r'''
def medidas(df, a, b):
    """Q e Y de Yule dos erros de a e b num fold (a = ambos erram, d = ambos acertam, b/c = só um erra)."""
    y = df.Target.values
    ea, eb = df[a].values != y, df[b].values != y
    n11, n00 = (ea & eb).sum(), (~ea & ~eb).sum()
    n10, n01 = (ea & ~eb).sum(), (~ea & eb).sum()
    ad, bc = float(n11 * n00), float(n10 * n01)
    return {'Q': (ad - bc) / (ad + bc),
            'Y': (np.sqrt(ad) - np.sqrt(bc)) / (np.sqrt(ad) + np.sqrt(bc)),
            'obs': n11 / len(y), 'indep': ea.mean() * eb.mean()}


def par(a, b):
    AL = R.aligned([a, b])
    per = pd.DataFrame([medidas(d, a, b) for d in AL.values()])
    q = per.Q.mean()
    return {'Q': q, 'Q min (fold)': per.Q.min(), 'Q máx (fold)': per.Q.max(), 'Y': per.Y.mean(),
            'OR ≈ (1+Q)/(1−Q)': (1 + q) / (1 - q), 'ambos erram / indep.': per.obs.mean() / per.indep.mean()}


rows = []
for f, reps in FAM.items():
    for ra, rb in itertools.combinations(REPS, 2):
        if ra in reps and rb in reps:
            rows.append({'Família': f, 'Par': f'{ra} × {rb}', **par(reps[ra], reps[rb])})
QY = pd.DataFrame(rows)
display(QY.set_index(['Família', 'Par']).round(2))

ORDEM = [f'{a} × {b}' for a, b in itertools.combinations(REPS, 2)]
PQ = QY.pivot(index='Par', columns='Família', values='Q').reindex(index=ORDEM, columns=list(FAM))
PY = QY.pivot(index='Par', columns='Família', values='Y').reindex(index=ORDEM, columns=list(FAM))
PQ['média'], PY['média'] = PQ.mean(axis=1), PY.mean(axis=1)
PQ, PY = PQ.sort_values('média', ascending=False), PY.loc[PQ.sort_values('média', ascending=False).index]
print('Q de Yule (linhas ordenadas pela média):'); display(PQ.round(2))
print('Y de Yule (colligação):'); display(PY.round(2))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, P, nome in ((axes[0], PQ, 'Q de Yule'), (axes[1], PY, 'Y de Yule (colligação)')):
    M = P.drop(columns='média')
    im = ax.imshow(M.values.astype(float), vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(M.shape[1]), M.columns); ax.set_yticks(range(M.shape[0]), M.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            ax.text(j, i, '—' if pd.isna(v) else f'{v:.2f}', ha='center', va='center', fontsize=8,
                    color='k' if pd.isna(v) or v >= 0.6 else 'w')
    fig.colorbar(im, ax=ax); ax.set_title(nome + ' dos erros, por família')
plt.tight_layout(); plt.show()
'''

VARIANTES_CODE = r'''
# Mesma entrada, arquitetura diferente (referência de "quase o mesmo modelo") e a CNN LOFAR de 5 s que colapsou
VAR = [('CNN HHT 1D × CNN HHT 2D', 'speccnn_hht_ctx0_e60p8_4c', 'hht_2d_cnn_e40p7_4c', 'mesma entrada'),
       ('CNN DEMON HCNN × CNN DEMON SpecCNN', 'hcnn_demon_tpsw_med5_e60p8_4c', 'speccnn_demon_ctx0_e60p8_4c', 'mesma entrada'),
       ('CNN MEL (repo) × CNN LOFAR 5 s (colapsou)', 'cnn', 'cnn2d_lofar5s_w32_b64_e15_4c', 'ilustração: modelo colapsado'),
       ('CNN HHT 2D × CNN LOFAR 5 s (colapsou)', 'hht_2d_cnn_e40p7_4c', 'cnn2d_lofar5s_w32_b64_e15_4c', 'ilustração: modelo colapsado')]
VQ = pd.DataFrame([{'Par': n, 'Tipo': t, 'SP a': R.summary(a)['SP'][0], 'SP b': R.summary(b)['SP'][0], **par(a, b)}
                   for n, a, b, t in VAR]).set_index('Par')
display(VQ.round(2))
'''

RESGATE_CODE = r'''
# Nas gravações em que o LOFAR erra, com que frequência o DEMON acerta (e vice-versa), por classe
res = []
for f, reps in FAM.items():
    if 'LOFAR' not in reps or 'DEMON' not in reps:
        continue
    l, d = reps['LOFAR'], reps['DEMON']
    D = pd.concat(R.aligned([l, d]).values())
    t = D.Target.map(CLS)
    el, okd = D[l] != D.Target, D[d] == D.Target
    for c in R.CLASSES:
        m = t == c
        res.append({'Família': f, 'Classe': c, 'recall LOFAR (%)': 100 * (~el[m]).mean(),
                    'recall DEMON (%)': 100 * okd[m].mean(),
                    'resgate: DEMON acerta | LOFAR erra (%)': 100 * okd[m & el].mean(),
                    'estrago: DEMON erra | LOFAR acerta (%)': 100 * (~okd[m & ~el]).mean(),
                    'oráculo (%)': 100 * ((~el) | okd)[m].mean()})
RES = pd.DataFrame(res).set_index(['Família', 'Classe'])
display(RES.round(1))
'''

COMBIN_CODE = r'''
# Combinações com DEMON (ou alternativa) já avaliadas no projeto, contra o melhor componente isolado
COMB = [('BiGRU LOFAR', 'bigru_att_lofar5s_h128_s15-5_e15_4c', 'BiGRU LOFAR + DEMON (fusão de features)', 'bigru_att_lofar5s-demon_h128_s15-5_e15_4c'),
        ('BiGRU LOFAR', 'bigru_att_lofar5s_h128_s15-5_e15_4c', 'BiGRU LOFAR + HHT (fusão de features)', 'bigru_att_lofar5s-hht_h128_s15-5_e15_4c'),
        ('MLP MEL 5 s', 'mlp_mel5s_h32-16_e35_4c', 'MLP MEL + DEMON (fusão de features)', 'mlp_mel5s-demon_h64-32_e35_4c'),
        ('RF LOFAR', 'rf_lofar5s_n100_balanced_4c', 'Ensemble RF MEL+LOFAR+HHT+DEMON (média de prob.)', 'ensemble_rf_prob_mel-lofar-hht-demon_4c'),
        ('RF LOFAR', 'rf_lofar5s_n100_balanced_4c', 'Ensemble RF MEL+LOFAR+HHT (sem DEMON)', 'ensemble_rf_prob_mel-lofar-hht_4c')]
cb = []
for la, a, lb, b in COMB:
    sa, sb = R.summary(a), R.summary(b)
    F, p = R.f_test_5x2(a, b)
    cb.append({'Combinação': lb, 'Base': la, 'SP base': R.fmt(sa['SP']), 'SP combinação': R.fmt(sb['SP']),
               'Δ SP (p.p.)': sb['SP'][0] - sa['SP'][0], 'F (5x2cv)': F, 'p': p,
               **{f'Δ {c}': sb['R_' + c][0] - sa['R_' + c][0] for c in R.CLASSES}})
CB = pd.DataFrame(cb).set_index('Combinação')
display(CB.round(3))
'''


def run(ns, src):
    exec(src, ns)


def build(filename=os.path.join(NOTEBOOKS_DIR, "30_correlacao_erros_entradas.ipynb")):
    ns = {}
    for src in (SETUP_CODE, DESEMP_CODE, QY_CODE, VARIANTES_CODE, RESGATE_CODE, COMBIN_CODE):
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    np = ns["np"]
    PQ, PY, QY, VQ, RES, CB, FAM = ns["PQ"], ns["PY"], ns["QY"], ns["VQ"], ns["RES"], ns["CB"], ns["FAM"]
    cnn_ok = ns["CNN_LOFAR_OK"]

    mq = PQ["média"]
    top, bot = mq.idxmax(), mq.idxmin()
    dem = [p for p in mq.index if "DEMON" in p]
    nao_dem = [p for p in mq.index if "DEMON" not in p]
    qmin, qmax = QY.Q.min(), QY.Q.max()
    ymin, ymax = QY.Y.min(), QY.Y.max()
    rmin, rmax = QY["ambos erram / indep."].min(), QY["ambos erram / indep."].max()
    neg = int((QY.Q < 0).sum())
    vq_same = VQ[VQ.Tipo == "mesma entrada"].Q
    vq_col = VQ[VQ.Tipo != "mesma entrada"]
    fam_lofar_demon = [f for f in FAM if "LOFAR" in FAM[f] and "DEMON" in FAM[f]]
    sm = RES.xs("SMALL", level="Classe")
    cb_sig = CB[CB.p < 0.05]
    cb_dem = CB[CB.index.str.contains("DEMON") & ~CB.index.str.contains("sem DEMON")]
    dem_sig = cb_dem[cb_dem.p < 0.05]
    out_sig = cb_sig[~cb_sig.index.isin(cb_dem.index)]

    def sig_txt(d):
        return ", ".join(f"{i} ({br(r['Δ SP (p.p.)'], 2)} p.p., {pv(r.p)})" for i, r in d.iterrows())
    svm_hht = QY[(QY["Família"] == "SVM") & QY.Par.str.contains("HHT")]

    def oq(x):
        return (1 + x) / (1 - x)

    def yq(x):
        o = oq(x)
        return (np.sqrt(o) - 1) / (np.sqrt(o) + 1)

    exemplo = [0.47, 0.63, 0.81, 0.96]

    cells = [md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 30: Correlação dos Erros entre Entradas (Q e Y de Yule)

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

O Notebook 29 mostrou que dez modelos de famílias e entradas diferentes erram nas mesmas gravações. Aqui fixamos o classificador e variamos só a entrada:

1. **Para o mesmo modelo, MEL, LOFAR, HHT e DEMON erram juntos?** Q e Y de Yule por par de entradas, em cada família (RF, SVM, MLP, BiGRU, CNN).
2. **Qual entrada é mais complementar ao LOFAR, e quanto ela poderia resgatar?** Taxas de resgate e de estrago do DEMON sobre o LOFAR, por classe.
3. **O que as combinações já testadas entregaram?** Fusões e ensembles com DEMON, com o F-test 5×2cv.

**Resumo:**
- Todos os {len(QY)} pares têm Q positivo ({br(qmin)} a {br(qmax)}; Y de {br(ymin)} a {br(ymax)}); {'nenhum' if neg == 0 else neg} negativo. Os dois modelos erram juntos {br(rmin, 1)}–{br(rmax, 1)}× mais do que se os erros fossem independentes.
- O par mais associado é **{top}** (Q médio {br(mq[top])}); o menos associado é **{bot}** ({br(mq[bot])}). Os pares com DEMON têm Q médio de {br(mq[dem].mean())}, contra {br(mq[nao_dem].mean())} entre MEL, LOFAR e HHT.
- O DEMON resgata parte dos erros do LOFAR, mas também erra muito onde o LOFAR acerta; {'nenhuma combinação com DEMON melhorou o SP de forma significativa (α = 0,05)' if dem_sig.empty else 'combinações com DEMON significativas: ' + sig_txt(dem_sig)}{('; entre as de referência, ' + sig_txt(out_sig) + ' é significativa') if not out_sig.empty else ''}.

Predição por gravação = voto majoritário das janelas; todas as tabelas são calculadas na execução (`notebooks/src/iara_results.py`).
"""), code(SETUP_CODE),
        md(f"""
## 1. As medidas

Para cada par de modelos e cada fold, montamos a tabela 2×2 dos erros por gravação:

| | B acerta | B erra |
|---|---|---|
| **A acerta** | d (ambos acertam) | c |
| **A erra** | b | a (ambos erram) |

- **Q de Yule** (coeficiente de associação, 1900): Q = (ad − bc) / (ad + bc).
- **Y de Yule** (coeficiente de colligação, 1912): Y = (√ad − √bc) / (√ad + √bc).

Os dois vão de −1 a +1: 0 = erros independentes; +1 = quando um erra, o outro também erra; negativo = os erros caem em gravações diferentes (o caso ideal para combinar). Os dois são funções da razão de chances OR = ad/bc: Q = (OR − 1)/(OR + 1) e Y = (√OR − 1)/(√OR + 1), de modo que Q = 2Y/(1 + Y²). O Y comprime os valores em direção a zero; a mesma associação dá números menores:

| Q | OR | Y |
|---|---|---|
""" + "\n".join(f"| {br(q)} | {br(oq(q), 1)} | {br(yq(q))} |" for q in exemplo) + """

**Escala de leitura (Davis, 1971, para o Q):** ≥ 0,70 muito forte; 0,50–0,69 substancial; 0,30–0,49 moderada; 0,10–0,29 fraca; < 0,10 desprezível. Esses limiares valem para o Q, não para o Y. O Q é a medida de diversidade de classificadores usada por Kuncheva e Whitaker (2003); entre classificadores treinados nos mesmos dados, Q positivo é a regra.

Q e Y são calculados em cada fold (639 gravações de teste) e depois feita a média dos 10 folds; a OR da tabela é a equivalente ao Q médio.
"""),
        md(f"""
## 2. Os modelos

Um modelo por família e entrada, todos de 4 classes nos 10 folds oficiais. {'A CNN LOFAR é a do artigo (`test_scripts/cnn_lofar.py`); a CNN MEL (`cnn`) é a do repositório (`test_scripts/cnn.py`: conv 128/64, ReLU, lr 1e-5, imagens sem sobreposição — não é a arquitetura do artigo).' if cnn_ok else 'A CNN LOFAR do artigo (`test_scripts/cnn_lofar.py`) ainda está em treinamento; a única CNN LOFAR disponível (`cnn2d_lofar5s_w32_b64_e15_4c`, SP ~35) colapsou e fica fora das tabelas principais (Seção 4).'} Na CNN, HHT = CNN 2D Hilbert e DEMON = HCNN (TPSW, mediana 5), as variantes de maior SP. No MLP, MEL e LOFAR são os do pipeline `iara` (janelas de 0,512 s, com early stopping), com a arquitetura do artigo, mas paciência 16 em vez de 8 e, no LOFAR, dropout 0,2 em vez de 0; HHT e DEMON usam janelas de 5 s.
"""), code(DESEMP_CODE),
        md("""
## 3. Q e Y de Yule entre entradas, por família
"""), code(QY_CODE),
        md(f"""
**Leitura.**
- **Nenhum par é independente:** Q de {br(qmin)} a {br(qmax)} (Y de {br(ymin)} a {br(ymax)}).
- **Ordem dos pares pelo Q médio:** """ + "; ".join(f"{p} {br(v)}" for p, v in mq.items()) + f""". A ordem é parecida em todas as famílias.
- **MEL e LOFAR são quase redundantes:** as duas são o espectro de potência do mesmo sinal, com escalas de frequência diferentes. O HHT fica no meio: outra decomposição da energia, ainda muito associada.
- **O DEMON é a entrada mais independente** (Q médio {br(mq[dem].mean())} contra {br(mq[nao_dem].mean())}): mede a modulação de baixa frequência (eixo e pás), outra informação física.
- **Instabilidade do SVM HHT:** nos pares com HHT, o Q do SVM varia entre folds de {br(svm_hht['Q min (fold)'].min())} a {br(svm_hht['Q máx (fold)'].max())}; os números do SVM com HHT são os menos confiáveis.
"""),
        md("""
## 4. Referências: mesma entrada, e um modelo colapsado
"""), code(VARIANTES_CODE),
        md(f"""
**Leitura.**
- **Trocar só a arquitetura quase não muda onde a CNN erra:** Q de {br(vq_same.min())} a {br(vq_same.max())} entre variantes com a mesma entrada, acima de qualquer par de entradas diferentes. A entrada pesa mais que o modelo.
- **Q baixo de um modelo colapsado não é complementaridade:** a CNN LOFAR de 5 s (SP {br(vq_col['SP b'].iloc[0], 1)}) dá Q de {br(vq_col.Q.min())} a {br(vq_col.Q.max())}, variando de {br(vq_col['Q min (fold)'].min())} a {br(vq_col['Q máx (fold)'].max())} entre folds. Ela erra quase ao acaso, e por isso não erra junto com ninguém. Por isso ela não entra nas tabelas principais.
"""),
        md("""
## 5. Quanto o DEMON poderia complementar o LOFAR?

Q menor é condição necessária, não suficiente: além de errar em gravações diferentes, o DEMON precisa **acertar** onde o LOFAR erra. Por classe:
- **resgate:** entre as gravações que o LOFAR erra, a fração que o DEMON acerta;
- **estrago:** entre as que o LOFAR acerta, a fração que o DEMON erra;
- **oráculo:** recall de um seletor perfeito, que sempre escolhe o modelo certo (teto teórico, não atingível).
"""), code(RESGATE_CODE),
        md(f"""
**Leitura.**
- **O teto é alto em `SMALL`:** o oráculo leva o recall de `SMALL` de """ + ", ".join(f"{br(sm.loc[f, 'recall LOFAR (%)'], 1)}% para {br(sm.loc[f, 'oráculo (%)'], 1)}% ({f})" for f in fam_lofar_demon) + f""".
- **Mas o DEMON também estraga:** em `SMALL`, ele erra {br(sm['estrago: DEMON erra | LOFAR acerta (%)'].min(), 0)}–{br(sm['estrago: DEMON erra | LOFAR acerta (%)'].max(), 0)}% das gravações que o LOFAR acerta. Combinar só ganha se o sistema souber **quando** confiar no DEMON.
"""),
        md("""
## 6. O que as combinações já testadas entregaram
"""), code(COMBIN_CODE),
        md(f"""
**Leitura.**
- {'Nenhuma combinação melhora o SP de forma significativa (α = 0,05)' if cb_sig.empty else 'Ganhos significativos (α = 0,05): ' + ', '.join(cb_sig.index)}; o menor p é {pv(CB.p.min())} ({CB.p.idxmin()}), com Δ SP de {br(CB.loc[CB.p.idxmin(), 'Δ SP (p.p.)'], 2)} p.p.
- No ensemble RF, tirar o DEMON dá SP {'maior' if CB.iloc[4]['Δ SP (p.p.)'] > CB.iloc[3]['Δ SP (p.p.)'] else 'menor'} ({br(CB.iloc[4]['Δ SP (p.p.)'], 2)} contra {br(CB.iloc[3]['Δ SP (p.p.)'], 2)} p.p. sobre o RF LOFAR); a média de probabilidades dilui a informação do DEMON, que é fraco sozinho.
- No gating do Notebook 26, o roteador por meta-features ficou abaixo do MLP LOFAR (efeito reverso: mandou para o DEMON gravações que o LOFAR acertaria).
"""),
        md(f"""
## 7. Conclusões

1. **Para o mesmo modelo, as entradas também erram juntas** (Q > 0 em todos os pares). Parte da associação vem das entradas — MEL e LOFAR são quase redundantes — e parte das gravações: há um núcleo de gravações difíceis para qualquer entrada (navio fora do treino, comprimento perto dos limiares de 50 m e 100 m; Notebook 29).
2. **O DEMON é a entrada mais complementar ao LOFAR e ao MEL**, com associação substancial (Q ≈ {br(mq[dem].mean())}) em vez de muito forte.
3. **{'O ganho das combinações com DEMON é pequeno e não significativo.' if dem_sig.empty else 'O ganho das combinações com DEMON é pequeno.'}** Fusão de features e média de probabilidades não sabem quando confiar no DEMON.{(' Por outro lado, ' + sig_txt(out_sig) + ' é significativa, embora LOFAR e HHT tenham Q bem maior que LOFAR e DEMON: Q baixo não garante ganho, e Q alto não o impede quando a entrada extra é mais forte sozinha.') if not out_sig.empty else ''} O próximo passo natural é um combinador treinado (stacking sobre probabilidades out-of-fold de LOFAR e DEMON).

**Limites:**
- No MLP, MEL/LOFAR (0,512 s) e HHT/DEMON (5 s) vêm de pipelines diferentes; RF e BiGRU usam o mesmo pipeline nas quatro entradas e mostram o mesmo padrão.
- {'A CNN LOFAR do artigo está incluída.' if cnn_ok else 'A CNN não tem LOFAR até a CNN LOFAR do artigo terminar de treinar; reexecutar este gerador a inclui automaticamente.'}
- Q e Y medem se os erros coincidem, não se os dois modelos erram para a mesma classe.
- Não há teste formal da diferença entre os Q; a amplitude entre folds está na tabela da Seção 3.
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
