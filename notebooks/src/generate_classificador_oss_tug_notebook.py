"""
Gera notebooks/38_classificador_oss_tug.ipynb: a intermitência do HHT 2D (NB37) ajuda um classificador simples a
separar Offshore Supply Ships de rebocadores, e MEDIUM de SMALL, em navios que ele nunca viu?

Desenho:
  - tarefas binárias: OSS (MEDIUM) × Tug (SMALL) e MEDIUM × SMALL (subconjuntos A–D);
  - atributos por gravação: LOFAR médio até 4 kHz (cache_lofar_espectro_medio.pkl), intermitência do HHT 2D
    (cache_hht_resumo.pkl), os dois juntos, e o nível médio do psd24k como controle (depende do ganho);
  - modelos: regressão logística (PCA ajustado dentro de cada fold) e floresta aleatória;
  - validação cruzada em 5 partes agrupadas por navio (nenhum navio no treino e no teste), repetida com 10
    sementes; os mesmos folds para todos os conjuntos de atributos (comparação pareada);
  - cada navio pesa 1 no treino (peso 1/nº de gravações do navio); a versão sem peso fica como comparação;
  - métrica principal: AUC por navio (média das probabilidades das gravações de cada navio no teste).

Rodar na raiz do repositório:  python notebooks/src/generate_classificador_oss_tug_notebook.py
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
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
warnings.filterwarnings('ignore')
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
pd.set_option('display.precision', 3); pd.set_option('display.width', 200)
PROC = os.path.join(R.ROOT, 'data', 'iara_processed')

meta = pd.read_csv(os.path.join(R.ROOT, 'src', 'iara', 'dataset_info', 'iara.csv'), dtype={'Ship ID': str, 'ID': str})
meta['Ship ID'] = meta['Ship ID'].str.strip(); meta['Length'] = pd.to_numeric(meta['Length'], errors='coerce')
meta = meta.set_index(meta.ID.astype(int)); meta = meta[meta.Dataset.isin(list('ABCD'))]

# atributos (caches criados pelos NB35 e NB37)
F = (np.arange(1024) + 1) * 16000 / 2048
LOF = pd.read_pickle(os.path.join(PROC, 'cache_lofar_espectro_medio.pkl')).loc[:, F <= 4000]
LOF.columns = [f'lofar_{i}' for i in range(LOF.shape[1])]
CV = pd.read_pickle(os.path.join(PROC, 'cache_hht_resumo.pkl'))['hht2d_cv'].copy()
CV.columns = [f'cv_{i}' for i in range(CV.shape[1])]
CV = CV.loc[:, CV.notna().mean() > 0.5]                   # faixas com intermitência definida na maioria das gravações
NV = pd.read_pickle(os.path.join(PROC, 'cache_psd24k_niveis.pkl'))[['nível médio']]
TAREFAS = {'OSS × Tug': ((meta['DETAILED TYPE'] == 'Offshore Supply Ship') & (meta.Length >= 50), (meta['DETAILED TYPE'] == 'Tug') & (meta.Length < 50)),
           'MEDIUM × SMALL': ((meta.Length >= 50) & (meta.Length < 100), meta.Length < 50)}
CONJ = {'LOFAR': ('l',), 'intermitência HHT': ('c',), 'LOFAR + intermitência': ('l', 'c'), 'nível (controle, depende do ganho)': ('n',)}
print(f'LOFAR: {LOF.shape[1]} faixas até 4 kHz; intermitência: {CV.shape[1]} faixas; nível: 1 valor')
'''

EXP_CODE = r'''
def pipe(partes, modelo):
    tr = []
    if 'l' in partes: tr.append(('l', make_pipeline(StandardScaler(), PCA(20, random_state=0)), list(LOF.columns)))
    if 'c' in partes: tr.append(('c', make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), PCA(10, random_state=0)), list(CV.columns)))
    if 'n' in partes: tr.append(('n', StandardScaler(), ['nível médio']))
    clf = (LogisticRegression(C=0.1, class_weight='balanced', max_iter=5000) if modelo == 'logística' else
           RandomForestClassifier(300, min_samples_leaf=3, class_weight='balanced', random_state=0, n_jobs=-1))
    return make_pipeline(ColumnTransformer(tr), clf)

_rows = []
for tarefa, (ma, mb) in TAREFAS.items():
    ids = meta.index[ma | mb].intersection(LOF.index).intersection(CV.index).intersection(NV.index)
    y = pd.Series(ma[ids].astype(int).values, index=ids)
    X = LOF.loc[ids].join(CV.loc[ids]).join(NV.loc[ids])
    g = meta.loc[ids, 'Ship ID'].values
    print(f'{tarefa}: {len(ids)} gravações de {len(set(g))} navios ({int(y.sum())} gravações da 1ª classe)')
    for rep in range(10):
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g))
        for peso in ('1 por navio', 'sem peso'):
            for modelo in ('logística', 'floresta'):
                for nome, partes in CONJ.items():
                    pr = pd.Series(np.nan, index=ids)
                    for tr, te in folds:
                        m = pipe(partes, modelo)
                        kw = {}
                        if peso == '1 por navio':
                            gtr = pd.Series(g[tr]); kw = {m.steps[-1][0] + '__sample_weight': (1.0 / gtr.map(gtr.value_counts())).values}
                        m.fit(X.iloc[tr], y.iloc[tr], **kw)
                        pr.iloc[te] = m.predict_proba(X.iloc[te])[:, 1]
                    ps, ys = pr.groupby(g).mean(), y.groupby(g).first()
                    _rows.append({'tarefa': tarefa, 'peso': peso, 'modelo': modelo, 'atributos': nome, 'repetição': rep,
                                  'AUC por navio': roc_auc_score(ys, ps), 'acurácia balanceada por navio': balanced_accuracy_score(ys, ps > 0.5),
                                  'AUC por gravação': roc_auc_score(y, pr)})
RES = pd.DataFrame(_rows)
'''

TAB_CODE = r'''
MET = ['AUC por navio', 'acurácia balanceada por navio', 'AUC por gravação']
PRINC = RES[RES.peso == '1 por navio'].groupby(['tarefa', 'modelo', 'atributos'])[MET].agg(['mean', 'std'])
print('Peso 1 por navio (principal); média e desvio entre as 10 repetições:'); display(PRINC.round(3))

_pv = RES[RES.peso == '1 por navio'].pivot_table(index=['tarefa', 'modelo', 'repetição'], columns='atributos', values='AUC por navio')
GANHO = (_pv['LOFAR + intermitência'] - _pv['LOFAR']).groupby(level=[0, 1]).agg(
    ganho_medio='mean', desvio='std', repeticoes_com_ganho=lambda x: 100 * (x > 0).mean())
GANHO.columns = ['ganho médio de AUC', 'desvio', '% das repetições com ganho']
print('Ganho de AUC por navio ao juntar a intermitência ao LOFAR (mesmos folds):'); display(GANHO.round(3))

SEMPESO = RES.groupby(['tarefa', 'modelo', 'atributos', 'peso'])['AUC por navio'].mean().unstack('peso')
print('AUC por navio, com e sem peso por navio:'); display(SEMPESO.round(3))

fig, ax = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
for a_, t in zip(ax, TAREFAS):
    d = RES[(RES.tarefa == t) & (RES.peso == '1 por navio')]
    nomes = list(CONJ)
    for k, mod in enumerate(('logística', 'floresta')):
        vals = [d[(d.modelo == mod) & (d.atributos == n)]['AUC por navio'].values for n in nomes]
        pos = np.arange(len(nomes)) + (k - 0.5) * 0.35
        bp = a_.boxplot(vals, positions=pos, widths=0.3, patch_artist=True, showfliers=False)
        for b in bp['boxes']:
            b.set_facecolor('tab:blue' if mod == 'logística' else 'tab:orange'); b.set_alpha(0.6)
    a_.axhline(0.5, color='gray', ls='--'); a_.set_xticks(range(len(nomes))); a_.set_xticklabels([n.replace(' (controle, depende do ganho)', '\n(controle)') for n in nomes], fontsize=8)
    a_.set_title(f'{t}: AUC por navio (azul: logística; laranja: floresta)', fontsize=10)
ax[0].set_ylabel('AUC por navio (10 repetições)')
plt.tight_layout(); plt.show()
'''


def build():
    filename = os.path.join(NOTEBOOKS_DIR, "38_classificador_oss_tug.ipynb")
    ns = {}
    for src in (SETUP_CODE, EXP_CODE, TAB_CODE):
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    P, G, SP = ns["PRINC"], ns["GANHO"], ns["SEMPESO"]

    def auc(t, m, a):
        return P.loc[(t, m, a), ("AUC por navio", "mean")]

    def bacc(t, m, a):
        return P.loc[(t, m, a), ("acurácia balanceada por navio", "mean")]

    def linha(t):
        partes = []
        for m in ("logística", "floresta"):
            partes.append(f"{m}: LOFAR {br(auc(t, m, 'LOFAR'), 2)}, intermitência {br(auc(t, m, 'intermitência HHT'), 2)}, "
                          f"juntos {br(auc(t, m, 'LOFAR + intermitência'), 2)}, nível (controle) {br(auc(t, m, 'nível (controle, depende do ganho)'), 2)}")
        return "; ".join(partes)

    def ganho(t):
        return "; ".join(f"{m}: {'+' if G.loc[(t, m), 'ganho médio de AUC'] >= 0 else ''}{br(G.loc[(t, m), 'ganho médio de AUC'], 3)} "
                         f"(ganho em {br(G.loc[(t, m), '% das repetições com ganho'], 0)}% das repetições)" for m in ("logística", "floresta"))
    ot, ms = "OSS × Tug", "MEDIUM × SMALL"
    best_ms = max(auc(ms, m, a) for m in ("logística", "floresta") for a in ("LOFAR", "intermitência HHT", "LOFAR + intermitência"))
    best_ms_bacc = max(bacc(ms, m, a) for m in ("logística", "floresta") for a in ("LOFAR", "intermitência HHT", "LOFAR + intermitência"))
    best_ot = max(auc(ot, m, a) for m in ("logística", "floresta") for a in ("LOFAR", "intermitência HHT", "LOFAR + intermitência"))
    g_pos = all(G.loc[(t, m), "ganho médio de AUC"] > 0 for t in (ot, ms) for m in ("logística", "floresta"))
    sp_nivel_ms = SP.loc[(ms, "logística", "nível (controle, depende do ganho)")]

    cells = [md("""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 38: A Intermitência do HHT Ajuda a Generalizar SMALL × MEDIUM?

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

**Pergunta.** No NB37, a única pista acústica que distingue Offshore Supply Ships (OSS, MEDIUM) de rebocadores (SMALL), independente do ganho, foi a **"intermitência" do HHT 2D** (coeficiente de variação da energia em passos de 80 ms). A Seção 6.1 do NB37 mostrou que ela mede sobretudo **energia relativa**: os OSS têm relativamente mais energia em ~100–200 Hz em relação ao componente dominante. O teste prático: juntar essa pista ao LOFAR melhora um classificador simples **em navios que ele nunca viu**?

**Desenho.**
- **Tarefas binárias:** OSS × Tug, e MEDIUM × SMALL (classes inteiras, subconjuntos A–D).
- **Atributos por gravação:** LOFAR médio até 4 kHz (512 faixas), intermitência do HHT 2D, os dois juntos e, como **controle**, o nível médio do psd24k (que depende do ganho do equipamento e, pelo NB37, é provavelmente instrumental).
- **Modelos:** regressão logística (PCA ajustado dentro de cada fold: 20 componentes do LOFAR e 10 da intermitência) e floresta aleatória.
- **Validação:** 5 partes agrupadas por navio (nenhum navio no treino e no teste), repetida com 10 sementes; os mesmos folds para todos os conjuntos de atributos, o que permite comparar par a par.
- **Peso:** cada navio pesa 1 no treino (peso 1/nº de gravações do navio). Sem isso, poucos navios com muitas gravações (os Motor Hoppers e os Pilot Vessels) dominam o ajuste.
- **Métrica principal:** AUC por navio (média das probabilidades das gravações de cada navio no teste). 0,5 = acaso.

**Cautela.** A intermitência foi escolhida olhando os mesmos navios OSS e Tug (NB37), então o resultado OSS × Tug é otimista para ela. Em MEDIUM × SMALL há muitos navios de outros tipos, o que torna esse teste mais independente.
"""), code(SETUP_CODE),
        md("""
## 1. Experimento

Leva alguns minutos: 2 tarefas × 10 repetições × 2 pesos × 2 modelos × 4 conjuntos de atributos × 5 folds.
"""), code(EXP_CODE),
        md("""
## 2. Resultados
"""), code(TAB_CODE),
        md(f"""
**Leitura.**
- **OSS × Tug (AUC por navio):** {linha(ot)}. Ganho ao juntar a intermitência ao LOFAR: {ganho(ot)}.
- **MEDIUM × SMALL (AUC por navio):** {linha(ms)}. Ganho ao juntar a intermitência ao LOFAR: {ganho(ms)}.
- **O LOFAR sozinho quase não separa MEDIUM de SMALL em navios novos** (AUC {br(auc(ms, 'logística', 'LOFAR'), 2)} na logística e {br(auc(ms, 'floresta', 'LOFAR'), 2)} na floresta), coerente com o NB29 e o NB35. {('A intermitência traz um ganho pequeno e consistente (positivo em todas as combinações)' if g_pos else 'O ganho da intermitência não é consistente')}, mas o melhor resultado em MEDIUM × SMALL fica em AUC {br(best_ms, 2)} e acurácia balanceada {br(best_ms_bacc, 2)}: pouco acima do acaso.
- **OSS × Tug chega a AUC ~{br(best_ot, 2)}** em navios novos, mesmo com o LOFAR sozinho, que no NB37 não mostrava nenhuma faixa diferente: combinando muitas faixas fracas, o classificador acha alguma separação. Mas o **controle de nível**, que não é acústico, dá AUC parecida ou maior: uma AUC de ~0,7 nessa comparação não basta para afirmar que o modelo aprendeu o som do navio.
- **Peso por navio:** sem ele, os resultados mudam bastante (por exemplo, o controle de nível em MEDIUM × SMALL com a logística vai de {br(sp_nivel_ms['1 por navio'], 2)} com peso para {br(sp_nivel_ms['sem peso'], 2)} sem peso), porque poucos navios muito gravados dominam o ajuste e o que vale para eles não vale para os outros.
"""),
        md(f"""
## 3. Conclusões

1. **A intermitência do HHT (na prática, a energia relativa das faixas no HHT 2D) ajuda pouco.** Juntá-la ao LOFAR aumenta a AUC por navio em {ganho(ms)} em MEDIUM × SMALL, mas o resultado continua perto do acaso (melhor AUC {br(best_ms, 2)}). Ela não resolve a generalização de MEDIUM.
2. **SMALL × MEDIUM continua sem solução acústica nestes dados.** Nenhuma combinação testada separa as duas classes em navios novos de forma útil; isso confirma, agora com um classificador, o que os NB35–37 mostraram faixa a faixa.
3. **Separações de AUC ~0,7 precisam de controle.** No par OSS × Tug, uma variável sabidamente instrumental (o nível sem calibração) separa tão bem quanto as acústicas. Qualquer resultado nesse patamar pode vir das condições de gravação, e não do navio.
4. **Para o artigo e a dissertação:** o recall de MEDIUM relatado com o split por gravação (~60%) vem sobretudo do reconhecimento de navios repetidos; com navios novos, nem o LOFAR, nem o MEL, nem o DEMON, nem o HHT, nem a combinação com a intermitência levam a uma classificação útil de MEDIUM.
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
