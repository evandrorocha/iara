"""
Código comum aos geradores dos notebooks 17, 18 e 19 (complementaridade MEL/LOFAR/HHT/DEMON no Random Forest).

* HELPER_SRC é o código de análise. Ele é inserido como célula nos notebooks e executado aqui (exec) para
  calcular os números citados no markdown. Assim, texto e células usam exatamente o mesmo cálculo.
* Todas as predições vêm de results/trainings/{multirep,tests}/<exp>/eval/fold_k via notebooks/iara_results.py.

Uso: importado por notebooks/generate_rf_{small,medium,large}_notebook.py (cwd = raiz do repositório).
"""
import json
import numbers
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

HELPER_SRC = r'''# Funções de análise de complementaridade (o gerador usa este mesmo código para os números do texto)
import os
import sys
import numpy as np
import pandas as pd

for _p in (os.getcwd(), os.path.join(os.getcwd(), 'notebooks')):
    if os.path.exists(os.path.join(_p, 'iara_results.py')) and _p not in sys.path:
        sys.path.insert(0, _p)
import iara_results as R

REPS = ['MEL', 'LOFAR', 'HHT', 'DEMON']
FAMILIAS = {
    # Random Forest (scratch/generate_rf_predictions_and_analysis.py): 100 árvores, sem max_depth,
    # class_weight='balanced'. MEL e LOFAR usam caches com média em 5,12 s (mel5s/lofar5s).
    'RF': {'MEL': 'rf_mel5s_n100_balanced_4c', 'LOFAR': 'rf_lofar5s_n100_balanced_4c',
           'HHT': 'rf_hht_n100_balanced_4c', 'DEMON': 'rf_demon_n100_balanced_4c'},
    # SVM Nyström (Notebooks 10/12/13; scratch/save_svm_all_representations.py)
    'SVM': {'MEL': 'svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0',
            'LOFAR': 'svm_nystroem_6000_lofar_elasticnet_l1r0.15',
            'HHT': 'svm_nystroem_4000_hht_C2.0_elasticnet_l1r0.15_4c',
            'DEMON': 'svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_4c'},
    # MLP (Notebooks 14/15/16; scratch/generate_mlp_predictions_and_analysis.py)
    'MLP': {'MEL': 'mlp', 'LOFAR': 'mlp_lofar',
            'HHT': 'mlp_hht_h32-16_e30_4c', 'DEMON': 'mlp_demon_h32-16_e30_4c'},
}
CLS = {nome: i for i, nome in enumerate(R.CLASSES)}


def nomes(familia):
    return [FAMILIAS[familia][r] for r in REPS]


def ship_ids():
    meta = pd.read_csv(os.path.join(R.ROOT, 'src', 'iara', 'dataset_info', 'iara.csv'),
                       dtype={'ID': str, 'Ship ID': str})
    return dict(zip(meta['ID'].astype(int), meta['Ship ID']))


def instancias(familia, classe):
    """Uma linha por instância (arquivo x fold de teste) da classe, com a predição de cada representação."""
    exps = FAMILIAS[familia]
    al = R.aligned(nomes(familia))
    df = pd.concat([d.assign(fold=k) for k, d in al.items()], ignore_index=True)
    df = df.rename(columns={exps[r]: r for r in REPS})
    df = df[df.Target == CLS[classe]].copy()
    df['Ship ID'] = df.File.map(ship_ids())
    return df


def analise_classe(familia, classe):
    c = CLS[classe]
    d = instancias(familia, classe)
    H = np.column_stack([d[r].values == c for r in REPS])   # acerto de cada representação
    k = H.sum(axis=1)                                         # nº de representações que acertam
    n = len(d)
    out = {'familia': familia, 'classe': classe, 'n_inst': n, 'n_grav': int(d.File.nunique()),
           'n_navios': int(d['Ship ID'].nunique()), 'n_sem_ship': int(d['Ship ID'].isna().sum())}

    rows = []
    for i, r in enumerate(REPS):
        s = R.summary(FAMILIAS[familia][r])
        rows.append({'Representação': r, 'Acertos (instâncias)': int(H[:, i].sum()),
                     'Recall agregado (%)': 100 * H[:, i].mean(),
                     'Recall médio nos folds (%)': s['R_' + classe][0], 'Desvio nos folds': s['R_' + classe][1],
                     'SP global 4c (%)': s['SP'][0], 'Desvio SP': s['SP'][1],
                     'Experimento': FAMILIAS[familia][r]})
    iso = pd.DataFrame(rows).set_index('Representação')
    out['isolado'] = iso
    best = iso['Recall médio nos folds (%)'].idxmax()
    out['melhor'] = best

    orc = R.oracle_recall(nomes(familia)).loc[classe]
    out['oraculo'] = {'acertos': int((k >= 1).sum()), 'agregado': 100 * (k >= 1).mean(),
                      'obs_folds': float(orc.iloc[0]), 'esp_indep': float(orc.iloc[1])}

    niveis = (4, 3, 2, 1, 0)
    cons = pd.DataFrame({'Nível': [f'{j} de 4 acertam' for j in niveis],
                         'Instâncias': [int((k == j).sum()) for j in niveis],
                         'Gravações distintas': [int(d.File[k == j].nunique()) for j in niveis]})
    cons['% da classe'] = 100 * cons['Instâncias'] / n
    out['consenso'] = cons.set_index('Nível')[['Instâncias', '% da classe', 'Gravações distintas']]

    rows = []
    for i, r in enumerate(REPS):
        m = (k == 1) & H[:, i]
        todas = d.assign(excl=m).groupby('File').excl.all()
        rows.append({'Representação': r, 'Instâncias': int(m.sum()), '% da classe': 100 * m.mean(),
                     'Gravações distintas': int(d.File[m].nunique()),
                     'Navios distintos': int(d['Ship ID'][m].nunique()),
                     'Gravações exclusivas em todas as aparições': int(todas.sum())})
    out['exclusivos'] = pd.DataFrame(rows).set_index('Representação')
    m1 = k == 1
    out['excl_total'] = {'inst': int(m1.sum()), 'grav': int(d.File[m1].nunique()),
                         'navios': int(d['Ship ID'][m1].nunique())}

    b = H[:, REPS.index(best)]
    rows = []
    for i, r in enumerate(REPS):
        if r == best:
            continue
        rows.append({'Troca': f'{best} → {r}', 'Mantém acertos': int((b & H[:, i]).sum()),
                     'Perde acertos': int((b & ~H[:, i]).sum()), 'Resgata erros': int((~b & H[:, i]).sum())})
    rev = pd.DataFrame(rows).set_index('Troca')
    rev['% dos acertos perdidos'] = 100 * rev['Perde acertos'] / b.sum()
    rev['% dos erros resgatados'] = 100 * rev['Resgata erros'] / (~b).sum()
    rev['Saldo (instâncias)'] = rev['Resgata erros'] - rev['Perde acertos']
    out['reverso'] = rev

    rows = []
    for i, r in enumerate(REPS):
        e = d.loc[~H[:, i], r]
        row = {'Representação': r, 'Erros (instâncias)': int(len(e))}
        for cc in range(4):
            if cc != c:
                row[f'→ {R.CLASSES[cc]} (%)'] = 100 * (e == cc).mean() if len(e) else np.nan
        rows.append(row)
    out['erros'] = pd.DataFrame(rows).set_index('Representação')

    # "≥2/4" conta como acerto os empates 2x2 (as duas erradas concordam numa mesma classe): não é um voto real
    P = d[REPS].values
    empate = 0
    for row, h in zip(P[k == 2], H[k == 2]):
        errados = row[~h]
        empate += int(errados[0] == errados[1])
    mv = R.majority_vote(nomes(familia))   # voto real: pluralidade, empate sorteado (seed fixa)
    rb = R.fold_metrics(FAMILIAS[familia][best])['R_' + classe]
    F, p = R.f_test_5x2(mv['R_' + classe].values, rb.values)
    best_sp = iso['SP global 4c (%)'].idxmax()
    sb = R.fold_metrics(FAMILIAS[familia][best_sp])['SP']
    Fs, ps = R.f_test_5x2(mv['SP'].values, sb.values)
    out['voto'] = {'ge2_n': int((k >= 2).sum()), 'ge2': 100 * (k >= 2).mean(),
                   'ge3_n': int((k >= 3).sum()), 'ge3': 100 * (k >= 3).mean(), 'empates_2x2': empate,
                   'recall': (mv['R_' + classe].mean(), mv['R_' + classe].std(ddof=0)),
                   'melhor': (rb.mean(), rb.std(ddof=0)), 'F': F, 'p': p,
                   'sp': (mv['SP'].mean(), mv['SP'].std(ddof=0)), 'melhor_sp_rep': best_sp,
                   'melhor_sp': (sb.mean(), sb.std(ddof=0)), 'F_sp': Fs, 'p_sp': ps}
    return out


def tabela_voto(a):
    v = a['voto']
    return pd.DataFrame([
        {'Critério': 'Unanimidade (4/4)', f'Recall {a["classe"]} (%)': a['consenso']['% da classe'].iloc[0],
         'Observação': 'contagem agregada'},
        {'Critério': 'Maioria estrita (≥3/4)', f'Recall {a["classe"]} (%)': v['ge3'],
         'Observação': 'contagem agregada'},
        {'Critério': '≥2/4 (empate 2×2 contado como acerto)', f'Recall {a["classe"]} (%)': v['ge2'],
         'Observação': f'NÃO realizável: inclui {v["empates_2x2"]} empates 2×2'},
        {'Critério': 'Voto real (pluralidade, empate sorteado)', f'Recall {a["classe"]} (%)': v['recall'][0],
         'Observação': f'média ± {v["recall"][1]:.2f} nos folds; F-test vs {a["melhor"]}: p = {v["p"]:.3f}'},
        {'Critério': f'Melhor isolado ({a["melhor"]})', f'Recall {a["classe"]} (%)': v['melhor'][0],
         'Observação': f'média ± {v["melhor"][1]:.2f} nos folds'},
        {'Critério': 'Oráculo (≥1/4)', f'Recall {a["classe"]} (%)': a['oraculo']['agregado'],
         'Observação': f'NÃO realizável; esperado c/ erros independentes: {a["oraculo"]["esp_indep"]:.2f}'},
    ]).set_index('Critério')


def resumo_familias(classe, familias=('SVM', 'MLP', 'RF')):
    A = {f: analise_classe(f, classe) for f in familias}
    rows = {}
    for f, a in A.items():
        iso, ex, v = a['isolado'], a['exclusivos'], a['voto']
        b = a['melhor']
        champ = ex['Instâncias'].idxmax()
        rows[f] = {
            'Melhor isolado: recall médio ± desvio (%)': f"{b}: {iso.loc[b, 'Recall médio nos folds (%)']:.2f} ± {iso.loc[b, 'Desvio nos folds']:.2f}",
            'Oráculo observado (agregado, %)': f"{a['oraculo']['agregado']:.2f}",
            'Oráculo esperado c/ erros independentes (%)': f"{a['oraculo']['esp_indep']:.2f}",
            'Unanimidade 4/4 (%)': f"{a['consenso']['% da classe'].iloc[0]:.2f}",
            'Erro comum 0/4 (%)': f"{a['consenso']['% da classe'].iloc[-1]:.2f}",
            'Maior resgate exclusivo': f"{champ}: {ex.loc[champ, 'Instâncias']} inst. / {ex.loc[champ, 'Gravações distintas']} grav.",
            '≥2/4 com empates (não realizável, %)': f"{v['ge2']:.2f}",
            'Voto real: recall (%) [p vs melhor isolado]': f"{v['recall'][0]:.2f} ± {v['recall'][1]:.2f} [p = {v['p']:.3f}]",
        }
    return pd.DataFrame(rows), A


def f_test_melhores(A, classe, ref='RF'):
    """F-test 5x2cv entre o recall da classe do melhor isolado de `ref` e o das outras famílias."""
    out = {}
    ra = R.fold_metrics(FAMILIAS[ref][A[ref]['melhor']])['R_' + classe].values
    for f, a in A.items():
        if f == ref:
            continue
        rb = R.fold_metrics(FAMILIAS[f][a['melhor']])['R_' + classe].values
        out[f] = R.f_test_5x2(ra, rb)
    return out
'''

SETUP_SRC = r'''# 1. Configuração do Ambiente e Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 15, 'figure.dpi': 150})
pd.set_option('display.precision', 2)
print("Ambiente configurado com sucesso! Bibliotecas prontas.")'''


def run_helper():
    """Executa HELPER_SRC no gerador (cwd = raiz do repositório) e devolve o namespace."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    ns = {}
    exec(HELPER_SRC, ns)
    return ns


def br(x, nd=2):
    """Número no formato brasileiro: 1.234,56 (inteiros: 1.234)."""
    if isinstance(x, numbers.Integral) or nd == 0:
        return f"{int(round(x)):,}".replace(",", ".")
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def p_txt(p):
    return "p < 0,001" if p < 0.001 else f"p = {br(p, 3)}"


def md(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]}


def code(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]}


def write_notebook(cells, filename):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"codemirror_mode": {"name": "ipython", "version": 3},
                                         "file_extension": ".py", "mimetype": "text/x-python", "name": "python",
                                         "nbconvert_exporter": "python", "pygments_lexer": "ipython3",
                                         "version": "3.10.12"}},
          "nbformat": 4, "nbformat_minor": 4}
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Notebook gerado com sucesso em '{filename}'!")


# --------------------------------------------------------------------------------------------------------------
# Construção comum dos notebooks 17-19
# --------------------------------------------------------------------------------------------------------------
NB_OUTRAS = {"SMALL": (10, 14), "MEDIUM": (12, 15), "LARGE": (13, 16)}


def aviso_modelos(ns):
    R = ns["R"]
    t7m, t7l = R.ARTICLE_TABLE7[("RF", "MEL")]["SP"], R.ARTICLE_TABLE7[("RF", "LOFAR")]["SP"]
    s = {r: R.summary(ns["FAMILIAS"]["RF"][r])["SP"] for r in ns["REPS"]}
    return (
        "> **Atenção — estes não são os RF do artigo.** Os quatro modelos foram treinados por "
        "`scratch/generate_rf_predictions_and_analysis.py` (100 árvores, sem `max_depth`, `class_weight='balanced'`). "
        "Para MEL e LOFAR, o script usa caches com **média em 5,12 s** (`mel5s`/`lofar5s`: o arquivo 1 tem 55 frames, "
        "contra 551 nas features oficiais de 0,512 s). Por isso, \"RF MEL\" e \"RF LOFAR\" aqui "
        f"(SP {br(s['MEL'][0])} ± {br(s['MEL'][1])} e {br(s['LOFAR'][0])} ± {br(s['LOFAR'][1])}) **não são o mesmo modelo** "
        f"do RF da Tabela 7 do artigo (MEL {br(t7m[0])} ± {br(t7m[1])}; LOFAR {br(t7l[0])} ± {br(t7l[1])}). "
        "Não compare esses valores como se fossem o mesmo experimento.\n"
        ">\n"
        "> O split do 5×2cv é **por gravação** (`exclusive_header='ID'`), como no artigo: o mesmo navio (`Ship ID`) "
        "pode aparecer no treino e no teste.\n"
        ">\n"
        f"> RF DEMON é fraco no problema de 4 classes (SP {br(s['DEMON'][0])} ± {br(s['DEMON'][1])}); "
        "seus resgates e acertos nesta classe devem ser lidos com isso em mente."
    )


def metodologia(a):
    cls = a["classe"]
    rep = a["n_inst"] / a["n_grav"]
    return f"""## 2. Metodologia Experimental e Base de Dados de Teste

Avaliação sobre a classe real **`{cls}` (Classe {['SMALL', 'MEDIUM', 'LARGE'].index(cls)})** nos conjuntos de teste dos 10 folds do 5×2cv, com as 4 representações (**MEL, LOFAR, HHT e DEMON**) no Random Forest.

**Como contar.** Cada linha das tabelas abaixo é uma **instância arquivo × fold de teste** (predição por arquivo = voto majoritário das janelas, como no pipeline `iara`):
- **{br(a['n_inst'])} instâncias** de teste de `{cls}`;
- **{br(a['n_grav'])} gravações distintas** (cada gravação aparece, em média, **{br(rep)} vezes** como teste nos 10 folds — e não 5 vezes: parte dos arquivos de cada fold vai para validação);
- **{br(a['n_navios'])} navios distintos** (`Ship ID` em `src/iara/dataset_info/iara.csv`).

Percentuais "agregados" usam as {br(a['n_inst'])} instâncias; "recall médio nos folds" é a média ± desvio (ddof=0) sobre os 10 folds, como no artigo.

> **Nota de reprodutibilidade.** O script de origem desempatava o voto das janelas pela menor classe; aqui usamos `Counter.most_common` (empate → classe da primeira janela), igual a `src/iara/ml/models/trainer.py`. Por isso as contagens diferem da versão anterior deste notebook em poucas instâncias (até ~1 p.p.)."""


def cells_comuns(ns, a, nb_num, nome_classe_txt, hipoteses=""):
    """Células das seções 2 a 7, iguais em estrutura para as três classes."""
    cls = a["classe"]
    iso, orc, cons, ex, rev, er, v = (a["isolado"], a["oraculo"], a["consenso"], a["exclusivos"], a["reverso"],
                                       a["erros"], a["voto"])
    b = a["melhor"]
    rb = iso.loc[b, "Recall médio nos folds (%)"]
    cons_pct = cons["% da classe"].values
    cons_n = cons["Instâncias"].values
    ex_sorted = ex.sort_values("Instâncias", ascending=False)
    champ = ex_sorted.index[0]
    voto_txt = ("**não difere significativamente**" if v["p"] >= 0.05 else
                ("é **significativamente maior**" if v["recall"][0] > v["melhor"][0] else "é **significativamente menor**"))
    sp_txt = ("equivalente" if v["p_sp"] >= 0.05 else ("significativamente maior" if v["sp"][0] > v["melhor_sp"][0]
                                                         else "significativamente menor"))
    err_cols = [c for c in er.columns if c.startswith("→")]
    dest = {r: er.loc[r, err_cols].astype(float).idxmax().replace("→ ", "").replace(" (%)", "") for r in er.index}

    dif = orc['esp_indep'] - orc['obs_folds']
    orc_txt = (f"O oráculo fica **{br(dif)} p.p. abaixo** do esperado sob independência: os erros das representações são "
               "**correlacionados** (tendem a cair nos mesmos áudios)." if dif > 0 else
               f"O oráculo fica {br(-dif)} p.p. acima do esperado sob independência.")
    cells = []
    cells.append(md(metodologia(a)))
    cells.append(code(f"""# 2. Desempenho global dos 4 RF e dados da classe {cls}
CLASSE = '{cls}'
display(R.table([(f'RF {{r}}', FAMILIAS['RF'][r]) for r in REPS], metrics=('SP', 'ACC', 'F1')))
print('Referência — RF da Tabela 7 do artigo (outro modelo: features de 0,512 s, outra configuração):')
display(pd.DataFrame({{f'RF {{rep}} (artigo)': {{m: R.fmt(v) for m, v in R.ARTICLE_TABLE7[('RF', rep)].items()}}
                      for rep in ('MEL', 'LOFAR')}}).T)

A = analise_classe('RF', CLASSE)
print(f"Instâncias arquivo×fold: {{A['n_inst']}} | gravações distintas: {{A['n_grav']}} | "
      f"navios distintos: {{A['n_navios']}} | aparições médias por gravação: {{A['n_inst']/A['n_grav']:.2f}}")"""))

    cells.append(md(f"""## 3. O Efeito Oráculo no Random Forest: Recall Individual vs. Oráculo

O melhor RF isolado em `{cls}` é o **{b}**, com recall médio de **{br(rb)}% ± {br(iso.loc[b, 'Desvio nos folds'])}** nos folds ({br(iso.loc[b, 'Acertos (instâncias)'])} de {br(a['n_inst'])} instâncias).

O **oráculo** (acerta se ALGUMA das 4 entradas acerta) chega a **{br(orc['agregado'])}% ({br(orc['acertos'])} instâncias)**. Dois cuidados:
- o oráculo **não é um sistema realizável**: ele usa o rótulo verdadeiro para escolher a entrada;
- se os erros das 4 entradas fossem **independentes**, o esperado seria **{br(orc['esp_indep'])}%** (1 − ∏(1 − rᵢ), média nos folds), contra **{br(orc['obs_folds'])}%** observados na mesma média por fold. {orc_txt}

O erro comum (as 4 florestas erram) atinge **{br(int(cons_n[-1]))} instâncias ({br(cons_pct[-1])}%)**."""))

    cells.append(code(f"""# 3. Gráfico: Desempenho isolado vs. oráculo (observado e esperado sob independência)
iso = A['isolado']
display(iso)
labels = [f'RF {{r}}' for r in iso.index] + ['ORÁCULO RF (não realizável)']
vals = list(iso['Recall agregado (%)']) + [A['oraculo']['agregado']]
plt.figure(figsize=(10, 5.5))
colors = ['#2980b9', '#e67e22', '#9b59b6', '#95a5a6', '#27ae60']
bars = plt.bar(labels, vals, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
bars[-1].set_hatch('//')
plt.axhline(A['oraculo']['esp_indep'], color='#c0392b', linestyle='--', lw=1.5, alpha=0.8,
            label=f"Oráculo esperado c/ erros independentes ({{A['oraculo']['esp_indep']:.2f}}%)")
plt.axhline(iso.loc[A['melhor'], 'Recall agregado (%)'], color='#2980b9', linestyle=':', lw=1.5, alpha=0.8,
            label=f"Melhor isolado {{A['melhor']}} ({{iso.loc[A['melhor'], 'Recall agregado (%)']:.2f}}%)")
plt.title(f'Recall Random Forest na Classe {{CLASSE}}: Modelos Isolados vs. Oráculo', fontweight='bold', pad=12)
plt.ylabel(f'Recall agregado em {{CLASSE}} (%)')
plt.ylim(0, 105)
for bar in bars:
    h = bar.get_height()
    plt.annotate(f'{{h:.1f}}%', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords='offset points',
                 ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#111')
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.show()"""))

    rev_lines = "\n".join(
        f"- **{t}:** mantém {br(int(rr['Mantém acertos']))}, perde {br(int(rr['Perde acertos']))} ({br(rr['% dos acertos perdidos'])}% dos acertos do {b}) e resgata {br(int(rr['Resgata erros']))} ({br(rr['% dos erros resgatados'])}% dos erros) — saldo **{int(rr['Saldo (instâncias)']):+d}** instâncias;"
        for t, rr in rev.iterrows())
    cells.append(md(f"""## 4. Grau de Consenso, Efeito Reverso e Votação em `{cls}` (RF)

### 4.1 Pirâmide de Consenso
- **Unanimidade (4/4 certas):** {br(int(cons_n[0]))} instâncias ({br(cons_pct[0])}%);
- **Alto consenso (3/4 certas):** {br(int(cons_n[1]))} instâncias ({br(cons_pct[1])}%);
- **Consenso dividido (2/4 certas):** {br(int(cons_n[2]))} instâncias ({br(cons_pct[2])}%);
- **Resgate exclusivo (1/4 certa):** {br(int(cons_n[3]))} instâncias ({br(cons_pct[3])}%);
- **Erro comum (0/4 certas):** {br(int(cons_n[4]))} instâncias ({br(cons_pct[4])}%).

### 4.2 Efeito reverso a partir do melhor isolado ({b})
{rev_lines}

Trocar de entrada sem um critério de seleção reduz os acertos quando o saldo é negativo.

### 4.3 Votação: "≥2/4" vs. voto real
- **≥3/4:** {br(v['ge3'])}% ({br(v['ge3_n'])} instâncias);
- **≥2/4:** {br(v['ge2'])}% ({br(v['ge2_n'])} instâncias). Esse número **não é um voto realizável**: conta como acerto os {br(v['empates_2x2'])} empates 2×2 (duas entradas certas contra duas que concordam numa mesma classe errada);
- **Voto real** (pluralidade entre as 4 entradas, empate sorteado; `R.majority_vote`): recall em `{cls}` de **{br(v['recall'][0])}% ± {br(v['recall'][1])}**, contra {br(v['melhor'][0])}% ± {br(v['melhor'][1])} do {b} isolado. Pelo F-test 5×2cv o voto real {voto_txt} do melhor isolado nesta classe ({p_txt(v['p'])});
- No SP global (4 classes), o voto real tem {br(v['sp'][0])} ± {br(v['sp'][1])}, contra {br(v['melhor_sp'][0])} ± {br(v['melhor_sp'][1])} do RF {v['melhor_sp_rep']} isolado: **{sp_txt}** ({p_txt(v['p_sp'])});
- **Oráculo (≥1/4):** {br(orc['agregado'])}% ({br(orc['acertos'])} instâncias) — teto não realizável."""))

    cells.append(code(f"""# 4. Visualização do Grau de Consenso e do Efeito Reverso em {cls} (RF)
display(A['consenso'])
display(A['reverso'])
display(tabela_voto(A))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
cons = A['consenso']
labels_cons = ['4 Entradas\\n(Unânime)', '3 Entradas\\n(Alto Consenso)', '2 Entradas\\n(Dividido)',
               '1 Entrada\\n(Exclusivo)', '0 Entradas\\n(Erro Comum)']
bars1 = ax1.bar(labels_cons, cons['Instâncias'], color=['#27ae60', '#2980b9', '#f39c12', '#e74c3c', '#7f8c8d'],
                edgecolor='black', linewidth=0.8, width=0.55)
ax1.set_title(f"Grau de Consenso RF em {{CLASSE}} ({{A['n_inst']}} instâncias arquivo×fold)", fontweight='bold', pad=12)
ax1.set_ylabel(f'Instâncias de {{CLASSE}}')
ax1.set_ylim(0, cons['Instâncias'].max() * 1.18)
for bar, pct in zip(bars1, cons['% da classe']):
    h = bar.get_height()
    ax1.annotate(f'{{int(h)}} ({{pct:.1f}}%)', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                 textcoords='offset points', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

rev = A['reverso']
x = np.arange(len(rev))
w = 0.35
r1 = ax2.bar(x - w / 2, rev['Perde acertos'], w, label=f"Perde acertos do {{A['melhor']}}", color='#c0392b', alpha=0.9, edgecolor='black')
r2 = ax2.bar(x + w / 2, rev['Resgata erros'], w, label=f"Resgata erros do {{A['melhor']}}", color='#27ae60', alpha=0.9, edgecolor='black')
ax2.set_title(f"Efeito Reverso no RF ao Trocar de Entrada a partir do {{A['melhor']}}", fontweight='bold', pad=12)
ax2.set_ylabel(f'Instâncias de {{CLASSE}}')
ax2.set_xticks(x)
ax2.set_xticklabels(rev.index, fontweight='bold', fontsize=11)
ax2.set_ylim(0, max(rev['Perde acertos'].max(), rev['Resgata erros'].max()) * 1.2)
ax2.legend(loc='upper right', frameon=True)
for rect in r1:
    ax2.annotate(f'-{{int(rect.get_height())}}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()), xytext=(0, 3),
                 textcoords='offset points', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#900')
for rect in r2:
    ax2.annotate(f'+{{int(rect.get_height())}}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()), xytext=(0, 3),
                 textcoords='offset points', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#006400')
plt.tight_layout()
plt.show()"""))

    ex_lines = "\n".join(
        f"{i + 1}. **{r}:** {br(int(rr['Instâncias']))} instâncias ({br(rr['% da classe'])}%), de {br(int(rr['Gravações distintas']))} gravações e {br(int(rr['Navios distintos']))} navios; dessas gravações, **{br(int(rr['Gravações exclusivas em todas as aparições']))}** são resgate exclusivo do {r} em **todas** as suas aparições de teste."
        for i, (r, rr) in enumerate(ex_sorted.iterrows()))
    et = a["excl_total"]
    cells.append(md(f"""## 5. Resgates Exclusivos no Random Forest em `{cls}`

Em **{br(et['inst'])} instâncias ({br(cons_pct[3])}%)** apenas uma das 4 florestas acertou. Elas correspondem a **{br(et['grav'])} gravações distintas** e **{br(et['navios'])} navios distintos** (a mesma gravação aparece em vários folds e pode ser resgatada por entradas diferentes em cada um):

{ex_lines}

O maior número de resgates exclusivos é do **{champ}**. Os resgates são, em grande parte, **instáveis entre folds**: poucas gravações são resgatadas pela mesma entrada em todas as aparições. Um resgate exclusivo também não é aproveitável sem um critério que saiba, sem o rótulo, quando confiar naquela entrada."""))

    cells.append(code(f"""# 5. Gráfico: Resgates Exclusivos na Classe {cls} (RF)
ex = A['exclusivos'].sort_values('Instâncias')
display(A['exclusivos'])
plt.figure(figsize=(10, 4.5))
bars_excl = plt.barh([f'{{r}} exclusivo' for r in ex.index], ex['Instâncias'], color='#16a085', height=0.5,
                     edgecolor='black', linewidth=0.8)
plt.title(f"Resgates Exclusivos RF em {{CLASSE}} ({{A['excl_total']['inst']}} instâncias = {{A['excl_total']['grav']}} gravações distintas)",
          fontweight='bold', pad=12)
plt.xlabel('Instâncias (arquivo×fold) resgatadas exclusivamente')
plt.xlim(0, ex['Instâncias'].max() * 1.6)
for bar, (r, row) in zip(bars_excl, ex.iterrows()):
    w = bar.get_width()
    plt.annotate(f"{{int(w)}} inst. ({{row['% da classe']:.1f}}%) | {{int(row['Gravações distintas'])}} grav. | "
                 f"{{int(row['Gravações exclusivas em todas as aparições'])}} estáveis",
                 xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(5, 0), textcoords='offset points',
                 ha='left', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()"""))

    rows_err = "\n".join(
        f"| **{r}** | {br(int(er.loc[r, 'Erros (instâncias)']))} | " + " | ".join(f"{br(er.loc[r, c])}%" for c in err_cols) +
        f" | `{dest[r]}` |" for r in er.index)
    head = " | ".join(f"`{c.replace('→ ', '').replace(' (%)', '')}`" for c in err_cols)
    cells.append(md(f"""## 6. Padrões de Erro em `{cls}`: Para Onde Cada Entrada Envia {nome_classe_txt} no RF?

Distribuição dos erros de cada entrada (instâncias agregadas):

| Entrada | Erros | {head} | Principal destino |
| :--- | :---: | {' | '.join([':---:'] * len(err_cols))} | :---: |
{rows_err}

{hipoteses}"""))

    stack_cols = [c for c in err_cols]
    cells.append(code(f"""# 6. Gráfico: Destino do Erro em {cls} por Representação (RF)
er = A['erros']
display(er)
cols = [c for c in er.columns if c.startswith('→')]
cores = {{'SMALL': '#3498db', 'MEDIUM': '#f39c12', 'LARGE': '#2ecc71', 'BACKGROUND': '#95a5a6'}}
plt.figure(figsize=(10, 5.5))
bottom = np.zeros(len(er))
for c in cols:
    alvo = c.replace('→ ', '').replace(' (%)', '')
    vals = er[c].values.astype(float)
    plt.bar(er.index, vals, 0.52, bottom=bottom, label=f'Confundido com {{alvo}}', color=cores[alvo], alpha=0.9,
            edgecolor='black', linewidth=0.6)
    for i, (vv, bb) in enumerate(zip(vals, bottom)):
        if vv > 3:
            plt.text(i, bb + vv / 2, f'{{vv:.1f}}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    bottom += vals
plt.xticks(range(len(er)), [f"{{r}} ({{n}} erros)" for r, n in zip(er.index, er['Erros (instâncias)'])])
plt.title(f'Padrão de Erro RF na Classe {{CLASSE}}: destino dos erros (%)', fontweight='bold', pad=12)
plt.ylabel('Distribuição Percentual dos Erros (%)')
plt.ylim(0, 118)
plt.legend(loc='upper center', ncol=3, frameon=True, fontsize=9)
plt.tight_layout()
plt.show()"""))

    cells.append(md(f"""## 7. Painel Consolidado (figura do script de origem)

A figura abaixo foi gerada por `scratch/generate_rf_predictions_and_analysis.py`, que desempata o voto das janelas pela menor classe. Por isso suas contagens podem diferir das células acima em poucas instâncias; **os valores de referência são os das células deste notebook**. A figura também usa a contagem agregada de instâncias, não de gravações."""))
    cells.append(code(f"""# 7. Exibição do painel salvo em disco (com fallback de caminho para cwd = notebooks/)
from IPython.display import Image
img_name = 'analise_complementaridade_rf_{cls.lower()}_oraculo.png'
for img_path in (os.path.join('reports', img_name), os.path.join('..', 'reports', img_name)):
    if os.path.exists(img_path):
        display(Image(filename=img_path))
        break
else:
    print(f"Imagem {{img_name}} não encontrada em reports/ nem ../reports/. "
          "Execute 'scratch/generate_rf_predictions_and_analysis.py' para gerá-la.")"""))
    return cells


def secao_trilateral(ns, cls, nb_num, sec_num):
    """Markdown + código da comparação SVM vs MLP vs RF na classe."""
    nb_svm, nb_mlp = NB_OUTRAS[cls]
    tab, A = ns["resumo_familias"](cls)
    ft = ns["f_test_melhores"](A, cls)
    rf = A["RF"]
    brf = rf["melhor"]
    linhas = []
    for f in ("SVM", "MLP"):
        a = A[f]
        F, p = ft[f]
        rr, ro = rf["isolado"].loc[brf, "Recall médio nos folds (%)"], a["isolado"].loc[a["melhor"], "Recall médio nos folds (%)"]
        rel = ("não difere significativamente" if p >= 0.05 else ("é significativamente maior" if rr > ro else "é significativamente menor"))
        linhas.append(f"- RF {brf} ({br(rr)}%) vs. {f} {a['melhor']} ({br(ro)}%): o recall do RF {rel} ({p_txt(p)}).")
    orcs = "; ".join(f"{f} {br(A[f]['oraculo']['agregado'])}% (esperado sob independência {br(A[f]['oraculo']['esp_indep'])}%)"
                     for f in ("SVM", "MLP", "RF"))
    abaixo = [f for f in ("SVM", "MLP", "RF") if A[f]['oraculo']['agregado'] < A[f]['oraculo']['esp_indep']]
    orc_cmp = ("Em todas as famílias o oráculo observado fica abaixo do esperado sob independência, isto é, os erros são correlacionados."
               if len(abaixo) == 3 else f"O oráculo observado fica abaixo do esperado sob independência em: {', '.join(abaixo) or 'nenhuma família'}.")
    unan = "; ".join(f"{f} {br(A[f]['consenso']['% da classe'].iloc[0])}%" for f in ("SVM", "MLP", "RF"))
    md_txt = f"""## {sec_num}. Comparativo Trilateral em `{cls}`: SVM Nyström vs. MLP vs. Random Forest

A tabela abaixo é calculada com a mesma função para as três famílias (SVM: Notebook {nb_svm}; MLP: Notebook {nb_mlp}; RF: Notebook {nb_num}). Os experimentos de cada família estão em `FAMILIAS`. As famílias **não são homogêneas**: SVM HHT/DEMON usam m = 4000, os mesmos do Notebook {nb_svm} (e são instáveis entre folds); MLP HHT/DEMON vêm de um script ad hoc (30 épocas, sem validação); RF MEL/LOFAR usam as features de 5,12 s.

**Melhor isolado (recall médio em `{cls}` nos folds), com F-test 5×2cv:**
{chr(10).join(linhas)}

**Oráculos (não realizáveis):** {orcs}. {orc_cmp} Com 4 entradas, um oráculo alto não indica um sistema realizável.

**Unanimidade 4/4:** {unan}."""
    code_txt = f"""# {sec_num}. Comparativo trilateral calculado (SVM vs MLP vs RF) na classe {cls}
tab, A_fam = resumo_familias(CLASSE)
display(tab)
for f, (F, p) in f_test_melhores(A_fam, CLASSE).items():
    print(f"F-test 5x2cv do recall {{CLASSE}}: RF {{A_fam['RF']['melhor']}} vs {{f}} {{A_fam[f]['melhor']}} -> F = {{F:.2f}}, p = {{p:.3f}}")"""
    return [md(md_txt), code(code_txt)], A
