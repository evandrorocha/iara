"""
Script de geração do notebook 13: notebooks/13_analise_complementaridade_small.ipynb
Análise de complementaridade, oráculo (observado vs. esperado sob independência), grau de consenso,
voto real e risco de efeito reverso das 4 entradas do SVM (MEL, LOFAR, HHT, DEMON) na classe SMALL.

Os geradores dos notebooks 14 (MEDIUM) e 13 (LARGE) reutilizam ANALYSIS_CODE e build_notebook() deste arquivo.
Todos os números citados no markdown são calculados aqui, a partir das predições por janela em
results/trainings/{tests,multirep} via notebooks/iara_results.py, com o MESMO código que vai para o notebook.
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------------------------------------------
# Código de análise: executado no gerador (para os números do markdown) e copiado para o notebook.
# Requer as variáveis CLASS_IDX (0/1/2) e CLASS_NAME definidas antes.
# --------------------------------------------------------------------------------------------------
ANALYSIS_CODE = r'''import numpy as np
import pandas as pd
import iara_results as R

# Experimentos em disco (predições por janela, 10 folds do 5x2cv oficial)
EXPS = {
    "MEL":   "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0",       # tests/ (pipeline iara), m=4000
    "LOFAR": "svm_nystroem_6000_lofar_elasticnet_l1r0.15",                  # tests/ (pipeline iara), m=6000
    "HHT":   "svm_nystroem_4000_hht_C2.0_elasticnet_l1r0.15_4c",            # multirep/ (scratch), m=4000
    "DEMON": "svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_4c",          # multirep/ (scratch), m=4000
}
M_OF = {"MEL": 4000, "LOFAR": 6000, "HHT": 4000, "DEMON": 4000}
LBL = {k: f"SVM {k} (m={M_OF[k]})" for k in EXPS}
REPS = list(EXPS)
NAMES = [EXPS[k] for k in REPS]
CLS = ["SMALL", "MEDIUM", "LARGE", "BACKGROUND"]

# Uma linha por instância arquivo x fold de teste (arquivos presentes nos 4 experimentos)
_al = R.aligned(NAMES)
ALL = pd.concat([df.assign(fold=k) for k, df in _al.items()], ignore_index=True)
ALL = ALL.rename(columns={EXPS[k]: k for k in REPS})
D = ALL[ALL.Target == CLASS_IDX].reset_index(drop=True)
HIT = pd.DataFrame({k: D[k].values == CLASS_IDX for k in REPS})
NH = HIT.sum(axis=1).values

N_INST = len(D)                                   # instâncias arquivo x fold
N_REC = D.File.nunique()                          # gravações únicas
EVAL_PER_FILE_ALL = ALL.groupby("File").size().mean()   # avaliações de teste por arquivo (todas as classes)
EVAL_PER_FILE_CLS = D.groupby("File").size().mean()
N_FILES_TOTAL = ALL.File.nunique()
TEST_PER_FOLD = int(ALL.groupby("fold").size().iloc[0])
REP_COVER = float(np.mean([len(set(_al[2 * r].File) | set(_al[2 * r + 1].File)) for r in range(5)]))
APPEAR_DIST = ALL.groupby("File").size().value_counts().sort_index().to_dict()

# 1) Desempenho individual (contagens agregadas nas 10 dobras) e recall médio por fold
hits = {k: int(HIT[k].sum()) for k in REPS}
rec_pool = {k: 100 * hits[k] / N_INST for k in REPS}
rec_fold = {k: R.summary(EXPS[k])[f"R_{CLASS_NAME}"] for k in REPS}
sp_fold = {k: R.summary(EXPS[k])["SP"] for k in REPS}
rec_range = {k: (R.fold_metrics(EXPS[k])[f"R_{CLASS_NAME}"].min(), R.fold_metrics(EXPS[k])[f"R_{CLASS_NAME}"].max()) for k in REPS}
UNSTABLE = [k for k in REPS if sp_fold[k][1] > 5]   # desvio do SP > 5 p.p. entre folds: modelo instável/colapsado
ORDER = sorted(REPS, key=lambda k: (-hits[k], REPS.index(k)))
BEST = ORDER[0]
ORACLE_HITS = int((NH >= 1).sum())
df_hits = pd.DataFrame({
    "Modelo": [LBL[k] for k in REPS] + ["ORÁCULO (≥1 de 4; não realizável)"],
    "Acertos (instâncias)": [hits[k] for k in REPS] + [ORACLE_HITS],
    f"Recall {CLASS_NAME} agregado (%)": [round(rec_pool[k], 2) for k in REPS] + [round(100 * ORACLE_HITS / N_INST, 2)],
    f"Recall {CLASS_NAME} média±dp por fold (%)": [R.fmt(rec_fold[k]) for k in REPS] + ["—"],
    f"Recall {CLASS_NAME} mín–máx por fold (%)": [f"{rec_range[k][0]:.1f}–{rec_range[k][1]:.1f}" for k in REPS] + ["—"],
    "SP global 4c (%)": [R.fmt(sp_fold[k]) for k in REPS] + ["—"],
    "Experimento": NAMES + ["—"],
}).set_index("Modelo")

# 2) Oráculo observado vs. esperado com erros independentes (média por fold)
ORC = R.oracle_recall(NAMES).loc[CLASS_NAME]
ORC_OBS, ORC_EXP = float(ORC["oráculo observado"]), float(ORC["esperado c/ erros independentes"])
ORC_RANDOM = 100 * (1 - (3 / 4) ** 4)            # 4 preditores uniformes ao acaso em 4 classes

# 3) Grau de consenso
cons = {n: int((NH == n).sum()) for n in range(4, -1, -1)}
df_consensus = pd.DataFrame({
    "Nível de consenso": ["4 de 4 (unânime)", "3 de 4", "2 de 4", "1 de 4 (resgate exclusivo)", "0 de 4 (erro comum)"],
    "Instâncias": [cons[n] for n in range(4, -1, -1)],
    f"% do total {CLASS_NAME}": [round(100 * cons[n] / N_INST, 2) for n in range(4, -1, -1)],
    "% dentro do oráculo": [round(100 * cons[n] / ORACLE_HITS, 2) for n in range(4, 0, -1)] + [np.nan],
    "Gravações únicas": [D.File[NH == n].nunique() for n in range(4, -1, -1)],
}).set_index("Nível de consenso")

# 4) Regras de limiar (contagem de acertos) vs. VOTO REAL (pluralidade, empate sorteado)
ge = {k: int((NH >= k).sum()) for k in (4, 3, 2, 1)}
VOTE = R.majority_vote(NAMES)
VOTE_REC = (VOTE[f"R_{CLASS_NAME}"].mean(), VOTE[f"R_{CLASS_NAME}"].std(ddof=0))
VOTE_SP = (VOTE["SP"].mean(), VOTE["SP"].std(ddof=0))
BEST_SP = max(REPS, key=lambda k: sp_fold[k][0])
F_REC, P_REC = R.f_test_5x2(VOTE[f"R_{CLASS_NAME}"].values, R.fold_metrics(EXPS[BEST])[f"R_{CLASS_NAME}"].values)
F_SP, P_SP = R.f_test_5x2(VOTE["SP"].values, R.fold_metrics(EXPS[BEST_SP])["SP"].values)
# empates 2x2 com a classe correta entre as empatadas
P = D[REPS].values
n_tie_right = 0
for row in P:
    c = np.bincount(row, minlength=4)
    if c[CLASS_IDX] == c.max() and (c == c.max()).sum() > 1:
        n_tie_right += 1
df_vote = pd.DataFrame({
    "Regra": ["Unanimidade (4/4 acertam)", "≥3/4 acertam", "≥2/4 acertam (conta empates 2×2 como acerto)",
              "Oráculo ≥1/4 (não realizável)",
              "VOTO REAL (pluralidade, empate sorteado) — média por fold",
              f"Melhor modelo isolado: {LBL[BEST]} — média por fold"],
    f"Recall {CLASS_NAME} (%)": [round(100 * ge[4] / N_INST, 2), round(100 * ge[3] / N_INST, 2),
                                 round(100 * ge[2] / N_INST, 2), round(100 * ge[1] / N_INST, 2),
                                 R.fmt(VOTE_REC), R.fmt(rec_fold[BEST])],
}).set_index("Regra")

# 5) Resgates exclusivos (instâncias, gravações, gravações exclusivas em TODAS as suas aparições)
excl_rows = []
for k in REPS:
    m_ex = (NH == 1) & HIT[k].values
    files_ex = set(D.File[m_ex])
    always = sum(1 for f in files_ex if m_ex[D.File.values == f].all())
    excl_rows.append({"Modelo": LBL[k], "Instâncias": int(m_ex.sum()),
                      f"% do total {CLASS_NAME}": round(100 * m_ex.sum() / N_INST, 2),
                      "Gravações únicas": len(files_ex),
                      "Gravações exclusivas em todas as aparições": always})
df_exclusive = pd.DataFrame(excl_rows).set_index("Modelo").sort_values("Instâncias")
EXCL_CHAMP = df_exclusive["Instâncias"].idxmax()

# 6) Efeito reverso: trocar cegamente o melhor modelo isolado por outro
rev_rows = []
for k in REPS:
    if k == BEST:
        continue
    keep = int((HIT[BEST] & HIT[k]).sum()); lose = int((HIT[BEST] & ~HIT[k]).sum())
    resc = int((~HIT[BEST] & HIT[k]).sum())
    rev_rows.append({"Troca": f"{BEST} → {k}", "Mantém acertos": keep, "Perde acertos (efeito reverso)": lose,
                     "% perda": round(100 * lose / hits[BEST], 2), "Resgata erros": resc,
                     "% resgate": round(100 * resc / (N_INST - hits[BEST]), 2), "Saldo líquido": resc - lose})
df_reverse = pd.DataFrame(rev_rows).set_index("Troca")

# 7) Padrão de confusão: para onde vão os erros
conf_rows = []
for k in REPS:
    err = D[k][D[k] != CLASS_IDX]
    row = {"Modelo": f"{LBL[k]} ({len(err)} erros)"}
    for c in range(4):
        if c != CLASS_IDX:
            row[f"→ {CLS[c]} (%)"] = round(100 * (err == c).mean(), 2)
    conf_rows.append(row)
df_confusion = pd.DataFrame(conf_rows).set_index("Modelo")

# 8) Correlação dos erros entre pares: P(B erra | A erra) vs. P(B erra), e erro para a MESMA classe
pair_rows = []
for i, a in enumerate(REPS):
    for b in REPS[i + 1:]:
        ea, eb = ~HIT[a].values, ~HIT[b].values
        both = ea & eb
        same = int((both & (D[a].values == D[b].values)).sum())
        pair_rows.append({"Par": f"{a} × {b}", "P(B erra)": round(100 * eb.mean(), 2),
                          "P(B erra | A erra)": round(100 * eb[ea].mean(), 2),
                          "Ambos erram": int(both.sum()),
                          "Ambos erram p/ a mesma classe": same,
                          "% mesma classe": round(100 * same / max(both.sum(), 1), 2)})
df_pairs = pd.DataFrame(pair_rows).set_index("Par")
'''

INSTABILITY_CODE = r'''# Os três SVMs HHT/DEMON existentes em disco (mesmo hiperparâmetro nominal, m diferente) — não são o mesmo modelo
VARIANTS = [
    ("SVM HHT m=2000 (exportação por arquivo, legado)", "svm_nystroem_2000_hht_C2.0_elasticnet_l1r0.15_4c"),
    ("SVM HHT m=4000 (usado neste notebook)", "svm_nystroem_4000_hht_C2.0_elasticnet_l1r0.15_4c"),
    ("SVM HHT m=6000", "svm_nystroem_6000_hht_C2.0_elasticnet_l1r0.15_4c"),
    ("SVM DEMON m=2000 (exportação por arquivo, legado)", "svm_nystroem_2000_demon_C2.0_elasticnet_l1r0.15_4c"),
    ("SVM DEMON m=4000 (usado neste notebook)", "svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_4c"),
    ("SVM DEMON m=4000, max_iter=1000", "svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_it1000_4c"),
]
display(R.table(VARIANTS))
import sklearn
print('scikit-learn neste kernel:', sklearn.__version__)
'''


def compute(class_idx, class_name):
    """Executa ANALYSIS_CODE e devolve o namespace com todos os resultados."""
    cwd = os.getcwd()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    try:
        ns = {"CLASS_IDX": class_idx, "CLASS_NAME": class_name}
        exec(ANALYSIS_CODE, ns)
    finally:
        os.chdir(cwd)
    return ns


def br(x, nd=2):
    """Número em formato brasileiro para o markdown (vírgula decimal, ponto de milhar)."""
    if isinstance(x, (int,)) or (hasattr(x, "is_integer") and float(x).is_integer() and nd == 0):
        return f"{int(x):,}".replace(",", ".")
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pm(ms, nd=2):
    return f"{br(ms[0], nd)} ± {br(ms[1], nd)}"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


SETUP_CODE = r'''# 1. Configuração do ambiente e imports
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print

sys.path.insert(0, os.path.abspath('.') if os.path.exists('iara_results.py') else os.path.abspath('notebooks'))
sys.path.insert(0, os.path.abspath('src'))  # módulos movidos para notebooks/src
sys.path.insert(0, os.path.abspath('notebooks/src'))

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 15, 'figure.dpi': 150})
'''


def common_cells(ns, class_idx, class_name, nb_num):
    """Células comuns aos três notebooks (metodologia, tabelas, oráculo, consenso, voto, exclusivos, reverso,
    confusão, correlação de erros). Textos com números vindos de ns."""
    n, B = ns["N_INST"], ns["BEST"]
    L = ns["LBL"]
    hits, rp = ns["hits"], ns["rec_pool"]
    oh = ns["ORACLE_HITS"]
    cons = ns["cons"]
    ge = ns["ge"]
    rev = ns["df_reverse"]
    excl = ns["df_exclusive"]
    conf = ns["df_confusion"]
    pairs = ns["df_pairs"]
    cells = []

    cells.append(md(f"""
## 2. Metodologia e base de teste

- **Protocolo:** validação cruzada $5\\times2$cv do artigo (10 partições, **estratificadas e isoladas por gravação de áudio (`ID`)**, `exclusive_header='ID'` em `src/iara/ml/experiment.py`). O isolamento é por gravação, **não** por navio: o mesmo navio pode aparecer em treino e teste.
- **Modelos (SVM Nyström RBF + SGD hinge, ElasticNet $\\rho=0{{,}}15$, $C=2$):**
  - {L['MEL']} — `results/trainings/tests/{ns['EXPS']['MEL']}` (pipeline iara oficial);
  - {L['LOFAR']} — `results/trainings/tests/{ns['EXPS']['LOFAR']}` (pipeline iara oficial);
  - {L['HHT']} — `results/trainings/multirep/{ns['EXPS']['HHT']}` (`scratch/analyze_small_errors_by_input.py`);
  - {L['DEMON']} — `results/trainings/multirep/{ns['EXPS']['DEMON']}` (idem).
- **Predição por arquivo:** voto majoritário das janelas (`collections.Counter.most_common`, como o pipeline iara).
- **Unidade de análise:** **instância arquivo × fold de teste**. Há **{br(n, 0)} instâncias** cuja classe real é `{class_name}`, correspondentes a **{br(ns['N_REC'], 0)} gravações únicas**. As contagens abaixo são, portanto, de instâncias e não de áudios distintos.
- **Quantas vezes cada arquivo é testado:** cada fold tem {br(ns['TEST_PER_FOLD'], 0)} arquivos de teste (o pipeline reserva parte de cada metade para validação), e cada repetição do $5\\times2$cv cobre em média {br(ns['REP_COVER'], 0)} dos {br(ns['N_FILES_TOTAL'], 0)} arquivos. Assim, cada arquivo aparece no teste em média **{br(ns['EVAL_PER_FILE_ALL'])} vezes** nas 10 dobras ({br(ns['EVAL_PER_FILE_CLS'])} em `{class_name}`), e **não 5 vezes**; distribuição do nº de aparições (todas as classes): """ + ", ".join(f"{k}×: {v}" for k, v in ns['APPEAR_DIST'].items()) + """ arquivos.

> **Aviso de reprodutibilidade (HHT/DEMON m=4000).** Os SVMs HHT e DEMON com $m=4000$ foram re-treinados em 20/10/2026 pelo `scratch/analyze_small_errors_by_input.py` (mesmo código e `random_state=42`). As novas predições **não reproduzem** as predições antigas usadas na versão anterior destes notebooks.""" + divergence_txt() + """ Possíveis causas (não isoladas): diferença de versão do scikit-learn e paralelismo (`n_jobs=-1`) no Nyström + SGD com *early stopping*; o `config.json` destes dois experimentos **não registra** a versão do scikit-learn (no host está instalada a 1.9.0 e no contêiner `iara-dev` a 1.7.2). Trate o SVM HHT/DEMON $m=4000$ como de **baixa reprodutibilidade entre execuções/versões**. Por isso **todos** os números envolvendo HHT/DEMON (recalls, interseções, resgates exclusivos, oráculos) mudaram em relação à versão anterior. Os números de MEL e LOFAR (pipeline oficial) não mudaram. A última célula deste notebook mostra que os SVMs HHT/DEMON com outros $m$ (que levavam o mesmo rótulo "SVM HHT/DEMON" em outros notebooks) também diferem entre si.
"""))
    cells.append(code("# 2. Análise completa a partir das predições em disco\n"
                      f"CLASS_IDX, CLASS_NAME = {class_idx}, '{class_name}'\n" + ANALYSIS_CODE +
                      "\nprint(f'{N_INST} instâncias arquivo×fold de {CLASS_NAME}; {N_REC} gravações únicas; '\n"
                      "      f'{EVAL_PER_FILE_ALL:.2f} avaliações de teste por arquivo (todas as classes)')\n"
                      "display(df_hits)"))

    ind = "; ".join(f"{L[k]} {br(rp[k])}%" for k in ns["ORDER"])
    tied = [k for k in ns["REPS"] if k != B and hits[k] == hits[B]]
    tie_txt = (" (empatado com o " + ", ".join(L[k] for k in tied) + f", {br(hits[B], 0)} acertos cada; adota-se o {B} como referência)") if tied else ""
    unstable_txt = ""
    if ns["UNSTABLE"]:
        unstable_txt = ("\n> **Modelos instáveis.** " + "; ".join(
            f"{L[k]}: SP global {pm(ns['sp_fold'][k])}%, recall `{class_name}` entre {br(ns['rec_range'][k][0], 1)}% e {br(ns['rec_range'][k][1], 1)}% conforme o fold"
            for k in ns["UNSTABLE"]) + ". O desvio entre folds é muito maior que o dos SVMs MEL/LOFAR "
            f"(SP {pm(ns['sp_fold']['MEL'])}% e {pm(ns['sp_fold']['LOFAR'])}%): em alguns folds o modelo colapsa para poucas classes. "
            "Recalls agregados desses modelos (e as interseções que os envolvem) misturam folds colapsados e não colapsados.\n")
    cells.append(md(f"""
## 3. Oráculo: recall individual vs. união dos acertos

Recall agregado em `{class_name}` (acertos / {br(n, 0)} instâncias; como todo fold tem o mesmo número de arquivos de `{class_name}`, coincide com a média por fold): {ind}. O melhor modelo isolado é o **{L[B]}**{tie_txt}.
{unstable_txt}

O **oráculo** (acerta se ao menos uma das 4 entradas acertar) cobre **{br(oh, 0)} instâncias ({br(100 * oh / n)}%)**; o erro comum (todas erram) é de **{br(cons[0], 0)} instâncias ({br(100 * cons[0] / n)}%)**.

**Cuidados de interpretação:**
- O oráculo **não é um sistema realizável**: ele usa o rótulo verdadeiro para escolher a entrada. A união de preditores fracos cresce mesmo sem informação — 4 preditores uniformemente aleatórios em 4 classes já dariam recall esperado de $1-(3/4)^4 = {br(ns['ORC_RANDOM'])}\\%$.
- A referência correta é o **esperado se os erros fossem independentes**, $1-\\prod_i(1-r_i)$. Média por fold: **oráculo observado {br(ns['ORC_OBS'])}% vs. esperado sob independência {br(ns['ORC_EXP'])}%**. """ +
                    ("O observado fica **abaixo** do esperado: os erros das 4 entradas são **positivamente correlacionados** (tendem a cair nos mesmos áudios), o que é o oposto de complementaridade forte." if ns['ORC_OBS'] < ns['ORC_EXP'] else
                     "O observado fica acima do esperado sob independência, indicando erros negativamente correlacionados (complementaridade real).") + """
"""))
    cells.append(code(r'''# 3. Oráculo observado vs. esperado com erros independentes (média por fold) + gráfico
display(R.oracle_recall(NAMES))
fig, ax = plt.subplots(figsize=(10.5, 5.5))
labels = [LBL[k] for k in REPS] + ['Oráculo\nobservado', 'Esperado c/ erros\nindependentes']
vals = [rec_pool[k] for k in REPS] + [100 * ORACLE_HITS / N_INST, ORC_EXP]
colors = ['#2980b9', '#e67e22', '#9b59b6', '#95a5a6', '#27ae60', '#bdc3c7']
bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
bars[-2].set_hatch('//'); bars[-1].set_hatch('..')
ax.axhline(ORC_RANDOM, color='#7f8c8d', ls=':', lw=1.3, label=f'4 preditores aleatórios ({ORC_RANDOM:.1f}%)')
for b in bars:
    ax.annotate(f'{b.get_height():.1f}%', xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4),
                textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title(f'Recall em {CLASS_NAME}: modelos isolados, oráculo e esperado sob independência', fontweight='bold')
ax.set_ylabel(f'Recall em {CLASS_NAME} (%)'); ax.set_ylim(0, 105); ax.legend(loc='upper left')
plt.xticks(fontsize=8.5); plt.tight_layout(); plt.show()'''))

    cells.append(md(f"""
## 4. Grau de consenso, regras de votação e voto real

### 4.1 Pirâmide de consenso (instâncias arquivo × fold)
- **4 de 4 corretas:** {br(cons[4], 0)} ({br(100 * cons[4] / n)}%);
- **3 de 4:** {br(cons[3], 0)} ({br(100 * cons[3] / n)}%);
- **2 de 4:** {br(cons[2], 0)} ({br(100 * cons[2] / n)}%);
- **1 de 4 (resgate exclusivo):** {br(cons[1], 0)} ({br(100 * cons[1] / n)}%; {br(100 * cons[1] / oh)}% dos acertos do oráculo);
- **0 de 4 (erro comum):** {br(cons[0], 0)} ({br(100 * cons[0] / n)}%).

### 4.2 Regras de limiar vs. voto real
- Unanimidade (4/4): {br(100 * ge[4] / n)}%; ≥3/4: {br(100 * ge[3] / n)}%; ≥2/4: {br(100 * ge[2] / n)}%; oráculo ≥1/4: {br(100 * ge[1] / n)}%.
- A regra "≥2/4" (`c4 + c3 + c2`) **não é uma votação**: ela conta como acerto os empates $2\\times2$, que um voto real só acerta por sorteio. Há **{br(ns['n_tie_right'], 0)} instâncias** em que a classe correta está empatada com outra no topo da votação.
- **Voto real** (pluralidade entre as 4 entradas, empate sorteado; `R.majority_vote`), média por fold: recall `{class_name}` = **{pm(ns['VOTE_REC'])}%**, contra **{pm(ns['rec_fold'][B])}%** do melhor modelo isolado, {L[B]}. F-test 5×2cv sobre o recall `{class_name}` por fold: $F={br(ns['F_REC'])}$, $p={br(ns['P_REC'], 3)}$ — """ +
                    ("diferença **estatisticamente significativa** a 5 %." if ns['P_REC'] < 0.05 else "diferença **não significativa** (equivalentes dentro do desvio).") + f"""
- Em **SP global** (4 classes), o voto real obtém **{pm(ns['VOTE_SP'])}%**, contra **{pm(ns['sp_fold'][ns['BEST_SP']])}%** do melhor SVM isolado em SP, {L[ns['BEST_SP']]}; F-test 5×2cv: $F={br(ns['F_SP'])}$, $p={br(ns['P_SP'], 3)}$ — """ +
                    ("significativo a 5 %." if ns['P_SP'] < 0.05 else "não significativo.") + """
"""))
    cells.append(code(r'''# 4. Tabelas de consenso e de votação + gráfico
display(df_consensus)
display(df_vote)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
labels_cons = ['4 entradas\n(unânime)', '3 entradas', '2 entradas', '1 entrada\n(exclusivo)', '0 entradas\n(erro comum)']
counts = df_consensus['Instâncias'].values
pcts = df_consensus[f'% do total {CLASS_NAME}'].values
b1 = ax1.bar(labels_cons, counts, color=['#27ae60', '#2980b9', '#f39c12', '#e74c3c', '#7f8c8d'], edgecolor='black', width=0.55)
for b, p in zip(b1, pcts):
    ax1.annotate(f'{int(b.get_height())} ({p:.1f}%)', xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4),
                 textcoords='offset points', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax1.set_title(f'Grau de consenso em {CLASS_NAME} ({N_INST} instâncias arquivo×fold; {N_REC} gravações)', fontweight='bold')
ax1.set_ylabel('Instâncias arquivo×fold'); ax1.set_ylim(0, counts.max() * 1.18)

dest = list(df_reverse.index)
r1 = ax2.bar(np.arange(len(dest)) - 0.175, df_reverse['Perde acertos (efeito reverso)'], 0.35, color='#c0392b', edgecolor='black', label=f'Perde acertos do {BEST}')
r2 = ax2.bar(np.arange(len(dest)) + 0.175, df_reverse['Resgata erros'], 0.35, color='#27ae60', edgecolor='black', label=f'Resgata erros do {BEST}')
for r, s in ((r1, '-'), (r2, '+')):
    for b in r:
        ax2.annotate(f'{s}{int(b.get_height())}', xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                     textcoords='offset points', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax2.set_xticks(np.arange(len(dest))); ax2.set_xticklabels(dest, fontweight='bold')
ax2.set_title(f'Efeito reverso ao trocar cegamente o {LBL[BEST]}', fontweight='bold')
ax2.set_ylabel('Instâncias arquivo×fold'); ax2.legend(loc='upper right')
ax2.set_ylim(0, max(df_reverse['Perde acertos (efeito reverso)'].max(), df_reverse['Resgata erros'].max()) * 1.2)
plt.tight_layout(); plt.show()
display(df_reverse)'''))

    rev_txt = "\n".join(
        f"- **{idx}:** mantém {br(r['Mantém acertos'], 0)}, perde {br(r['Perde acertos (efeito reverso)'], 0)} acertos ({br(r['% perda'])}%) e resgata {br(r['Resgata erros'], 0)} erros ({br(r['% resgate'])}%); saldo **{int(r['Saldo líquido']):+d}**."
        for idx, r in rev.iterrows())
    cells.append(md(f"""
### 4.3 Efeito reverso (troca cega a partir do melhor modelo isolado, {L[B]}, {br(hits[B], 0)} acertos)
{rev_txt}

Nenhuma troca cega é vantajosa{"" if (rev['Saldo líquido'] <= 0).all() else " em todos os casos (há saldo positivo em ao menos uma troca)"}: qualquer seletor precisaria identificar, **sem o rótulo**, as instâncias em que o {B} erra — o que é exatamente o que o oráculo pressupõe.
"""))

    excl_txt = "\n".join(
        f"- **{idx}:** {br(r['Instâncias'], 0)} instâncias ({br(r[f'% do total {class_name}'])}%), {br(r['Gravações únicas'], 0)} gravações únicas; gravações que são resgate exclusivo em **todas** as suas aparições no teste: **{br(r['Gravações exclusivas em todas as aparições'], 0)}**."
        for idx, r in excl.sort_values("Instâncias", ascending=False).iterrows())
    cells.append(md(f"""
## 5. Resgates exclusivos

Em **{br(cons[1], 0)} instâncias** exatamente uma das entradas acerta:
{excl_txt}

As contagens são de instâncias arquivo × fold, não de áudios distintos. A mesma gravação pode ser resgate exclusivo em um fold e não em outro (os modelos são re-treinados a cada fold), por isso a última coluna é bem menor que a primeira. Como as predições HHT/DEMON m=4000 **não são reprodutíveis** entre execuções (ver aviso na seção 2), o ranking de "resgates exclusivos" entre HHT e DEMON não deve ser tomado como propriedade física das representações. Explicações físicas para resgates individuais (cavitação, rotação de eixo etc.) **não foram verificadas** nos áudios e são apenas hipóteses.
"""))
    cells.append(code(r'''# 5. Resgates exclusivos
display(df_exclusive)
fig, ax = plt.subplots(figsize=(9.5, 4.5))
b = ax.barh(df_exclusive.index, df_exclusive['Instâncias'], color='#2980b9', height=0.5, edgecolor='black')
for bar, (_, r) in zip(b, df_exclusive.iterrows()):
    ax.annotate(f"{int(r['Instâncias'])} inst. / {int(r['Gravações únicas'])} grav. ({int(r['Gravações exclusivas em todas as aparições'])} sempre exclusivas)",
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2), xytext=(5, 0), textcoords='offset points',
                ha='left', va='center', fontsize=9)
ax.set_title(f'Resgates exclusivos em {CLASS_NAME} ({int(df_exclusive["Instâncias"].sum())} instâncias arquivo×fold)', fontweight='bold')
ax.set_xlabel('Instâncias arquivo×fold'); ax.set_xlim(0, df_exclusive['Instâncias'].max() * 1.9)
plt.tight_layout(); plt.show()'''))

    others = [c for c in conf.columns]
    conf_txt = "\n".join(
        f"| {idx} | " + " | ".join(f"{br(r[c])}%" for c in others) + " |" for idx, r in conf.iterrows())
    pair_txt = "\n".join(
        f"| {idx} | {br(r['P(B erra)'])}% | {br(r['P(B erra | A erra)'])}% | {br(r['Ambos erram'], 0)} | {br(r['Ambos erram p/ a mesma classe'], 0)} ({br(r['% mesma classe'])}%) |"
        for idx, r in pairs.iterrows())
    cells.append(md(f"""
## 6. Para onde vão os erros e correlação entre as entradas

| Modelo | """ + " | ".join(c.replace(" (%)", "") for c in others) + " |\n| :--- | " + " | ".join(":---:" for _ in others) + " |\n" + conf_txt + f"""

**Correlação dos erros entre pares de entradas** (instâncias de `{class_name}`). Se os erros fossem independentes, $P(B\\text{{ erra}}\\mid A\\text{{ erra}}) \\approx P(B\\text{{ erra}})$:

| Par (A × B) | P(B erra) | P(B erra \\| A erra) | Ambos erram | Ambos erram para a mesma classe |
| :--- | :---: | :---: | :---: | :---: |
{pair_txt}

""" + ("Em todos os pares a probabilidade condicional é maior que a marginal: quando uma entrada erra, a outra tende a errar também, e frequentemente para a **mesma** classe. A afirmação de versões anteriores de que as entradas \"nunca concordam no erro\" é falsa." if (pairs['P(B erra | A erra)'] > pairs['P(B erra)']).all() else
       "Em parte dos pares a probabilidade condicional é maior que a marginal (erros correlacionados); ver tabela.") + """
As descrições físicas dos mecanismos de confusão (bandas, raias, IMFs, pás) são hipóteses não testadas neste notebook.
"""))
    cells.append(code(r'''# 6. Padrão de confusão e correlação dos erros
display(df_confusion)
display(df_pairs)
fig, ax = plt.subplots(figsize=(10.5, 5.5))
bottom = np.zeros(len(df_confusion))
pal = ['#3498db', '#f39c12', '#e74c3c', '#95a5a6']
for i, c in enumerate(df_confusion.columns):
    v = df_confusion[c].values
    ax.bar(df_confusion.index, v, 0.5, bottom=bottom, label=c.replace(' (%)', ''), color=pal[i], edgecolor='black', linewidth=0.5)
    for j, (bb, vv) in enumerate(zip(bottom, v)):
        if vv > 4:
            ax.text(j, bb + vv / 2, f'{vv:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    bottom += v
ax.set_title(f'Destino dos erros em {CLASS_NAME} por entrada (%)', fontweight='bold')
ax.set_ylabel('% dos erros'); ax.set_ylim(0, 118); ax.legend(loc='upper right')
plt.xticks(fontsize=8.5); plt.tight_layout(); plt.show()'''))

    cells.append(md("""
## 7. Variantes do SVM HHT/DEMON em disco

Há mais de um SVM HHT/DEMON no projeto, com o mesmo rótulo nominal em outros notebooks mas $m$ (número de componentes Nyström) e/ou `max_iter` diferentes. A tabela abaixo (SP e ACC = acurácia balanceada, média ± desvio sobre os 10 folds) mostra que eles **não são intercambiáveis**; neste notebook usa-se sempre o $m=4000$ indicado explicitamente.
"""))
    cells.append(code(INSTABILITY_CODE))
    return cells


def notebook_json(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py",
                              "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python",
                              "pygments_lexer": "ipython3", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def write(filename, cells):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notebook_json(cells), f, indent=2, ensure_ascii=False)
    print(f"Notebook gerado com sucesso em '{filename}'!")


BACKUP_DIR = "data/iara_processed/_backup_20261020"


def old_divergence():
    """% de instâncias arquivo×fold em que a predição HHT/DEMON m=4000 atual difere da usada na versão anterior
    (pickles antigos de MEDIUM/LARGE no backup). Só para documentar a não-reprodutibilidade; não entra em nenhuma métrica."""
    import pickle
    import pandas as pd
    import iara_results as R
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for cls in ("medium", "large"):
        p = os.path.join(root, BACKUP_DIR, f"{cls}_complementarity_analysis.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            old = pickle.load(f)["df"]
        for rep, exp in (("HHT", "svm_nystroem_4000_hht_C2.0_elasticnet_l1r0.15_4c"),
                         ("DEMON", "svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_4c")):
            new = pd.concat([v.assign(fold=k) for k, v in R.file_predictions(exp).items()])
            m = old.merge(new, left_on=["fold", "file_id"], right_on=["fold", "File"])
            out[(cls.upper(), rep)] = 100 * (m[f"pred_{rep.lower()}"] != m.Prediction).mean()
    return out


def divergence_txt():
    dv = old_divergence()
    if not dv:
        return ""
    return (" Comparando com as predições antigas (pickles `medium/large_complementarity_analysis.pkl` em `"
            + BACKUP_DIR + "/`, que não são mais usados), a predição por arquivo muda em "
            + "; ".join(f"{rep} {br(v, 1)}% das instâncias de `{cls}`" for (cls, rep), v in dv.items()) + ".")


def summary_cell(ns, nb_num, class_name, class_desc, question):
    n, B, L, cons, oh = ns["N_INST"], ns["BEST"], ns["LBL"], ns["cons"], ns["ORACLE_HITS"]
    conf = ns["df_confusion"]
    top = {}
    for k in ns["REPS"]:
        row = conf.loc[[i for i in conf.index if i.startswith(L[k] + " ")][0]]
        top[k] = (row.idxmax().replace("→ ", "").replace(" (%)", ""), row.max())
    pairs = ns["df_pairs"]
    ml = pairs.loc["MEL × LOFAR"]
    unstable = ""
    if ns["UNSTABLE"]:
        unstable = (" Atenção: " + " e ".join(L[k] for k in ns["UNSTABLE"])
                    + " são instáveis entre folds (SP global " + " e ".join(pm(ns["sp_fold"][k]) + "%" for k in ns["UNSTABLE"]) + ").")
    corr = ("os erros são **positivamente correlacionados**, isto é, tendem a cair nos mesmos áudios."
            if ns["ORC_OBS"] < ns["ORC_EXP"] else "os erros são menos correlacionados que o acaso.")
    return md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook {nb_num}: Complementaridade das entradas do SVM, oráculo e grau de consenso na classe {class_name}

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

## Resumo

{class_desc} {question}

A análise é feita por **instância arquivo × fold de teste** ({br(n, 0)} instâncias de `{class_name}`, {br(ns['N_REC'], 0)} gravações únicas; em média cada arquivo aparece {br(ns['EVAL_PER_FILE_ALL'])} vezes no teste ao longo das 10 dobras). Todos os números abaixo são recalculados das predições em disco por `notebooks/iara_results.py`.

### Principais resultados
1. **Recall isolado em `{class_name}`:** """ + "; ".join(f"{L[k]} {br(ns['rec_pool'][k])}%" for k in ns["ORDER"]) + f".{unstable}" + f"""
2. **Oráculo (não realizável):** ao menos uma entrada acerta {br(oh, 0)} instâncias ({br(100 * oh / n)}%); todas erram em {br(cons[0], 0)} ({br(100 * cons[0] / n)}%). Média por fold: **observado {br(ns['ORC_OBS'])}% vs. {br(ns['ORC_EXP'])}% esperado se os erros fossem independentes** — {corr} Para comparação, 4 preditores aleatórios dariam {br(ns['ORC_RANDOM'])}%.
3. **Consenso:** {br(cons[4], 0)} instâncias ({br(100 * cons[4] / n)}%) são acertadas pelas 4 entradas; {br(cons[1], 0)} ({br(100 * cons[1] / oh)}% dos acertos do oráculo) dependem de uma única entrada.
4. **Voto real** entre as 4 entradas: recall `{class_name}` {pm(ns['VOTE_REC'])}% vs. {pm(ns['rec_fold'][B])}% do {L[B]} (F-test 5×2cv $p={br(ns['P_REC'], 3)}$, """ + ("significativo" if ns["P_REC"] < 0.05 else "não significativo") + f"""); SP global {pm(ns['VOTE_SP'])}% vs. {pm(ns['sp_fold'][ns['BEST_SP']])}% do {L[ns['BEST_SP']]} ($p={br(ns['P_SP'], 3)}$, """ + ("significativo" if ns["P_SP"] < 0.05 else "não significativo") + f"""). A regra "≥2/4" ({br(100 * ns['ge'][2] / n)}%) conta empates como acerto e não é um voto realizável.
5. **Efeito reverso:** trocar cegamente o {L[B]} por outra entrada — """ + "; ".join(f"{idx.split('→ ')[1]}: perde {br(r['Perde acertos (efeito reverso)'], 0)} e resgata {br(r['Resgata erros'], 0)} (saldo {int(r['Saldo líquido']):+d})" for idx, r in ns['df_reverse'].iterrows()) + f""".
6. **Destino dos erros:** """ + "; ".join(f"{k} → `{top[k][0]}` ({br(top[k][1])}%)" for k in ns["REPS"]) + f""". Os erros **coincidem com frequência**: MEL e LOFAR erram ambos em {br(ml['Ambos erram'], 0)} instâncias, {br(ml['Ambos erram p/ a mesma classe'], 0)} delas para a mesma classe; $P(\\text{{LOFAR erra}}\\mid\\text{{MEL erra}}) = {br(ml['P(B erra | A erra)'])}\\%$ contra $P(\\text{{LOFAR erra}}) = {br(ml['P(B erra)'])}\\%$.

> Os números de HHT/DEMON desta versão diferem da versão anterior porque o SVM HHT/DEMON $m=4000$ não é reprodutível entre execuções (ver seção 2).
""")


def conclusion_cell(ns, class_name, extra=""):
    B, L, cons, oh, n = ns["BEST"], ns["LBL"], ns["cons"], ns["ORACLE_HITS"], ns["N_INST"]
    if ns["P_REC"] >= 0.05:
        vote = "não difere significativamente d"
    elif ns["VOTE_REC"][0] > ns["rec_fold"][B][0]:
        vote = "supera significativamente "
    else:
        vote = "fica significativamente abaixo d"
    return md(f"""
## 8. Implicações para um seletor de entrada

- O teto do oráculo **não** é uma meta atingível: ele pressupõe saber, sem o rótulo, quando cada entrada está certa. A referência honesta é o voto real e o melhor modelo isolado ({L[B]}, recall `{class_name}` {pm(ns['rec_fold'][B])}%).
- Como os erros são correlacionados e a troca cega """ + ("sempre perde mais do que ganha" if (ns['df_reverse']['Saldo líquido'] < 0).all() else "não traz ganho líquido") + """, qualquer roteamento precisa de um sinal de confiança (margem do SVM, por exemplo) ajustado **fora** do conjunto de teste (folds internos), e o ganho só pode ser afirmado com o F-test 5×2cv.
- Ideias como *soft gating*/MoE ou *stacking* sobre os scores das 4 entradas são hipóteses de trabalho; este notebook não as avalia.
""" + extra + f"""
## Conclusão

1. Uma parte grande dos erros em `{class_name}` é **comum** às entradas: o oráculo observado ({br(ns['ORC_OBS'])}%) fica """ + ("abaixo" if ns['ORC_OBS'] < ns['ORC_EXP'] else "acima") + f""" do esperado sob independência ({br(ns['ORC_EXP'])}%). Não se pode concluir que "a informação está presente no áudio, apenas dividida entre representações".
2. O consenso pleno cobre {br(100 * cons[4] / n)}% das instâncias e {br(100 * cons[1] / oh)}% dos acertos do oráculo dependem de uma única entrada — mas esses resgates exclusivos são instáveis entre folds e, para HHT/DEMON, entre execuções do treino.
3. O voto real entre as 4 entradas {vote}o melhor modelo isolado em recall `{class_name}` ({pm(ns['VOTE_REC'])}% vs. {pm(ns['rec_fold'][B])}%, p = {br(ns['P_REC'], 3)}) e, em SP global, obtém {pm(ns['VOTE_SP'])}% vs. {pm(ns['sp_fold'][ns['BEST_SP']])}% (p = {br(ns['P_SP'], 3)}).
""")


def cross_class_md(nss, nb_nums):
    """Tabela de síntese entre classes (todas calculadas agora)."""
    cols = list(nss)
    hdr = "| Métrica (instâncias arquivo × fold) | " + " | ".join(f"`{c}` (NB {nb_nums[c]})" for c in cols) + " |\n"
    hdr += "| :--- | " + " | ".join(":---:" for _ in cols) + " |\n"

    def row(name, f):
        return f"| {name} | " + " | ".join(f(nss[c], c) for c in cols) + " |\n"

    t = hdr
    t += row("Instâncias / gravações únicas", lambda s, c: f"{br(s['N_INST'], 0)} / {br(s['N_REC'], 0)}")
    t += row("Melhor modelo isolado (recall)", lambda s, c: f"{s['BEST']} {br(s['rec_pool'][s['BEST']])}%")
    t += row("Oráculo observado (média por fold)", lambda s, c: f"{br(s['ORC_OBS'])}%")
    t += row("Esperado c/ erros independentes", lambda s, c: f"{br(s['ORC_EXP'])}%")
    t += row("Erro comum (0/4)", lambda s, c: f"{br(100 * s['cons'][0] / s['N_INST'])}% ({br(s['cons'][0], 0)})")
    t += row("Unanimidade (4/4)", lambda s, c: f"{br(100 * s['cons'][4] / s['N_INST'])}% ({br(s['cons'][4], 0)})")
    t += row("Resgates exclusivos (1/4)", lambda s, c: f"{br(100 * s['cons'][1] / s['N_INST'])}% ({br(s['cons'][1], 0)})")
    t += row("Maior nº de resgates exclusivos", lambda s, c: f"{s['EXCL_CHAMP'].split(' (')[0].replace('SVM ', '')} ({br(s['df_exclusive'].loc[s['EXCL_CHAMP'], 'Instâncias'], 0)})")
    t += row("Regra ≥2/4 (conta empates)", lambda s, c: f"{br(100 * s['ge'][2] / s['N_INST'])}%")
    t += row("Voto real (média ± dp por fold)", lambda s, c: f"{pm(s['VOTE_REC'])}%")
    t += row("Melhor isolado (média ± dp por fold)", lambda s, c: f"{pm(s['rec_fold'][s['BEST']])}%")
    t += row("F-test 5×2cv voto real vs. melhor isolado", lambda s, c: f"p = {br(s['P_REC'], 3)}")
    return t


def create_notebook(filename="notebooks/13_analise_complementaridade_small.ipynb"):
    ns = compute(0, "SMALL")
    cells = [summary_cell(ns, 10, "SMALL",
                          "A classe **`SMALL`** (embarcações de pequeno porte, comprimento $< 50\\text{ m}$) é a de menor recall para os SVMs deste estudo (dataset IARA, *Silva et al.*, IEEE Access 2025).",
                          "Este notebook pergunta se os áudios de `SMALL` que o SVM erra são os mesmos para as 4 entradas (MEL, LOFAR, HHT, DEMON) ou se os erros mudam com a representação, e qual seria o risco de um seletor de entrada (efeito reverso)."),
             code(SETUP_CODE)]
    cells += common_cells(ns, 0, "SMALL", 10)
    cells.append(conclusion_cell(ns, "SMALL"))
    write(filename, cells)


if __name__ == "__main__":
    create_notebook()
