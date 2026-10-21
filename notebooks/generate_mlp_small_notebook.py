"""
Script de geração do notebook 14: notebooks/14_analise_complementaridade_mlp_small.ipynb
Análise de complementaridade (oráculo, consenso e resgates exclusivos) do MLP [32, 16] para a classe SMALL
(Dataset IARA), com MEL, LOFAR, HHT e DEMON.

Todos os números citados no markdown são calculados AQUI, na geração, a partir das predições em disco
(notebooks/iara_results.py); as células de código do notebook refazem as mesmas contas.
Uso (cwd = raiz do repositório):  python notebooks/generate_mlp_small_notebook.py
"""

import os
import sys
import json

CLASS = 0
FILENAME = "notebooks/14_analise_complementaridade_mlp_small.ipynb"

# Código de análise compartilhado entre a geração e o notebook (executado nos dois lugares).
HELPER = r'''# Funções de análise de complementaridade, calculadas a partir das predições em disco (iara_results).
import os
import sys
import numpy as np
import pandas as pd
for _p in ('.', 'notebooks'):
    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))
import iara_results as R

CLASS_NAMES = {0: 'SMALL', 1: 'MEDIUM', 2: 'LARGE', 3: 'BACKGROUND'}
REPS = ['MEL', 'LOFAR', 'HHT', 'DEMON']
# MLP [32, 16]: MEL/LOFAR = pipeline iara oficial (com validação e early stopping);
#               HHT/DEMON = scratch/generate_mlp_predictions_and_analysis.py (30 épocas, seed 42, sem validação)
MLP_EXPS = {'MEL': 'mlp', 'LOFAR': 'mlp_lofar',
            'HHT': 'mlp_hht_h32-16_e30_4c', 'DEMON': 'mlp_demon_h32-16_e30_4c'}
# SVM Nyström declarado nos NBs 10/12/13 (m=4000 para MEL/HHT/DEMON, m=6000 para LOFAR)
SVM_EXPS = {'MEL': 'svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0',
            'LOFAR': 'svm_nystroem_6000_lofar_elasticnet_l1r0.15',
            'HHT': 'svm_nystroem_4000_hht_C2.0_elasticnet_l1r0.15_4c',
            'DEMON': 'svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_4c'}
VOTE_SEEDS = list(range(10))  # sementes do desempate aleatório do voto de pluralidade


def ship_id_map():
    meta = pd.read_csv(os.path.join(R.ROOT, 'src', 'iara', 'dataset_info', 'iara.csv'), dtype={'ID': str})
    return dict(zip(meta['ID'].astype(int), meta['Ship ID']))


def stacked(exps):
    """Instâncias (arquivo, fold) presentes nas 4 representações, empilhadas sobre os 10 folds."""
    A = R.aligned([exps[r] for r in REPS])
    return pd.concat([d.assign(fold=k) for k, d in A.items()], ignore_index=True)


def complementarity(exps, c):
    cls = CLASS_NAMES[c]
    names = [exps[r] for r in REPS]
    d = stacked(exps)
    d = d[d.Target == c].reset_index(drop=True)
    ships = ship_id_map()
    d['Ship'] = d.File.map(ships)
    H = np.column_stack([d[n].values == c for n in names])
    nh = H.sum(axis=1)
    N = len(d)
    per_file = d.groupby('File').size()
    S = {'cls': cls, 'N': N, 'n_files': int(d.File.nunique()), 'n_ships': int(d.Ship.nunique()),
         'evals_per_file': N / d.File.nunique(), 'evals_min': int(per_file.min()), 'evals_max': int(per_file.max())}
    S['hits'] = {r: int(H[:, i].sum()) for i, r in enumerate(REPS)}
    S['recall'] = {r: 100 * H[:, i].mean() for i, r in enumerate(REPS)}        # agregado sobre as instâncias
    S['recall_fold'] = {r: R.summary(exps[r])[f'R_{cls}'] for r in REPS}       # média ± desvio por fold
    S['best'] = max(REPS, key=S['recall'].get)
    S['oracle_hits'] = int((nh > 0).sum())
    S['oracle'] = 100 * (nh > 0).mean()
    orc = R.oracle_recall(names).loc[cls]
    S['oracle_fold'], S['oracle_indep'] = float(orc.iloc[0]), float(orc.iloc[1])
    S['cons'] = {j: int((nh == j).sum()) for j in (4, 3, 2, 1, 0)}
    S['ge3'] = int((nh >= 3).sum())
    S['ge2'] = int((nh >= 2).sum())
    votes = [R.majority_vote(names, seed=s)[f'R_{cls}'].mean() for s in VOTE_SEEDS]
    S['vote'], S['vote_min'], S['vote_max'] = votes[0], min(votes), max(votes)
    mv0 = R.majority_vote(names, seed=VOTE_SEEDS[0])[f'R_{cls}'].values
    best_fold = R.fold_metrics(exps[S['best']])[f'R_{cls}'].values
    S['vote_fold'] = (mv0.mean(), mv0.std(ddof=0))
    S['vote_F'], S['vote_p'] = R.f_test_5x2(mv0, best_fold)   # F-test 5x2cv sobre o recall da classe
    ex = {}
    for i, r in enumerate(REPS):
        m = H[:, i] & (nh == 1)
        cnt = d.File[m].value_counts()
        always = [f for f, k in cnt.items() if k == per_file[f]]
        ex[r] = {'inst': int(m.sum()), 'files': int(d.File[m].nunique()), 'ships': int(d.Ship[m].nunique()),
                 'files_always': len(always), 'ships_always': len({ships[f] for f in always})}
    S['excl'] = ex
    S['excl_top'] = max(REPS, key=lambda r: ex[r]['inst'])
    conf = {}
    for i, r in enumerate(REPS):
        p = d[exps[r]][~H[:, i]]
        vc = p.value_counts(normalize=True) * 100
        conf[r] = {'n_err': int((~H[:, i]).sum()),
                   **{CLASS_NAMES[k]: float(vc.get(k, 0.0)) for k in range(4) if k != c}}
    S['conf'] = conf
    iM, iL = REPS.index('MEL'), REPS.index('LOFAR')
    both = (~H[:, iM]) & (~H[:, iL])
    S['ml_both_err'] = int(both.sum())
    S['ml_same_err'] = int((d[exps['MEL']].values[both] == d[exps['LOFAR']].values[both]).sum())
    # troca a partir do melhor isolado: o que se perde e o que se resgata ao usar outra entrada
    ib = REPS.index(S['best'])
    S['switch'] = {r: {'keep': int((H[:, ib] & H[:, i]).sum()), 'lose': int((H[:, ib] & ~H[:, i]).sum()),
                       'gain': int((~H[:, ib] & H[:, i]).sum())} for i, r in enumerate(REPS) if i != ib}
    return S


def vote_global(exps):
    """SP global (4 classes) do voto real de pluralidade vs. melhor modelo isolado em SP, com F-test 5x2cv."""
    names = [exps[r] for r in REPS]
    mv = R.majority_vote(names, seed=VOTE_SEEDS[0])
    sp = {r: R.summary(exps[r])['SP'] for r in REPS}
    best = max(REPS, key=lambda r: sp[r][0])
    F, p = R.f_test_5x2(mv['SP'].values, exps[best])
    return {'vote_sp': (mv['SP'].mean(), mv['SP'].std(ddof=0)), 'best': best, 'best_sp': sp[best],
            'F': F, 'p': p, 'sp': sp}


def reprodutibilidade():
    """Compara as predições atuais de HHT/DEMON com as da versão anterior (pickle de backup de 15/10/2026)."""
    import pickle
    from collections import Counter
    pth = os.path.join(R.ROOT, 'data', 'iara_processed', '_backup_20261020', 'mlp_all_representations_predictions.pkl')
    if not os.path.exists(pth):
        return None
    with open(pth, 'rb') as f:
        old = pickle.load(f)
    rows = []
    for rep, key in [('HHT', 'hht'), ('DEMON', 'demon')]:
        exp = MLP_EXPS[rep]
        diff = tot = 0
        sp_old, sp_low = [], []
        for k in range(10):
            o, tg = old[key][k], old['targets'][k]
            files = list(o)
            sp_old.append(R._metrics(np.array([tg[f] for f in files]), np.array([o[f] for f in files]))['SP'])
            w = pd.read_csv(os.path.join(R.exp_dir(exp), 'eval', f'fold_{k}', f'{exp}_multiclass_test.csv'),
                            usecols=['File', 'Target', 'Prediction'])
            g = w.groupby('File')
            y = g.Target.agg(lambda s: Counter(s).most_common(1)[0][0])
            p = g.Prediction.agg(lambda s: int(np.argmax(np.bincount(s, minlength=4))))  # mesmo desempate do pickle
            sp_low.append(R._metrics(y.values, p.values)['SP'])
            tot += len(files)
            diff += sum(int(o[f]) != int(p.get(f, -1)) for f in files)
        rows.append({'Representação': rep, 'SP versão anterior (pickle 15/10)': round(float(np.mean(sp_old)), 2),
                     'SP atual (empate -> 1ª janela, pipeline iara)': round(R.summary(exp)['SP'][0], 2),
                     'SP atual (empate -> menor índice, como o script)': round(float(np.mean(sp_low)), 2),
                     'Instâncias c/ predição diferente (mesmo desempate)': diff, 'Instâncias': tot})
    return pd.DataFrame(rows).set_index('Representação')


def tabela_acertos(S):
    rows = [{'Representação': r, 'Acertos': S['hits'][r], 'Recall agregado (%)': round(S['recall'][r], 2),
             'Recall por fold (%)': R.fmt(S['recall_fold'][r])} for r in REPS]
    rows.append({'Representação': 'ORÁCULO (≥1 de 4 acerta; não realizável)', 'Acertos': S['oracle_hits'],
                 'Recall agregado (%)': round(S['oracle'], 2), 'Recall por fold (%)': f"{S['oracle_fold']:.2f}"})
    rows.append({'Representação': 'Oráculo esperado c/ erros independentes', 'Acertos': np.nan,
                 'Recall agregado (%)': np.nan, 'Recall por fold (%)': f"{S['oracle_indep']:.2f}"})
    rows.append({'Representação': 'Voto real de pluralidade (empate sorteado)', 'Acertos': np.nan,
                 'Recall agregado (%)': np.nan,
                 'Recall por fold (%)': f"{S['vote']:.2f} (10 sementes: {S['vote_min']:.2f}–{S['vote_max']:.2f})"})
    return pd.DataFrame(rows).set_index('Representação')


def tabela_consenso(S):
    lab = {4: '4 de 4 (unanimidade)', 3: '3 de 4', 2: '2 de 4', 1: '1 de 4 (resgate exclusivo)',
           0: '0 de 4 (erro comum)'}
    return pd.DataFrame([{'Nível de consenso': lab[j], 'Instâncias': n, '% do total': round(100 * n / S['N'], 2),
                          '% dos acertos do oráculo': round(100 * n / S['oracle_hits'], 2) if j > 0 else np.nan}
                         for j, n in S['cons'].items()]).set_index('Nível de consenso')


def tabela_exclusivos(S):
    return pd.DataFrame([{'Representação': r, 'Instâncias (arquivo×fold)': e['inst'],
                          '% das instâncias': round(100 * e['inst'] / S['N'], 2),
                          'Gravações únicas': e['files'], 'Navios únicos (Ship ID)': e['ships'],
                          'Gravações exclusivas em todas as aparições': e['files_always'],
                          'Navios dessas gravações': e['ships_always']}
                         for r, e in S['excl'].items()]).set_index('Representação')


def tabela_confusao(S):
    rows = []
    for r in REPS:
        c = S['conf'][r]
        rows.append({'Representação': r, 'Erros': c['n_err'],
                     **{f'→ {k} (%)': round(v, 2) for k, v in c.items() if k != 'n_err'}})
    return pd.DataFrame(rows).set_index('Representação')


def tabela_troca(S):
    return pd.DataFrame([{'Trocar para': r, f"Mantém acertos do {S['best']}": v['keep'],
                          f"Perde acertos do {S['best']}": v['lose'], f"Resgata erros do {S['best']}": v['gain'],
                          'Saldo líquido': v['gain'] - v['lose']} for r, v in S['switch'].items()]).set_index('Trocar para')


def tabela_svm_vs_mlp(Ssvm, Smlp):
    def col(S):
        b = S['best']
        return {'Melhor isolado (recall agregado)': f"{S['recall'][b]:.2f}% ({b})",
                'Oráculo observado (média por fold)': f"{S['oracle_fold']:.2f}%",
                'Oráculo esperado c/ erros independentes': f"{S['oracle_indep']:.2f}%",
                'Unanimidade 4/4': f"{100 * S['cons'][4] / S['N']:.2f}% ({S['cons'][4]})",
                'Resgate exclusivo 1/4': f"{100 * S['cons'][1] / S['N']:.2f}% ({S['cons'][1]})",
                'Erro comum 0/4': f"{100 * S['cons'][0] / S['N']:.2f}% ({S['cons'][0]})",
                'Maior resgatador exclusivo (instâncias)': f"{S['excl_top']} ({S['excl'][S['excl_top']]['inst']})",
                '"≥2/4" (conta empates 2×2 como acerto)': f"{100 * S['ge2'] / S['N']:.2f}%",
                'Voto real de pluralidade (média por fold)': f"{S['vote']:.2f}%"}
    return pd.DataFrame({'SVM Nyström': col(Ssvm), 'MLP [32, 16]': col(Smlp)})'''

META = {
    0: {"nb": 14, "svm_nb": 10, "cls": "SMALL", "porte": "pequeno porte, comprimento $< 50\\text{ m}$"},
    1: {"nb": 15, "svm_nb": 12, "cls": "MEDIUM", "porte": "médio porte, comprimento de $50$ a $100\\text{ m}$"},
    2: {"nb": 16, "svm_nb": 13, "cls": "LARGE", "porte": "grande porte, comprimento $> 100\\text{ m}$"},
}


# ----------------------------------------------------------------------------------------------
# Formatação (pt-BR, LaTeX)
def P(x, nd=2):
    """Percentual em LaTeX com vírgula decimal."""
    return "$" + f"{x:.{nd}f}".replace(".", "{,}") + "\\%$"


def D(x, nd=2):
    return "$" + f"{x:.{nd}f}".replace(".", "{,}") + "$"


def I(n):
    return f"{int(n):,}".replace(",", ".")


def PV(p):
    return "$p = " + f"{p:.3f}".replace(".", "{,}") + "$"


def MS(ms, nd=2):
    return "$" + f"{ms[0]:.{nd}f}".replace(".", "{,}") + " \\pm " + f"{ms[1]:.{nd}f}".replace(".", "{,}") + "$"


def md(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def comparar(a, b, p, rotulo_a, rotulo_b):
    """Frase de comparação baseada no F-test 5x2cv."""
    if p < 0.05:
        lado = "acima" if a > b else "abaixo"
        return f"a diferença entre {rotulo_a} e {rotulo_b} é significativa ({lado}; F-test 5x2cv, {PV(p)})"
    return (f"{rotulo_a} e {rotulo_b} são estatisticamente equivalentes "
            f"(F-test 5x2cv, {PV(p)}; diferença dentro da variação entre folds)")


def create_notebook(filename=FILENAME):
    sys.path.insert(0, "notebooks")
    ns = {}
    exec(HELPER, ns)
    R = ns["R"]
    m = META[CLASS]
    cls, nb, svm_nb = m["cls"], m["nb"], m["svm_nb"]
    REPS = ns["REPS"]

    S = ns["complementarity"](ns["MLP_EXPS"], CLASS)
    T = ns["complementarity"](ns["SVM_EXPS"], CLASS)
    VG = ns["vote_global"](ns["MLP_EXPS"])
    REP = ns["reprodutibilidade"]()
    sp_mlp = {r: R.summary(ns["MLP_EXPS"][r])["SP"] for r in REPS}
    sp_svm = {r: R.summary(ns["SVM_EXPS"][r])["SP"] for r in REPS}
    art = {r: R.ARTICLE_TABLE7[("MLP", r)]["SP"] for r in ("MEL", "LOFAR")}
    allS = {c: ns["complementarity"](ns["MLP_EXPS"], c) for c in (0, 1, 2)} if CLASS == 2 else None
    allT = {c: ns["complementarity"](ns["SVM_EXPS"], c) for c in (0, 1, 2)} if CLASS == 2 else None

    N, best = S["N"], S["best"]
    rb = S["recall"][best]
    pct = lambda n, tot=N: 100 * n / tot
    others = sorted([r for r in REPS if r != best], key=lambda r: -S["recall"][r])
    excl_sorted = sorted(REPS, key=lambda r: -S["excl"][r]["inst"])
    top = S["excl_top"]
    et = S["excl"][top]
    main_err = {r: max((k for k in S["conf"][r] if k != "n_err"), key=lambda k: S["conf"][r][k]) for r in REPS}
    same_frac = pct(S["ml_same_err"], S["ml_both_err"])
    sp_m = lambda r: MS(sp_mlp[r])
    unstable_svm = [r for r in REPS if sp_svm[r][1] > 5]
    svm_fm = {r: R.fold_metrics(ns["SVM_EXPS"][r])["SP"] for r in REPS}
    collapsed = {r: int((svm_fm[r] == 0).sum()) for r in REPS}

    hhtd = REP.loc["HHT"] if REP is not None else None
    demd = REP.loc["DEMON"] if REP is not None else None
    if REP is not None:
        repro_txt = (
            f"- **Não determinismo (GPU):** HHT e DEMON foram re-treinados em 20/10/2026 e **não reproduzem exatamente** "
            f"a versão anterior (pickle de 15/10): SP do HHT {D(hhtd.iloc[0])} $\\rightarrow$ {D(hhtd.iloc[1])} "
            f"({I(hhtd.iloc[3])} das {I(hhtd.iloc[4])} instâncias arquivo×fold com predição diferente) e do DEMON "
            f"{D(demd.iloc[0])} $\\rightarrow$ {D(demd.iloc[2])} com o desempate do script ({I(demd.iloc[3])} instâncias "
            f"diferentes). Mesmo com `seed=42`, o treino na GPU não é determinístico; todas as interseções, resgates e "
            f"oráculos abaixo foram recalculados com as predições atuais.\n")
    else:
        repro_txt = ("- **Não determinismo (GPU):** HHT e DEMON foram re-treinados em 20/10/2026 e não reproduzem "
                     "exatamente a versão anterior; o pickle de backup não foi encontrado para quantificar a diferença.\n")

    # ------------------------------------------------------------------------------------------
    cells = []
    vote_txt = comparar(S["vote_fold"][0], S["recall_fold"][best][0], S["vote_p"],
                        "o voto real", f"o {best} isolado")
    cells.append(md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook {nb}: Análise de Complementaridade, Efeito Oráculo e Grau de Consenso no MLP — Classe {cls}

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

## Resumo Executivo da Investigação com Redes Neurais (MLP)

A classe **`{cls}`** (embarcações de {m['porte']}) é analisada arquivo por arquivo nos 10 folds do protocolo $5\\times2$cv, com **particionamento por gravação de áudio (`ID`)**. Atenção: o split é por gravação, **não por navio** — gravações diferentes do mesmo navio (`Ship ID`) podem estar no treino e no teste.

A base são **{I(N)} instâncias de teste (arquivo × fold)** da classe `{cls}`, que correspondem a **{I(S['n_files'])} gravações únicas** de **{I(S['n_ships'])} navios distintos**. Cada gravação aparece no teste **{D(S['evals_per_file'])} vezes em média** (de {S['evals_min']} a {S['evals_max']}), e não 5 vezes.

As 4 representações usam a mesma arquitetura **MLP `[32, 16]`**, mas com protocolos de treino diferentes:
1. **MEL (256-D)** e **LOFAR (1024-D):** predições oficiais do pipeline `iara` (`results/trainings/tests/mlp` e `mlp_lofar`), com conjunto de validação e *early stopping*;
2. **HHT (256-D)** e **DEMON (253-D):** MLP PyTorch `[32, 16]` de `scratch/generate_mlp_predictions_and_analysis.py`, **30 épocas fixas**, `seed=42`, **sem validação nem early stopping**.

### Principais resultados (recalculados):
1. **Melhor modelo isolado:** {best} com recall de {P(rb)} ({I(S['hits'][best])} acertos); depois {', '.join(f'{r} ({P(S["recall"][r])})' for r in others)}.
2. **Unanimidade (4/4 acertam):** {P(pct(S['cons'][4]))} ({I(S['cons'][4])} instâncias); **erro comum (0/4):** {P(pct(S['cons'][0]))} ({I(S['cons'][0])}).
3. **Oráculo (≥1 de 4 acerta):** {P(S['oracle'])} ({I(S['oracle_hits'])} instâncias). O oráculo **não é um sistema realizável** (exige saber a resposta certa) e fica **abaixo** do esperado se os erros fossem independentes ({P(S['oracle_indep'])}): os erros das 4 entradas são **positivamente correlacionados**, e não ortogonais.
4. **Voto real de pluralidade** (empate sorteado) atinge recall de {P(S['vote'])} em `{cls}`; {vote_txt}. O antigo critério "≥2/4" ({P(pct(S['ge2']))}) contava empates $2\\times2$ como acerto e não corresponde a nenhum classificador.
5. **Resgates exclusivos:** o {top} tem o maior número de instâncias em que só ele acerta ({I(et['inst'])}), mas isso corresponde a {I(et['files'])} gravações ({I(et['ships'])} navios), e só **{I(et['files_always'])} gravações** são resgatadas exclusivamente pelo {top} em **todas** as suas aparições no teste.
6. **MEL e LOFAR erram juntos e para o mesmo lado:** quando ambos erram uma instância de `{cls}` ({I(S['ml_both_err'])} casos), preveem a **mesma** classe errada em {P(same_frac)} deles ({I(S['ml_same_err'])}).
"""))

    cells.append(code("""
# 1. Configuração do ambiente, imports e funções de análise (calculadas a partir das predições em disco)
""" + HELPER.strip("\n") + """

import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 15, 'figure.dpi': 150})
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
""" + f"""
CLASSE = {CLASS}  # {cls}
print('Experimentos MLP:', MLP_EXPS)
"""))

    art_txt = (f"Para referência, a Tabela 7 do artigo (Silva et al., IEEE Access 2025) reporta SP {MS(art['MEL'])} "
               f"(MLP MEL) e {MS(art['LOFAR'])} (MLP LOFAR); os resultados locais do pipeline oficial são "
               f"{sp_m('MEL')} e {sp_m('LOFAR')}.")
    cells.append(md(f"""
## 2. Metodologia Experimental e Base de Dados de Teste

- **Instâncias:** cada linha é um par (gravação, fold) da classe `{cls}` presente no teste das 4 representações: {I(N)} instâncias = {I(S['n_files'])} gravações únicas = {I(S['n_ships'])} navios (`Ship ID` de `src/iara/dataset_info/iara.csv`). As contagens de "acertos" e "resgates" abaixo são **instâncias**, não gravações nem navios distintos.
- **Predição por arquivo:** voto majoritário das janelas com `Counter.most_common` (empate $\\rightarrow$ classe da 1ª janela), como em `src/iara/ml/models/trainer.py`. O script de HHT/DEMON desempatava pelo menor índice de classe (`argmax(bincount)`), por isso o SP do DEMON pode diferir ligeiramente do log do script. O experimento HHT (`mlp_hht_h32-16_e30_4c`) está disponível só em granularidade de arquivo (restaurado do pickle; ver `config.json`), o que é suficiente para todas as análises por arquivo deste notebook.
- **Protocolos diferentes:** MEL/LOFAR usam validação e early stopping (pipeline `iara`); HHT/DEMON usam 30 épocas fixas sem validação. Parte da diferença entre representações pode vir do protocolo de treino, e não só da entrada.
{repro_txt}- **SP global (4 classes):** MEL {sp_m('MEL')}, LOFAR {sp_m('LOFAR')}, HHT {sp_m('HHT')}, DEMON {sp_m('DEMON')}. {art_txt}
"""))

    cells.append(code("""
# 2. Métricas globais dos 4 MLPs (SP, ACC = acurácia balanceada; média ± desvio nos 10 folds) e reprodutibilidade
display(R.table([(f'MLP {r}', MLP_EXPS[r]) for r in REPS], metrics=('SP', 'ACC', 'F1')))
print('Tabela 7 do artigo (SP):', {r: R.ARTICLE_TABLE7[('MLP', r)]['SP'] for r in ('MEL', 'LOFAR')})
display(reprodutibilidade())

S = complementarity(MLP_EXPS, CLASSE)
print(f"{S['cls']}: {S['N']} instâncias (arquivo×fold) = {S['n_files']} gravações únicas = {S['n_ships']} navios; "
      f"{S['evals_per_file']:.2f} aparições por gravação em média (mín {S['evals_min']}, máx {S['evals_max']})")
display(tabela_acertos(S))
display(tabela_consenso(S))
"""))

    cells.append(md(f"""
## 3. O Efeito Oráculo no MLP: Recall Individual vs. Oráculo

Isoladamente, o **{best}** lidera em `{cls}` com recall de {P(rb)} ({I(S['hits'][best])} acertos), seguido por {', '.join(f'{r} ({P(S["recall"][r])})' for r in others)}.

A união das 4 representações (**oráculo**: conta acerto se qualquer uma acerta) chega a {P(S['oracle'])} ({I(S['oracle_hits'])} instâncias), {D(S['oracle'] - rb, 1)} p.p. acima do melhor isolado. Esse número deve ser lido com cautela:
- o oráculo **não é realizável** — escolher a entrada certa exige conhecer o rótulo;
- mesmo preditores sem relação entre si elevam o oráculo: com os recalls individuais acima e **erros independentes**, o esperado seria {P(S['oracle_indep'])} (média por fold, $1-\\prod_i(1-r_i)$), contra {P(S['oracle_fold'])} observados. O oráculo observado fica **abaixo** do esperado, ou seja, as entradas tendem a errar **nos mesmos áudios** — a evidência é de erros correlacionados, não de informação ortogonal.
"""))

    cells.append(code(f"""
# 3. Gráfico: recall isolado vs. oráculo observado, oráculo esperado (erros independentes) e voto real
labels = [f'{{r}}' for r in REPS] + ['Voto real', 'Oráculo\\n(não realizável)']
vals = [S['recall'][r] for r in REPS] + [S['vote'], S['oracle']]
colors = ['#2980b9', '#e67e22', '#9b59b6', '#95a5a6', '#16a085', '#27ae60']
plt.figure(figsize=(10, 5.5))
bars = plt.bar(labels, vals, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
bars[-1].set_hatch('//')
plt.axhline(S['oracle_indep'], color='#c0392b', linestyle='--', lw=1.5, alpha=0.8,
            label=f"Oráculo esperado c/ erros independentes ({{S['oracle_indep']:.1f}}%)")
plt.axhline(S['recall'][S['best']], color='#e67e22', linestyle=':', lw=1.5, alpha=0.8,
            label=f"Melhor isolado {{S['best']}} ({{S['recall'][S['best']]:.1f}}%)")
plt.title(f"Recall MLP na classe {{S['cls']}}: isolados, voto real e oráculo", fontweight='bold', pad=12)
plt.ylabel(f"Recall em {{S['cls']}} (%)")
plt.ylim(0, 105)
for bar in bars:
    h = bar.get_height()
    plt.annotate(f'{{h:.1f}}%', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords='offset points',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.show()
"""))

    lab = {4: "Unanimidade (4/4 certas)", 3: "3/4 certas", 2: "Consenso dividido (2/4 certas)",
           1: "Resgate exclusivo (1/4 certa)", 0: "Erro comum (0/4 certas)"}
    cons_lines = "\n".join(f"- **{lab[j]}:** {I(n)} instâncias ({P(pct(n))})" for j, n in S["cons"].items())
    sec4 = f"""
## 4. Análise de Grau de Consenso em `{cls}` no MLP

{cons_lines}

{P(pct(S['cons'][1], S['oracle_hits']))} dos acertos do oráculo dependem de uma única entrada.

### Votação por consenso no MLP:
- **Unanimidade (4/4):** {P(pct(S['cons'][4]))};
- **Maioria estrita ($\\ge 3/4$):** {P(pct(S['ge3']))} ({I(S['ge3'])} instâncias);
- **"$\\ge 2/4$" (`c4 + c3 + c2`):** {P(pct(S['ge2']))} — **não é um classificador**: conta como acerto os empates $2\\times2$, nos quais um voto real escolheria a classe errada metade das vezes, e ignora como os votos errados se distribuem;
- **Voto real de pluralidade** (empate sorteado, `R.majority_vote`): recall de {MS(S['vote_fold'])} em `{cls}` (variação entre 10 sementes de desempate: {D(S['vote_min'])}–{D(S['vote_max'])}), contra {MS(S['recall_fold'][best])} do {best} isolado — {vote_txt};
- **Oráculo ($\\ge 1/4$):** {P(S['oracle'])} (não realizável).

No SP global de 4 classes, o voto real das 4 MLPs obtém {MS(VG['vote_sp'])}, contra {MS(VG['best_sp'])} do melhor isolado ({VG['best']}): {comparar(VG['vote_sp'][0], VG['best_sp'][0], VG['p'], 'o voto', 'o melhor isolado')}.
"""
    if CLASS == 2:
        sw_lines = "\n".join(f"- **Trocar para {r}:** perde {I(v['lose'])} e resgata {I(v['gain'])} (saldo líquido: {v['gain'] - v['lose']:+d} instâncias);"
                             for r, v in S["switch"].items())
        sec4 = sec4.replace("## 4. Análise de Grau de Consenso", "## 4. Análise de Grau de Consenso e Risco do Efeito Reverso")
        sec4 += f"""
### 4.2 O efeito reverso a partir do {best} ({I(S['hits'][best])} acertos)
{sw_lines}

Trocar o {best} por outra representação, sem um critério de escolha por instância, reduz os acertos em `{cls}`.
"""
    cells.append(md(sec4))

    cons_plot = """
# 4. Visualização do grau de consenso no MLP
labels_cons = ['4 Entradas\\n(Unânime)', '3 Entradas', '2 Entradas\\n(Dividido)', '1 Entrada\\n(Exclusivo)', '0 Entradas\\n(Erro Comum)']
counts_cons = [S['cons'][j] for j in (4, 3, 2, 1, 0)]
colors_cons = ['#27ae60', '#2980b9', '#f39c12', '#e74c3c', '#7f8c8d']
"""
    if CLASS == 2:
        cons_plot += """
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
"""
    else:
        cons_plot += """
fig, ax1 = plt.subplots(figsize=(9, 5))
"""
    cons_plot += """
bars = ax1.bar(labels_cons, counts_cons, color=colors_cons, edgecolor='black', linewidth=0.8, width=0.55)
ax1.set_title(f"Grau de consenso MLP em {S['cls']} ({S['N']} instâncias arquivo×fold)", fontweight='bold', pad=12)
ax1.set_ylabel(f"Instâncias de {S['cls']}")
ax1.set_ylim(0, max(counts_cons) * 1.15)
for bar, n in zip(bars, counts_cons):
    ax1.annotate(f'{n} ({100 * n / S["N"]:.1f}%)', xy=(bar.get_x() + bar.get_width() / 2, n), xytext=(0, 4),
                 textcoords='offset points', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
"""
    if CLASS == 2:
        cons_plot += """
display(tabela_troca(S))
dest = list(S['switch'])
perdas = [S['switch'][r]['lose'] for r in dest]
ganhos = [S['switch'][r]['gain'] for r in dest]
x = np.arange(len(dest)); w = 0.35
r1 = ax2.bar(x - w / 2, perdas, w, label=f"Perde acertos do {S['best']}", color='#c0392b', alpha=0.9, edgecolor='black')
r2 = ax2.bar(x + w / 2, ganhos, w, label=f"Resgata erros do {S['best']}", color='#27ae60', alpha=0.9, edgecolor='black')
ax2.set_title(f"Efeito reverso ao trocar a entrada a partir do {S['best']}", fontweight='bold', pad=12)
ax2.set_ylabel(f"Instâncias de {S['cls']}")
ax2.set_xticks(x); ax2.set_xticklabels([f'Para {r}' for r in dest], fontweight='bold')
ax2.set_ylim(0, max(perdas + ganhos) * 1.2)
ax2.legend(loc='upper right', frameon=True)
for rect, sign, col in [(r, '-', '#900') for r in r1] + [(r, '+', '#006400') for r in r2]:
    h = rect.get_height()
    ax2.annotate(f'{sign}{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                 textcoords='offset points', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=col)
"""
    cons_plot += """
plt.tight_layout()
plt.show()
"""
    cells.append(code(cons_plot))

    excl_lines = "\n".join(
        f"- **{r}:** {I(S['excl'][r]['inst'])} instâncias ({P(pct(S['excl'][r]['inst']))}) = {I(S['excl'][r]['files'])} gravações / "
        f"{I(S['excl'][r]['ships'])} navios; exclusivas em todas as aparições: {I(S['excl'][r]['files_always'])} gravação(ões)"
        for r in excl_sorted)
    err_lines = "\n".join(
        f"- **{r}** ({I(S['conf'][r]['n_err'])} erros): " + ", ".join(
            f"{P(v)} $\\rightarrow$ `{k}`" for k, v in sorted(((k, v) for k, v in S['conf'][r].items() if k != 'n_err'),
                                                              key=lambda kv: -kv[1]))
        for r in REPS)
    excl_note = (f"Os resgates exclusivos são contados por **instância (arquivo × fold)** e são instáveis entre folds: "
                 f"a mesma gravação pode ser resgatada só pelo {top} num fold e por outra entrada (ou por nenhuma) em outro. "
                 f"Também não são evidência de informação ortogonal: uma entrada mais fraca e com erros diferentes gera "
                 f"resgates exclusivos mesmo ao acaso (ver o oráculo esperado na seção 3).")
    if main_err["MEL"] != main_err["LOFAR"]:
        mel_lof = (f"MEL erra `{cls}` principalmente para `{main_err['MEL']}` e LOFAR para `{main_err['LOFAR']}`, mas isso **não** "
                   f"significa que errem em direções opostas nos mesmos áudios: nas {I(S['ml_both_err'])} instâncias em que **ambos** "
                   f"erram, preveem a **mesma** classe errada em {I(S['ml_same_err'])} ({P(same_frac)}).")
    else:
        mel_lof = (f"MEL e LOFAR erram `{cls}` principalmente para a mesma classe (`{main_err['MEL']}`), e nas {I(S['ml_both_err'])} "
                   f"instâncias em que **ambos** erram preveem a **mesma** classe errada em {I(S['ml_same_err'])} ({P(same_frac)}): "
                   f"os erros das duas entradas de Fourier são fortemente acoplados.")

    if CLASS == 2:
        cells.append(md(f"""
## 5. Resgates Exclusivos no MLP em `{cls}`

Em **{I(S['cons'][1])} instâncias de `{cls}` ({P(pct(S['cons'][1]))})**, três entradas falharam e apenas uma acertou:

{excl_lines}

O **{top}** tem o maior número de resgates exclusivos ({I(et['inst'])} instâncias). O texto anterior falava em "117 navios"; na verdade a contagem é de instâncias arquivo×fold: hoje são {I(et['inst'])} instâncias = **{I(et['files'])} gravações** de **{I(et['ships'])} navios**, e apenas **{I(et['files_always'])} gravações** ({I(et['ships_always'])} navios) são resgatadas exclusivamente pelo {top} em todas as suas aparições no teste.

{excl_note}

A hipótese de que o DEMON capta a modulação de eixo/pás de motores lentos é plausível, mas **não foi testada** aqui (não há análise das frequências de modulação desses arquivos).
"""))
        cells.append(code("""
# 5. Resgates exclusivos por representação: instâncias vs. gravações únicas vs. navios
display(tabela_exclusivos(S))
order = sorted(REPS, key=lambda r: S['excl'][r]['inst'])
inst = [S['excl'][r]['inst'] for r in order]
grav = [S['excl'][r]['files'] for r in order]
alw = [S['excl'][r]['files_always'] for r in order]
y = np.arange(len(order)); h = 0.27
plt.figure(figsize=(10, 4.8))
plt.barh(y + h, inst, h, label='Instâncias (arquivo×fold)', color='#16a085', edgecolor='black')
plt.barh(y, grav, h, label='Gravações únicas', color='#2980b9', edgecolor='black')
plt.barh(y - h, alw, h, label='Gravações exclusivas em todas as aparições', color='#7f8c8d', edgecolor='black')
for yy, vals in zip(y, zip(inst, grav, alw)):
    for off, v in zip((h, 0, -h), vals):
        plt.annotate(str(v), xy=(v, yy + off), xytext=(4, 0), textcoords='offset points', va='center', fontsize=9)
plt.yticks(y, order)
plt.xlabel(f"Contagem em {S['cls']}")
plt.title(f"Resgates exclusivos MLP em {S['cls']} ({S['cons'][1]} instâncias com 1/4 acerto)", fontweight='bold', pad=12)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
"""))
        cells.append(md(f"""
## 6. Padrões de Erro em `{cls}`: Para Onde Cada Entrada Envia a Embarcação no MLP?

{err_lines}

{mel_lof}

As explicações mecânicas antes associadas a cada destino de erro (energia de propulsores, exaustão, dispersão entre IMFs) eram especulativas e foram retiradas: estes dados mostram **para onde** vão os erros, não **por quê**.
"""))
        cells.append(code("""
# 6. Destino do erro em LARGE por representação (MLP)
display(tabela_confusao(S))
dests = [k for k in S['conf'][REPS[0]] if k != 'n_err']
cores = {'SMALL': '#3498db', 'MEDIUM': '#f39c12', 'LARGE': '#e74c3c', 'BACKGROUND': '#95a5a6'}
plt.figure(figsize=(10, 5.5))
bottom = np.zeros(len(REPS))
for k in dests:
    v = np.array([S['conf'][r][k] for r in REPS])
    plt.bar([f"{r}\\n({S['conf'][r]['n_err']} erros)" for r in REPS], v, 0.52, bottom=bottom,
            label=f'Confundido com {k}', color=cores[k], alpha=0.9, edgecolor='black', linewidth=0.6)
    for i, (b, vv) in enumerate(zip(bottom, v)):
        if vv > 4:
            plt.text(i, b + vv / 2, f'{vv:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    bottom += v
plt.title(f"Padrão de erro MLP na classe {S['cls']} (% dos erros de cada entrada)", fontweight='bold', pad=12)
plt.ylabel('Distribuição percentual dos erros (%)')
plt.ylim(0, 110)
plt.legend(loc='upper right', frameon=True, fontsize=9)
plt.tight_layout()
plt.show()
"""))
        cells.append(md(f"""
## 7. Painel Consolidado (MLP)

O painel de 4 quadrantes antes exibido aqui (`reports/analise_complementaridade_mlp_large_oraculo.png`) foi gerado em 15/10/2026 com as predições anteriores de HHT/DEMON e **não corresponde aos números atuais**; por isso não é mais exibido. Os gráficos das seções 3 a 6 são gerados a partir das predições em disco.
"""))
    else:
        cells.append(md(f"""
## 5. Resgates Exclusivos e Padrão de Erro no MLP

### Resgates exclusivos ({I(S['cons'][1])} instâncias com 1/4 acerto):
{excl_lines}

{excl_note}

### Padrão de erro:
{err_lines}

{mel_lof}

> O painel consolidado antes exibido aqui (`reports/analise_complementaridade_mlp_{cls.lower()}_oraculo.png`) foi gerado em 15/10/2026 com as predições anteriores de HHT/DEMON e não corresponde aos números atuais; foi substituído pelas tabelas abaixo.
"""))
        cells.append(code("""
# 5. Resgates exclusivos (instâncias vs. gravações vs. navios) e destino dos erros
display(tabela_exclusivos(S))
display(tabela_confusao(S))
"""))

    # ------------------------------------------------------------------------------------------
    svm_flag = ""
    if unstable_svm:
        svm_flag = ("**Atenção:** no SVM, " + "; ".join(
                        f"{r} tem SP {MS(sp_svm[r])} (mín. por fold {D(svm_fm[r].min(), 1)}"
                        + (f"; SP $= 0$, isto é, alguma classe com recall nulo, em {collapsed[r]} dos 10 folds)"
                           if collapsed[r] else ")")
                        for r in unstable_svm) +
                    " — modelo(s) instável(is) entre folds, que não deve(m) servir de referência para anunciar ganho. "
                    "Isso afeta o consenso, os resgates exclusivos e o oráculo do SVM; a comparação abaixo deve ser lida "
                    "com essa ressalva.")
    sec_n = 8 if CLASS == 2 else 6
    cmp_rows = [
        ("Melhor modelo isolado", f"{P(T['recall'][T['best']])} ({T['best']})", f"{P(rb)} ({best})"),
        ("Oráculo observado (média por fold)", P(T['oracle_fold']), P(S['oracle_fold'])),
        ("Oráculo esperado c/ erros independentes", P(T['oracle_indep']), P(S['oracle_indep'])),
        ("Unanimidade (4/4)", f"{P(pct(T['cons'][4], T['N']))} ({I(T['cons'][4])})", f"{P(pct(S['cons'][4]))} ({I(S['cons'][4])})"),
        ("Resgate exclusivo (1/4)", f"{P(pct(T['cons'][1], T['N']))} ({I(T['cons'][1])})", f"{P(pct(S['cons'][1]))} ({I(S['cons'][1])})"),
        ("Erro comum (0/4)", f"{P(pct(T['cons'][0], T['N']))} ({I(T['cons'][0])})", f"{P(pct(S['cons'][0]))} ({I(S['cons'][0])})"),
        ("Maior resgatador exclusivo", f"{T['excl_top']} ({I(T['excl'][T['excl_top']]['inst'])} inst. / {I(T['excl'][T['excl_top']]['files'])} grav.)",
         f"{top} ({I(et['inst'])} inst. / {I(et['files'])} grav.)"),
        ("\"$\\ge 2/4$\" (empates contados como acerto)", P(pct(T['ge2'], T['N'])), P(pct(S['ge2']))),
        ("Voto real de pluralidade", P(T['vote']), P(S['vote'])),
    ]
    tbl = "\n".join(f"| **{a}** | {b} | {c} |" for a, b, c in cmp_rows)
    cells.append(md(f"""
## {sec_n}. Comparativo Cruzado: `{cls}` no SVM (Notebook {svm_nb}) vs. `{cls}` no MLP (Notebook {nb})

SVM Nyström recalculado com os experimentos declarados no Notebook {svm_nb} ($m=4000$ para MEL, HHT e DEMON; $m=6000$ para LOFAR): {', '.join(f'{r} SP {MS(sp_svm[r])}' for r in REPS)}.

{svm_flag}

| Dimensão | SVM Nyström (NB {svm_nb}) | MLP `[32, 16]` (NB {nb}) |
| :--- | :---: | :---: |
{tbl}

Nas duas famílias o oráculo fica **abaixo** do esperado sob independência (erros correlacionados) e o voto real fica abaixo do "$\\ge 2/4$". Em relação ao melhor modelo isolado, o voto real fica {D(T['vote'] - T['recall'][T['best']], 1)} p.p. no SVM e {D(S['vote'] - rb, 1)} p.p. no MLP (recall de `{cls}`). A semelhança entre SVM e MLP é **consistente** com uma dificuldade ligada aos dados de `{cls}` (e não ao classificador), mas não demonstra uma causa física.
"""))
    cells.append(code("""
# Comparativo SVM vs. MLP (mesmas contas, SVM com os experimentos do notebook correspondente)
display(R.table([(f'SVM {r}', SVM_EXPS[r]) for r in REPS], metrics=('SP', 'ACC'), recalls=False))
T = complementarity(SVM_EXPS, CLASSE)
display(tabela_svm_vs_mlp(T, S))
"""))

    if CLASS == 2:
        u = lambda X, c: pct(X[c]["cons"][4], X[c]["N"])
        hier = " $<$ ".join(f"`{META[c]['cls']}` (SVM {P(u(allT, c), 1)}, MLP {P(u(allS, c), 1)})" for c in (0, 1, 2))
        orc_rows = ", ".join(f"`{META[c]['cls']}` MLP {P(allS[c]['oracle_fold'], 1)} vs. {P(allS[c]['oracle_indep'], 1)} esperado"
                             for c in (0, 1, 2))
        vote_rows = ", ".join(f"`{META[c]['cls']}` {P(allS[c]['vote'], 1)} vs. {allS[c]['best']} {P(allS[c]['recall'][allS[c]['best']], 1)} ({PV(allS[c]['vote_p'])})"
                              for c in (0, 1, 2))
        excl_rows = "; ".join(f"`{META[c]['cls']}`: MLP {allS[c]['excl_top']}, SVM {allT[c]['excl_top']}" for c in (0, 1, 2))
        cells.append(md(f"""
---

## 9. Conclusão do Estudo de Consenso Multi-Entrada (SVM e MLP)

1. **A hierarquia de unanimidade se repete nas duas famílias:** {hier}. É um padrão consistente entre classificadores, compatível com `SMALL` ser a classe mais difícil; não é, por si só, uma "lei física".
2. **Oráculos não demonstram complementaridade:** em todas as classes o oráculo observado fica abaixo do esperado com erros independentes ({orc_rows}); os erros das entradas são positivamente correlacionados, e o oráculo não é realizável.
3. **Voto real vs. melhor isolado (recall da classe, MLP):** {vote_rows}. No SP global, o voto real das 4 MLPs ({MS(VG['vote_sp'])}) é {('equivalente ao' if VG['p'] >= 0.05 else 'diferente do')} melhor isolado {VG['best']} ({MS(VG['best_sp'])}, {PV(VG['p'])}).
4. **Maior resgatador exclusivo por classe** — {excl_rows}. Os resgates são contagens de instâncias, instáveis entre folds, e poucas gravações são resgatadas pela mesma entrada em todas as aparições; não sustentam a afirmação de que o DEMON seja "indispensável".
5. **Teto de desempenho:** os tetos de oráculo do SVM e do MLP são próximos, mas não são alcançáveis; ganhos reais precisam ser medidos com um combinador realizável (voto, *stacking*, gating) e testados com o F-test 5x2cv.
"""))

    nb_data = {
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
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=2, ensure_ascii=False)
    print(f"Notebook gerado com sucesso em '{filename}'!")


if __name__ == "__main__":
    create_notebook()
