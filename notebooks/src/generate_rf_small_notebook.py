"""
Script de geração do notebook 19: notebooks/19_analise_complementaridade_rf_small.ipynb
Análise de Ortogonalidade, Efeito Oráculo e Grau de Consenso no Random Forest para a Classe SMALL (Dataset IARA).

Os números do markdown são calculados na geração (notebooks/rf_complementaridade_comum.py + iara_results.py)
a partir das predições em results/trainings/multirep/rf_*_4c. Rodar na raiz do repositório:
    python notebooks/generate_rf_small_notebook.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rf_complementaridade_comum as C  # noqa: E402
from rf_complementaridade_comum import br, p_txt, md, code  # noqa: E402

CLASSE = "SMALL"
NB = 17

HIPOTESES = """> **Hipóteses físicas (não testadas nesta análise).** As explicações abaixo são conjecturas para orientar trabalho futuro; nada nas predições as confirma:
> - DEMON: a cadência de pás/eixo de motores de popa (ex.: 2.500–5.000 RPM ⇒ ~42–83 Hz de rotação de eixo) poderia gerar partições limpas nas árvores;
> - MEL: a energia em altas frequências poderia aproximar embarcações pequenas e `MEDIUM`;
> - LOFAR/HHT/DEMON: a falta de raias tonais fortes ou a modulação por ondas de superfície poderia levar o áudio para `LARGE`."""


def create_notebook(filename="notebooks/19_analise_complementaridade_rf_small.ipynb"):
    ns = C.run_helper()
    tri_cells, A_fam = C.secao_trilateral(ns, CLASSE, NB, 8)
    a = A_fam["RF"]
    iso, orc, cons, ex, v = a["isolado"], a["oraculo"], a["consenso"], a["exclusivos"], a["voto"]
    b = a["melhor"]
    champ = ex["Instâncias"].idxmax()
    unan = {f: A_fam[f]["consenso"]["% da classe"].iloc[0] for f in A_fam}
    orc_f = {f: A_fam[f]["oraculo"]["agregado"] for f in A_fam}
    maior_orc = max(orc_f, key=orc_f.get)
    voto_rel = ("não difere significativamente do" if v["p"] >= 0.05 else
                ("é significativamente maior que o" if v["recall"][0] > v["melhor"][0] else "é significativamente menor que o"))

    header = f"""# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook {NB}: Análise de Ortogonalidade, Efeito Oráculo e Grau de Consenso no Random Forest — Classe SMALL

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

## Resumo Executivo da Investigação com Random Forest (RF)

A classe **`SMALL`** (embarcações com comprimento < 50 m, conforme `src/iara/default.py`) foi avaliada em **{br(a['n_inst'])} instâncias de teste** (arquivo × fold; **{br(a['n_grav'])} gravações distintas** de **{br(a['n_navios'])} navios**) nos 10 folds do protocolo 5×2cv, com **Random Forest (100 árvores, `class_weight='balanced'`)** nas 4 representações (**MEL, LOFAR, HHT e DEMON**).

O objetivo é confrontar os resultados do SVM (Notebook 13) e do MLP (Notebook 16) com os do Random Forest na classe `SMALL`.

{C.aviso_modelos(ns)}

### Principais Descobertas no Random Forest:
1. **Oráculo RF de {br(orc['agregado'])}% ({br(orc['acertos'])} / {br(a['n_inst'])} instâncias), não realizável; sob erros independentes o esperado seria {br(orc['esp_indep'])}%:**
   - Entre as três famílias, o maior oráculo em `SMALL` é o do {maior_orc} (SVM {br(orc_f['SVM'])}%, MLP {br(orc_f['MLP'])}%, RF {br(orc_f['RF'])}%). O oráculo usa o rótulo para escolher a entrada.
   - Erro comum (as 4 erram): **{br(cons['Instâncias'].iloc[-1])} instâncias ({br(cons['% da classe'].iloc[-1])}%)**.
2. **Unanimidade 4/4 de {br(unan['RF'])}%** ({br(cons['Instâncias'].iloc[0])} instâncias), contra {br(unan['SVM'])}% no SVM e {br(unan['MLP'])}% no MLP.
3. **Maior número de resgates exclusivos: {champ}** ({br(ex.loc[champ, 'Instâncias'])} instâncias = {br(ex.loc[champ, 'Gravações distintas'])} gravações; só {br(ex.loc[champ, 'Gravações exclusivas em todas as aparições'])} gravações são exclusivas do {champ} em todas as aparições).
4. **Votação:** o critério "≥2/4" dá {br(v['ge2'])}%, mas conta {br(v['empates_2x2'])} empates 2×2 como acerto e não é realizável. O **voto real** (pluralidade, empate sorteado) tem recall em `SMALL` de **{br(v['recall'][0])}% ± {br(v['recall'][1])}** e {voto_rel} melhor isolado ({b}, {br(v['melhor'][0])}% ± {br(v['melhor'][1])}; F-test 5×2cv, {p_txt(v['p'])}). No SP global, o voto real ({br(v['sp'][0])}) fica contra {br(v['melhor_sp'][0])} do RF {v['melhor_sp_rep']} ({p_txt(v['p_sp'])})."""

    cells = [C.md(header), C.code(C.SETUP_SRC), C.code(C.HELPER_SRC)]
    cells += C.cells_comuns(ns, a, NB, "a Embarcação Pequena", HIPOTESES)
    cells += tri_cells
    cells += secao_ensemble(ns, v)
    C.write_notebook(cells, filename)


ENS4 = "ensemble_rf_prob_mel-lofar-hht-demon_4c"
ENS3 = "ensemble_rf_prob_mel-lofar-hht_4c"

ENSEMBLE_SRC = r'''# 9. Ensemble RF por média de probabilidades (scratch/ensemble_rf_prob_5x2cv.py)
RF = FAMILIAS['RF']
ENS4, ENS3 = 'ensemble_rf_prob_mel-lofar-hht-demon_4c', 'ensemble_rf_prob_mel-lofar-hht_4c'
linhas = [(f'RF {r}', RF[r]) for r in REPS] + [('Ensemble RF (4 rep., média de prob.)', ENS4),
                                              ('Ensemble RF sem DEMON (exploratório)', ENS3)]
tab_ens = R.table(linhas, metrics=('SP', 'ACC', 'F1'))
mv = R.majority_vote(nomes('RF'))
tab_ens.loc['Voto real (pluralidade, empate sorteado)'] = (
    [f"{mv[m].mean():.2f} ± {mv[m].std(ddof=0):.2f}" for m in ('SP', 'ACC', 'F1')]
    + [round(mv['R_' + c].mean(), 2) for c in R.CLASSES] + ['(calculado aqui)'])
display(tab_ens)

testes = []
for ens_lab, ens in (('4 rep.', ENS4), ('sem DEMON', ENS3)):
    fe = R.fold_metrics(ens)
    for ref in ('LOFAR', 'MEL'):
        fr = R.fold_metrics(RF[ref])
        for m in ['SP'] + ['R_' + c for c in R.CLASSES]:
            F, p = R.f_test_5x2(fe[m].values, fr[m].values)
            testes.append({'Ensemble': ens_lab, 'vs': f'RF {ref}', 'Métrica': m.replace('R_', 'recall '),
                           'Δ (p.p.)': round(fe[m].mean() - fr[m].mean(), 2), 'F': round(F, 2), 'p': round(p, 3)})
    F, p = R.f_test_5x2(fe['SP'].values, mv['SP'].values)
    testes.append({'Ensemble': ens_lab, 'vs': 'voto real', 'Métrica': 'SP',
                   'Δ (p.p.)': round(fe['SP'].mean() - mv['SP'].mean(), 2), 'F': round(F, 2), 'p': round(p, 3)})
display(pd.DataFrame(testes).set_index(['Ensemble', 'vs', 'Métrica']))'''


def secao_ensemble(ns, v):
    R = ns["R"]
    rf = ns["FAMILIAS"]["RF"]
    s4, s3 = R.summary(ENS4), R.summary(ENS3)
    lo, me = R.summary(rf["LOFAR"]), R.summary(rf["MEL"])
    fe4, fe3 = R.fold_metrics(ENS4), R.fold_metrics(ENS3)
    flo, fme = R.fold_metrics(rf["LOFAR"]), R.fold_metrics(rf["MEL"])
    pt = lambda a, b, m: p_txt(R.f_test_5x2(a[m].values, b[m].values)[1])  # noqa: E731
    # Pontos de comparação citados no texto (os sinais e significâncias saem dos dados)
    perdas = [c for c in ("MEDIUM", "BACKGROUND", "LARGE")
              if s4["R_" + c][0] < lo["R_" + c][0]]
    sig = lambda a, b, m: R.f_test_5x2(a[m].values, b[m].values)[1] < 0.05  # noqa: E731
    sp_txt = lambda a, b: ("**sem diferença significativa**" if not sig(a, b, "SP") else  # noqa: E731
                           "**diferença significativa**")

    intro = f"""## 9. Ensemble RF por Média de Probabilidades

A Seção 4.3 mostra que o critério "≥2/4 acertam" ({br(v['ge2'])}% em `SMALL`) usa o rótulo e não é uma regra de decisão. Uma regra realizável que usa a mesma ideia é combinar as probabilidades dos RF: o modelo **`ensemble_rf`** (`scratch/ensemble_rf_prob_5x2cv.py`) não treina nada novo. Para cada gravação de teste:

1. em cada RF, tira a média das probabilidades (`predict_proba`) das janelas da gravação — as janelas não coincidem entre representações (MEL/LOFAR em 5 s, HHT/DEMON em 5,12 s), por isso a média é por gravação;
2. tira a média dessas probabilidades entre as representações (pesos iguais);
3. prediz a classe de maior probabilidade.

O empate entre classes, problema do voto por maioria (2 RF contra 2), praticamente não ocorre com probabilidades contínuas. Os folds e o treino de cada RF são os mesmos dos modelos isolados; os números são de 4 classes (não há RF MEL/LOFAR em 3 classes).

Duas variantes:
- **4 representações** (MEL + LOFAR + HHT + DEMON): a proposta original, definida antes de ver os resultados;
- **sem DEMON**: **exploratória** — a exclusão do DEMON (SP {br(R.summary(rf['DEMON'])['SP'][0])} isolado) foi decidida depois de ver o desempenho no teste, então seu resultado está otimista e não deve ser apresentado como escolha a priori."""

    leitura = f"""**Leitura.**
- **SP global:** o ensemble de 4 representações tem SP {br(s4['SP'][0])} ± {br(s4['SP'][1])}, contra {br(lo['SP'][0])} ± {br(lo['SP'][1])} do RF LOFAR ({sp_txt(fe4, flo)}; F-test 5×2cv, {pt(fe4, flo, 'SP')}) e {br(me['SP'][0])} ± {br(me['SP'][1])} do RF MEL ({sp_txt(fe4, fme)}; {pt(fe4, fme, 'SP')}). A variante sem DEMON chega a {br(s3['SP'][0])} ± {br(s3['SP'][1])} ({pt(fe3, flo, 'SP')} vs RF LOFAR).
- **`SMALL`:** o recall sobe de {br(lo['R_SMALL'][0])}% (RF LOFAR) para {br(s4['R_SMALL'][0])}% ({pt(fe4, flo, 'R_SMALL')}). Contra o RF MEL ({br(me['R_SMALL'][0])}%), a diferença é {pt(fe4, fme, 'R_SMALL')}. Os {br(v['ge2'])}% do critério "≥2/4" não são alcançados: parte deles vinha de empates 2×2 contados como acerto.
- **Troca, não ganho líquido:** em relação ao RF LOFAR, o ensemble perde recall em {', '.join(f'`{c}` ({br(s4["R_" + c][0] - lo["R_" + c][0])} p.p.; {pt(fe4, flo, "R_" + c)})' for c in perdas) or 'nenhuma outra classe'}. O ganho em `SMALL` vem, em parte, de redistribuir acertos entre as classes.
- **Uso recomendado:** linha de base simples de fusão (sem treino adicional) para comparar com os sistemas de gating do Notebook 26, e não como um modelo que supera o melhor RF isolado."""

    return [C.md(intro), C.code(ENSEMBLE_SRC), C.md(leitura)]


if __name__ == "__main__":
    create_notebook()
