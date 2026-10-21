"""
Script de geração do notebook 17: notebooks/17_analise_complementaridade_rf_small.ipynb
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


def create_notebook(filename="notebooks/17_analise_complementaridade_rf_small.ipynb"):
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

O objetivo é confrontar os resultados do SVM (Notebook 10) e do MLP (Notebook 14) com os do Random Forest na classe `SMALL`.

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
    C.write_notebook(cells, filename)


if __name__ == "__main__":
    create_notebook()
