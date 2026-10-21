"""
Script de geração do notebook 18: notebooks/18_analise_complementaridade_rf_medium.ipynb
Análise de Ortogonalidade, Efeito Oráculo e Grau de Consenso no Random Forest para a Classe MEDIUM (Dataset IARA).

Os números do markdown são calculados na geração (notebooks/rf_complementaridade_comum.py + iara_results.py)
a partir das predições em results/trainings/multirep/rf_*_4c. Rodar na raiz do repositório:
    python notebooks/generate_rf_medium_notebook.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rf_complementaridade_comum as C  # noqa: E402
from rf_complementaridade_comum import br, p_txt  # noqa: E402

CLASSE = "MEDIUM"
NB = 18

HIPOTESES = """> **Hipóteses físicas (não testadas nesta análise).** Conjecturas para orientar trabalho futuro; nada nas predições as confirma:
> - a assinatura de motores diesel de média rotação de embarcações de 50–100 m poderia cair, nas partições das árvores, em regiões próximas às de embarcações `SMALL`;
> - o nível de ruído ambiente/de superfície poderia aproximar assinaturas intermediárias da classe `SMALL`."""


def create_notebook(filename="notebooks/18_analise_complementaridade_rf_medium.ipynb"):
    ns = C.run_helper()
    tri_cells, A_fam = C.secao_trilateral(ns, CLASSE, NB, 8)
    a = A_fam["RF"]
    orc, cons, ex, v, er = a["oraculo"], a["consenso"], a["exclusivos"], a["voto"], a["erros"]
    b = a["melhor"]
    ex_s = ex.sort_values("Instâncias", ascending=False)
    unan = {f: A_fam[f]["consenso"]["% da classe"].iloc[0] for f in A_fam}
    orc_f = {f: A_fam[f]["oraculo"]["agregado"] for f in A_fam}
    to_small = er["→ SMALL (%)"]
    voto_rel = ("não difere significativamente do" if v["p"] >= 0.05 else
                ("é significativamente maior que o" if v["recall"][0] > v["melhor"][0] else "é significativamente menor que o"))
    excl_txt = "; ".join(f"{r} {br(int(row['Instâncias']))} inst. ({br(int(row['Gravações distintas']))} grav.)"
                         for r, row in ex_s.iterrows())

    header = f"""# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook {NB}: Análise de Ortogonalidade, Efeito Oráculo e Grau de Consenso no Random Forest — Classe MEDIUM

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

## Resumo Executivo da Investigação com Random Forest (RF)

A classe **`MEDIUM`** (embarcações de 50 m a 100 m, conforme `src/iara/default.py`) foi avaliada em **{br(a['n_inst'])} instâncias de teste** (arquivo × fold; **{br(a['n_grav'])} gravações distintas** de **{br(a['n_navios'])} navios**) nos 10 folds do protocolo 5×2cv, com **Random Forest (100 árvores, `class_weight='balanced'`)** nas 4 representações (**MEL, LOFAR, HHT e DEMON**).

O objetivo é confrontar os resultados do SVM (Notebook 12) e do MLP (Notebook 15) com os do Random Forest em navios de porte intermediário.

{C.aviso_modelos(ns)}

### Principais Descobertas no Random Forest:
1. **Oráculo RF de {br(orc['agregado'])}% ({br(orc['acertos'])} / {br(a['n_inst'])} instâncias), não realizável; sob erros independentes o esperado seria {br(orc['esp_indep'])}%:**
   - Melhor isolado: **{b}** ({br(v['melhor'][0])}% ± {br(v['melhor'][1])} nos folds). Oráculos das três famílias: SVM {br(orc_f['SVM'])}%, MLP {br(orc_f['MLP'])}%, RF {br(orc_f['RF'])}%.
   - Erro comum (as 4 erram): **{br(int(cons['Instâncias'].iloc[-1]))} instâncias ({br(cons['% da classe'].iloc[-1])}%)**.
2. **Unanimidade 4/4 de {br(unan['RF'])}%** ({br(int(cons['Instâncias'].iloc[0]))} instâncias), contra {br(unan['SVM'])}% no SVM e {br(unan['MLP'])}% no MLP.
3. **Resgates exclusivos:** {excl_txt}. São instâncias arquivo × fold, em grande parte instáveis entre folds (ver Seção 5).
4. **Destino dos erros:** quando erra em `MEDIUM`, o RF envia a maior parte para `SMALL` em todas as entradas ({'; '.join(f'{r} {br(x)}%' for r, x in to_small.items())}).
5. **Votação:** "≥2/4" dá {br(v['ge2'])}%, mas conta {br(v['empates_2x2'])} empates 2×2 como acerto e não é realizável. O **voto real** tem recall em `MEDIUM` de **{br(v['recall'][0])}% ± {br(v['recall'][1])}** e {voto_rel} melhor isolado ({b}; F-test 5×2cv, {p_txt(v['p'])})."""

    assert (to_small == er[[c for c in er.columns if c.startswith("→")]].max(axis=1)).all(), \
        "o texto do item 4 supõe que SMALL é o principal destino em todas as entradas"

    cells = [C.md(header), C.code(C.SETUP_SRC), C.code(C.HELPER_SRC)]
    cells += C.cells_comuns(ns, a, NB, "o Navio Médio", HIPOTESES)
    cells += tri_cells
    C.write_notebook(cells, filename)


if __name__ == "__main__":
    create_notebook()
