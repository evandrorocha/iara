"""
Script de geração do notebook 14: notebooks/14_analise_complementaridade_medium.ipynb
Complementaridade das 4 entradas do SVM (MEL, LOFAR, HHT, DEMON) na classe MEDIUM: oráculo observado vs. esperado
sob independência, grau de consenso, voto real, resgates exclusivos e efeito reverso.

Reutiliza o código de análise e os construtores de células de generate_small_complementarity_notebook.py
(mesma metodologia do Notebook 13). Todos os números vêm das predições em disco via notebooks/iara_results.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_small_complementarity_notebook as G  # noqa: E402


def create_notebook(filename="notebooks/14_analise_complementaridade_medium.ipynb"):
    ns = G.compute(1, "MEDIUM")
    ns_small = G.compute(0, "SMALL")
    cells = [G.summary_cell(
        ns, 12, "MEDIUM",
        "A classe **`MEDIUM`** (embarcações de porte intermediário, comprimento entre $50\\text{ m}$ e $100\\text{ m}$) fica entre `SMALL` e `LARGE` no espaço de características do dataset IARA (*Silva et al.*, IEEE Access 2025).",
        "Seguindo o Notebook 13 (`SMALL`), este notebook pergunta se os áudios de `MEDIUM` que o SVM erra mudam com a entrada (MEL, LOFAR, HHT, DEMON), qual é o grau de consenso entre as entradas e qual seria o risco de um seletor de entrada (efeito reverso)."),
        G.code(G.SETUP_CODE)]
    cells += G.common_cells(ns, 1, "MEDIUM", 12)

    # Canal SMALL <-> MEDIUM (calculado)
    def pct(nsx, rep, col):
        conf = nsx["df_confusion"]
        row = conf.loc[[i for i in conf.index if i.startswith(nsx["LBL"][rep] + " ")][0]]
        return row[col]
    canal = (f"- No Notebook 13, o {ns['LBL']['MEL']} envia {G.br(pct(ns_small, 'MEL', '→ MEDIUM (%)'))}% dos erros de `SMALL` para `MEDIUM`; "
             f"aqui, envia {G.br(pct(ns, 'MEL', '→ SMALL (%)'))}% dos erros de `MEDIUM` para `SMALL`. "
             f"Para o {ns['LBL']['LOFAR']}: {G.br(pct(ns_small, 'LOFAR', '→ MEDIUM (%)'))}% (SMALL→MEDIUM) e {G.br(pct(ns, 'LOFAR', '→ SMALL (%)'))}% (MEDIUM→SMALL).\n"
             "- Isso indica uma confusão recíproca `SMALL` ↔ `MEDIUM`. Explicações físicas (perfil energético de lanchas vs. rebocadores, hélices de passo variável etc.) são hipóteses não verificadas nos áudios.\n")
    extra = ("\n## 9. Síntese comparativa: SMALL (NB 13) vs. MEDIUM (NB 14)\n\n"
             "Todas as linhas abaixo são recalculadas agora para as duas classes (instâncias arquivo × fold; média de " + G.br(ns["EVAL_PER_FILE_ALL"]) + " aparições de teste por arquivo, não 5).\n\n"
             + G.cross_class_md({"SMALL": ns_small, "MEDIUM": ns}, {"SMALL": 10, "MEDIUM": 12}) + "\n" + canal)
    cells.append(G.conclusion_cell(ns, "MEDIUM", extra=extra))
    G.write(filename, cells)


if __name__ == "__main__":
    create_notebook()
