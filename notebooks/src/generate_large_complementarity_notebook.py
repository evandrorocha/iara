"""
Script de geração do notebook 13: notebooks/13_analise_complementaridade_large.ipynb
Complementaridade das 4 entradas do SVM (MEL, LOFAR, HHT, DEMON) na classe LARGE: oráculo observado vs. esperado
sob independência, grau de consenso, voto real, resgates exclusivos e efeito reverso.

Reutiliza o código de análise e os construtores de células de generate_small_complementarity_notebook.py
(mesma metodologia dos Notebooks 10 e 12). Todos os números vêm das predições em disco via notebooks/iara_results.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_small_complementarity_notebook as G  # noqa: E402


def create_notebook(filename="notebooks/13_analise_complementaridade_large.ipynb"):
    ns = G.compute(2, "LARGE")
    nss = {"SMALL": G.compute(0, "SMALL"), "MEDIUM": G.compute(1, "MEDIUM"), "LARGE": ns}
    cells = [G.summary_cell(
        ns, 13, "LARGE",
        "A classe **`LARGE`** (embarcações de grande porte — mercantes, graneleiros, petroleiros, porta-contêineres —, comprimento $> 100\\text{ m}$ segundo *Silva et al.*, IEEE Access 2025) é a de maior recall entre as classes de navio para os SVMs deste estudo.",
        "Completando os Notebooks 10 (`SMALL`) e 12 (`MEDIUM`), este notebook aplica a mesma análise de complementaridade das 4 entradas (MEL, LOFAR, HHT, DEMON) à classe `LARGE`."),
        G.code(G.SETUP_CODE)]
    cells += G.common_cells(ns, 2, "LARGE", 13)
    oracle_mean = sum(s["ORC_OBS"] for s in nss.values()) / 3
    exp_mean = sum(s["ORC_EXP"] for s in nss.values()) / 3
    extra = ("\n## 9. Síntese entre as classes de navio (NB 10, 12 e 13)\n\n"
             "Todas as linhas abaixo são recalculadas agora para as três classes (instâncias arquivo × fold; média de " + G.br(ns["EVAL_PER_FILE_ALL"]) + " aparições de teste por arquivo, não 5).\n\n"
             + G.cross_class_md(nss, {"SMALL": 10, "MEDIUM": 12, "LARGE": 13}) +
             f"\nMédia simples dos oráculos observados nas três classes de navio: {G.br(oracle_mean)}% (esperado sob independência: {G.br(exp_mean)}%). "
             "Em todas as classes o oráculo observado fica " + ("abaixo" if all(s["ORC_OBS"] < s["ORC_EXP"] for s in nss.values()) else "ora abaixo, ora acima") +
             " do esperado com erros independentes; o oráculo não é um sistema realizável e não indica, por si só, que um roteador possa elevar o SP. "
             "Afirmações de que \"o gargalo pode ser superado com roteamento dinâmico\" ou de SP > 70% não são sustentadas por esta análise.\n"
             "\nOs papéis físicos atribuídos a cada entrada em versões anteriores (LOFAR = raias de maquinário, DEMON = rotação de eixo a 1,3–2,0 Hz, HHT = transientes não lineares) são hipóteses: este notebook não mede frequências nem verifica os áudios.\n")
    cells.append(G.conclusion_cell(ns, "LARGE", extra=extra))
    G.write(filename, cells)


if __name__ == "__main__":
    create_notebook()
