"""
Script de geração do notebook 19: notebooks/19_analise_complementaridade_rf_large.ipynb
Análise de Ortogonalidade, Efeito Oráculo e Grau de Consenso no Random Forest para a Classe LARGE (Dataset IARA).

Os números do markdown são calculados na geração (notebooks/rf_complementaridade_comum.py + iara_results.py)
a partir das predições em results/trainings/multirep/rf_*_4c. Rodar na raiz do repositório:
    python notebooks/generate_rf_large_notebook.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rf_complementaridade_comum as C  # noqa: E402
from rf_complementaridade_comum import br, p_txt, md, code  # noqa: E402

CLASSE = "LARGE"
NB = 19
FAMS = ("SVM", "MLP", "RF")

HIPOTESES = """> **Hipóteses físicas (não testadas nesta análise).** Conjecturas para orientar trabalho futuro; nada nas predições as confirma:
> - DEMON: a modulação pela rotação lenta de eixo de navios grandes (< 2 Hz) poderia separar parte dos `LARGE`; quando essa modulação não é captada, o envelope residual poderia se parecer com o de embarcações pequenas;
> - LOFAR: raias tonais de maquinário poderiam explicar a vantagem do LOFAR nesta classe."""

SINTESE_CODE = """# 9. Síntese: SVM vs MLP vs RF nas três classes de navio
linhas = []
for cl in ('SMALL', 'MEDIUM', 'LARGE'):
    for f in ('SVM', 'MLP', 'RF'):
        a = analise_classe(f, cl)
        ex = a['exclusivos']
        linhas.append({'Classe': cl, 'Família': f,
                       'Melhor isolado': a['melhor'], 'Recall melhor isolado (%)': a['voto']['melhor'][0],
                       'Unanimidade 4/4 (%)': a['consenso']['% da classe'].iloc[0],
                       'Erro comum 0/4 (%)': a['consenso']['% da classe'].iloc[-1],
                       'Oráculo obs. (%)': a['oraculo']['agregado'],
                       'Oráculo esperado indep. (%)': a['oraculo']['esp_indep'],
                       'Maior resgate exclusivo': ex['Instâncias'].idxmax(),
                       'Voto real (%)': a['voto']['recall'][0], 'p voto vs melhor': a['voto']['p']})
sintese = pd.DataFrame(linhas).set_index(['Classe', 'Família'])
display(sintese)"""


def create_notebook(filename="notebooks/19_analise_complementaridade_rf_large.ipynb"):
    ns = C.run_helper()
    tri_cells, A_fam = C.secao_trilateral(ns, CLASSE, NB, 8)
    a = A_fam["RF"]
    orc, cons, ex, v = a["oraculo"], a["consenso"], a["exclusivos"], a["voto"]
    b = a["melhor"]
    champ = ex["Instâncias"].idxmax()
    unan = {f: A_fam[f]["consenso"]["% da classe"].iloc[0] for f in A_fam}
    orc_f = {f: A_fam[f]["oraculo"]["agregado"] for f in A_fam}
    voto_rel = ("não difere significativamente do" if v["p"] >= 0.05 else
                ("é significativamente maior que o" if v["recall"][0] > v["melhor"][0] else "é significativamente menor que o"))

    header = f"""# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook {NB}: Análise de Ortogonalidade, Efeito Oráculo e Grau de Consenso no Random Forest — Classe LARGE

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

## Resumo Executivo da Investigação com Random Forest (RF)

A classe **`LARGE`** (embarcações com comprimento ≥ 100 m, conforme `src/iara/default.py`) foi avaliada em **{br(a['n_inst'])} instâncias de teste** (arquivo × fold; **{br(a['n_grav'])} gravações distintas** de **{br(a['n_navios'])} navios**) nos 10 folds do protocolo 5×2cv, com **Random Forest (100 árvores, `class_weight='balanced'`)** nas 4 representações (**MEL, LOFAR, HHT e DEMON**).

O objetivo é confrontar os resultados do SVM (Notebook 13) e do MLP (Notebook 16) com os do Random Forest na classe `LARGE`.

{C.aviso_modelos(ns)}

### Principais Descobertas no Random Forest:
1. **Oráculo RF de {br(orc['agregado'])}% ({br(orc['acertos'])} / {br(a['n_inst'])} instâncias), não realizável; sob erros independentes o esperado seria {br(orc['esp_indep'])}%:**
   - Oráculos das três famílias: SVM {br(orc_f['SVM'])}%, MLP {br(orc_f['MLP'])}%, RF {br(orc_f['RF'])}%.
   - Erro comum (as 4 erram): **{br(int(cons['Instâncias'].iloc[-1]))} instâncias ({br(cons['% da classe'].iloc[-1])}%)**.
2. **Unanimidade 4/4 de {br(unan['RF'])}%:** {br(int(cons['Instâncias'].iloc[0]))} instâncias arquivo × fold, que correspondem a **{br(int(cons['Gravações distintas'].iloc[0]))} gravações distintas** (não "navios"). SVM: {br(unan['SVM'])}%; MLP: {br(unan['MLP'])}%. Com o alto consenso, **{br(v['ge3'])}% ({br(v['ge3_n'])} instâncias)** são acertadas por pelo menos 3 entradas.
3. **Maior número de resgates exclusivos: {champ}** ({br(int(ex.loc[champ, 'Instâncias']))} instâncias = {br(int(ex.loc[champ, 'Gravações distintas']))} gravações de {br(int(ex.loc[champ, 'Navios distintos']))} navios; {br(int(ex.loc[champ, 'Gravações exclusivas em todas as aparições']))} dessas gravações são exclusivas do {champ} em todas as aparições).
4. **Votação:** "≥2/4" dá {br(v['ge2'])}%, mas conta {br(v['empates_2x2'])} empates 2×2 como acerto e não é realizável. O **voto real** tem recall em `LARGE` de **{br(v['recall'][0])}% ± {br(v['recall'][1])}** e {voto_rel} melhor isolado ({b}, {br(v['melhor'][0])}% ± {br(v['melhor'][1])}; F-test 5×2cv, {p_txt(v['p'])})."""

    # ---- Síntese (Seção 9): calculada aqui com o mesmo código da célula ----
    S = {cl: (A_fam if cl == CLASSE else {f: ns["analise_classe"](f, cl) for f in FAMS}) for cl in ("SMALL", "MEDIUM", "LARGE")}
    un = {cl: [S[cl][f]["consenso"]["% da classe"].iloc[0] for f in FAMS] for cl in S}
    faixa = "; ".join(f"`{cl}` {br(min(x))}–{br(max(x))}%" for cl, x in un.items())
    ordem_ok = max(un["SMALL"]) < min(un["MEDIUM"]) and max(un["MEDIUM"]) < min(un["LARGE"])
    ordem_txt = ("Em todas as famílias, a unanimidade cresce de `SMALL` para `MEDIUM` e para `LARGE`, sem sobreposição das faixas."
                 if ordem_ok else "As faixas de unanimidade das classes se sobrepõem entre famílias.")
    top_orc = {cl: max(FAMS, key=lambda f: S[cl][f]["oraculo"]["agregado"]) for cl in S}
    champs = {cl: {f: S[cl][f]["exclusivos"]["Instâncias"].idxmax() for f in FAMS} for cl in S}
    champ_txt = "; ".join(f"`{cl}`: " + ", ".join(f"{f} → {champs[cl][f]}" for f in FAMS) for cl in S)
    sig = [(cl, f) for cl in S for f in FAMS if S[cl][f]["voto"]["p"] < 0.05]
    sig_txt = ("Em nenhuma combinação classe × família o voto real difere significativamente do melhor isolado (F-test 5×2cv, p ≥ 0,05)."
               if not sig else "O voto real difere significativamente do melhor isolado (p < 0,05) em: " +
               ", ".join(f"{f}/{cl} ({br(S[cl][f]['voto']['recall'][0])}% vs {br(S[cl][f]['voto']['melhor'][0])}%, {p_txt(S[cl][f]['voto']['p'])})" for cl, f in sig) +
               "; nas demais combinações, não.")
    abaixo = all(S[cl][f]["oraculo"]["agregado"] < S[cl][f]["oraculo"]["esp_indep"] for cl in S for f in FAMS)

    sintese_md = f"""## 9. Síntese: SVM vs. MLP vs. Random Forest nas Três Classes de Navio

Consolidação dos Notebooks 10 a 19, calculada na célula abaixo com o mesmo código (instâncias arquivo × fold; as famílias não são homogêneas — ver Seção 8):

1. **Oráculos (não realizáveis):** o maior oráculo é do {top_orc['SMALL']} em `SMALL`, do {top_orc['MEDIUM']} em `MEDIUM` e do {top_orc['LARGE']} em `LARGE`. {"Em todos os 9 casos o oráculo observado fica abaixo do esperado sob erros independentes: as entradas erram de forma correlacionada." if abaixo else "Nem sempre o oráculo observado fica abaixo do esperado sob erros independentes."}
2. **Maior número de resgates exclusivos por família:** {champ_txt}. Esses resgates são contados em instâncias e são, em grande parte, instáveis entre folds.
3. **Unanimidade 4/4 por classe (faixa entre famílias):** {faixa}. {ordem_txt} Isso descreve o consenso dos modelos; não identifica a causa física (marulho, distância, etc.).
4. **Votação real:** {sig_txt} Um oráculo alto não se converte em ganho com voto simples."""

    cells = [C.md(header), C.code(C.SETUP_SRC), C.code(C.HELPER_SRC)]
    cells += C.cells_comuns(ns, a, NB, "o Navio Grande", HIPOTESES)
    cells += tri_cells
    cells += [md(sintese_md), code(SINTESE_CODE)]
    C.write_notebook(cells, filename)


if __name__ == "__main__":
    create_notebook()
