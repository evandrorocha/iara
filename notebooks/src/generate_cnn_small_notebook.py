"""
Script de geração do notebook 22: notebooks/22_analise_complementaridade_cnn_small.ipynb
(e construtor comum dos notebooks 23 e 24 — ver build_cnn_notebook).

Análise de complementaridade, efeito oráculo e grau de consenso na CNN 2D para uma classe.
Todos os números vêm das predições em disco (notebooks/iara_results.py + notebooks/cnn_complementaridade_comum.py);
os valores citados no markdown são calculados aqui, na geração.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iara_results as R  # noqa: E402
import cnn_complementaridade_comum as C  # noqa: E402

CLASS_INFO = {
    0: {"nb": 20, "desc": "embarcações de pequeno porte, comprimento < 50 m", "item": "lanchas"},
    1: {"nb": 21, "desc": "embarcações de 50 m a 100 m (rebocadores, apoio offshore)", "item": "rebocadores"},
    2: {"nb": 22, "desc": "embarcações de grande porte, comprimento ≥ 100 m (mercantes, graneleiros, petroleiros)", "item": "navios"},
}


def pc(x, nd=2):
    """Percentual com vírgula decimal, fora de LaTeX."""
    return f"{x:.{nd}f}".replace(".", ",") + "%"


def th(x):
    return f"{x:,}".replace(",", ".")


def nm(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def pv(p):
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def md(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def build_cnn_notebook(c, filename):
    cls = C.CLS[c]
    info = CLASS_INFO[c]
    NB = info["nb"]
    A = C.analyze(C.CNN, c)
    n, nf = A["n"], A["n_files"]
    base = A["base"]
    S = {r: R.summary(e) for r, e in C.CNN.items()}
    colt = C.collapse_table()
    tb = {r: C.sp_lowest_index_tiebreak(e) for r, e in C.CNN.items() if r != "MEL"}
    art_cnn_lofar = R.ARTICLE_TABLE7[("CNN", "LOFAR")]["SP"]
    art_cnn_mel = R.ARTICLE_TABLE7[("CNN", "MEL")]["SP"]
    frames = C.lofar_cache_frames()
    imgs = C.images_per_file()
    p_pad, p_one, p_two = (frames < 32).mean() * 100, ((frames >= 32) & (frames < 48)).mean() * 100, (frames >= 48).mean() * 100
    cfg = R.config(C.CNN["LOFAR"])["params"]
    img_s = cfg["image_windows"] * cfg["frame_step_s"]
    fam_df, fam = C.family_table(c)
    import pandas as pd
    _trn = os.path.join(R.ROOT, "results", "trainings", "tests", "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0", "eval",
                        "fold_0", "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_trn.csv")
    _trn_files = set(pd.read_csv(_trn, usecols=["File"]).File.unique())
    _fr = C.lofar_cache_frames_by_file()
    n_files_trn = sum(1 for f in _trn_files if f in _fr)
    n_img_trn = sum((1 if _fr[f] < 32 else (_fr[f] - 32) // 16 + 1) for f in _trn_files if f in _fr)
    k4 = A["consensus"][4]
    excl_sorted = sorted(A["exclusive"], key=A["exclusive"].get, reverse=True)
    champ = excl_sorted[0]
    F_vote, p_vote = A["vote_vs_best_sp"]
    F_vr, p_vr = A["vote_vs_base_recall"]
    collapsed = [r for r in C.REPS if r != "MEL"]
    nfold_bad = {r: int(colt.loc[r, "folds c/ recall < 10% em alguma classe"]) for r in C.REPS}
    rec_rng = {r: (R.fold_metrics(C.CNN[r])[f"R_{cls}"].min(), R.fold_metrics(C.CNN[r])[f"R_{cls}"].max()) for r in C.REPS}

    cells = []
    # ------------------------------------------------------------------ 0. resumo
    col_txt = "; ".join(f"**{r}** SP {nm(S[r]['SP'][0])} ± {nm(S[r]['SP'][1])}% ({nfold_bad[r]}/10 folds com recall < 10% em alguma classe)" for r in collapsed)
    cells.append(md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook {NB}: Análise de Complementaridade, Efeito Oráculo e Grau de Consenso na CNN 2D — Classe {cls}

**Autor:** Evandro Rocha (UFRJ / COPPE / PEE)
**Data:** Outubro de 2026

---

## Resumo Executivo

A classe **`{cls}`** ({info['desc']}) é analisada nas **{th(n)} instâncias de teste** acumuladas nos 10 folds do 5×2cv oficial (as mesmas **{nf} gravações** aparecem, em média, {nm(n / nf)} vezes no teste; não são {th(n)} áudios distintos), com a CNN 2D em 4 representações (**MEL, LOFAR, HHT e DEMON**).

> **Aviso principal — três das quatro CNNs colapsaram.** A CNN MEL foi treinada pelo pipeline `iara`, mas **não tem a arquitetura do artigo** (`test_scripts/cnn.py`: conv 128/64, ReLU, lr 1e-5, imagens sem sobreposição — não é a arquitetura do artigo; SP {nm(S['MEL']['SP'][0])} ± {nm(S['MEL']['SP'][1])}%; CNN MEL do artigo: {nm(art_cnn_mel[0])}%). As CNNs LOFAR, HHT e DEMON são uma **CNN ad hoc** (`scratch/generate_cnn_predictions_and_analysis.py`) treinada sobre caches com média em 5,12 s: {col_txt}. A CNN LOFAR do artigo tem SP **{nm(art_cnn_lofar[0])} ± {nm(art_cnn_lofar[1])}%** (Tabela 7). Com modelos assim, a "complementaridade" abaixo mede sobretudo a **instabilidade** desses modelos (em cada fold eles favorecem uma classe diferente), e **não diz nada sobre a informação contida nas representações** LOFAR, HHT e DEMON.

### Principais resultados (recalculados)
1. **Oráculo não é um sistema realizável.** O oráculo (acerta se ao menos uma CNN acerta) chega a **{pc(A['oracle_hits'] / n * 100)}** ({th(A['oracle_hits'])} / {th(n)}), mas o esperado se os erros fossem independentes seria **{pc(A['oracle_exp'])}**: os erros são **correlacionados**, e não ortogonais. O **voto real** das 4 CNNs (pluralidade, empate sorteado) tem recall de `{cls}` de **{pc(A['vote_recall'])}**, contra {pc(A['recall'][base])} da melhor CNN isolada ({base}; F-test 5×2cv sobre o recall, p = {pv(p_vr)}). Em SP global, o voto fica em **{nm(A['vote_sp'][0])} ± {nm(A['vote_sp'][1])}%**, abaixo da CNN {A['best_sp_rep']} isolada ({nm(A['best_sp'][0])}%; p = {pv(p_vote)}{', diferença significativa' if p_vote < 0.05 else ''}).
2. **Unanimidade:** só **{k4}** instâncias ({pc(k4 / n * 100)}) são acertadas pelas 4 CNNs, e **{A['consensus'][0]}** ({pc(A['consensus'][0] / n * 100)}) são erradas por todas. A baixa unanimidade é o esperado quando três modelos estão perto do acaso; ela não indica "especialização" de filtros.
3. **Resgates exclusivos:** {A['consensus'][1]} instâncias ({pc(A['consensus'][1] / n * 100)}) são acertadas por uma única CNN (líder: **{champ}**, {A['exclusive'][champ]}). Elas correspondem a {A['exclusive_files']} gravações, das quais só **{A['exclusive_files_always']}** são resgate exclusivo em **todas** as suas aparições no teste: os resgates são instáveis entre folds.
"""))

    cells.append(code(r'''
# 1. Configuração do Ambiente e Imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('notebooks'))
sys.path.insert(0, os.path.abspath('src'))  # módulos movidos para notebooks/src
sys.path.insert(0, os.path.abspath('notebooks/src'))
import iara_results as R
import cnn_complementaridade_comum as C

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 150
})
print("Ambiente configurado. Resultados lidos de", R.ROOT)
'''))

    # ------------------------------------------------------------------ 2. metodologia + colapso
    exps_txt = "\n".join(f"  - **{r}** — `{e}` ({'pipeline iara, `results/trainings/tests/`, `test_scripts/cnn.py` (128/64, lr 1e-5 — não é a arquitetura do artigo)' if r == 'MEL' else 'CNN ad hoc, `results/trainings/multirep/`'})" for r, e in C.CNN.items())
    tb_txt = "; ".join(f"{r}: SP {nm(S[r]['SP'][0])}% → {nm(tb[r]['SP'].mean())}%, recall `{cls}` {nm(S[r][f'R_{cls}'][0])}% → {nm(tb[r][f'R_{cls}'].mean())}%" for r in collapsed)
    cells.append(md(f"""
## 2. Metodologia e Diagnóstico dos Modelos

- **Experimentos** (predições por janela/imagem em disco, agregadas por arquivo com voto majoritário):
{exps_txt}
- **Protocolo:** 5×2cv oficial (10 folds); a partição é por **gravação** (`exclusive_header='ID'`). O mesmo navio pode aparecer em treino e teste.
- **Unidade de contagem:** "instância" = (fold, gravação de teste). Os {th(n)} casos de `{cls}` vêm de {nf} gravações distintas.
- **ACC** = acurácia balanceada (média dos recalls).

### 2.1 Por que as CNNs ad hoc colapsaram
- Cada imagem da CNN ad hoc tem {cfg['image_windows']} frames de {nm(cfg['frame_step_s'])} s, ou seja, **≈ {nm(img_s, 0)} s de áudio**, com passo de {cfg['image_stride']} frames. No cache LOFAR, **{pc(p_pad, 1)}** das gravações têm menos de 32 frames e viram uma única imagem completada por *wrap padding*; {pc(p_one, 1)} geram uma imagem sem padding; {pc(p_two, 1)} geram 2 imagens (o máximo, já que nenhuma gravação passa de {int(frames.max())} frames). Nos testes, {pc((imgs == 1).mean() * 100, 1)} dos arquivos têm 1 imagem e o restante, 2. Isso dá ≈ {th(n_img_trn)} imagens de treino no fold 0 ({n_files_trn} gravações), para uma CNN treinada do zero, por {cfg['epochs']} épocas fixas, sem validação nem *early stopping*.
- Em cada fold o modelo tende a "escolher" uma classe: o recall de `{cls}` da CNN LOFAR varia de {nm(rec_rng['LOFAR'][0], 1)}% a {nm(rec_rng['LOFAR'][1], 1)}% entre folds; o da HHT, de {nm(rec_rng['HHT'][0], 1)}% a {nm(rec_rng['HHT'][1], 1)}%; o da DEMON, de {nm(rec_rng['DEMON'][0], 1)}% a {nm(rec_rng['DEMON'][1], 1)}%. A CNN MEL do repositório fica entre {nm(rec_rng['MEL'][0], 1)}% e {nm(rec_rng['MEL'][1], 1)}%.
- **Sensibilidade ao desempate:** com 2 imagens por arquivo os empates são frequentes. Este notebook usa a regra do pipeline `iara` (`Counter.most_common`: empate → classe da primeira imagem). O script de scratch usa `argmax(bincount)` (empate → menor índice de classe, ou seja, favorece `SMALL`). Trocando a regra: {tb_txt}. Uma métrica que muda tanto com a regra de desempate não é estável o suficiente para sustentar conclusões sobre as representações.
"""))
    cells.append(code(r'''
# 2.1 Diagnóstico: métricas globais e faixa (mín–máx) por fold de cada CNN
display(C.collapse_table())
tb = pd.DataFrame({r: {'SP (iara: empate -> 1ª imagem)': R.summary(e)['SP'][0],
                       'SP (script: empate -> menor índice)': C.sp_lowest_index_tiebreak(e)['SP'].mean(),
                       'R_SMALL (iara)': R.summary(e)['R_SMALL'][0],
                       'R_SMALL (script)': C.sp_lowest_index_tiebreak(e)['R_SMALL'].mean()}
                   for r, e in C.CNN.items() if r != 'MEL'}).T
display(tb.round(2))
print('Imagens por arquivo de teste (CNN LOFAR ad hoc):', C.images_per_file().value_counts().sort_index().to_dict())
print('CNN do artigo (Tabela 7):', {k: v['SP'] for k, v in R.ARTICLE_TABLE7.items() if k[0] == 'CNN'})
'''))

    cells.append(code(f'''
# 2.2 Análise da classe {cls}: tabelas recalculadas das predições em disco
CLASSE = {c}
A = C.analyze(C.CNN, CLASSE)
df_hits, df_consensus, df_exclusive, df_reverse, df_confusion_pattern = C.tables(A, CLASSE)
total = A['n']
print(f"{{total}} instâncias de teste de {cls} ({{A['n_files']}} gravações distintas)")
display(df_hits.round(2))
display(df_consensus.round(2))
display(df_reverse.round(2))
'''))

    # ------------------------------------------------------------------ 3. oráculo
    order = sorted(C.REPS, key=lambda r: A["recall"][r], reverse=True)
    cells.append(md(f"""
## 3. Efeito Oráculo: Recall Isolado, Teto Teórico e Voto Real

- Recall isolado em `{cls}`: {", ".join(f"{r} {pc(A['recall'][r])}" for r in order)}.
- **Oráculo (≥ 1/4): {pc(A['oracle_hits'] / n * 100)}** ({th(A['oracle_hits'])} instâncias). O oráculo usa o rótulo verdadeiro para escolher o modelo e **não é um sistema realizável**: é só um limite superior.
- **Esperado com erros independentes:** {pc(A['oracle_exp'])} (1 − ∏(1 − rᵢ), média por fold). O oráculo observado fica {nm(A['oracle_exp'] - A['oracle_obs'])} p.p. abaixo: os erros das CNNs são correlacionados.
- Mesmo quatro preditores ao acaso dariam um oráculo alto (com 4 classes, ≈ 1 − 0,75⁴ ≈ 68%). Por isso o "ganho do oráculo" sobre o melhor modelo isolado ({nm(A['oracle_hits'] / n * 100 - A['recall'][base])} p.p. sobre {base}) não mede complementaridade útil.
- **Voto real** (pluralidade, empate sorteado): recall de `{cls}` = **{pc(A['vote_recall'])}** (p = {pv(p_vr)} contra {base}).
"""))
    cells.append(code(f'''
# 3. Gráfico: recall isolado vs. oráculo, esperado sob independência e voto real
plt.figure(figsize=(11, 5.5))
colors = ['#95a5a6'] * (len(df_hits) - 3) + ['#27ae60', '#7f8c8d', '#c0392b']
bars = plt.bar(df_hits['Representacao'], df_hits['Recall_Pct'], color=colors, width=0.55, edgecolor='black', linewidth=0.8)
bars[-3].set_hatch('//')
plt.axhline(A['recall'][A['base']], color='#e67e22', linestyle=':', lw=1.5, label=f"Melhor CNN isolada: {{A['base']}} ({{A['recall'][A['base']]:.1f}}%)")
plt.title('Recall CNN 2D na Classe {cls}: isolados, oráculo, esperado e voto real', fontweight='bold', pad=12)
plt.ylabel('Recall em {cls} (%)')
plt.ylim(0, 105)
plt.xticks(rotation=15, ha='right')
for bar in bars:
    h = bar.get_height()
    plt.annotate(f'{{h:.1f}}%', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.show()
'''))

    # ------------------------------------------------------------------ 4. consenso
    cons = A["consensus"]
    cells.append(md(f"""
## 4. Grau de Consenso

- **4/4 certas (unanimidade):** {cons[4]} ({pc(cons[4] / n * 100)});
- **3/4:** {cons[3]} ({pc(cons[3] / n * 100)});
- **2/4:** {cons[2]} ({pc(cons[2] / n * 100)});
- **1/4 (resgate exclusivo):** {cons[1]} ({pc(cons[1] / n * 100)});
- **0/4 (erro comum):** {cons[0]} ({pc(cons[0] / n * 100)}).

### Regras de votação
- **Unanimidade (4/4):** {pc(cons[4] / n * 100)};
- **Maioria estrita (≥ 3/4):** {pc(A['ge3'] / n * 100)} ({th(A['ge3'])});
- **"≥ 2/4":** {pc(A['ge2'] / n * 100)} ({th(A['ge2'])}). Essa contagem trata empates 2×2 como acerto, então é um oráculo parcial, não um voto;
- **Voto real (pluralidade, empate sorteado):** {pc(A['vote_recall'])};
- **Oráculo (≥ 1/4):** {pc(A['oracle_hits'] / n * 100)}.
"""))
    cells.append(code(f'''
# 4. Visualização do grau de consenso
plt.figure(figsize=(10, 5.5))
labels_cons = ['4 Entradas\\n(Unânime)', '3 Entradas\\n(Alto Consenso)', '2 Entradas\\n(Dividido)', '1 Entrada\\n(Exclusivo)', '0 Entradas\\n(Erro Comum)']
counts_cons = df_consensus['Instancias'].values
pcts_cons = df_consensus['Pct_Total'].values
bars = plt.bar(labels_cons, counts_cons, color=['#27ae60', '#2980b9', '#f39c12', '#e74c3c', '#7f8c8d'], edgecolor='black', linewidth=0.8, width=0.55)
plt.title(f'Grau de Consenso CNN 2D em {cls} ({{total}} instâncias de teste, {{A["n_files"]}} gravações)', fontweight='bold', pad=12)
plt.ylabel('Instâncias de {cls}')
plt.ylim(0, max(counts_cons) * 1.15)
for bar, pct in zip(bars, pcts_cons):
    h = bar.get_height()
    plt.annotate(f'{{int(h)}} ({{pct:.1f}}%)', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold')
plt.tight_layout()
plt.show()
'''))

    # ------------------------------------------------------------------ 5. exclusivos + confusão
    others = [C.CLS[j] for j in C.CLS if j != c]
    conf_lines = "\n".join(
        f"| **{r}** | {A['confusion'][r]['n_err']} | " + " | ".join(pc(A['confusion'][r][o]) for o in others) + " |"
        for r in C.REPS)
    cells.append(md(f"""
## 5. Resgates Exclusivos e Padrão de Erro

### Resgates exclusivos em `{cls}` (apenas uma CNN acerta)
{chr(10).join(f"- **{r}:** {A['exclusive'][r]} instâncias ({pc(A['exclusive'][r] / n * 100)})" for r in excl_sorted)}

No total são {cons[1]} instâncias, de {A['exclusive_files']} gravações. Só **{A['exclusive_files_always']}** gravações são resgate exclusivo em todas as suas aparições no teste; as demais dependem do fold. Nas CNNs colapsadas, um "resgate exclusivo" costuma ser apenas o fold em que o modelo passou a favorecer `{cls}`. Explicações mecânicas (raias espectrais, modulação de pás etc.) para esses resgates **não têm suporte** nestes dados e foram retiradas.

### Para onde vão os erros (percentual dos erros de cada CNN)
| Entrada | Erros | {' | '.join(others)} |
| :--- | ---: | {' | '.join([':---:'] * len(others))} |
{conf_lines}
"""))
    cells.append(code(f'''
# 5. Resgates exclusivos e padrão de confusão
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
bars_excl = ax1.barh(df_exclusive['Representacao'], df_exclusive['Instancias_Salvas'], color='#2980b9', height=0.5, edgecolor='black', linewidth=0.8)
ax1.set_title(f'Resgates Exclusivos CNN 2D em {cls} ({{int(df_exclusive.Instancias_Salvas.sum())}} instâncias, {{A["exclusive_files"]}} gravações)', fontweight='bold', pad=12)
ax1.set_xlabel('Instâncias salvas por apenas UMA entrada')
ax1.set_xlim(0, df_exclusive['Instancias_Salvas'].max() * 1.35)
for bar, pct in zip(bars_excl, df_exclusive['Percentual']):
    w = bar.get_width()
    ax1.annotate(f'{{int(w)}} ({{pct:.1f}}%)', xy=(w, bar.get_y() + bar.get_height()/2), xytext=(5, 0), textcoords="offset points",
                 ha='left', va='center', fontsize=9.5, fontweight='bold')

reps = df_confusion_pattern['Representacao']
cols = [c for c in df_confusion_pattern.columns if c.startswith('Confundido_')]
bottom = np.zeros(len(reps))
for col, color in zip(cols, ['#f39c12', '#2ecc71', '#95a5a6']):
    vals = df_confusion_pattern[col].values
    ax2.bar(reps, vals, 0.52, bottom=bottom, label=col.replace('Confundido_', 'Confundido com ').replace('_Pct', ''), color=color, edgecolor='black', linewidth=0.6)
    for i, v in enumerate(vals):
        if v > 4:
            ax2.text(i, bottom[i] + v/2, f'{{v:.1f}}%', ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')
    bottom += vals
ax2.set_title('Destino dos erros em {cls} por entrada (%)', fontweight='bold', pad=12)
ax2.set_ylabel('Distribuição percentual dos erros (%)')
ax2.set_ylim(0, 118)
ax2.tick_params(axis='x', rotation=12)
ax2.legend(loc='upper right', frameon=True, fontsize=8.5)
plt.tight_layout()
plt.show()
'''))

    # ------------------------------------------------------------------ 6. efeito reverso
    rev_lines = "\n".join(
        f"- **Trocar para {r}:** perde {v['lose']} acertos do {base} ({pc(v['lose_pct'])}) e resgata {v['rescue']} de seus erros ({pc(v['rescue_pct'])}); saldo **{v['net']:+d}**."
        for r, v in A["reverse"].items())
    cells.append(md(f"""
## 6. Efeito Reverso: o Risco de Trocar de Entrada

Tomando como base a melhor CNN isolada em `{cls}`, **{base}** ({A['hits'][base]} acertos e {n - A['hits'][base]} erros):
{rev_lines}

Nenhuma troca estática de entrada tem saldo positivo. Lembre-se, porém, de que três dos modelos de destino estão colapsados: o saldo negativo reflete a qualidade desses modelos, e não um limite das representações.
"""))
    cells.append(code(f'''
# 6. Visualização do efeito reverso a partir da melhor CNN isolada
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(df_reverse))
width = 0.35
lose_col = [c for c in df_reverse.columns if c.startswith('Perde_do_')][0]
resc_col = [c for c in df_reverse.columns if c.startswith('Resgata_Erros_')][0]
b1 = ax.bar(x - width/2, df_reverse[lose_col], width, label=f"Perde acertos do {{A['base']}}", color='#c0392b', edgecolor='black')
b2 = ax.bar(x + width/2, df_reverse[resc_col], width, label=f"Resgata erros do {{A['base']}}", color='#27ae60', edgecolor='black')
ax.set_title(f"Efeito reverso na CNN a partir de {{A['base']}} ({cls})", fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(df_reverse['Destino_Troca'])
ax.set_ylabel('Instâncias')
ax.set_ylim(0, max(df_reverse[lose_col].max(), df_reverse[resc_col].max()) * 1.2)
for bars, sign, color in [(b1, '-', '#c0392b'), (b2, '+', '#27ae60')]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{{sign}}{{int(h)}}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
ax.legend(frameon=True)
plt.tight_layout()
plt.show()
'''))

    # ------------------------------------------------------------------ 7. figura
    cells.append(md(f"""
## 7. Figura Consolidada

A figura `reports/analise_complementaridade_cnn_{cls.lower()}_oraculo.png`, gerada por `scratch/generate_cnn_predictions_and_analysis.py`, **não é mais exibida aqui**. Ela usa a regra de desempate do script (menor índice de classe), e por isso os números não batem com os deste notebook (Seção 2.1); além disso, os títulos dela falam em "Áudios Únicos" quando os valores são instâncias de fold. Os quatro painéis da figura estão nas Seções 3–5 acima, recalculados.
"""))

    # ------------------------------------------------------------------ 8. comparação entre famílias
    fam_txt = []
    for f, a in fam.items():
        nn = a["n"]
        fam_txt.append(f"| **{f}** (NB {C.FAMILY_NB[f][c]}) | {pc(a['recall'][a['base']])} ({a['base']}) | {pc(a['oracle_hits'] / nn * 100)} / {pc(a['oracle_exp'])} | {pc(a['consensus'][4] / nn * 100)} | {pc(a['consensus'][0] / nn * 100)} | {pc(a['ge2'] / nn * 100)} | {pc(a['vote_recall'])} | {nm(a['vote_sp'][0])} vs. {nm(a['best_sp'][0])} ({a['best_sp_rep']}), p = {pv(a['vote_vs_best_sp'][1])} |")
    cnnf = fam["CNN 2D"]
    others_f = {f: a for f, a in fam.items() if f != "CNN 2D"}
    sig_better = [f for f, a in fam.items() if a["vote_vs_best_sp"][1] < 0.05 and a["vote_sp"][0] > a["best_sp"][0]]
    vote_txt = ("Em nenhuma família o voto real das 4 entradas supera, com significância (F-test 5×2cv, p < 0,05), o melhor modelo isolado em SP global."
                if not sig_better else "O voto real supera significativamente o melhor isolado em: " + ", ".join(sig_better) + ".")
    max_unan_other = max(a["consensus"][4] / a["n"] * 100 for a in others_f.values())
    cells.append(md(f"""
## 8. Comparativo entre Famílias em `{cls}`: SVM vs. MLP vs. RF vs. CNN 2D

Experimentos: SVM — MEL m=4000, LOFAR m=6000, HHT/DEMON m=4000; MLP — `mlp`, `mlp_lofar`, `mlp_hht_h32-16_e30_4c`, `mlp_demon_h32-16_e30_4c`; RF — `rf_{{mel5s,lofar5s,hht,demon}}_n100_balanced_4c`; CNN — como acima. A lista completa está em `C.FAMILIES`.

| Família | Melhor isolado (recall) | Oráculo obs. / esperado indep. | Unanimidade 4/4 | Erro comum 0/4 | "≥2/4" (empates = acerto) | Voto real (recall) | SP global: voto vs. melhor isolado (F-test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
{chr(10).join(fam_txt)}

**Leitura:**
- {vote_txt} Na CNN, o voto é {'**significativamente pior**' if cnnf['vote_vs_best_sp'][1] < 0.05 else 'pior, mas sem significância,'} ({nm(cnnf['vote_sp'][0])}% vs. {nm(cnnf['best_sp'][0])}%, p = {pv(cnnf['vote_vs_best_sp'][1])}), o que é coerente com 3 dos 4 votantes estarem colapsados.
- A unanimidade da CNN ({pc(cnnf['consensus'][4] / cnnf['n'] * 100)}) é bem menor que a das outras famílias (até {pc(max_unan_other)}). Isso decorre do colapso das CNNs ad hoc, e não de uma "ortogonalidade" das representações.
- Em todas as famílias o oráculo observado fica abaixo do esperado sob independência (erros correlacionados).
"""))
    cells.append(code(f'''
# 8. Tabela comparativa recalculada (todas as famílias, classe {cls})
fam_df, _ = C.family_table(CLASSE)
display(fam_df)
'''))

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


def create_notebook(filename="notebooks/22_analise_complementaridade_cnn_small.ipynb"):
    build_cnn_notebook(0, filename)


if __name__ == "__main__":
    create_notebook()
