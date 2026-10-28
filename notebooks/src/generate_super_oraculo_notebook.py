"""
Script de geração do notebook 27: notebooks/27_analise_falhas_super_oraculo.ipynb
Análise das instâncias (arquivo x fold) em que nenhum dos 16 especialistas (4 famílias x 4 representações) acerta.

Fontes: data/iara_processed/{rf,mlp,svm,cnn}_all_representations_predictions.pkl (predições por arquivo dos 16
especialistas, as mesmas do gate do NB26), cache_gating_metafeatures.pkl, src/iara/dataset_info/iara.csv e
results_heterogeneous_gating.pkl. Todos os números do markdown são calculados aqui, na geração
(notebooks/gating_comum.py). Rodar na raiz do repositório:
    python notebooks/generate_super_oraculo_notebook.py
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gating_comum as G  # noqa: E402
from gating_comum import br, p_txt, pm  # noqa: E402


def md(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]}


def code(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]}


# Grupos de confusão (definidos pela classe real e pela predição de consenso dos 16 especialistas)
GROUPS = [
    ("A1", "LARGE com consenso SMALL", "(df_fail['classe_real'] == 'Large') & (df_fail['predicao_dominante'] == 'Small')"),
    ("A2", "SMALL com consenso LARGE", "(df_fail['classe_real'] == 'Small') & (df_fail['predicao_dominante'] == 'Large')"),
    ("A3", "Navio com consenso BACKGROUND", "(df_fail['classe_real'] != 'Background') & (df_fail['predicao_dominante'] == 'Background')"),
    ("A4", "MEDIUM com consenso LARGE", "(df_fail['classe_real'] == 'Medium') & (df_fail['predicao_dominante'] == 'Large')"),
]


def ships(d):
    s = d["ship_id"].astype(str).str.strip()
    return s[s != "-"].nunique()


def compute():
    df_fail, total = G.unanimous_failures()
    sup = G.oracle(G.ALL16)
    # acurácias do Super-Oráculo por fold
    bal, acc = [], []
    for k in range(10):
        fids, y = G.fold_arrays(k)
        H = (G.experts_preds(G.ALL16, k, fids) == y[None, :]).any(0)
        bal.append(np.mean([H[y == c].mean() for c in range(4)]) * 100)
        acc.append(H.mean() * 100)
    med = G.class_medians()
    grp = {}
    for gid, _, expr in GROUPS:
        d = df_fail[eval(expr)]
        grp[gid] = {"n": len(d), "rec": d["file_id"].nunique(), "ships": ships(d),
                    "Lmin": d["comprimento_m"].min(), "Lmax": d["comprimento_m"].max(),
                    "types": d.drop_duplicates("file_id")["tipo_embarcacao"].str.strip().value_counts(),
                    "hlr": d["high_low_ratio"].median(), "rms": d["rms_db"].median(),
                    "hlr_5_19": int(((d["high_low_ratio"] >= 5) & (d["high_low_ratio"] <= 19.5)).sum()),
                    "real": d["classe_real"].value_counts()}
    het = pickle.load(open(os.path.join(G.PROC, "results_heterogeneous_gating.pkl"), "rb"))
    mlp_lofar = G.single_folds("MLP", "lofar")[0]
    hfix = G.gate_folds(G.FIXED_HMOE)[0]
    hauto = G.hmoe_auto_folds()
    assert abs(float(str(het["df"].iloc[4]["SP (%)"]).split()[0]) - hfix.mean()) < 0.006
    assert abs(float(str(het["df"].iloc[5]["SP (%)"]).split()[0]) - sup["sp"].mean()) < 0.006
    return {"df": df_fail, "total": total, "sup": sup, "bal": np.array(bal), "acc": np.array(acc), "med": med,
            "grp": grp, "hfix": hfix, "hauto": hauto, "mlp_lofar": mlp_lofar,
            "t_fix": G.f_test_5x2(hfix, mlp_lofar), "t_auto": G.f_test_5x2(hauto, mlp_lofar),
            "counts": G.dataset_counts()}


def create_notebook(filename="notebooks/27_analise_falhas_super_oraculo.ipynb"):
    X = compute()
    df, total, sup, med, grp = X["df"], X["total"], X["sup"], X["med"], X["grp"]
    n = len(df)
    n_rec = df["file_id"].nunique()
    n_ship = ships(df)
    n_bg_rec = df[df["classe_real"] == "Background"]["file_id"].nunique()
    n_tie = int(df["empate_no_consenso"].sum())
    rand16 = G.random_oracle_sp(16)
    covered = sum(grp[g]["n"] for g in grp)
    types_all = df.drop_duplicates("file_id")["tipo_geral"].str.strip().value_counts()
    real_counts = df["classe_real"].value_counts()

    def types_txt(g, k=3):
        t = grp[g]["types"].head(k)
        return ", ".join(f"{name} ({cnt})" for name, cnt in t.items())

    def L(g):
        return f"{br(grp[g]['Lmin'], 0)}–{br(grp[g]['Lmax'], 0)} m"

    def gline(g, label):
        a = grp[g]
        return f"{a['n']} instâncias ({a['rec']} gravações, {a['ships']} navios)"

    cells = []
    cells.append(md(f"""
# Análise das Falhas Unânimes do Super-Oráculo – Dataset IARA
### Notebook 27: Instâncias em que nenhum dos 16 especialistas acerta

No estudo de roteamento dinâmico (Gating, Notebook 26) sobre o dataset **IARA** (*Silva et al.*, IEEE Access 2025), o **Super-Oráculo** combina 16 especialistas (4 representações espectrais $\\times$ 4 famílias de classificadores) e conta acerto quando **pelo menos um** dos 16 acerta. Ele usa o rótulo verdadeiro para escolher o especialista: **não é um sistema realizável**. Nos 10 folds do $5\\times2\\text{{cv}}$:
* **Índice SP:** {pm(sup['sp'])}%
* **Acurácia Global (micro):** {pm(X['acc'])}%
* **Acurácia Balanceada:** {pm(X['bal'])}%

> **Referências de acaso.** Um oráculo de 16 preditores **uniformemente aleatórios** (4 classes) teria SP $= 1-(3/4)^{{16}} = {br(rand16)}\\%$, e o esperado se os erros dos 16 especialistas fossem independentes é {br(sup['sp_exp'], 3)}%. O valor observado fica {'abaixo de ambos' if sup['sp'].mean() < min(rand16, sup['sp_exp']) else 'próximo deles'}: com 16 tentativas por áudio, um oráculo perto de 100% é esperado mesmo sem complementaridade real, e os erros dos especialistas são correlacionados. O SP do Super-Oráculo não é um "teto" de desempenho.

Em **{br(total - n)} das {br(total)} avaliações de teste** (arquivo × fold) pelo menos um especialista acertou. O split do protocolo é por gravação (`ID`), não por navio: o mesmo navio pode ter gravações em treino e teste.

Nas outras **{n} instâncias ({br(n / total * 100)}%)**, **todos os 16 especialistas erraram**. Como cada gravação aparece em vários folds de teste, essas {n} instâncias correspondem a **{n_rec} gravações distintas**: {n_rec - n_bg_rec} gravações de **{n_ship} navios** e {n_bg_rec} gravações de *Background*.

### Objetivo deste Notebook:
1. **Identificar** cada uma dessas gravações cruzando o `ID` com os metadados do IARA (`iara.csv`: nome, tipo AIS, comprimento, estado do mar);
2. **Mostrar o que os 16 especialistas previram** em consenso e comparar com a classe real;
3. **Descrever** os descritores do sinal (potência RMS, razão de energia alta/baixa frequência, índice de modulação) dessas gravações em relação às medianas de cada classe;
4. Agrupar as falhas nos **4 principais grupos de confusão**.
"""))
    cells.append(code("""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

try:
    from IPython.display import display, HTML
except ImportError:
    display = print

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.abspath('src'))  # módulos movidos para notebooks/src
sys.path.insert(0, os.path.abspath('notebooks/src'))
import gating_comum as G   # predições por arquivo dos 16 especialistas, meta-features e iara.csv

# Configuração de estilo visual (sem dependência externa de seaborn)
if 'seaborn-v0_8-whitegrid' in plt.style.available:
    plt.style.use('seaborn-v0_8-whitegrid')
elif 'seaborn-whitegrid' in plt.style.available:
    plt.style.use('seaborn-whitegrid')
else:
    plt.style.use('default')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 120
})

print("Ambiente configurado com sucesso.")
"""))
    cells.append(md("""
## 1. Carregamento dos Dados de Predição dos 16 Especialistas e Metadados do IARA

Carregamos os 10 folds de validação cruzada para as 4 famílias de modelos nas 4 representações:
* **Modelos:** Random Forest (RF), Multi-Layer Perceptron (MLP), Support Vector Machine (SVM Nyström RBF) e CNN 2D.
* **Representações:** MEL, LOFAR, HHT e DEMON.
* **Total:** $4 \\times 4 = 16$ especialistas para cada áudio. A procedência de cada especialista (pipeline oficial ou script ad hoc, especialistas colapsados) está na Seção 3.1 do Notebook 26.

Cruzamos cada áudio com:
1. `src/iara/dataset_info/iara.csv` (nome do navio, comprimento, tipo AIS, estado do mar, vento e chuva);
2. `data/iara_processed/cache_gating_metafeatures.pkl` (7 descritores acústicos dos primeiros 10 s).

A predição de consenso é a classe mais votada entre os 16 (`Counter.most_common`; em empate, a do primeiro especialista na ordem RF, SVM, MLP, CNN × MEL, LOFAR, HHT, DEMON).
"""))
    cells.append(code("""
df_fail, total_evaluations = G.unanimous_failures()
ship_ids = df_fail['ship_id'].astype(str).str.strip()

print(f"Total de avaliações de teste (arquivo x fold, 10 folds 5x2cv): {total_evaluations}")
print(f"Instâncias em que TODOS os 16 especialistas erraram: {len(df_fail)} ({len(df_fail)/total_evaluations*100:.2f}%)")
print(f"  = {df_fail['file_id'].nunique()} gravações distintas; {ship_ids[ship_ids != '-'].nunique()} navios (Ship ID)")
print(f"  instâncias com empate na predição de consenso: {int(df_fail['empate_no_consenso'].sum())}")
print(f"Acertos do Super-Oráculo: {total_evaluations - len(df_fail)} ({((total_evaluations - len(df_fail))/total_evaluations)*100:.2f}%)")

sup = G.oracle(G.ALL16)
print(f"\\nSuper-Oráculo SP: {sup['sp'].mean():.2f} ± {sup['sp'].std():.2f}")
print(f"  esperado com erros independentes: {sup['sp_exp']:.3f} | oráculo de 16 aleatórios: {G.random_oracle_sp(16):.2f}")
"""))
    cells.append(md(f"""
## 2. Visão Estatística Global das Falhas Unânimes

Abaixo analisamos a classe real das {n} instâncias e a **predição de consenso dos 16 especialistas** (a classe para a qual a maioria dos modelos convergiu). {n_tie} instâncias têm empate no consenso.
"""))
    cells.append(code("""
cm_consensus = pd.crosstab(
    df_fail['classe_real'],
    df_fail['predicao_dominante'],
    rownames=['Classe Real'],
    colnames=['Predição de Consenso (16 Especialistas)'],
    margins=True
)

print(f"=== DISTRIBUIÇÃO DAS CLASSES REAIS NAS {len(df_fail)} INSTÂNCIAS DE FALHA UNÂNIME ===")
dist = df_fail.groupby('classe_real').agg(**{'Instâncias': ('file_id', 'size'), 'Gravações distintas': ('file_id', 'nunique')})
display(dist.sort_values('Instâncias', ascending=False))

print("\\n=== MATRIZ DE CONFUSÃO DO CONSENSO (instâncias) ===")
display(cm_consensus)
"""))

    def grp_md(g, title):
        a = grp[g]
        return a, title

    a1, a2, a3, a4 = grp["A1"], grp["A2"], grp["A3"], grp["A4"]
    m = med
    cells.append(md(f"""
## 3. Os 4 Principais Grupos de Confusão

Os quatro grupos abaixo somam {covered} das {n} instâncias. As descrições vêm das tabelas da próxima célula; os valores de referência são as **medianas por classe** sobre as {br(X['counts']['n_uo'])} gravações do UO (`high_low_ratio`: SMALL {br(m.loc['Small', 'high_low_ratio'], 3)}, MEDIUM {br(m.loc['Medium', 'high_low_ratio'], 3)}, LARGE {br(m.loc['Large', 'high_low_ratio'], 3)}, BACKGROUND {br(m.loc['Background', 'high_low_ratio'], 3)}; `rms_db`: SMALL {br(m.loc['Small', 'rms_db'], 1)}, MEDIUM {br(m.loc['Medium', 'rms_db'], 1)}, LARGE {br(m.loc['Large', 'rms_db'], 1)}, BACKGROUND {br(m.loc['Background', 'rms_db'], 1)} dB).

> As causas físicas (propulsores auxiliares, regime de máquina, distância de passagem etc.) **não podem ser verificadas** com os dados disponíveis (não há registro de manobra, rotação ou potência de máquina) e por isso não são afirmadas aqui.

---

### Grupo 1: Real = LARGE $\\rightarrow$ Consenso = SMALL — {gline('A1', '')}
* Comprimentos de {L('A1')}. Tipos mais frequentes (por gravação): {types_txt('A1')}.
* `high_low_ratio` mediano de {br(a1['hlr'], 3)} (mediana de LARGE: {br(m.loc['Large', 'high_low_ratio'], 3)}); apenas {a1['hlr_5_19']} das {a1['n']} instâncias têm razão entre 5 e 19. `rms_db` mediano de {br(a1['rms'], 1)} dB.

---

### Grupo 2: Real = SMALL $\\rightarrow$ Consenso = LARGE — {gline('A2', '')}
* Comprimentos de {L('A2')}. Tipos mais frequentes (por gravação): {types_txt('A2')}.
* `high_low_ratio` mediano de {br(a2['hlr'], 3)} (mediana de SMALL: {br(m.loc['Small', 'high_low_ratio'], 3)}; de LARGE: {br(m.loc['Large', 'high_low_ratio'], 3)}). `rms_db` mediano de {br(a2['rms'], 1)} dB.

---

### Grupo 3: Real = NAVIO $\\rightarrow$ Consenso = BACKGROUND — {gline('A3', '')}
* Classes reais: {', '.join(f"{k.upper()} {v}" for k, v in a3['real'].items())} instâncias.
* `rms_db` mediano de {br(a3['rms'], 1)} dB: abaixo das medianas das classes de navio e acima da mediana de BACKGROUND ({br(m.loc['Background', 'rms_db'], 1)} dB). Ou seja, sinais mais fracos que o típico de navios.

---

### Grupo 4: Real = MEDIUM $\\rightarrow$ Consenso = LARGE — {gline('A4', '')}
* Comprimentos de {L('A4')}. Tipos mais frequentes (por gravação): {types_txt('A4')}.
* `high_low_ratio` mediano de {br(a4['hlr'], 3)} (mediana de MEDIUM: {br(m.loc['Medium', 'high_low_ratio'], 3)}; de LARGE: {br(m.loc['Large', 'high_low_ratio'], 3)}). `rms_db` mediano de {br(a4['rms'], 1)} dB.
"""))
    group_code = "\n".join(
        f"""
print("\\n" + "=" * 90)
print("{gid}: {title}")
print("=" * 90)
d = df_fail[{expr}]
print(f"{{len(d)}} instâncias, {{d['file_id'].nunique()}} gravações distintas")
display(d.drop_duplicates('file_id')[cols_view].head(10))
display(d[['comprimento_m', 'high_low_ratio', 'rms_db', 'mod_index']].describe().round(3))"""
        for gid, title, expr in GROUPS)
    cells.append(code(f"""
# Casos por grupo (uma linha por gravação) e resumo dos descritores
cols_view = ['file_id', 'nome_navio', 'classe_real', 'tipo_embarcacao', 'comprimento_m', 'high_low_ratio', 'rms_db', 'estado_mar']
{group_code}

print("\\nMedianas por classe sobre as gravações do UO (referência):")
display(G.class_medians()[['high_low_ratio', 'rms_db', 'mod_index']].round(3))
"""))
    cells.append(md(f"""
## 4. Visualização Gráfica das Falhas Unânimes

Painel de 4 vias:
1. **(A) Classe real vs. predição de consenso** (instâncias);
2. **(B) `high_low_ratio` vs. `rms_db`** das instâncias de falha, com as medianas de cada classe no UO (×);
3. **(C) Distribuição de comprimento** (AIS) dos navios nas falhas, com os limites de classe de 50 m e 100 m;
4. **(D) Categorias AIS** (por gravação distinta): {', '.join(f"{k} ({v})" for k, v in types_all.head(3).items())} são as mais frequentes.
"""))
    cells.append(code("""
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Gráfico de Barras: Classes Reais vs. Predições de Consenso
classes = ['Small', 'Medium', 'Large', 'Background']
true_counts = [len(df_fail[df_fail['classe_real'] == c]) for c in classes]
pred_counts = [len(df_fail[df_fail['predicao_dominante'] == c]) for c in classes]

x = np.arange(len(classes))
w = 0.35
axes[0, 0].bar(x - w/2, true_counts, w, label='Classe Real', color='#2b5c8f', alpha=0.9, edgecolor='black')
axes[0, 0].bar(x + w/2, pred_counts, w, label='Predição de Consenso dos Modelos', color='#e07a5f', alpha=0.9, edgecolor='black')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(classes, fontweight='bold')
axes[0, 0].set_ylabel('Número de Instâncias (arquivo x fold)', fontweight='bold')
axes[0, 0].set_title('(A) Comparativo: Classe Real vs. Predição de Consenso', fontweight='bold')
axes[0, 0].legend(frameon=True, facecolor='white')
for i in range(len(classes)):
    axes[0, 0].text(x[i] - w/2, true_counts[i] + 1.0, str(true_counts[i]), ha='center', fontweight='bold')
    axes[0, 0].text(x[i] + w/2, pred_counts[i] + 1.0, str(pred_counts[i]), ha='center', fontweight='bold')

# 2. Dispersão: high_low_ratio vs. rms_db (+ medianas de cada classe no UO)
color_map = {'Large': '#d90429', 'Medium': '#f4a261', 'Small': '#2a9d8f', 'Background': '#264653'}
med = G.class_medians()
for c in classes:
    sub = df_fail[df_fail['classe_real'] == c]
    axes[0, 1].scatter(sub['rms_db'], sub['high_low_ratio'], label=f'Real: {c}', color=color_map[c],
                       alpha=0.85, s=65, edgecolor='black', linewidth=0.6)
    axes[0, 1].scatter(med.loc[c, 'rms_db'], med.loc[c, 'high_low_ratio'], marker='X', s=220,
                       color=color_map[c], edgecolor='black', linewidth=1.2)
axes[0, 1].set_yscale('log')
axes[0, 1].set_xlabel('Potência RMS (dB, primeiros 10 s)', fontweight='bold')
axes[0, 1].set_ylabel('Razão Alta/Baixa Frequência (Escala Log)', fontweight='bold')
axes[0, 1].set_title('(B) rms_db vs. high_low_ratio (× = mediana da classe no UO)', fontweight='bold')
axes[0, 1].axhline(1.0, color='gray', linestyle='--', alpha=0.7)
axes[0, 1].legend(frameon=True, facecolor='white', loc='upper left')

# 3. Histograma de Comprimentos dos Navios nas Falhas
lengths = df_fail['comprimento_m'].dropna()
axes[1, 0].hist(lengths, bins=25, color='#3d405b', alpha=0.85, edgecolor='black')
axes[1, 0].axvline(50, color='#e76f51', linestyle='--', linewidth=1.5, label='Limite Small / Medium (50m)')
axes[1, 0].axvline(100, color='#2a9d8f', linestyle='--', linewidth=1.5, label='Limite Medium / Large (100m)')
axes[1, 0].set_xlabel('Comprimento do Navio pelo AIS (metros)', fontweight='bold')
axes[1, 0].set_ylabel('Instâncias de Falha', fontweight='bold')
axes[1, 0].set_title('(C) Distribuição de Comprimento dos Navios que Falharam', fontweight='bold')
axes[1, 0].legend(frameon=True, facecolor='white')

# 4. Distribuição por Categoria de Embarcação (SHIPTYPE), por gravação distinta
types = df_fail.drop_duplicates('file_id')['tipo_geral'].str.strip().value_counts().head(6)
y_pos = np.arange(len(types))
axes[1, 1].barh(y_pos, types.values, color='#457b9d', alpha=0.9, edgecolor='black')
axes[1, 1].set_yticks(y_pos)
axes[1, 1].set_yticklabels(types.index, fontweight='bold')
axes[1, 1].invert_yaxis()
axes[1, 1].set_xlabel('Gravações distintas com falha unânime', fontweight='bold')
axes[1, 1].set_title('(D) Principais Categorias AIS nas Falhas', fontweight='bold')
for i, v in enumerate(types.values):
    axes[1, 1].text(v + 0.5, i, str(v), va='center', fontweight='bold')

plt.tight_layout()
plt.show()
"""))
    cells.append(md(f"""
## 5. Tabela Completa das {n} Instâncias com Falha Unânime

Tabela com todas as instâncias (uma linha por arquivo × fold; {n_rec} gravações distintas). Pode ser filtrada e ordenada pelo nome do navio, classe real, comprimento ou descritores.
"""))
    cells.append(code("""
cols_full = [
    'file_id', 'fold', 'nome_navio', 'ship_id', 'classe_real', 'predicao_dominante',
    'votos_dominantes', 'empate_no_consenso', 'tipo_embarcacao', 'comprimento_m',
    'high_low_ratio', 'crest_factor', 'rms_db', 'estado_mar'
]
df_full_display = df_fail[cols_full].sort_values(['classe_real', 'nome_navio']).reset_index(drop=True)
print(f"Exibindo as primeiras 25 linhas de um total de {len(df_full_display)} instâncias:")
display(df_full_display.head(25))
"""))
    hF, hp = X["t_fix"]
    aF, ap = X["t_auto"]
    cells.append(md(f"""
## 6. Conclusões

1. **O SP de {br(sup['sp'].mean())}% do Super-Oráculo não é um teto físico:**
   * É a fração de áudios em que *algum* de 16 especialistas acerta, escolhido com o rótulo. Com 16 tentativas, preditores aleatórios já chegariam a {br(rand16)}%, e erros independentes dariam {br(sup['sp_exp'], 3)}%. O valor observado, abaixo dessas referências, indica erros **correlacionados** entre os especialistas, e não um limite imposto pela física do sinal.
   * As {n} instâncias ({br(n / total * 100)}%) em que todos erram vêm de {n_rec} gravações; a mesma gravação costuma reaparecer em vários folds. Elas mostram onde os 16 especialistas concordam no erro, mas não provam que essas gravações sejam "inclassificáveis".
2. **Comprimento AIS vs. assinatura acústica (hipótese):**
   * As classes são definidas por comprimento (< 50 m, 50–100 m, > 100 m). Nos grupos 2 e 4, embarcações de {L('A2')} (principalmente {grp['A2']['types'].index[0]}) e de {L('A4')} (principalmente {grp['A4']['types'].index[0]}) foram classificadas como LARGE. Isso é compatível com a hipótese de que o comprimento nem sempre se reflete na assinatura acústica, mas os dados disponíveis não permitem testá-la (não há potência de máquina, rotação ou regime de operação).
3. **Relação com o Gating (Notebook 26):**
   * Os sistemas de gating sem vazamento ficam em SP {pm(X['hfix'])} (H-MoE especializado) e {pm(X['hauto'])} (H-MoE automático), contra {pm(X['mlp_lofar'])} do MLP-LOFAR isolado (F-test 5×2cv: {p_txt(hp)} e {p_txt(ap)}). Nenhum deles se aproxima do Super-Oráculo, e a distância entre eles e o oráculo não deve ser lida como "espaço a ser conquistado": boa parte dela vem do próprio mecanismo do oráculo.
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
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1, ensure_ascii=False)
    print(f"Notebook '{filename}' criado com sucesso!")


if __name__ == "__main__":
    create_notebook()
