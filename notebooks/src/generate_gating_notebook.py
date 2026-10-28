"""
Script de geração do notebook 26: notebooks/26_sistema_gating_frontend.ipynb
Sistema de Gating Front-End (roteamento de entradas por meta-features físicas) sob o protocolo 5x2cv.

Fontes (nenhum número do markdown é digitado; todos são calculados aqui, na geração):
  * data/iara_processed/results_competence_gating_{all_models,svm}.pkl e results_heterogeneous_gating.pkl
    (scratch/test_competence_gating_all_models.py, test_competence_gating_svm.py,
    run_heterogeneous_gating_experiment.py; roteador treinado só em fold ^ 1);
  * notebooks/gating_comum.py, que reproduz o roteador a partir das predições por arquivo
    data/iara_processed/{rf,mlp,svm,cnn}_all_representations_predictions.pkl e devolve o SP por fold
    (F-test 5x2cv e sensibilidade aos hiperparâmetros);
  * notebooks/iara_results.py (SP dos experimentos isolados recalculado dos CSVs por janela).

Rodar na raiz do repositório:  python notebooks/generate_gating_notebook.py   (~5 min: a grade de 18 roteadores)
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gating_comum as G  # noqa: E402
from gating_comum import br, p_txt, pm  # noqa: E402

FAM_LABEL = {"RF": "Random Forest", "MLP": "MLP", "SVM": "SVM (Nyström)", "CNN": "CNN 2D"}
REP_UP = {r: r.upper() for r in G.REPS}


def md(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]}


def code(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]}


def _sp_str(s):
    return float(str(s).split()[0])


def compute():
    """Todos os números usados no markdown."""
    c = G.dataset_counts()
    fam = {}
    for f in G.FAMS:
        base = {r: G.single_folds(f, r) for r in G.REPS}
        g_sp, g_rec, g_rt = G.gate_folds(G.homog(f))
        best = max(G.REPS, key=lambda r: base[r][0].mean())
        F, p = G.f_test_5x2(g_sp, base[best][0])
        unstable = [r for r in G.REPS if base[r][0].std() > G.COLLAPSE_STD]
        fam[f] = {"base": base, "gate": g_sp, "rec": g_rec, "routes": g_rt, "best": best, "F": F, "p": p,
                  "unstable": unstable, "oracle": G.oracle(G.homog(f))}
    mlp_lofar = fam["MLP"]["base"]["lofar"][0]
    for f in fam:
        fam[f]["F_ml"], fam[f]["p_ml"] = G.f_test_5x2(fam[f]["gate"], mlp_lofar)

    # Conferência com os pickles gerados pelos scripts de scratch/ (detecta pickle desatualizado)
    proc = G.PROC
    res_all = pickle.load(open(os.path.join(proc, "results_competence_gating_all_models.pkl"), "rb"))
    res_svm = pickle.load(open(os.path.join(proc, "results_competence_gating_svm.pkl"), "rb"))
    het = pickle.load(open(os.path.join(proc, "results_heterogeneous_gating.pkl"), "rb"))
    pk = {"RF": res_all["RANDOM FOREST"], "MLP": res_all["MLP"], "SVM": res_svm, "CNN": res_all["CNN 2D"]}
    for f in fam:
        assert abs(_sp_str(pk[f]["df"].iloc[4]["SP (%)"]) - fam[f]["gate"].mean()) < 0.006, f
        for i, r in enumerate(G.REPS):
            assert abs(_sp_str(pk[f]["df"].iloc[i]["SP (%)"]) - fam[f]["base"][r][0].mean()) < 0.006, (f, r)

    hfix_sp, hfix_rec, _ = G.gate_folds(G.FIXED_HMOE)
    hauto_sp = G.hmoe_auto_folds()
    assert abs(_sp_str(het["df"].iloc[4]["SP (%)"]) - hfix_sp.mean()) < 0.006
    assert abs(_sp_str(het["df"].iloc[3]["SP (%)"]) - hauto_sp.mean()) < 0.006
    sup = G.oracle(G.ALL16)
    assert abs(_sp_str(het["df"].iloc[5]["SP (%)"]) - sup["sp"].mean()) < 0.006

    grid = {}
    for name, ex in [*[(f, G.homog(f)) for f in G.FAMS], ("H-MoE", G.FIXED_HMOE)]:
        v = {cfg: G.gate_folds(ex, depth=cfg[0], leaf=cfg[1], w_prior=cfg[2])[0].mean() for cfg in G.GRID}
        rep = v[G.REPORTED_CFG]
        grid[name] = {"rep": rep, "rank": 1 + sum(x > rep + 1e-9 for x in v.values()), "min": min(v.values()),
                      "max": max(v.values()), "med": float(np.median(list(v.values())))}

    all_single = {(f, r): fam[f]["base"][r][0].mean() for f in fam for r in G.REPS}
    best_single = max(all_single, key=all_single.get)
    return {"c": c, "fam": fam, "hfix": hfix_sp, "hfix_rec": hfix_rec, "hauto": hauto_sp,
            "hfix_test": G.f_test_5x2(hfix_sp, mlp_lofar), "hauto_test": G.f_test_5x2(hauto_sp, mlp_lofar),
            "hfix_vs_rf": G.f_test_5x2(hfix_sp, fam["RF"]["gate"]),
            "sup": sup, "grid": grid, "best_single": best_single, "mlp_lofar": mlp_lofar}


def create_gating_notebook(filename="notebooks/26_sistema_gating_frontend.ipynb"):
    X = compute()
    c, fam, grid, sup = X["c"], X["fam"], X["grid"], X["sup"]
    bs_f, bs_r = X["best_single"]
    n_sig = sum(fam[f]["p"] < 0.05 for f in fam)
    rf, mlp, svm, cnn = fam["RF"], fam["MLP"], fam["SVM"], fam["CNN"]
    unstable_txt = "; ".join(f"{f} {', '.join(REP_UP[r] for r in fam[f]['unstable'])}"
                             for f in fam if fam[f]["unstable"]) or "nenhum"

    def gtxt(f):
        a = fam[f]
        return f"{pm(a['gate'])}"

    def fam_line(f):
        a = fam[f]
        b = a["base"][a["best"]][0]
        d = a["gate"].mean() - b.mean()
        return (f"| {FAM_LABEL[f]} | {gtxt(f)} | {REP_UP[a['best']]} {pm(b)} | {'+' if d >= 0 else '−'}{br(abs(d))} | "
                f"{br(a['F'])} | {br(a['p'], 2)} |")

    def grid_line(n):
        g = grid[n]
        return (f"| {FAM_LABEL.get(n, n + ' especializado')} | {br(g['rep'])} | {g['rank']}ª de 18 | "
                f"{br(g['min'])}–{br(g['max'])} | {br(g['med'])} |")

    rt = {f: fam[f]["routes"] / fam[f]["routes"].sum() * 100 for f in fam}
    rf_base = rf["base"]
    rf_lof_small = rf_base["lofar"][1][0]
    sup_rand16 = G.random_oracle_sp(16)
    hfix_F, hfix_p = X["hfix_test"]
    hauto_F, hauto_p = X["hauto_test"]
    hrf_F, hrf_p = X["hfix_vs_rf"]
    ml = X["mlp_lofar"]
    all_below = all(fam[f]['oracle']['sp'].mean() < fam[f]['oracle']['sp_exp'] for f in fam)

    cells = []
    cells.append(md(f"""
# Sistema de Gating Front-End (Roteamento Dinâmico de Entradas) – Dataset IARA
### Notebook 26: Arquitetura Multimodal Agnóstica ao Modelo Classificador com Suporte a 4 Representações

Neste notebook, investigamos a arquitetura de **Gating Front-End (Multiplexador Dinâmico de Entradas)** no problema de reconhecimento de alvos acústicos submarinos passivos (*Underwater Acoustic Target Recognition* - UATR) sobre o dataset **IARA** (*Silva et al.*, IEEE Access 2025).

### Princípios da Arquitetura de Roteamento e Protocolo:
1. **O Foco é o Processo de Roteamento da Entrada:** O roteador avalia **7 meta-características físicas** extraídas dos primeiros $10\\text{{ s}}$ do sinal acústico bruto e seleciona a representação espectral (**MEL**, **LOFAR**, **HHT** ou **DEMON**) cuja predição será usada.
2. **Modelo Downstream Agnóstico e Modular:** O classificador final não é fixo. A arquitetura aceita qualquer família de modelo (Random Forest, MLP, SVM Nyström ou CNN 2D) ou combinações heterogêneas (**H-MoE**).
3. **Especialistas já treinados:** cada ramo usa as predições por arquivo de um modelo treinado em uma única representação (Seção 3.1 lista a procedência de cada um). Os especialistas **não são homogêneos dentro da mesma família** (pipelines, hiperparâmetros e resoluções diferentes).
4. **Protocolo $5\\times2\\text{{cv}}$ (*Silva et al.*, 2025) e Prevenção de Vazamento Inter-Repetição (`trn_fold = fold ^ 1`):**
   * O particionamento oficial $5\\times2\\text{{cv}}$ (5 repetições de *2-fold CV*) é feito por **gravação de áudio (`ID`)**, não por navio (`Ship ID`): o mesmo navio pode aparecer em gravações de treino e de teste. O conjunto UO tem **{br(c['n_uo'])} gravações**; cada fold de teste tem **{br(c['n_fold'])} gravações**, e os dois folds de teste de uma repetição somam {br(c['n_par'])} gravações ({br(c['n_fold'])} × 2); as outras {br(c['n_uo'] - c['n_par'])} gravações não entram no teste daquela repetição. Somando os 10 folds são {br(c['n_aval'])} avaliações, ou seja, cada gravação aparece em média {br(c['aval_por_arq'])} vezes num conjunto de teste.
   * Para um fold de teste $k$, os 8 folds das **outras 4 repetições** contêm em média ${br(c['overlap_outras_rep'], 1)}\\%$ dos arquivos de teste de $k$. Treinar o roteador nos "outros 9 folds" faria o roteador memorizar as 7 meta-features de cada `file_id`.
   * Por isso o roteador e a seleção de modelos (*Nested CV*) são treinados **exclusivamente no fold complementar da mesma repetição (`trn_fold = fold ^ 1`)**, com **0% de sobreposição de arquivos** (`assert` no código).
"""))
    cells.append(code("""
import os
import sys
import pickle
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display, HTML
except ImportError:
    display = print

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.abspath('src'))  # módulos movidos para notebooks/src
sys.path.insert(0, os.path.abspath('notebooks/src'))
import iara_results as R   # métricas dos experimentos isolados a partir dos CSVs por janela
import gating_comum as G   # réplica do roteador (SP por fold), oráculos e F-test

# Configuração de estilo visual (sem dependência do seaborn)
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

print("Ambiente configurado com sucesso para visualização científica (sem dependência de seaborn).")
"""))
    cells.append(md("""
## 1. Arquitetura do Sistema Gating Front-End

Abaixo apresentamos o fluxo de processamento do sistema. O módulo `Acoustic Front-End Gating` atua como um multiplexador dinâmico na entrada, roteando o áudio entre as 4 representações a partir de 7 descritores extraídos dos primeiros $10\\text{ s}$ do áudio:

```
                            Sinal de Áudio Submarino (.wav, janela 10s)
                                           │
                                           ▼
                     ┌───────────────────────────────────────────┐
                     │    Acoustic Front-End Gating (Router)     │
                     │  7 Meta-features físicas (janela 10s):    │
                     │  • High / Low Energy Ratio & RMS (dB)     │
                     │  • Crest Factor (< 250 Hz) & ZCR          │
                     │  • Spectral Flatness & Spectral Flux      │
                     │  • Modulation Index (2 - 8 kHz)           │
                     └─────────────────────┬─────────────────────┘
                                           │
                                           │ Seleção Dinâmica da Entrada
                                           ▼
               ┌───────────────┬───────────────────┬───────────────┐
               ▼               ▼                   ▼               ▼
        [    MEL    ]   [   LOFAR    ]      [    HHT    ]   [   DEMON   ]
        (Banda Larga /  (Raias Tonais /     (Freq. Instant. (Modulação /
         Ruído Geral)    Diesel Lento)       / Transitórios) Cavitação)
               │               │                   │               │
               └───────────────┼───────────────────┼───────────────┘
                               │
                               ▼
            ┌─────────────────────────────────────────────┐
            │       Classificador Downstream Modular      │
            │   (Homogêneo: RF, MLP, SVM, CNN ou H-MoE)   │
            └──────────────────────┬──────────────────────┘
                                   │
                                   ▼
                        Predição Final da Classe
                 [Small | Medium | Large | Background]
```

### Motivação física de cada ramo (hipóteses de projeto, não testadas aqui):
* **`MEL`:** banco de filtros psicoacústico de banda larga; representação genérica de ruído hidrodinâmico contínuo.
* **`LOFAR`:** análise de banda estreita em baixa frequência; pensada para raias harmônicas tonais de máquinas.
* **`HHT`:** Transformada de Hilbert-Huang via Decomposição em Modos Empíricos (EMD); frequências instantâneas de eventos não-estacionários.
* **`DEMON`:** demodulação de envoltória na banda de cavitação; frequência de rotação do eixo e de passagem de pás.
"""))
    cells.append(md(f"""
## 2. Análise Estatística das Meta-Características Acústicas Físicas

Carregamos as 7 meta-características físicas extraídas dos primeiros $10\\text{{ s}}$ de cada um dos ${br(c['n_meta'])}$ arquivos de áudio do IARA (`cache_gating_metafeatures.pkl`, todos os subconjuntos; os experimentos usam só as {br(c['n_uo'])} gravações do UO) e inspecionamos sua distribuição.
"""))
    cells.append(code("""
# Carregar meta-características acústicas físicas (janela de 10s por gravação)
meta_paths = [
    'data/iara_processed/cache_gating_metafeatures.pkl',
    '../data/iara_processed/cache_gating_metafeatures.pkl'
]
meta_path = next((p for p in meta_paths if os.path.exists(p)), None)

if meta_path is None:
    raise FileNotFoundError("Arquivo cache_gating_metafeatures.pkl não encontrado.")

with open(meta_path, 'rb') as f:
    meta_dict = pickle.load(f)

df_meta = pd.DataFrame.from_dict(meta_dict, orient='index')
df_meta.index.name = 'file_id'
df_meta.reset_index(inplace=True)

print(f"Total de áudios com meta-características físicas extraídas (janela 10s): {len(df_meta)}")
df_meta[['crest_factor', 'spectral_flatness', 'mod_index', 'flux_std', 'high_low_ratio', 'rms_db', 'zcr']].describe().round(3)
"""))
    cells.append(md("""
## 3. Desempenho Experimental Comparativo Livre de Vazamento (10 Folds $5\\times2\\text{cv}$, `trn_fold = fold ^ 1`)

Avaliamos o **RF Competence Router** (`n_estimators=50, max_depth=3, min_samples_leaf=5`, escore = 0,9 × probabilidade de acerto + 0,1 × SP do especialista no fold de treino; treinado exclusivamente no fold disjunto `trn_fold = fold ^ 1`) atuando como multiplexador de entrada nas 4 famílias de classificadores:
1. **Random Forest (RF)**
2. **Multi-Layer Perceptron (MLP)**
3. **Support Vector Machine (SVM Nyström RBF)**
4. **CNN 2D**

> **Atenção — escolha dos hiperparâmetros do roteador.** Esta configuração (`max_depth=3`, `min_samples_leaf=5`, peso do prior 0,1) não foi escolhida num conjunto de validação próprio: ela é a melhor ou quase a melhor de 18 variantes **avaliadas no próprio teste** (Seção 3.2). Os resultados abaixo são, portanto, otimistas.

Em cada família comparamos:
* as **4 entradas isoladas** (sem roteamento), destacando a **melhor entrada isolada** da família;
* o **Gated Model** (roteamento via meta-features treinado apenas em `fold ^ 1`), com o **F-test 5×2cv** (Alpaydin, o mesmo teste do artigo) contra a melhor entrada isolada;
* o **Oráculo** (acerta se *alguma* das 4 entradas acertar). O oráculo usa o rótulo verdadeiro para escolher a entrada: **não é um sistema realizável**. Mostramos também o SP esperado se os erros das 4 entradas fossem independentes.
"""))
    cells.append(md(f"""
### 3.1 Procedência dos 16 especialistas

As predições por arquivo usadas pelo gate (`data/iara_processed/*_all_representations_predictions.pkl`) vêm dos experimentos abaixo. A coluna `SP iara_results` recalcula o SP a partir dos CSVs por janela com o desempate do pipeline iara (`Counter.most_common`); a coluna `SP gate` usa as predições por arquivo do pickle (empate → menor classe). As duas diferem só nos arquivos com empate de votos.

Especialistas com desvio do SP entre folds acima de {br(G.COLLAPSE_STD, 0)} p.p. são marcados como **instáveis/colapsados** (folds com recall ≈ 0 em alguma classe) e não devem ser usados como baseline para anunciar ganho: **{unstable_txt}**.
"""))
    cells.append(code("""
rows = []
for (fam, rep), exp in G.EXPERTS_EXP.items():
    sp_gate, rec_gate = G.single_folds(fam, rep)
    s = R.summary(exp)
    rows.append({
        'Família': fam, 'Entrada': rep.upper(), 'Experimento': exp,
        'SP gate (%)': f"{sp_gate.mean():.2f} ± {sp_gate.std():.2f}",
        'SP iara_results (%)': R.fmt(s['SP']),
        'ACC bal. (%)': R.fmt(s['ACC']),
        'Recall SMALL (%)': round(rec_gate[0], 1),
        'Status': 'INSTÁVEL/COLAPSADO' if sp_gate.std() > G.COLLAPSE_STD else '',
    })
df_experts = pd.DataFrame(rows)
display(df_experts)
for fam, note in G.EXPERTS_NOTE.items():
    print(f"{fam}: {note}")
"""))
    cells.append(code("""
# Resultados dos scripts de scratch/ (pickles) + SP por fold recalculado pela réplica do roteador (gating_comum)
def find_file(rel_paths):
    for p in rel_paths:
        if os.path.exists(p):
            return p
    return None

res_all_path = find_file([
    'data/iara_processed/results_competence_gating_all_models.pkl',
    '../data/iara_processed/results_competence_gating_all_models.pkl'
])
res_svm_path = find_file([
    'data/iara_processed/results_competence_gating_svm.pkl',
    '../data/iara_processed/results_competence_gating_svm.pkl'
])
with open(res_all_path, 'rb') as f:
    res_all = pickle.load(f)
with open(res_svm_path, 'rb') as f:
    res_svm = pickle.load(f)

models_data = [
    ('Random Forest', 'RF', res_all['RANDOM FOREST']),
    ('MLP', 'MLP', res_all['MLP']),
    ('SVM (Nyström)', 'SVM', res_svm),
    ('CNN 2D', 'CNN', res_all['CNN 2D'])
]

def sp_of(s):
    return float(str(s).split()[0])

summary_rows, gate_folds = [], {}
for model_name, fam, data in models_data:
    df_m = data['df']
    gated, oracle_row = df_m.iloc[4], df_m.iloc[5]
    # SP por fold (réplica) -- confere com o pickle
    g_sp, g_rec, _ = G.gate_folds(G.homog(fam))
    assert abs(g_sp.mean() - sp_of(gated['SP (%)'])) < 0.006, model_name
    gate_folds[fam] = g_sp
    base = {r: G.single_folds(fam, r)[0] for r in G.REPS}
    best = max(G.REPS, key=lambda r: base[r].mean())
    F, p = G.f_test_5x2(g_sp, base[best])
    orc = G.oracle(G.homog(fam))

    routes = data['routes']
    tot = sum(routes.values())
    route_str = ", ".join(f"{r} {routes[r]/tot*100:.1f}%" for r in ['MEL', 'LOFAR', 'HHT', 'DEMON'])
    lofar_flag = ' (colapsado)' if base['lofar'].std() > G.COLLAPSE_STD else ''

    summary_rows.append({
        'Classificador Downstream': model_name,
        'Baseline MEL SP (%)': df_m.iloc[0]['SP (%)'],
        'Baseline LOFAR SP (%)': df_m.iloc[1]['SP (%)'] + lofar_flag,
        'Melhor entrada isolada': f"{best.upper()} {base[best].mean():.2f} ± {base[best].std():.2f}",
        'Gated SP (%)': gated['SP (%)'],
        'Δ vs melhor isolada (p.p.)': f"{g_sp.mean() - base[best].mean():+.2f}",
        'F-test 5x2cv': f"F={F:.2f}, p={p:.2f}",
        'Recall SMALL (%)': gated['Recall SMALL (%)'],
        'Recall MEDIUM (%)': gated['Recall MEDIUM (%)'],
        'Recall LARGE (%)': gated['Recall LARGE (%)'],
        'Recall BG (%)': gated['Recall BG (%)'],
        'Oráculo SP (%)': oracle_row['SP (%)'],
        'Oráculo esperado c/ erros indep. (%)': f"{orc['sp_exp']:.2f}",
        'Rotas Selecionadas': route_str
    })

df_master = pd.DataFrame(summary_rows)
print("=== TABELA MASTER (LEAK-FREE trn_fold = fold ^ 1): GATING FRONT-END NOS 4 CLASSIFICADORES ===")
print("Oráculo: usa o rótulo para escolher a entrada (não realizável). F-test: gate vs. melhor entrada isolada da família.")
display(df_master)
"""))
    cells.append(md(f"""
**Leitura da tabela (valores calculados na geração deste notebook):**

| Família | Gate SP (%) | Melhor entrada isolada | Δ (p.p.) | F | p |
|---|---|---|---|---|---|
{chr(10).join(fam_line(f) for f in fam)}

* **{n_sig} de 4** gates diferem significativamente (p < 0,05) da melhor entrada isolada da própria família.{' As diferenças estão dentro da variação entre folds: o gate é **equivalente** à melhor entrada isolada.' if n_sig == 0 else ''}
* Na CNN, a comparação contra o LOFAR ({pm(cnn['base']['lofar'][0])}) não é informativa: esse especialista colapsou. O gate da CNN manda {br(rt['CNN'][0], 1)}% dos áudios para o MEL e fica equivalente à CNN-MEL isolada.
* O oráculo de cada família ({', '.join(f"{f} {br(fam[f]['oracle']['sp'].mean())}" for f in fam)}) fica **{'abaixo' if all_below else 'em parte abaixo'}** do esperado se os erros das 4 entradas fossem independentes ({', '.join(f"{f} {br(fam[f]['oracle']['sp_exp'])}" for f in fam)}): os erros das entradas são positivamente correlacionados. Para comparação, 4 preditores uniformemente aleatórios já dão um oráculo de SP {br(G.random_oracle_sp(4))}%.
"""))
    cells.append(md("""
### 3.1b Visualização Gráfica do Desempenho e Distribuição de Rotas

Geramos gráficos lendo dinamicamente de `df_master` e dos dicionários de rotas para comparar o Índice SP (Baseline MEL, Baseline LOFAR, Gating Front-End e Oráculo), o Recall por Classe e a distribuição percentual das rotas selecionadas. O oráculo é um limite não realizável (usa o rótulo).
"""))
    cells.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Extração dinâmica a partir de df_master e models_data
classifiers = df_master['Classificador Downstream'].tolist()
sp_mel = [float(str(v).split()[0]) for v in df_master['Baseline MEL SP (%)']]
sp_lofar = [float(str(v).split()[0]) for v in df_master['Baseline LOFAR SP (%)']]
sp_gated = [float(str(v).split()[0]) for v in df_master['Gated SP (%)']]
sp_oracle = [float(str(v).split()[0]) for v in df_master['Oráculo SP (%)']]

x = np.arange(len(classifiers))
w = 0.20

axes[0].bar(x - 1.5*w, sp_mel, w, label='Baseline MEL', color='#8d99ae', alpha=0.9)
axes[0].bar(x - 0.5*w, sp_lofar, w, label='Baseline LOFAR', color='#457b9d', alpha=0.9)
axes[0].bar(x + 0.5*w, sp_gated, w, label='Gating Front-End', color='#2b5c8f', alpha=0.95)
axes[0].bar(x + 1.5*w, sp_oracle, w, label='Oráculo (não realizável)', color='#e07a5f', alpha=0.6, linestyle='--', edgecolor='#c0523b')

axes[0].set_xticks(x)
axes[0].set_xticklabels(classifiers, rotation=15, ha='right', fontweight='bold')
axes[0].set_ylabel('Índice SP (%)', fontweight='bold')
axes[0].set_title('(A) Índice SP: Baselines vs. Gating vs. Oráculo', fontweight='bold')
axes[0].set_ylim(30, 95)
axes[0].legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

for i in range(len(classifiers)):
    axes[0].text(x[i] + 0.5*w, sp_gated[i] + 1.2, f"{sp_gated[i]:.2f}%", ha='center', fontsize=8.5, fontweight='bold', color='#1d3557')

# 2. Recall por Classe sob o Sistema Gated (lido dinamicamente de df_master)
rec_small = [float(str(v).replace('%', '')) for v in df_master['Recall SMALL (%)']]
rec_medium = [float(str(v).replace('%', '')) for v in df_master['Recall MEDIUM (%)']]
rec_large = [float(str(v).replace('%', '')) for v in df_master['Recall LARGE (%)']]
rec_bg = [float(str(v).replace('%', '')) for v in df_master['Recall BG (%)']]

w2 = 0.20
axes[1].bar(x - 1.5*w2, rec_small, w2, label='Small (Crítica)', color='#d90429', alpha=0.9)
axes[1].bar(x - 0.5*w2, rec_medium, w2, label='Medium', color='#f4a261', alpha=0.9)
axes[1].bar(x + 0.5*w2, rec_large, w2, label='Large', color='#2a9d8f', alpha=0.9)
axes[1].bar(x + 1.5*w2, rec_bg, w2, label='Background', color='#264653', alpha=0.9)

axes[1].set_xticks(x)
axes[1].set_xticklabels(classifiers, rotation=15, ha='right', fontweight='bold')
axes[1].set_ylabel('Recall por Classe (%)', fontweight='bold')
axes[1].set_title('(B) Recall por Classe sob Gating (trn_fold = fold ^ 1)', fontweight='bold')
axes[1].set_ylim(30, 100)
axes[1].legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')

# 3. Distribuição Percentual de Rotas por Classificador (lida dinamicamente)
route_mel = [d['routes']['MEL'] / sum(d['routes'].values()) * 100 for _, _, d in models_data]
route_lofar = [d['routes']['LOFAR'] / sum(d['routes'].values()) * 100 for _, _, d in models_data]
route_hht = [d['routes']['HHT'] / sum(d['routes'].values()) * 100 for _, _, d in models_data]
route_demon = [d['routes']['DEMON'] / sum(d['routes'].values()) * 100 for _, _, d in models_data]

axes[2].bar(x - 1.5*w2, route_mel, w2, label='MEL', color='#457b9d', alpha=0.9)
axes[2].bar(x - 0.5*w2, route_lofar, w2, label='LOFAR', color='#1d3557', alpha=0.9)
axes[2].bar(x + 0.5*w2, route_hht, w2, label='HHT', color='#e63946', alpha=0.9)
axes[2].bar(x + 1.5*w2, route_demon, w2, label='DEMON', color='#a8dadc', alpha=0.9)

axes[2].set_xticks(x)
axes[2].set_xticklabels(classifiers, rotation=15, ha='right', fontweight='bold')
axes[2].set_ylabel('Áudios Roteados (%)', fontweight='bold')
axes[2].set_title('(C) Distribuição das Rotas Selecionadas', fontweight='bold')
axes[2].set_ylim(0, 95)
axes[2].legend(frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.show()
"""))
    cells.append(md(f"""
### 3.2 Sensibilidade aos hiperparâmetros do roteador (18 variantes avaliadas no teste)

Repetimos o gate com `max_depth` ∈ {{3, 5, None}} × `min_samples_leaf` ∈ {{1, 5}} × peso do prior ∈ {{0; 0,1; 0,3}} (o restante idêntico). A configuração reportada é `(3, 5, 0,1)`.

| Gate | SP reportado (%) | Posição entre as 18 | Faixa das 18 (%) | Mediana (%) |
|---|---|---|---|---|
{chr(10).join(grid_line(n) for n in grid)}

A configuração reportada é a {', '.join(f"{grid[n]['rank']}ª" + ' no ' + n for n in grid)} entre as 18, e a faixa entre variantes ({br(min(g['max'] - g['min'] for g in grid.values()))} a {br(max(g['max'] - g['min'] for g in grid.values()))} p.p.) é maior que as diferenças entre o gate e a melhor entrada isolada. Como a escolha foi feita olhando o teste, os SPs reportados do gate são otimistas; uma estimativa honesta exige escolher os hiperparâmetros por validação interna em `fold ^ 1` (*nested CV*).

A célula abaixo recalcula a grade (leva alguns minutos). Os valores da tabela acima foram calculados na geração do notebook; nas variantes sem limite de profundidade (`max_depth=None`) a célula pode diferir em ~0,1–0,15 p.p. conforme a versão do scikit-learn (desempates na construção das árvores). A configuração reportada (`max_depth=3`) reproduz exatamente os pickles.
"""))
    cells.append(code("""
grid_rows = []
for name, ex in [*[(f, G.homog(f)) for f in G.FAMS], ('H-MoE especializado', G.FIXED_HMOE)]:
    vals = {cfg: G.gate_folds(ex, depth=cfg[0], leaf=cfg[1], w_prior=cfg[2])[0].mean() for cfg in G.GRID}
    rep = vals[G.REPORTED_CFG]
    best_cfg = max(vals, key=vals.get)
    grid_rows.append({
        'Gate': name,
        'SP reportado (3, 5, 0.1)': round(rep, 2),
        'Posição (de 18)': 1 + sum(v > rep + 1e-9 for v in vals.values()),
        'Mín': round(min(vals.values()), 2), 'Mediana': round(float(np.median(list(vals.values()))), 2),
        'Máx': round(max(vals.values()), 2), 'Config. do máximo (depth, leaf, w)': best_cfg,
    })
display(pd.DataFrame(grid_rows))
"""))
    cells.append(md(f"""
## 4. Arquitetura Heterogênea: Gating Front-End com Especialistas Distintos (H-MoE)

Na Seção 3, a mesma família de classificador foi usada nos 4 ramos. Os especialistas isolados de maior SP por representação são:
* **LOFAR:** o especialista isolado de maior SP entre os 16 é o **{bs_f}-{REP_UP[bs_r]}** (**{pm(fam[bs_f]['base'][bs_r][0])}**); o MLP-LOFAR vem do pipeline oficial;
* **HHT:** o **Random Forest** alcança **{pm(rf_base['hht'][0])}**, com {br(rf_base['hht'][1][0], 1)}% de recall em *Small* (MLP-HHT: {br(mlp['base']['hht'][1][0], 1)}%; SVM-HHT: {br(svm['base']['hht'][1][0], 1)}%);
* **MEL:** o **Random Forest** tem o maior recall de *Small* entre os MEL ({br(rf_base['mel'][1][0], 1)}%), mas SP {pm(rf_base['mel'][0])}, abaixo de MLP-MEL ({pm(mlp['base']['mel'][0])}) e SVM-MEL ({pm(svm['base']['mel'][0])}).

Abaixo avaliamos o sistema **H-MoE (Heterogeneous Mixture of Experts)** de duas formas:
* **Automático:** o especialista de cada representação é escolhido entre RF, MLP e SVM pelo SP no fold disjunto (*Nested CV* em `trn_fold = fold ^ 1`);
* **Especializado fixo** (`MEL:RF, LOFAR:MLP, HHT:RF, DEMON:MLP`): este mapa foi escolhido pelo autor **depois** de ver os resultados isolados de teste, logo não é uma estimativa independente.
"""))
    cells.append(code("""
# Carregar resultados do H-MoE Heterogêneo (treinado estritamente em trn_fold = fold ^ 1)
hmoe_paths = [
    'data/iara_processed/results_heterogeneous_gating.pkl',
    '../data/iara_processed/results_heterogeneous_gating.pkl'
]
hmoe_path = next((p for p in hmoe_paths if os.path.exists(p)), None)
with open(hmoe_path, 'rb') as f:
    hmoe_data = pickle.load(f)
print("=== TABELA COMPARATIVA (LEAK-FREE trn_fold = fold ^ 1): H-MoE VS. GATING HOMOGÊNEO ===")
print("Linha 6 (Super-Oráculo): usa o rótulo para escolher entre 16 especialistas -> não realizável.")
display(hmoe_data['df'])
print("\\nDescoberta Automática de Especialistas por Representação (Nested CV em trn_fold = fold ^ 1):")
for r, models in hmoe_data['selected_models_per_fold'].items():
    counts_str = ", ".join([f"{m} ({c}/10 folds)" for m, c in Counter(models).most_common()])
    print(f"  {r.upper()}: {counts_str}")

# F-test 5x2cv contra o melhor especialista isolado (MLP-LOFAR), com SP por fold
mlp_lofar = G.single_folds('MLP', 'lofar')[0]
hfix = G.gate_folds(G.FIXED_HMOE)[0]
hauto = G.hmoe_auto_folds()
assert abs(hfix.mean() - sp_of(hmoe_data['df'].iloc[4]['SP (%)'])) < 0.006
assert abs(hauto.mean() - sp_of(hmoe_data['df'].iloc[3]['SP (%)'])) < 0.006
tests = []
for nome, a in [('H-MoE Automático', hauto), ('H-MoE Especializado', hfix),
                ('Gated RF', gate_folds['RF']), ('Gated MLP', gate_folds['MLP']),
                ('Gated SVM', gate_folds['SVM']), ('Gated CNN', gate_folds['CNN'])]:
    F, p = G.f_test_5x2(a, mlp_lofar)
    tests.append({'Sistema': nome, 'SP (%)': f"{a.mean():.2f} ± {a.std():.2f}",
                  'Δ vs MLP-LOFAR (p.p.)': f"{a.mean() - mlp_lofar.mean():+.2f}", 'F': round(F, 2), 'p': round(p, 3)})
print(f"\\nF-test 5x2cv contra o MLP-LOFAR isolado ({mlp_lofar.mean():.2f} ± {mlp_lofar.std():.2f}):")
display(pd.DataFrame(tests))
print(f"Referência — MLP LOFAR na Tabela 7 do artigo: SP {R.ARTICLE_TABLE7[('MLP', 'LOFAR')]['SP'][0]:.2f} ± {R.ARTICLE_TABLE7[('MLP', 'LOFAR')]['SP'][1]:.2f}")

# Super-Oráculo vs. referências de acaso
sup = G.oracle(G.ALL16)
print(f"\\nSuper-Oráculo (16 especialistas): SP {sup['sp'].mean():.2f} ± {sup['sp'].std():.2f}")
print(f"  esperado se os erros dos 16 fossem independentes: SP {sup['sp_exp']:.3f}")
print(f"  oráculo de 16 preditores uniformemente aleatórios: SP {G.random_oracle_sp(16):.2f}")
print(f"  oráculo de 4 preditores uniformemente aleatórios:  SP {G.random_oracle_sp(4):.2f}")
"""))
    cells.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

df_h = hmoe_data['df']
sp_mel_rf = float(str(df_h.iloc[0]['SP (%)']).split()[0])
sp_lof_mlp = float(str(df_h.iloc[1]['SP (%)']).split()[0])
sp_gated_rf = float(str(df_h.iloc[2]['SP (%)']).split()[0])
sp_hmoe_auto = float(str(df_h.iloc[3]['SP (%)']).split()[0])
sp_hmoe_fix = float(str(df_h.iloc[4]['SP (%)']).split()[0])
sp_super_ora = float(str(df_h.iloc[5]['SP (%)']).split()[0])
sp_gated_svm = float(str(res_svm['df'].iloc[4]['SP (%)']).split()[0])
sp_gated_mlp = float(str(res_all['MLP']['df'].iloc[4]['SP (%)']).split()[0])

# 1. Índice SP (Leak-Free)
archs = [
    'Baseline MEL (RF)', 'Gated SVM', 'Gated RF', 'H-MoE Auto',
    'H-MoE Especializado', 'Gated MLP', 'LOFAR Isolado (MLP)', 'Super-Oráculo\\n(não realizável)'
]
sps = [sp_mel_rf, sp_gated_svm, sp_gated_rf, sp_hmoe_auto, sp_hmoe_fix, sp_gated_mlp, sp_lof_mlp, sp_super_ora]
colors = ['#8d99ae', '#457b9d', '#2b5c8f', '#2a9d8f', '#2a9d8f', '#1d3557', '#f4a261', '#e07a5f']

bars = axes[0].bar(range(len(archs)), sps, color=colors, alpha=0.92, edgecolor='black', linewidth=0.8)
axes[0].set_xticks(range(len(archs)))
axes[0].set_xticklabels(archs, rotation=28, ha='right', fontweight='bold', fontsize=8.5)
axes[0].set_ylabel('Índice SP (%)', fontweight='bold')
axes[0].set_title('(A) Comparativo Global de SP (Leak-Free 5x2cv)', fontweight='bold')
axes[0].set_ylim(50, 105)
axes[0].axhline(sp_lof_mlp, color='#d90429', linestyle='--', linewidth=1.2, label=f'Baseline LOFAR MLP ({sp_lof_mlp:.2f}%)')
axes[0].axhline(G.random_oracle_sp(16), color='gray', linestyle=':', linewidth=1.2, label=f'Oráculo de 16 aleatórios ({G.random_oracle_sp(16):.2f}%)')
axes[0].legend(loc='upper left', frameon=True, facecolor='white', fontsize=8.5)
for bar, s in zip(bars, sps):
    axes[0].text(bar.get_x() + bar.get_width()/2, s + 1.2, f'{s:.2f}%', ha='center', fontsize=8, fontweight='bold')

# 2. Recall por Classe: Gated RF vs H-MoE Especializado vs LOFAR MLP
classes = ['Small (Crítica)', 'Medium', 'Large', 'Background']
rec_cols = ['Recall SMALL (%)', 'Recall MEDIUM (%)', 'Recall LARGE (%)', 'Recall BG (%)']
rec_rf = [float(str(df_h.iloc[2][c]).replace('%', '')) for c in rec_cols]
rec_hmoe = [float(str(df_h.iloc[4][c]).replace('%', '')) for c in rec_cols]
rec_lof = [float(str(df_h.iloc[1][c]).replace('%', '')) for c in rec_cols]
x = np.arange(len(classes))
w = 0.26
axes[1].bar(x - w, rec_rf, w, label='Gated RF Homogêneo', color='#2b5c8f', alpha=0.9)
axes[1].bar(x, rec_hmoe, w, label='H-MoE Especializado', color='#2a9d8f', alpha=0.95)
axes[1].bar(x + w, rec_lof, w, label='LOFAR Isolado (MLP)', color='#f4a261', alpha=0.9)
axes[1].set_xticks(x)
axes[1].set_xticklabels(classes, fontweight='bold')
axes[1].set_ylabel('Recall por Classe (%)', fontweight='bold')
axes[1].set_title('(B) Recall por Classe: Gated RF vs. H-MoE vs. LOFAR MLP', fontweight='bold')
axes[1].set_ylim(35, 100)
axes[1].legend(loc='upper left', frameon=True, facecolor='white', fontsize=8.5)
for i in range(len(classes)):
    axes[1].text(x[i] - w, rec_rf[i] + 1.0, f'{rec_rf[i]:.1f}%', ha='center', fontsize=7.5, fontweight='bold')
    axes[1].text(x[i], rec_hmoe[i] + 1.0, f'{rec_hmoe[i]:.1f}%', ha='center', fontsize=7.5, fontweight='bold')
    axes[1].text(x[i] + w, rec_lof[i] + 1.0, f'{rec_lof[i]:.1f}%', ha='center', fontsize=7.5, fontweight='bold')

# 3. Distribuição das Rotas e Modelos no H-MoE Especializado (lida dinamicamente)
rf_cnt = hmoe_data['routes_fixed']
tot_rf_cnt = sum(rf_cnt.values())
route_keys = ['LOFAR:MLP', 'MEL:RF', 'HHT:RF', 'DEMON:MLP']
routes_labels = ['LOFAR\\n(MLP)', 'MEL\\n(RF)', 'HHT\\n(RF)', 'DEMON\\n(MLP)']
pcts = [rf_cnt[k] / tot_rf_cnt * 100 for k in route_keys]
pie_colors = ['#1d3557', '#457b9d', '#e63946', '#a8dadc']
axes[2].bar(range(len(routes_labels)), pcts, color=pie_colors, alpha=0.9, edgecolor='black', linewidth=0.8)
axes[2].set_xticks(range(len(routes_labels)))
axes[2].set_xticklabels(routes_labels, fontweight='bold')
axes[2].set_ylabel('Áudios Roteados (%)', fontweight='bold')
axes[2].set_title('(C) Alocação Dinâmica no H-MoE Especializado', fontweight='bold')
axes[2].set_ylim(0, 70)
for i, p in enumerate(pcts):
    axes[2].text(i, p + 1.2, f'{p:.1f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
"""))

    def eq(p):
        return "não difere significativamente" if p >= 0.05 else "difere significativamente"

    cells.append(md(f"""
## 5. Conclusões e Recomendações

### Principais Conclusões (Protocolo $5\\times2\\text{{cv}}$ com roteador treinado em `fold ^ 1`):

1. **Por que o treino estritamente disjunto (`trn_fold = fold ^ 1`) é obrigatório em $5\\times2\\text{{cv}}$:**
   * Cada fold de teste tem {br(c['n_fold'])} das {br(c['n_uo'])} gravações do UO, e o split é por gravação (`ID`), não por navio. As 7 meta-características (`crest_factor`, `spectral_flatness`, `mod_index`, `flux_std`, `high_low_ratio`, `rms_db`, `zcr`) são extraídas dos primeiros $10\\text{{ s}}$ de cada arquivo; treinar o roteador nos "outros 9 folds" vazaria ${br(c['overlap_outras_rep'], 1)}\\%$ dos arquivos de teste, presentes nas outras 4 repetições.
   * Restringindo o roteador e a seleção *Nested CV* ao fold complementar da mesma repetição (`trn_fold = fold ^ 1`, 0% de sobreposição de arquivos), eliminamos a memorização por arquivo.

2. **Gate homogêneo: {'equivalente à melhor entrada isolada de cada família (nenhum ganho demonstrado)' if n_sig == 0 else f'{n_sig} de 4 famílias com diferença significativa vs. a melhor entrada isolada'}.**
   * **Gated Random Forest ({gtxt('RF')}):** contra o RF-{REP_UP[rf['best']]} isolado ({pm(rf['base'][rf['best']][0])}), F-test 5×2cv {p_txt(rf['p'])}. O roteador distribui os contatos entre LOFAR ({br(rt['RF'][1], 1)}%), MEL ({br(rt['RF'][0], 1)}%) e HHT ({br(rt['RF'][2], 1)}%), e o recall de `SMALL` fica em {br(rf['rec'][0], 1)}% (RF-LOFAR isolado: {br(rf_lof_small, 1)}%).
   * **Gated SVM ({gtxt('SVM')}):** contra o SVM-{REP_UP[svm['best']]} isolado ({pm(svm['base'][svm['best']][0])}), {p_txt(svm['p'])}; rotas LOFAR {br(rt['SVM'][1], 1)}% e MEL {br(rt['SVM'][0], 1)}%.
   * **Gated MLP ({gtxt('MLP')}):** contra o MLP-{REP_UP[mlp['best']]} isolado ({pm(mlp['base'][mlp['best']][0])}), {p_txt(mlp['p'])}.
   * **Gated CNN 2D ({gtxt('CNN')}):** contra a CNN-{REP_UP[cnn['best']]} isolada ({pm(cnn['base'][cnn['best']][0])}), {p_txt(cnn['p'])}. O gate manda {br(rt['CNN'][0], 1)}% das rotas ao MEL; os especialistas CNN-LOFAR/HHT/DEMON ad hoc estão colapsados, então não cabe anunciar ganho sobre eles.
   * Além disso, a configuração do roteador foi escolhida no teste (Seção 3.2), o que torna esses SPs otimistas.

3. **H-MoE e o melhor especialista isolado (MLP-LOFAR, {pm(ml)}):**
   * O **H-MoE Especializado ({pm(X['hfix'])})** {eq(hfix_p)} do MLP-LOFAR isolado (F = {br(hfix_F)}, {p_txt(hfix_p)}) nem do Gated RF homogêneo (F = {br(hrf_F)}, {p_txt(hrf_p)}). Seu recall de `SMALL` é {br(X['hfix_rec'][0], 1)}% contra {br(fam['MLP']['base']['lofar'][1][0], 1)}% do MLP-LOFAR.
   * O **H-MoE Automático ({pm(X['hauto'])})** {eq(hauto_p)} do MLP-LOFAR (F = {br(hauto_F)}, {p_txt(hauto_p)}).
   * O **Gated MLP** também {eq(mlp['p_ml'])} do MLP-LOFAR ({p_txt(mlp['p_ml'])}).
   * *Hipótese (não testada):* quando um especialista domina (`LOFAR:MLP`), 7 descritores globais de uma única janela de $10\\text{{ s}}$ podem não ter informação suficiente para identificar os poucos áudios em que outro especialista acerta e ele erra; desviar áudios para especialistas de SP menor tende a gerar trocas líquidas nulas ou negativas.

4. **Super-Oráculo ({pm(sup['sp'])}) não é teto alcançável nem evidência de complementaridade.**
   * O Super-Oráculo escolhe, com o rótulo verdadeiro, o especialista que acerta: não é um sistema realizável.
   * O oráculo de **16 preditores uniformemente aleatórios** já teria SP de {br(sup_rand16)}%, e o esperado se os erros dos 16 especialistas fossem independentes é {br(sup['sp_exp'], 3)}%. O valor observado fica **{'abaixo' if sup['sp'].mean() < min(sup_rand16, sup['sp_exp']) else 'perto'}** dessas duas referências: os erros dos especialistas são correlacionados, e o número de 16 tentativas por áudio, sozinho, já explica um oráculo próximo de 100%.
   * {'O que este notebook mostra é que, com os dados atuais, **nenhum** roteador baseado nas 7 meta-features supera de forma estatisticamente significativa o melhor especialista isolado.' if n_sig == 0 and min(hfix_p, hauto_p, mlp['p_ml']) >= 0.05 else 'Ver os p-valores acima antes de afirmar qualquer ganho.'} Para avançar: (i) escolher os hiperparâmetros do roteador por validação interna; (ii) usar especialistas homogêneos e não colapsados; (iii) testar descritores mais informativos para o roteamento.
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
        json.dump(nb_data, f, indent=1, ensure_ascii=False)
    print(f"Notebook '{filename}' criado com sucesso!")


if __name__ == "__main__":
    create_gating_notebook()
