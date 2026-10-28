"""
Gera notebooks/28_correcao_extracao_hht.ipynb: auditoria e correção da extração HHT.

O cache HHT original (data/iara_processed/hht_features + cache_hht.pkl, 15/10/2026) misturava dois limites
de iterações de peneiramento da EMD: o script scratch/extract_all_hht.py foi editado no meio da extração e a
extração foi retomada pulando os arquivos já gravados. Nenhuma gravação BACKGROUND ficou no grupo antigo,
o que confunde configuração de extração com classe. O HHT foi re-extraído com uma configuração única
(8 IMFs, MAX_ITERATION=30) e todos os experimentos HHT foram retreinados.

Fontes:
  reports/correcao_hht/{arquivos,reproducao}.csv, contexto.json   (scratch/audit_hht_mixed_extraction.py)
  results/trainings/_backup_hht_iter_misto_20261015/<exp>          predições ANTES (cache misto)
  results/trainings/multirep/<exp>                                 predições DEPOIS (cache re-extraído)
  reports/correcao_hht/gating_antes/results_*gating*.pkl (antes) e data/iara_processed/results_*gating*.pkl (depois)
Como nos outros geradores, o mesmo código das células roda aqui para calcular os números citados no markdown.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")
NOTEBOOKS_DIR = os.path.abspath(os.path.join(HERE, ".."))


def br(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pm(ms, nd=2):
    return f"{br(ms[0], nd)} ± {br(ms[1], nd)}"


def pv(p):
    return "p < 0,001" if p < 0.001 else f"p = {br(p, 3)}"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


SETUP_CODE = r'''
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print

for _p in ('.', 'src', 'notebooks', 'notebooks/src'):
    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))
import iara_results as R

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11, 'figure.dpi': 120})

AUD = os.path.join(R.ROOT, 'reports', 'correcao_hht')
BACKUP = '_backup_hht_iter_misto_20261015'          # results/trainings/<BACKUP>/<exp> = ANTES
CLS = {0: 'SMALL', 1: 'MEDIUM', 2: 'LARGE', 3: 'BACKGROUND'}
'''

EVID_CODE = r'''
ctx = json.load(open(os.path.join(AUD, 'contexto.json'), encoding='utf-8'))
arq = pd.read_csv(os.path.join(AUD, 'arquivos.csv'))
rep = pd.read_csv(os.path.join(AUD, 'reproducao.csv'))

# classe de cada gravação (os folds 0 e 1 formam uma partição completa do conjunto rotulado)
_fp = R.file_predictions('rf_hht_n100_balanced_4c')
classe = pd.concat([_fp[0], _fp[1]]).drop_duplicates('File').set_index('File').Target.map(CLS)
arq['classe'] = arq.ID.map(classe)
tab_grupo = pd.crosstab(arq.dropna(subset=['classe']).classe, arq.dropna(subset=['classe']).grupo, margins=True)
tab_grupo = tab_grupo.reindex(['SMALL', 'MEDIUM', 'LARGE', 'BACKGROUND', 'All'])

ITERS = ctx['amostra_reproducao']['iteracoes']
tab_rep = pd.DataFrame({f'MAX_ITERATION={it}': rep.groupby('grupo')[f'max_dif_iter{it}'].apply(lambda v: 100 * (v < 1e-6).mean())
                        for it in ITERS}).round(1)
tab_rep_tol = pd.DataFrame({f'MAX_ITERATION={it}': rep.groupby('grupo')[f'max_dif_iter{it}'].apply(lambda v: 100 * (v < 1e-3).mean())
                            for it in ITERS}).round(1)
tab_mud = arq.groupby('grupo').agg(gravacoes=('ID', 'size'),
                                   janelas_mudaram_pct=('frac_janelas_mudou', lambda v: 100 * v.mean()),
                                   max_dif_mediana=('max_dif', 'median'), media_dif_media=('media_dif', 'mean')).round(4)
'''

COMPARE_CODE = r'''
# (rótulo, experimento, semente fixa?, script) — os 22 experimentos HHT com versão antes e depois
EXPS = [
    ('RF 100 árv.', 'rf_hht_n100_balanced', 'sim (random_state=42, CPU)', ('4c', '3c')),
    ('SVM Nyström m=6000', 'svm_nystroem_6000_hht_C2.0_elasticnet_l1r0.15', 'sim (random_state=42, CPU)', ('4c', '3c')),
    ('SVM Nyström m=4000 (NBs 13–15)', 'svm_nystroem_4000_hht_C2.0_elasticnet_l1r0.15', 'sim (random_state=42, CPU)', ('4c',)),
    ('SVM Nyström m=2000 (gate)', 'svm_nystroem_2000_hht_C2.0_elasticnet_l1r0.15', 'sim (random_state=42, CPU)', ('4c',)),
    ('MLP [32,16] 30 ép., seed 42 (gate)', 'mlp_hht_h32-16_e30', 'sim (GPU)', ('4c',)),
    ('MLP [32,16] 30 ép., sem semente', 'mlp_hht_h32-16_e30_noseed', 'não', ('4c', '3c')),
    ('CNN espectral 1D (1 janela)', 'speccnn_hht_ctx0_e60p8', 'sim, por fold (GPU)', ('4c', '3c')),
    ('CNN espectral 1D (5 janelas)', 'speccnn_hht_ctx2_e60p8', 'sim, por fold (GPU)', ('4c', '3c')),
    ('CNN 2D ad hoc b32 (colapsada)', 'cnn2d_hht_w32_b32_e25', 'não', ('4c', '3c')),
    ('CNN 2D ad hoc b64 (gate)', 'cnn2d_hht_w32_b64_e20', 'sim (GPU)', ('4c',)),
    ('BiGRU + atenção | HHT', 'bigru_att_hht_h64_s15-5_e20', 'não', ('4c', '3c')),
    ('BiGRU + atenção | MEL + HHT', 'bigru_att_mel5s-hht_h64_s15-5_e20', 'não', ('4c', '3c')),
    ('BiGRU + atenção | LOFAR + HHT', 'bigru_att_lofar5s-hht_h128_s15-5_e15', 'não', ('4c', '3c')),
]
MET = ['SP', 'R_SMALL', 'R_MEDIUM', 'R_LARGE', 'R_BACKGROUND']
rows = []
for lab, e, seed, modes in EXPS:
    for m in modes:
        name = f'{e}_{m}'
        old, new = R.fold_metrics(f'{BACKUP}/{name}', m), R.fold_metrics(name, m)
        r = {'Modelo': lab, 'Modo': m, 'Semente fixa': seed}
        for k in MET:
            if k == 'R_BACKGROUND' and m == '3c':
                continue
            r[f'{k} antes'], r[f'{k} depois'] = old[k].mean(), new[k].mean()
            r[f'Δ {k}'] = new[k].mean() - old[k].mean()
            r[f'p {k}'] = R.f_test_5x2(new[k].values, old[k].values)[1]
        r['SP std antes'], r['SP std depois'] = old['SP'].std(ddof=0), new['SP'].std(ddof=0)
        rows.append(r)
cmp = pd.DataFrame(rows)
cols = ['Modelo', 'Modo', 'Semente fixa', 'SP antes', 'SP depois', 'Δ SP', 'p SP', 'R_SMALL antes', 'R_SMALL depois', 'p R_SMALL',
        'R_BACKGROUND antes', 'R_BACKGROUND depois', 'p R_BACKGROUND']
'''

PLOT_CODE = r'''
d = cmp.copy()
d['rot'] = d.Modelo + ' [' + d.Modo + ']'
d = d.sort_values('Δ SP')
fig, ax = plt.subplots(figsize=(11, 0.42 * len(d) + 1.5))
cores = ['#c0392b' if p < 0.05 else '#7f8c8d' for p in d['p SP']]
ax.barh(d.rot, d['Δ SP'], color=cores)
ax.axvline(0, color='k', lw=1)
ax.set_xlabel('Δ SP = depois − antes (p.p.)')
ax.set_title('Efeito da re-extração do HHT no SP (vermelho: F-test 5x2cv p < 0,05)', fontweight='bold')
plt.tight_layout()
plt.show()
'''

GATING_CODE = r'''
import pickle
PROC = os.path.join(R.ROOT, 'data', 'iara_processed')
PROC_OLD = os.path.join(AUD, 'gating_antes')   # cópia versionada dos pickles de gating antes da correção

def _tabelas(nome):
    """[(família, df antes, df depois)] de um pickle de resultados de gating (antes = backup)."""
    o = pickle.load(open(os.path.join(PROC_OLD, nome + '.pkl'), 'rb'))
    n = pickle.load(open(os.path.join(PROC, nome + '.pkl'), 'rb'))
    return [(k, o[k]['df'], n[k]['df']) for k in o] if 'df' not in o else [('', o['df'], n['df'])]

def _num(s):
    return float(str(s).split('±')[0])

rows = []
for nome in ['results_competence_gating_all_models', 'results_competence_gating_svm', 'results_heterogeneous_gating']:
    for fam, a, b in _tabelas(nome):
        col = a.columns[0]
        for (_, ra), (_, rb) in zip(a.iterrows(), b.iterrows()):
            assert ra[col] == rb[col]
            rows.append({'Pickle': nome.replace('results_', ''), 'Família': fam, 'Sistema': ra[col],
                         'SP antes': ra['SP (%)'], 'SP depois': rb['SP (%)'], 'Δ SP': _num(rb['SP (%)']) - _num(ra['SP (%)']),
                         'Recall SMALL antes': ra['Recall SMALL (%)'], 'Recall SMALL depois': rb['Recall SMALL (%)']})
gate = pd.DataFrame(rows)
gate_sis = gate[gate.Sistema.str.contains('GATED|Gated|H-MoE|Oráculo|Super')]

# H-MoE especializado: único sistema de gating com SP por fold no pickle -> F-test 5x2cv
_ho = pickle.load(open(os.path.join(PROC_OLD, 'results_heterogeneous_gating.pkl'), 'rb'))['res_hetero_expert_fixed']
_hn = pickle.load(open(os.path.join(PROC, 'results_heterogeneous_gating.pkl'), 'rb'))['res_hetero_expert_fixed']
hmoe_old, hmoe_new = np.array([r[0] for r in _ho]), np.array([r[0] for r in _hn])
F_hmoe, p_hmoe = R.f_test_5x2(hmoe_new, hmoe_old)
'''


def build(filename=os.path.join(NOTEBOOKS_DIR, "28_correcao_extracao_hht.ipynb")):
    ns = {}
    exec(SETUP_CODE, ns)
    exec(EVID_CODE, ns)
    exec(COMPARE_CODE, ns)
    R, ctx, arq, rep, cmp = ns["R"], ns["ctx"], ns["arq"], ns["rep"], ns["cmp"]
    tab_grupo, tab_rep, tab_rep_tol, tab_mud = ns["tab_grupo"], ns["tab_rep"], ns["tab_rep_tol"], ns["tab_mud"]

    g = tab_grupo
    n_antes, n_depois = int(g.loc["All", "antes"]), int(g.loc["All", "depois"])
    bg_antes, bg_depois = int(g.loc["BACKGROUND", "antes"]), int(g.loc["BACKGROUND", "depois"])
    navios_antes = n_antes - bg_antes
    navios_tot = navios_antes + int(g.loc["All", "depois"]) - bg_depois
    n_arq = {k: int(v) for k, v in arq.grupo.value_counts().items()}
    rp = {(gr, it): tab_rep.loc[gr, f"MAX_ITERATION={it}"] for gr in tab_rep.index for it in ns["ITERS"]}
    rpt = {(gr, it): tab_rep_tol.loc[gr, f"MAX_ITERATION={it}"] for gr in tab_rep_tol.index for it in ns["ITERS"]}
    mud = tab_mud

    sig = cmp[cmp["p SP"] < 0.05]
    c4 = cmp[cmp.Modo == "4c"]
    seeded = cmp[cmp["Semente fixa"].str.startswith("sim (random_state")]
    rf4 = cmp[(cmp.Modelo == "RF 100 árv.") & (cmp.Modo == "4c")].iloc[0]
    rf3 = cmp[(cmp.Modelo == "RF 100 árv.") & (cmp.Modo == "3c")].iloc[0]
    pure4 = c4[~c4.Modelo.str.contains(r"\| (?:MEL|LOFAR) \+ HHT")]          # só HHT (sem as fusões MEL/LOFAR + HHT)
    best_old4 = pure4.loc[pure4["SP antes"].idxmax()]
    best_new4 = pure4.loc[pure4["SP depois"].idxmax()]
    _e = {lab: e for lab, e, _, _ in ns["EXPS"]}
    p_best_rf = R.f_test_5x2(_e[best_new4.Modelo] + "_4c", "rf_hht_n100_balanced_4c")[1] if best_new4.Modelo != "RF 100 árv." else None
    fus = c4[c4.Modelo.str.contains(r"\| (?:MEL|LOFAR) \+ HHT")]
    n_down = int((cmp["Δ SP"] < 0).sum())
    sig_txt = "; ".join(f"{r.Modelo} [{r.Modo}]: {br(r['SP antes'])} → {br(r['SP depois'])} ({pv(r['p SP'])})"
                        for _, r in sig.iterrows()) or "nenhum"
    seeded_txt = "; ".join(f"{r.Modelo} [{r.Modo}] Δ = {br(r['Δ SP'])} p.p. ({pv(r['p SP'])})" for _, r in seeded.iterrows())
    bgsig = cmp[(cmp.Modo == "4c") & (cmp["p R_BACKGROUND"] < 0.05)]
    bg_txt = "; ".join(f"{r.Modelo}: {br(r['R_BACKGROUND antes'])} → {br(r['R_BACKGROUND depois'])}% ({pv(r['p R_BACKGROUND'])})"
                       for _, r in bgsig.iterrows()) or "nenhum modelo"

    cells = []
    cells.append(md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 28: Auditoria e Correção da Extração HHT

**Resumo.** O cache HHT usado até 23/10/2026 (`data/iara_processed/hht_features/` e `cache_hht.pkl`, extraídos em 15/10) **não** foi gerado com uma configuração única: o script `scratch/extract_all_hht.py` foi editado no meio da extração ({ctx['ultima_edicao_extract_all_hht'].replace('T', ' ')}), e a extração foi retomada pulando os arquivos já gravados ({ctx['primeiro_pkl_antigo'].replace('T', ' ')} a {ctx['ultimo_pkl_antigo'].replace('T', ' ')}). As {n_arq.get('antes', 0)} gravações extraídas antes da edição usaram um limite de iterações de peneiramento da EMD de pelo menos 100; as {n_arq.get('depois', 0)} seguintes, 30. **Nenhuma das {bg_antes + bg_depois} gravações BACKGROUND está no primeiro grupo**, que tem {navios_antes} dos {navios_tot} navios rotulados: a configuração de extração ficou associada à classe.

O HHT foi re-extraído com uma configuração única (8 IMFs, `MAX_ITERATION=30`, o valor do script atual) e os 22 experimentos HHT foram retreinados. Este notebook documenta a evidência, a correção e o efeito em cada resultado.

> Todos os números são recalculados das predições por janela (`notebooks/iara_results.py`; voto majoritário por arquivo, ACC balanceada, desvio populacional nos 10 folds). **Antes** = `results/trainings/_backup_hht_iter_misto_20261015/<exp>`; **depois** = `results/trainings/multirep/<exp>`. As evidências da extração vêm de `scratch/audit_hht_mixed_extraction.py` (`reports/correcao_hht/`).
"""))
    cells.append(code(SETUP_CODE))

    cells.append(md(f"""
## 1. A evidência

### 1.1 Duas configurações de EMD no mesmo cache
`scratch/audit_hht_mixed_extraction.py` recalculou, com a cadeia de `extract_all_hht.py`, {ctx['amostra_reproducao']['por_grupo']} gravações sorteadas de cada grupo (janelas {', '.join(map(str, ctx['amostra_reproducao']['janelas']))}) com `MAX_ITERATION` = {', '.join(map(str, ns['ITERS']))} e comparou com o espectro antigo. Percentual de janelas reproduzidas **exatamente** (diferença máxima < 1e-6):

| Grupo do arquivo antigo | {' | '.join(f'MAX_ITERATION={it}' for it in ns['ITERS'])} |
| :--- | {' | '.join([':---:'] * len(ns['ITERS']))} |
| antes da edição ({n_arq.get('antes', 0)} gravações) | {' | '.join(br(rp[('antes', it)], 1) + '%' for it in ns['ITERS'])} |
| depois da edição ({n_arq.get('depois', 0)} gravações) | {' | '.join(br(rp[('depois', it)], 1) + '%' for it in ns['ITERS'])} |

* O grupo "antes" é reproduzido com 100 ou mais iterações e só parcialmente com 30 (as janelas em que o peneiramento converge antes de 30 iterações dão o mesmo resultado nas duas configurações). O grupo "depois" é reproduzido com 30.
* O grupo "depois" não é reproduzido bit a bit em todas as janelas: com tolerância de 1e-3, {br(rpt[('depois', 30)], 1)}% das janelas batem com 30 iterações. A EMD é sensível a diferenças numéricas mínimas (o ambiente pode ter mudado desde 15/10). A re-extração reproduz a si mesma exatamente no ambiente atual (conferido nas gravações 1 e 1706).
* {br(100 - rp[('antes', 100)], 1)}% das janelas testadas do grupo "antes" não batem com nenhuma das configurações testadas; pode haver outras diferenças não identificadas.
"""))
    cells.append(code(EVID_CODE + r'''
print('Janelas reproduzidas exatamente (%, max|dif| < 1e-6):'); display(tab_rep)
print('Janelas reproduzidas com tolerância 1e-3 (%):'); display(tab_rep_tol)
'''))
    cells.append(md(f"""
### 1.2 Por que isso é grave: a configuração está associada à classe

| Classe | antes (≥ 100 iterações) | depois (30 iterações) |
| :--- | :---: | :---: |
""" + "".join(f"| {c} | {int(g.loc[c, 'antes'])} | {int(g.loc[c, 'depois'])} |\n" for c in ['SMALL', 'MEDIUM', 'LARGE', 'BACKGROUND']) + f"""| **Total** | **{n_antes}** | **{n_depois}** |

Um classificador pode aprender a "assinatura" da configuração de extração em vez da acústica: um espectro com cara de ≥ 100 iterações nunca é BACKGROUND. Isso tende a inflar a separação navio × BACKGROUND de qualquer modelo HHT e de tudo o que usa suas predições.
"""))
    cells.append(code(r'''
display(tab_grupo)
'''))
    cells.append(md(f"""
### 1.3 Quanto a feature mudou na re-extração

| Grupo | Gravações | Janelas que mudaram (> 1e-3) | Mediana da diferença máxima por gravação |
| :--- | :---: | :---: | :---: |
| antes | {int(mud.loc['antes', 'gravacoes'])} | {br(mud.loc['antes', 'janelas_mudaram_pct'], 1)}% | {br(mud.loc['antes', 'max_dif_mediana'], 3)} |
| depois | {int(mud.loc['depois', 'gravacoes'])} | {br(mud.loc['depois', 'janelas_mudaram_pct'], 1)}% | {br(mud.loc['depois', 'max_dif_mediana'], 3)} |

(Escala da feature: cada janela é normalizada pelo próprio máximo, de 0 a 1.)
"""))
    cells.append(code(r'''
display(tab_mud)
'''))

    cells.append(md(f"""
## 2. A correção
* `scratch/extract_all_hht.py` sem alterações (8 IMFs, `MAX_ITERATION=30`), rodado do zero em todas as {len(arq)} gravações, com o cache antigo movido para `data/iara_processed/_backup_hht_iter_misto_20261015/`.
* Os 22 experimentos HHT foram retreinados com os mesmos scripts, hiperparâmetros e nomes: `scratch/eval_hht_models_5x2cv.py` (RF, SVM m=6000, MLP sem semente, CNN 2D b32), `scratch/eval_hht_spectral_cnn_5x2cv.py` (CNN espectral 1D) e `scratch/rerun_hht_after_reextraction.py` (só o ramo HHT de `save_svm_all_representations.py`, `analyze_small_errors_by_input.py`, `generate_mlp_predictions_and_analysis.py`, `generate_cnn_predictions_and_analysis.py` e das duas BiGRUs, sem retreinar MEL, LOFAR ou DEMON).
* As predições antigas ficam em `results/trainings/_backup_hht_iter_misto_20261015/` (versionadas em `results/predicoes/`).

## 3. Antes × depois em cada experimento HHT
F-test 5×2cv (Alpaydin) entre a versão depois e a antes. Nos modelos **com semente fixa em CPU** (RF e SVMs), a diferença vem só da feature. Nos modelos **sem semente**, parte da diferença é aleatoriedade do treino, que o teste não separa.
"""))
    cells.append(code(COMPARE_CODE + r'''
display(cmp[cols].round(3))
'''))
    cells.append(code(PLOT_CODE))
    cells.append(md(f"""
### 3.1 Leitura
* **{n_down} de {len(cmp)}** combinações modelo × modo têm SP menor depois da correção. Diferenças significativas a 5%: {sig_txt}.
* **Modelos com semente fixa em CPU (a comparação mais limpa):** {seeded_txt}.
* **RF HHT**, o melhor modelo HHT antes da correção: 4c {br(rf4['SP antes'])} → **{br(rf4['SP depois'])}** ({pv(rf4['p SP'])}); 3c {br(rf3['SP antes'])} → {br(rf3['SP depois'])} ({pv(rf3['p SP'])}).
* **Melhor modelo só com HHT em 4c:** antes, {best_old4.Modelo} (SP {br(best_old4['SP antes'])}); depois, {best_new4.Modelo} (SP {br(best_new4['SP depois'])}){'' if p_best_rf is None else f", contra o RF corrigido {pv(p_best_rf)}" + (" (empate estatístico)" if p_best_rf >= 0.05 else "")}.
* **Fusões com HHT (BiGRU MEL + HHT e LOFAR + HHT):** {'; '.join(f"{r.Modelo}: {br(r['SP antes'])} → {br(r['SP depois'])} ({pv(r['p SP'])})" for _, r in fus.iterrows())}.
* **Recall de BACKGROUND (4c), onde o confundimento deveria aparecer:** mudança significativa em {bg_txt}.
"""))

    exec(GATING_CODE, ns)
    gs = ns["gate_sis"]
    big = gs.loc[gs["Δ SP"].abs().idxmax()]
    gated = gs[~gs.Sistema.str.contains("Oráculo|Super")]
    cells.append(md(f"""
## 4. Efeito no gating (Notebook 26)
A entrada `hht` dos pickles de predições por arquivo foi trocada pelas predições corrigidas e os três experimentos de gating (`scratch/test_competence_gating_all_models.py`, `test_competence_gating_svm.py`, `run_heterogeneous_gating_experiment.py`) foram refeitos. Os baselines MEL, LOFAR e DEMON não mudam; os roteadores são retreinados, porque a competência do especialista HHT mudou.

* Nos {len(gated)} sistemas com roteador, o SP muda entre {br(gated['Δ SP'].min())} e {br(gated['Δ SP'].max())} p.p.; a maior mudança entre todos os sistemas (incluindo os oráculos) é em {big.Família + ' ' if big.Família else ''}{big.Sistema} ({big['SP antes']} → {big['SP depois']}).
* H-MoE especializado (MEL:RF, LOFAR:MLP, HHT:RF, DEMON:MLP), o único com SP por fold no pickle: {br(ns['hmoe_old'].mean())} → {br(ns['hmoe_new'].mean())} (F-test 5×2cv: {pv(ns['p_hmoe'])}).
* Nenhum sistema de gating muda mais que {br(gated['Δ SP'].abs().max())} p.p.; o efeito da correção do HHT no gating é pequeno.
"""))
    cells.append(code(GATING_CODE + r'''
display(gate_sis)
print(f'H-MoE especializado, SP por fold antes: {hmoe_old.round(2).tolist()}')
print(f'H-MoE especializado, SP por fold depois: {hmoe_new.round(2).tolist()}  (F-test: F = {F_hmoe:.2f}, p = {p_hmoe:.3f})')
'''))

    cells.append(md(r"""
## 5. Outros resultados derivados
Notebooks que usam predições HHT e foram regenerados com as predições corrigidas:
* **10** (análise HHT) e **11** (comparação de modelos HHT);
* **13–15** (complementaridade SMALL/MEDIUM/LARGE com o SVM HHT m=4000) e **16–24** (complementaridade por família, MLP/RF/CNN);
* **25** (BiGRU, inclusive as fusões MEL + HHT e LOFAR + HHT);
* **26** (gating): a entrada `hht` de `data/iara_processed/{rf,mlp,svm,cnn}_all_representations_predictions.pkl` foi atualizada com `scratch/update_hht_in_all_representations_pickles.py` e os experimentos de gating foram refeitos;
* **27** (super-oráculo).

Os números HHT de `reports/comparacao_modelos.md` foram recalculados pelo seu script; os relatórios com números digitados à mão (`reports/estudo_viabilidade_hht.md`, `analise_bigru_multiclasse_demon_hht.md`, `analise_bigru_completa_mel_lofar.md`, `implementacao_gating_frontend.md`) receberam uma nota de correção datada com os valores atuais.
"""))
    cells.append(md(f"""
## 6. Conclusões
1. **O problema era real:** o cache HHT original misturava duas configurações de EMD, e a configuração estava associada à classe (nenhuma gravação BACKGROUND no grupo extraído com ≥ 100 iterações). Isso viola a premissa de que todas as amostras passam pela mesma extração.
2. **O efeito mensurável ficou concentrado no Random Forest:** SP 4c {br(rf4['SP antes'])} → {br(rf4['SP depois'])} ({pv(rf4['p SP'])}) e recall de BACKGROUND {br(rf4['R_BACKGROUND antes'])} → {br(rf4['R_BACKGROUND depois'])}% ({pv(rf4['p R_BACKGROUND'])}). É o padrão esperado do confundimento. *Hipótese (não testada):* árvores separam a "assinatura" da extração com cortes em poucos bins, e essa assinatura dizia "não é BACKGROUND".
3. **Nos outros {len(cmp) - len(sig)} casos a diferença não é significativa a 5%** (|Δ SP| ≤ {br(cmp.loc[cmp['p SP'] >= 0.05, 'Δ SP'].abs().max())} p.p.). Isso não prova que esses modelos não usaram a assinatura, só que o efeito, se existiu, ficou dentro da variação entre folds; nos modelos sem semente a comparação também carrega a aleatoriedade do treino.
4. **Ranking HHT corrigido (4c, só HHT):** o melhor em SP médio passa a ser {best_new4.Modelo} (SP {br(best_new4['SP depois'])}); o RF corrigido fica em {br(rf4['SP depois'])}{'' if p_best_rf is None else f' (diferença {pv(p_best_rf)}' + (', empate estatístico)' if p_best_rf >= 0.05 else ')')}.
5. **Gating:** efeito pequeno (≤ {br(gated['Δ SP'].abs().max())} p.p. nos sistemas com roteador; H-MoE especializado {pv(ns['p_hmoe'])}).
6. **Prevenção:** o cache de features deve ser gerado de uma vez, com a configuração gravada junto (como o `config.json` das features LOFAR/MEL do pipeline iara), e conferido por recálculo de uma amostra antes de treinar.
"""))

    notebook = {
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
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Notebook {filename} gerado com sucesso!")


if __name__ == "__main__":
    build()
