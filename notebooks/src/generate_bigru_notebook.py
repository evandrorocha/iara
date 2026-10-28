"""
Gera notebooks/25_avaliacao_multiclasse_bigru.ipynb: avaliação do BiGRU com Atenção Temporal
(MEL, LOFAR, DEMON, HHT e fusões) em 4 e 3 classes sob o 5x2cv oficial.

Todos os números vêm das predições gravadas em disco, via notebooks/iara_results.py:
  * BiGRU: results/trainings/multirep/bigru_att_*_{4c,3c}
      - scratch/eval_bigru_multiclass_5x2cv.py       (h64:  demon, hht, mel5s, mel5s-demon, mel5s-hht)
      - scratch/eval_bigru_lofar_multiclass_5x2cv.py (h128: lofar5s, lofar5s-demon, lofar5s-hht)
  * baselines locais: results/trainings/tests/{mlp_lofar, mlp, cnn, svm_nystroem_4000_log_melgram_...,
    svm_nystroem_6000_lofar_...} e a CRNN 2D LOFAR (crnn2d_multiclass_lofar_..., arquivos <exp>_test.csv);
  * artigo: R.ARTICLE_TABLE7.
Os números citados no markdown são calculados aqui, na geração (nada digitado).
"""
import os
import sys
import json
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iara_results as R  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------------------------------
BIGRU = [  # (rótulo curto, rótulo, tag do experimento sem o sufixo _4c/_3c, dimensão)
    ("DEMON", "DEMON", "bigru_att_demon_h64_s15-5_e20", 253),
    ("HHT", "HHT", "bigru_att_hht_h64_s15-5_e20", 256),
    ("MEL", "MEL (mel5s)", "bigru_att_mel5s_h64_s15-5_e20", 256),
    ("MEL+DEM", "FUSÃO MEL + DEMON", "bigru_att_mel5s-demon_h64_s15-5_e20", 509),
    ("MEL+HHT", "FUSÃO MEL + HHT", "bigru_att_mel5s-hht_h64_s15-5_e20", 512),
    ("LOFAR", "LOFAR (lofar5s)", "bigru_att_lofar5s_h128_s15-5_e15", 1024),
    ("LOF+DEM", "FUSÃO LOFAR + DEMON", "bigru_att_lofar5s-demon_h128_s15-5_e15", 1277),
    ("LOF+HHT", "FUSÃO LOFAR + HHT", "bigru_att_lofar5s-hht_h128_s15-5_e15", 1280),
]
BASELINES = [
    ("MLP LOFAR (≈ artigo)", "mlp_lofar"),
    ("MLP MEL (≈ artigo)", "mlp"),
    ("CNN MEL (repo, ≠ artigo)", "cnn"),
    ("SVM Nyström m=4000 MEL", "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0"),
    ("SVM Nyström m=6000 LOFAR", "svm_nystroem_6000_lofar_elasticnet_l1r0.15"),
]
CRNN = "crnn2d_multiclass_lofar_s10_fpool32_h128_lr0.0002"
REF = "mlp_lofar"

# Código de apoio compartilhado entre a geração e o notebook (fonte única).
HELPER_SRC = r'''
def crnn_fold_metrics(name="crnn2d_multiclass_lofar_s10_fpool32_h128_lr0.0002"):
    """A CRNN 2D grava <exp>_test.csv (sem '_multiclass'); aplica as mesmas convenções de iara_results:
    voto por arquivo com Counter.most_common, SP, ACC balanceada, desvio ddof=0."""
    d = os.path.join(R.ROOT, "results", "trainings", "tests", name)
    rows = []
    for k in range(10):
        df = pd.read_csv(os.path.join(d, "eval", f"fold_{k}", f"{name}_test.csv"), usecols=["File", "Target", "Prediction"])
        y, p = [], []
        for fid, g in df.groupby("File", sort=True):
            y.append(Counter(g["Target"]).most_common(1)[0][0])
            p.append(Counter(g["Prediction"].tolist()).most_common(1)[0][0])
        rows.append({"fold": k, **R._metrics(np.array(y), np.array(p))})
    return pd.DataFrame(rows).set_index("fold")


def ms(series):
    return series.mean(), series.std(ddof=0)
'''
exec(HELPER_SRC)


def f2(x, nd=2):
    """Número com vírgula decimal para LaTeX: 63,80 -> 63{,}80."""
    return f"{x:.{nd}f}".replace(".", "{,}")


# ---------------------------------------------------------------------------------------------
# Números usados no markdown
# ---------------------------------------------------------------------------------------------
S4 = {lab: R.summary(tag + "_4c", "4c") for _, lab, tag, _ in BIGRU}
S3 = {lab: R.summary(tag + "_3c", "3c") for _, lab, tag, _ in BIGRU}
SB = {name: R.summary(name) for _, name in BASELINES}
crnn = crnn_fold_metrics(CRNN)
SCRNN = {c: ms(crnn[c]) for c in crnn.columns}
FT = {lab: R.f_test_5x2(tag + "_4c", REF) for _, lab, tag, _ in BIGRU}
ART_MLP_LOFAR = R.ARTICLE_TABLE7[("MLP", "LOFAR")]["SP"]
ART_CNN_LOFAR = R.ARTICLE_TABLE7[("CNN", "LOFAR")]["SP"]

best4 = max(S4, key=lambda k: S4[k]["SP"][0])
best3 = max(S3, key=lambda k: S3[k]["SP"][0])
n_sig_worse = [lab for lab in FT if FT[lab][1] < 0.05 and S4[lab]["SP"][0] < SB[REF]["SP"][0]]
n_equiv = [lab for lab in FT if FT[lab][1] >= 0.05]
p_equiv_min = min(FT[lab][1] for lab in n_equiv) if n_equiv else float("nan")

tag_of = {lab: tag for _, lab, tag, _ in BIGRU}
F_small_lofar_mel = R.f_test_5x2(tag_of["LOFAR (lofar5s)"] + "_4c", tag_of["MEL (mel5s)"] + "_4c", metric="R_SMALL")
F_small_ld_lofar = R.f_test_5x2(tag_of["FUSÃO LOFAR + DEMON"] + "_4c", tag_of["LOFAR (lofar5s)"] + "_4c", metric="R_SMALL")
F_small_ld_mlp = R.f_test_5x2(tag_of["FUSÃO LOFAR + DEMON"] + "_4c", REF, metric="R_SMALL")
F_med_lofar_mel = R.f_test_5x2(tag_of["LOFAR (lofar5s)"] + "_4c", tag_of["MEL (mel5s)"] + "_4c", metric="R_MEDIUM")
F_crnn_bigru_lofar = R.f_test_5x2(crnn["SP"].values, tag_of["LOFAR (lofar5s)"] + "_4c")
F_md_mel = R.f_test_5x2(tag_of["FUSÃO MEL + DEMON"] + "_4c", tag_of["MEL (mel5s)"] + "_4c")
F_mh_mel = R.f_test_5x2(tag_of["FUSÃO MEL + HHT"] + "_4c", tag_of["MEL (mel5s)"] + "_4c")
F_ld_lofar = R.f_test_5x2(tag_of["FUSÃO LOFAR + DEMON"] + "_4c", tag_of["LOFAR (lofar5s)"] + "_4c")

cfgA = R.config(tag_of["MEL (mel5s)"] + "_4c")["params"]
cfgB = R.config(tag_of["LOFAR (lofar5s)"] + "_4c")["params"]
step = cfgA["frame_step_s"]
L, stride = cfgA["seq_len_frames"], cfgA["seq_stride_frames"]
assert (cfgB["frame_step_s"], cfgB["seq_len_frames"], cfgB["seq_stride_frames"]) == (step, L, stride)

# arquivos de validação dos folds oficiais (não usados pelos scripts do BiGRU)
_val = os.path.join(R.ROOT, "results", "trainings", "tests", "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0",
                    "eval", "fold_0", "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_val.csv")
N_VAL = pd.read_csv(_val, usecols=["File"]).File.nunique()
N_TEST4 = len(R.file_predictions(tag_of["MEL (mel5s)"] + "_4c")[0])
N_TEST3 = len(R.file_predictions(tag_of["MEL (mel5s)"] + "_3c")[0])
_nw = pd.concat(R.file_predictions(tag_of["MEL (mel5s)"] + "_4c").values()).n_windows
NSEQ_MIN, NSEQ_MAX, NSEQ_MEAN = int(_nw.min()), int(_nw.max()), _nw.mean()

lab_lofar, lab_mel, lab_ld = "LOFAR (lofar5s)", "MEL (mel5s)", "FUSÃO LOFAR + DEMON"
neural_small = {**{f"BiGRU {lab} (4c)": S4[lab]["R_SMALL"][0] for lab in S4},
                "MLP LOFAR (≈ artigo)": SB["mlp_lofar"]["R_SMALL"][0], "MLP MEL (≈ artigo)": SB["mlp"]["R_SMALL"][0],
                "CNN MEL (repo, ≠ artigo)": SB["cnn"]["R_SMALL"][0], "CRNN 2D LOFAR (local)": SCRNN["R_SMALL"][0]}
CRNN_SUMMARY_SP = pd.read_csv(os.path.join(R.ROOT, 'results', 'trainings', 'tests', CRNN, 'metrics_summary.csv')).set_index('metric').loc['sp', 'mean']
best_small_name = max(neural_small, key=neural_small.get)


def sig(p):
    return "diferença significativa" if p < 0.05 else "diferença não significativa"


def pfmt(p):
    return f"{p:.3f}".replace(".", "{,}") if p >= 0.001 else "< 0{,}001"


# ---------------------------------------------------------------------------------------------
# Células
# ---------------------------------------------------------------------------------------------
def md(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def create_bigru_notebook(filename="notebooks/25_avaliacao_multiclasse_bigru.ipynb"):
    cells = []
    cells.append(md(f"""
# Estudo de Redes Neurais Recorrentes — Dataset IARA
### Notebook 25: Avaliação Completa do BiGRU com Atenção Temporal (MEL, LOFAR, DEMON, HHT e Fusões Multimodais)

Neste notebook, investigamos a arquitetura **Bidirectional Gated Recurrent Unit (BiGRU) com Mecanismo de Atenção Temporal** no problema de classificação acústica submarina passiva (*Underwater Acoustic Target Recognition* - UATR) sobre o dataset **IARA** (*Silva et al.*, IEEE Access 2025).

### Objetivos da Investigação:
1. **Avaliação Temporal vs. Estática:** Verificar se a modelagem explícita da dinâmica temporal de sequências de $L={L}$ frames supera os classificadores estáticos (SVM Nyström, MLP e CNN). Os frames dos caches usados aqui têm passo de **${f2(step)}\\text{{ s}}$** (média das features em blocos de 5,12 s), logo cada sequência cobre **$\\approx {f2(L * step, 1)}\\text{{ s}}$** de áudio, com deslocamento de {stride} frames ($\\approx {f2(stride * step, 1)}\\text{{ s}}$). Gravações com menos de {L} frames viram uma única sequência completada com zeros. Cada arquivo de teste gera de {NSEQ_MIN} a {NSEQ_MAX} sequências (média ${f2(NSEQ_MEAN)}$), e a predição do arquivo é o voto majoritário das sequências.
2. **Comparação de Representações Espectrais e de Envoltória:**
   - **MEL (256-D)** e **LOFAR (1024-D)**: caches `mel5s` / `lofar5s`, com resolução temporal **10× menor** que as features oficiais de 0,512 s do artigo. Os resultados não são, portanto, diretamente comparáveis aos da Tabela 7 do artigo;
   - **DEMON (253-D):** demodulação de envoltória (passagem de pás do hélice);
   - **HHT Pura (256-D):** espectro marginal de Hilbert-Huang (EMD + Transformada de Hilbert);
   - **Fusões:** `MEL + DEMON` (509-D), `MEL + HHT` (512-D), `LOFAR + DEMON` (1277-D) e `LOFAR + HHT` (1280-D), por concatenação frame a frame (truncadas ao menor número de frames).
3. **Análise de Escopo:** **4 Classes** (`SMALL`, `MEDIUM`, `LARGE`, `BACKGROUND`; {N_TEST4} arquivos de teste por fold) e **3 Classes** (modelo **retreinado** sem `BACKGROUND`; {N_TEST3} arquivos de teste por fold).

**Protocolo:** validação cruzada $5\\times2\\text{{cv}}$ oficial (10 folds, os mesmos arquivos de `results/trainings/tests/*/eval/fold_k`), com particionamento por **gravação de áudio** (`exclusive_header='ID'`, `src/iara/ml/experiment.py`): janelas de uma mesma gravação nunca se misturam entre treino e teste, **mas o mesmo navio pode aparecer em treino e teste** (não há isolamento por `Ship ID`).

**Limitações de treino (declaradas):** não há conjunto de validação nem *early stopping*. Os modelos treinam por um número **fixo** de épocas, e os {N_VAL} arquivos de validação de cada fold oficial são descartados (não entram nem no treino). Os scripts não fixam semente (`torch.manual_seed` ausente), então uma nova execução não reproduz os números exatamente.
"""))

    cells.append(code(r'''
import os
import sys
from collections import Counter
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

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
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
''' + HELPER_SRC + r'''
print('Ambiente configurado. Resultados lidos de', R.ROOT)
'''))

    cells.append(md(f"""
## 1. Configuração dos Dois Scripts (lida do código e de `config.json`)

Há **duas arquiteturas diferentes**, uma por script. Toda comparação MEL × LOFAR neste notebook mistura o efeito da representação com o efeito da arquitetura.

| Item | `eval_bigru_multiclass_5x2cv.py` (DEMON, HHT, MEL, MEL+DEMON, MEL+HHT) | `eval_bigru_lofar_multiclass_5x2cv.py` (LOFAR, LOFAR+DEMON, LOFAR+HHT) |
| :--- | :--- | :--- |
| GRU | 2 camadas, bidirecional, `hidden_size={cfgA['hidden_size']}` (saída {2 * cfgA['hidden_size']}-D), dropout 0,3 entre camadas | 2 camadas, bidirecional, `hidden_size={cfgB['hidden_size']}` (saída {2 * cfgB['hidden_size']}-D), dropout 0,3 entre camadas |
| Atenção | Linear({2 * cfgA['hidden_size']}→{cfgA['hidden_size']}) → tanh → Linear(→1), softmax no tempo | Linear({2 * cfgB['hidden_size']}→{cfgB['hidden_size']}) → tanh → Linear(→1), softmax no tempo |
| Classificador | **BatchNorm1d**({2 * cfgA['hidden_size']}) → Dropout → Linear(→**{cfgA['fc_hidden']}**) → ReLU → **BatchNorm1d**({cfgA['fc_hidden']}) → Dropout → Linear | Linear(→**64**) → ReLU → Dropout → Linear (**sem BatchNorm**) |
| Épocas (fixas) | {cfgA['epochs']} | {cfgB['epochs']} |
| Pesos de classe | `compute_class_weight('balanced')` = $n/(K\\cdot n_c)$ | $1/n_c$ normalizado para somar $K$ |
| `drop_last` no treino | `False` | `True` |
| Comum | AdamW (lr={cfgA['lr']}, wd=1e-3), CosineAnnealingLR, *clip* de gradiente 1,0, lote {cfgA['batch_size']}, `StandardScaler` ajustado nas janelas de treino, $L={L}$, passo {stride} | idem |

**Agregação por arquivo:** os scripts usam `argmax(bincount)` (empate → menor índice de classe). As métricas abaixo são **recalculadas** a partir das predições por sequência gravadas em disco com a convenção do pipeline `iara` (`Counter.most_common`, empate → classe da primeira sequência), por isso podem diferir ligeiramente dos valores impressos pelos scripts. **ACC = acurácia balanceada** (média dos recalls), como a coluna ACC da Tabela 7 do artigo; desvio-padrão populacional sobre os 10 folds.
"""))

    rows_py = ",\n".join(f"    ({lab!r}, {tag!r}, {dim})" for _, lab, tag, dim in BIGRU)
    base_py = ",\n".join(f"    ({lab!r}, {name!r})" for lab, name in BASELINES)
    cells.append(md("""
## 2. Tabela Consolidada de Resultados (10 Folds $5\\times2\\text{cv}$)
"""))
    cells.append(code(f'''
BIGRU = [
{rows_py}
]
BASELINES = [
{base_py}
]

def linha(escopo, modelo, rep, dim, s, exp):
    return {{'Escopo': escopo, 'Modelo': modelo, 'Representacao': rep, 'Dim': dim,
            'SP (%)': s['SP'][0], 'SP_std': s['SP'][1], 'Bal Acc (%)': s['ACC'][0], 'Bal Acc_std': s['ACC'][1],
            'Rec SMALL (%)': s['R_SMALL'][0], 'Rec MED (%)': s['R_MEDIUM'][0], 'Rec LARGE (%)': s['R_LARGE'][0],
            'Rec BG (%)': s['R_BACKGROUND'][0] if 'R_BACKGROUND' in s else np.nan, 'Experimento': exp}}

rows = []
for escopo, suf in [('4 Classes', '4c'), ('3 Classes', '3c')]:
    for rep, tag, dim in BIGRU:
        exp = f'{{tag}}_{{suf}}'
        rows.append(linha(escopo, 'BiGRU + Atenção', rep, dim, R.summary(exp, suf), exp))
for modelo, exp in BASELINES:
    rows.append(linha('4 Classes', modelo, '', np.nan, R.summary(exp), exp))
crnn = crnn_fold_metrics()
rows.append(linha('4 Classes', 'CRNN 2D LOFAR (re-execução local)', 'LOFAR', 1024,
                  {{c: ms(crnn[c]) for c in crnn.columns}}, 'crnn2d_multiclass_lofar_s10_fpool32_h128_lr0.0002'))
for (mod, rep), v in R.ARTICLE_TABLE7.items():
    if mod in ('MLP', 'CNN'):
        rows.append({{'Escopo': '4 Classes', 'Modelo': f'{{mod}} (Tabela 7 do artigo)', 'Representacao': rep, 'Dim': np.nan,
                     'SP (%)': v['SP'][0], 'SP_std': v['SP'][1], 'Bal Acc (%)': v['ACC'][0], 'Bal Acc_std': v['ACC'][1],
                     'Rec SMALL (%)': np.nan, 'Rec MED (%)': np.nan, 'Rec LARGE (%)': np.nan, 'Rec BG (%)': np.nan,
                     'Experimento': 'artigo (features de 0,512 s)'}})

df_results = pd.DataFrame(rows)
display(df_results.round(2))
'''))

    cells.append(md(f"""
## 3. Teste Estatístico: F-test $5\\times2\\text{{cv}}$ contra a MLP LOFAR

O artigo compara modelos pelo F-test $5\\times2\\text{{cv}}$ combinado (Alpaydin). Abaixo, cada BiGRU (4 classes) é testado contra a **MLP LOFAR re-executada localmente** (`mlp_lofar`, SP ${f2(SB[REF]['SP'][0])} \\pm {f2(SB[REF]['SP'][1])}\\%$, mesmos folds). Contra o **artigo** não é possível testar (não há SP por fold publicado); a MLP LOFAR do artigo tem SP ${f2(ART_MLP_LOFAR[0])} \\pm {f2(ART_MLP_LOFAR[1])}\\%$ (Tabela 7), com features de 0,512 s.
"""))
    cells.append(code('''
ft = []
for rep, tag, dim in BIGRU:
    exp = f'{tag}_4c'
    s = R.summary(exp)
    F, p = R.f_test_5x2(exp, 'mlp_lofar')
    ft.append({'BiGRU (4c)': rep, 'SP (%)': R.fmt(s['SP']), 'Δ SP vs MLP LOFAR local (p.p.)': s['SP'][0] - R.summary('mlp_lofar')['SP'][0],
               'F': F, 'p': p, 'Diferença significativa (p<0,05)?': 'sim' if p < 0.05 else 'não',
               'Δ SP vs MLP LOFAR do artigo (p.p.)': s['SP'][0] - R.ARTICLE_TABLE7[('MLP', 'LOFAR')]['SP'][0]})
F, p = R.f_test_5x2(crnn['SP'].values, 'mlp_lofar')
ft.append({'BiGRU (4c)': 'CRNN 2D LOFAR (referência)', 'SP (%)': R.fmt(ms(crnn['SP'])),
           'Δ SP vs MLP LOFAR local (p.p.)': crnn['SP'].mean() - R.summary('mlp_lofar')['SP'][0], 'F': F, 'p': p,
           'Diferença significativa (p<0,05)?': 'sim' if p < 0.05 else 'não',
           'Δ SP vs MLP LOFAR do artigo (p.p.)': crnn['SP'].mean() - R.ARTICLE_TABLE7[('MLP', 'LOFAR')]['SP'][0]})
display(pd.DataFrame(ft).round(3))
'''))

    sig_txt = ", ".join(f"{lab} (p = ${pfmt(FT[lab][1])}$)" for lab in n_sig_worse) or "nenhuma"
    eq_txt = ", ".join(f"{lab} (p = ${pfmt(FT[lab][1])}$)" for lab in n_equiv) or "nenhuma"
    cells.append(md(f"""
**Leitura:** em 4 classes, são **significativamente piores** que a MLP LOFAR local: {sig_txt}. As demais configurações ficam **abaixo em média, mas sem diferença significativa** pelo F-test: {eq_txt}. O melhor BiGRU em 4 classes é **{best4}** (SP ${f2(S4[best4]['SP'][0])} \\pm {f2(S4[best4]['SP'][1])}\\%$), ${f2(SB[REF]['SP'][0] - S4[best4]['SP'][0])}$ p.p. abaixo da MLP LOFAR local e ${f2(ART_MLP_LOFAR[0] - S4[best4]['SP'][0])}$ p.p. abaixo da MLP LOFAR do artigo.
"""))

    cells.append(md(f"""
## 4. Gráfico Comparativo: Índice SP (%) e Acurácia Balanceada por Representação

SP com barras de desvio-padrão entre os 10 folds, para as 8 configurações do BiGRU em **4 Classes** e **3 Classes**. As linhas de referência (só no painel de 4 classes) são a **MLP LOFAR local** (${f2(SB[REF]['SP'][0])}\\%$), a **MLP LOFAR do artigo** (${f2(ART_MLP_LOFAR[0])}\\%$) e o **SVM MEL m=4000 local** (${f2(SB['svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0']['SP'][0])}\\%$). Os rótulos mostram SP e, entre parênteses, a acurácia balanceada.
"""))
    cells.append(code(f'''
df_4c = df_results[(df_results['Escopo'] == '4 Classes') & (df_results['Modelo'] == 'BiGRU + Atenção')]
df_3c = df_results[(df_results['Escopo'] == '3 Classes') & (df_results['Modelo'] == 'BiGRU + Atenção')]
sp_ref = R.summary('mlp_lofar')['SP'][0]
sp_art = R.ARTICLE_TABLE7[('MLP', 'LOFAR')]['SP'][0]
sp_svm = R.summary('svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0')['SP'][0]

reps_labels = {[s for s, _, _, _ in BIGRU]!r}
colors = ['#95a5a6', '#e67e22', '#3498db', '#9b59b6', '#2ecc71', '#34495e', '#e74c3c', '#1abc9c']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
bars1 = ax1.bar(np.arange(len(reps_labels)), df_4c['SP (%)'], yerr=df_4c['SP_std'], capsize=5, color=colors, alpha=0.85, edgecolor='black', lw=1.2)
ax1.axhline(sp_svm, color='blue', linestyle='--', label=f'SVM MEL m=4000 local ({{sp_svm:.1f}}%)', lw=1.6)
ax1.axhline(sp_ref, color='red', linestyle='--', label=f'MLP LOFAR local ({{sp_ref:.1f}}%)', lw=1.6)
ax1.axhline(sp_art, color='darkred', linestyle=':', label=f'MLP LOFAR artigo ({{sp_art:.1f}}%)', lw=1.6)
ax1.set_title("BiGRU com Atenção — 4 Classes (SP Global)", fontsize=13, fontweight='bold', pad=12)
ax1.set_ylabel("Standard Product (SP) %", fontsize=11, fontweight='bold')
ax1.set_xticks(np.arange(len(reps_labels)))
ax1.set_xticklabels(reps_labels, fontsize=10, fontweight='bold', rotation=20)
ax1.set_ylim(45, 72)
ax1.legend(loc='lower left', frameon=True)
for bar, sp, ba in zip(bars1, df_4c['SP (%)'], df_4c['Bal Acc (%)']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.6, f"{{sp:.1f}}%\\n({{ba:.1f}}%)",
             ha='center', va='bottom', fontsize=8.5, fontweight='bold')

bars2 = ax2.bar(np.arange(len(reps_labels)), df_3c['SP (%)'], yerr=df_3c['SP_std'], capsize=5, color=colors, alpha=0.85, edgecolor='black', lw=1.2)
ax2.set_title("BiGRU com Atenção — 3 Classes (modelo retreinado sem BACKGROUND)", fontsize=13, fontweight='bold', pad=12)
ax2.set_ylabel("Standard Product (SP) %", fontsize=11, fontweight='bold')
ax2.set_xticks(np.arange(len(reps_labels)))
ax2.set_xticklabels(reps_labels, fontsize=10, fontweight='bold', rotation=20)
ax2.set_ylim(45, 70)
for bar, sp, ba in zip(bars2, df_3c['SP (%)'], df_3c['Bal Acc (%)']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.6, f"{{sp:.1f}}%\\n({{ba:.1f}}%)",
             ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.show()
'''))

    cells.append(md("""
## 5. Recalls por Classe: o Alvo Crítico SMALL

A principal dificuldade no IARA é reconhecer embarcações de pequeno porte (**SMALL**). O gráfico mostra os recalls médios por classe de cada configuração:
"""))
    cells.append(code('''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
w = 0.25
x = np.arange(len(reps_labels))
for ax, d, titulo in [(ax1, df_4c, "Recalls por Classe em 4 Classes"), (ax2, df_3c, "Recalls por Classe em 3 Classes (retreinado sem BACKGROUND)")]:
    ax.bar(x - w, d['Rec SMALL (%)'], width=w, label='SMALL', color='#3498db', edgecolor='black', alpha=0.85)
    ax.bar(x, d['Rec MED (%)'], width=w, label='MEDIUM', color='#f39c12', edgecolor='black', alpha=0.85)
    ax.bar(x + w, d['Rec LARGE (%)'], width=w, label='LARGE', color='#2ecc71', edgecolor='black', alpha=0.85)
    ax.set_title(titulo, fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel("Recall (%)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(reps_labels, fontsize=9.5, fontweight='bold', rotation=20)
    ax.set_ylim(20, 85)
    ax.legend(loc='upper left', frameon=True)
    for i in range(len(reps_labels)):
        for dx, col in [(-w, 'Rec SMALL (%)'), (0, 'Rec MED (%)'), (w, 'Rec LARGE (%)')]:
            ax.text(x[i] + dx, d[col].iloc[i] + 1, f"{d[col].iloc[i]:.0f}%", ha='center', fontsize=7.5, fontweight='bold')
plt.tight_layout()
plt.show()

# F-test 5x2cv sobre o recall de SMALL / MEDIUM (por fold)
pares = [('LOFAR vs MEL (SMALL)', 'bigru_att_lofar5s_h128_s15-5_e15_4c', 'bigru_att_mel5s_h64_s15-5_e20_4c', 'R_SMALL'),
         ('LOFAR+DEMON vs LOFAR (SMALL)', 'bigru_att_lofar5s-demon_h128_s15-5_e15_4c', 'bigru_att_lofar5s_h128_s15-5_e15_4c', 'R_SMALL'),
         ('LOFAR+DEMON vs MLP LOFAR local (SMALL)', 'bigru_att_lofar5s-demon_h128_s15-5_e15_4c', 'mlp_lofar', 'R_SMALL'),
         ('LOFAR vs MEL (MEDIUM)', 'bigru_att_lofar5s_h128_s15-5_e15_4c', 'bigru_att_mel5s_h64_s15-5_e20_4c', 'R_MEDIUM')]
out = []
for nome, a, b, m in pares:
    F, p = R.f_test_5x2(a, b, metric=m)
    out.append({'Comparação': nome, 'A (%)': R.summary(a)[m][0], 'B (%)': R.summary(b)[m][0], 'F': F, 'p': p})
display(pd.DataFrame(out).round(3))
'''))

    s_mel, s_lof, s_ld = S4[lab_mel], S4[lab_lofar], S4[lab_ld]
    cells.append(md(f"""
## 6. Discussão Científica e Conclusões

### A. Existe um "teto estrutural" dos modelos temporais?
- A versão anterior deste notebook afirmava um teto de ~62% para BiGRU/CRNN. Com as predições re-executadas, o melhor BiGRU em 4 classes é **{best4}** com SP ${f2(S4[best4]['SP'][0])} \\pm {f2(S4[best4]['SP'][1])}\\%$, e em 3 classes **{best3}** com ${f2(S3[best3]['SP'][0])} \\pm {f2(S3[best3]['SP'][1])}\\%$.
- Contra a MLP LOFAR local (${f2(SB[REF]['SP'][0])} \\pm {f2(SB[REF]['SP'][1])}\\%$), só {sig_txt} são significativamente piores. As outras configurações são **estatisticamente equivalentes** pelo F-test $5\\times2\\text{{cv}}$ (p ≥ ${pfmt(p_equiv_min)}$), embora fiquem abaixo em média. Não há, portanto, evidência de um "teto" específico dos modelos recorrentes; também não há evidência de que eles superem os estáticos.
- As fusões não trazem ganho significativo sobre a representação base: MEL+DEMON × MEL p = ${pfmt(F_md_mel[1])}$; MEL+HHT × MEL p = ${pfmt(F_mh_mel[1])}$; LOFAR+DEMON × LOFAR p = ${pfmt(F_ld_lofar[1])}$.
- A CRNN 2D LOFAR re-executada (`results/trainings/tests/crnn2d_multiclass_lofar_s10_fpool32_h128_lr0.0002`, recalculada com a convenção `iara`; o `metrics_summary.csv` gravado pelo script de treino dá SP ${f2(CRNN_SUMMARY_SP)}\\%$) tem SP ${f2(SCRNN['SP'][0])} \\pm {f2(SCRNN['SP'][1])}\\%$ (recalls SMALL/MEDIUM/LARGE = ${f2(SCRNN['R_SMALL'][0], 1)}$ / ${f2(SCRNN['R_MEDIUM'][0], 1)}$ / ${f2(SCRNN['R_LARGE'][0], 1)}\\%$); contra o BiGRU LOFAR (${f2(s_lof['SP'][0])}\\%$), p = ${pfmt(F_crnn_bigru_lofar[1])}$.
- **Ressalvas que limitam a interpretação:** (i) as features MEL/LOFAR do BiGRU têm resolução de 5,12 s, 10× menor que a do artigo, então a comparação com a Tabela 7 mistura modelo e representação; (ii) épocas fixas, sem validação nem *early stopping*; (iii) as variantes MEL e LOFAR usam arquiteturas diferentes (Seção 1); (iv) o protocolo separa gravações, não navios. Hipóteses como "contaminação dos estados ocultos por variações de SNR" ou "cadência temporal específica dos navios de treino" **não foram testadas** neste estudo.

### B. Recall de SMALL
- O BiGRU MEL tem recall de SMALL de ${f2(s_mel['R_SMALL'][0], 1)}\\%$ e o BiGRU LOFAR, ${f2(s_lof['R_SMALL'][0], 1)}\\%$ (F-test sobre o recall de SMALL: p = ${pfmt(F_small_lofar_mel[1])}$, {sig(F_small_lofar_mel[1])}). Mas a troca de entrada também troca a arquitetura, e o recall de MEDIUM cai de ${f2(s_mel['R_MEDIUM'][0], 1)}\\%$ para ${f2(s_lof['R_MEDIUM'][0], 1)}\\%$ (p = ${pfmt(F_med_lofar_mel[1])}$, {sig(F_med_lofar_mel[1])}).
- A fusão LOFAR + DEMON chega a ${f2(s_ld['R_SMALL'][0], 1)}\\%$ em 4 classes (vs. LOFAR puro, p = ${pfmt(F_small_ld_lofar[1])}$, {sig(F_small_ld_lofar[1])}) e ${f2(S3[lab_ld]['R_SMALL'][0], 1)}\\%$ em 3 classes. Contra a MLP LOFAR local (${f2(SB[REF]['R_SMALL'][0], 1)}\\%$), p = ${pfmt(F_small_ld_mlp[1])}$ ({sig(F_small_ld_mlp[1])}).
- Entre as redes neurais desta tabela, o maior recall de SMALL em 4 classes é o de **{best_small_name}** (${f2(neural_small[best_small_name], 1)}\\%$). Isso vale só para os modelos listados aqui; não é uma afirmação sobre "todas as redes do estudo".

### C. Recomendação Prática
Os classificadores estáticos (MLP e SVM) continuam sendo a opção mais simples: SP estatisticamente equivalente ao dos melhores BiGRU (a MLP LOFAR local ainda é maior em média), menor custo e sem dependência de sequências de $\\approx {f2(L * step, 0)}\\text{{ s}}$. O BiGRU com LOFAR (+ DEMON) mostra um perfil de recalls diferente (mais SMALL, menos MEDIUM; só a queda de MEDIUM é significativa nos testes acima). Isso o torna um **candidato** a especialista de SMALL, mas a hipótese só se sustenta depois de um teste num sistema realizável (por exemplo, uma cascata avaliada no 5x2cv), e não apenas comparando recalls.
"""))

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.10.12"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Notebook gerado com sucesso em '{filename}'!")


if __name__ == "__main__":
    create_bigru_notebook()
