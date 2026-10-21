"""Gera os notebooks 1–4 (comparação geral, opção de rejeição, CPA e trajetória do SVM).

Nenhum número é digitado: tudo é calculado a partir das predições por janela em disco
(results/trainings/tests/... e results/trainings/cpa_exp/...) via notebooks/iara_results.py.
Os números citados no markdown são calculados aqui, na hora da geração, e as células de código
dos notebooks recalculam os mesmos valores durante a execução.
Valores do artigo vêm de R.ARTICLE_TABLE7 ou são citados explicitamente como "Tabela X do artigo".

Uso (cwd = raiz do repositório): python notebooks/generate_notebooks.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import iara_results as R  # noqa: E402


def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Notebook {filename} criado com sucesso!")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


# =====================================================================================
# Funções auxiliares compartilhadas: executadas aqui (para os números do markdown) e
# copiadas literalmente para uma célula dos notebooks (para o recálculo na execução).
# =====================================================================================
HELPERS = r'''
import glob
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

if os.path.abspath('.') not in sys.path:
    sys.path.insert(0, os.path.abspath('.'))   # cwd = notebooks/
import iara_results as R

TESTS_DIR = os.path.join(R.ROOT, 'results', 'trainings', 'tests')
CPA_DIR = os.path.join(R.ROOT, 'results', 'trainings', 'cpa_exp')
LIMIARES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]


def ms(v):
    """(média, desvio populacional ddof=0) — mesma convenção de iara_results."""
    v = np.asarray(v, dtype=float)
    return (float(np.mean(v)), float(np.std(v, ddof=0)))


def votos_csv(path):
    """Uma linha por arquivo: voto majoritário das janelas (Counter.most_common, igual a
    R.file_predictions) e a concordância = fração das janelas que votaram na classe vencedora."""
    df = pd.read_csv(path, usecols=['File', 'Target', 'Prediction'])
    rows = []
    for fid, g in df.groupby('File', sort=True):
        c = Counter(g['Prediction'].tolist())
        pred = c.most_common(1)[0][0]
        rows.append((fid, Counter(g['Target']).most_common(1)[0][0], pred, len(g), c[pred] / len(g)))
    return pd.DataFrame(rows, columns=['File', 'Target', 'Prediction', 'n_windows', 'agreement'])


def metricas_csvs(paths):
    """DataFrame por fold (SP, ACC balanceada, ACC_micro, F1, recalls) a partir de 10 CSVs por janela."""
    return pd.DataFrame([R._metrics(d.Target.values, d.Prediction.values) for d in map(votos_csv, paths)])


def fold_metrics_exp(exp):
    """R.fold_metrics(exp); se o CSV tiver outro prefixo (ex.: 'svm_nystroem' grava
    'svm_nystroem_300_multiclass_test.csv'), lê o único *_multiclass_test.csv de cada fold."""
    try:
        return R.fold_metrics(exp).reset_index(drop=True)
    except FileNotFoundError:
        paths = []
        for k in range(10):
            found = glob.glob(os.path.join(TESTS_DIR, exp, 'eval', f'fold_{k}', '*_multiclass_test.csv'))
            assert len(found) == 1, (exp, k, found)
            paths.append(found[0])
        return metricas_csvs(paths)


def resumo_exp(exp):
    fm = fold_metrics_exp(exp)
    return {c: ms(fm[c]) for c in fm.columns}


def curva_rejeicao(exp, limiares=LIMIARES):
    """Rejeição por concordância de janelas: aceita o arquivo se agreement >= t.
    Por fold: cobertura, ACC balanceada, ACC micro e SP nos arquivos aceitos; depois média ± desvio."""
    linhas = []
    for t in limiares:
        acc = {'Cobertura': [], 'ACC': [], 'ACC_micro': [], 'SP': []}
        for k, df in R.file_predictions(exp).items():
            a = df[df.agreement >= t]
            m = R._metrics(a.Target.values, a.Prediction.values)
            acc['Cobertura'].append(len(a) / len(df) * 100)
            for key in ('ACC', 'ACC_micro', 'SP'):
                acc[key].append(m[key])
        linhas.append({'t': t, **{key: ms(v) for key, v in acc.items()}})
    return linhas


def rejeicao_por_classe(exp, t):
    """Por classe: nº médio de arquivos por fold, cobertura e recall nos aceitos (média ± desvio nos folds)."""
    out = []
    for c, nome in enumerate(R.CLASSES):
        tot, cob, rec = [], [], []
        for k, df in R.file_predictions(exp).items():
            dc = df[df.Target == c]
            a = dc[dc.agreement >= t]
            tot.append(len(dc))
            cob.append(len(a) / len(dc) * 100)
            rec.append((a.Prediction == c).mean() * 100 if len(a) else np.nan)
        out.append({'Classe': nome, 'Arquivos/fold': float(np.mean(tot)), 'Cobertura': ms(cob),
                    'Recall nos aceitos': (float(np.nanmean(rec)), float(np.nanstd(rec)))})
    return out


def cpa_paths(sub, modelo, treino, teste):
    """CSVs por janela do experimento CPA (training_scripts/cpa_presence*.py + CrossComparator):
    treino == teste -> <sub>/<treino>_<modelo>/eval/fold_k/*_multiclass_test.csv
    treino != teste -> <sub>/comparison/<treino>_<modelo>/*_multiclass_test_<k>.csv"""
    paths = []
    for k in range(10):
        if treino == teste:
            pat = os.path.join(CPA_DIR, sub, f'{treino}_{modelo}', 'eval', f'fold_{k}', '*_multiclass_test.csv')
        else:
            pat = os.path.join(CPA_DIR, sub, 'comparison', f'{treino}_{modelo}', f'*_multiclass_test_{k}.csv')
        found = glob.glob(pat)
        assert len(found) == 1, (pat, found)
        paths.append(found[0])
    return paths


def cpa_metricas(sub, modelo, treino, teste):
    return metricas_csvs(cpa_paths(sub, modelo, treino, teste))
'''

exec(HELPERS)  # define as mesmas funções aqui no gerador
F = R.fmt


def p_txt(p):
    return f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}"


SETUP_PLOT = '''
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})
'''

# Experimentos do pipeline iara oficial (results/trainings/tests/)
SVM_MEL = "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0"
SVM_MEL_PCA = "svm_nystroem_4000_log_melgram_pca64_elasticnet_l1r0.15_C2.0"
SVM_LOFAR = "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0"
SVM_LOFAR_NOPCA = "svm_nystroem_4000_lofar_C2.0"
SVM_LOFAR_6000 = "svm_nystroem_6000_lofar_elasticnet_l1r0.15"
CNN_MEL = "cnn"
MLP_MEL = "mlp"
MLP_LOFAR = "mlp_lofar"

S_MEL, S_LOFAR = R.summary(SVM_MEL), R.summary(SVM_LOFAR)
S_MEL_PCA, S_LOFAR_NOPCA, S_LOFAR_6000 = R.summary(SVM_MEL_PCA), R.summary(SVM_LOFAR_NOPCA), R.summary(SVM_LOFAR_6000)
S_CNN, S_MLP, S_MLP_LOFAR = R.summary(CNN_MEL), R.summary(MLP_MEL), R.summary(MLP_LOFAR)
A7 = R.ARTICLE_TABLE7

FT = {
    "mel_vs_cnn": R.f_test_5x2(SVM_MEL, CNN_MEL),
    "mel_vs_cnn_acc": R.f_test_5x2(SVM_MEL, CNN_MEL, metric="ACC"),
    "mel_vs_mlp": R.f_test_5x2(SVM_MEL, MLP_MEL),
    "lofar_vs_mlp_lofar": R.f_test_5x2(SVM_LOFAR, MLP_LOFAR),
    "lofar_pca_4000": R.f_test_5x2(SVM_LOFAR, SVM_LOFAR_NOPCA),
    "lofar_pca_4000_acc": R.f_test_5x2(SVM_LOFAR, SVM_LOFAR_NOPCA, metric="ACC"),
    "lofar_pca_vs_6000": R.f_test_5x2(SVM_LOFAR, SVM_LOFAR_6000, metric="ACC"),
    "mel_pca_4000": R.f_test_5x2(SVM_MEL_PCA, SVM_MEL),
    "mel_pca_4000_acc": R.f_test_5x2(SVM_MEL_PCA, SVM_MEL, metric="ACC"),
}
RATIO_STD = S_MEL["ACC"][1] / A7[("CNN", "MEL")]["ACC"][1]
# Guarda: o texto do NB1 afirma que nenhuma destas comparações é significativa
assert all(v[1] >= 0.05 for v in FT.values()), FT

# ==================== NOTEBOOK 1: COMPARAÇÃO GERAL ====================
art_rows = "\n".join(
    f"| **{m}** *(Tabela 7 do artigo)* | {f} | {F(A7[(m, f)]['SP'])} | {F(A7[(m, f)]['ACC'])} | {F(A7[(m, f)]['F1'])} |"
    for m in ("RF", "MLP", "CNN") for f in ("MEL", "LOFAR"))

cells_1 = [
    md(r"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 1: Comparação Geral de Desempenho (Sem Opção de Rejeição)

Neste notebook, é apresentada a análise comparativa global entre os modelos baselines estabelecidos no artigo (*Silva et al., 2025*) e a proposta baseada em **Support Vector Machines (SVM) com Aproximação de Nyström** nas representações espectrais **MEL** e **LOFAR**.

> **Nota Metodológica sobre Particionamento e Métricas:**
> * **Particionamento por Gravação de Áudio (`ID`):** Todos os experimentos seguem o protocolo oficial de validação cruzada estratificada $5\times2$cv (10 folds) implementado em `src/iara/ml/experiment.py` (`exclusive_header='ID'`), em conformidade com *Silva et al. (2025)* (Seção V-D): o particionamento é realizado no nível de **gravação de áudio (`ID`)**, garantindo que janelas espectrais extraídas do mesmo arquivo de áudio jamais coexistam nos conjuntos de treino e teste. O isolamento **não** é por navio: o mesmo navio pode aparecer em gravações diferentes no treino e no teste.
> * **Distinção entre Acurácia Balanceada (`Bal. Acc`) e Acurácia Global/Micro (`UACC`):** Na Tabela 7 do artigo original (*Silva et al., 2025*) e no compilador oficial `iara.ml.metrics.GridCompiler`, a coluna denominada `ACC (%)` corresponde à **Acurácia Balanceada (`Bal. Acc`)**, definida como a média aritmética dos recalls (probabilidades de detecção) das 4 classes, e `F1-Score (%)` corresponde ao **Macro-F1**. Essa métrica não deve ser confundida com a **Acurácia Global/Micro (`UACC`, `sklearn.metrics.accuracy_score`)**, que mede a fração bruta de áudios classificados corretamente sem ponderação por classe.
> * **Origem dos números:** as linhas do SVM são recalculadas neste notebook a partir das predições por janela em `results/trainings/tests/<experimento>/eval/fold_k/` (voto majoritário por arquivo, média ± desvio-padrão populacional nos 10 folds), via `notebooks/iara_results.py`. As linhas RF/MLP/CNN são os valores publicados na Tabela 7 do artigo (`R.ARTICLE_TABLE7`).
"""),
    md(r"""
### Parâmetros de Configuração e Treinamento do SVM Nyström:
Para garantir a replicabilidade científica dos experimentos, as seguintes configurações de hiperparâmetros foram empregadas no estimador (conforme o nome dos experimentos `svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0` e `svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0`):
* **Aproximação de Kernel:** Kernel Gaussiano RBF aproximado pelo método de Nyström com $m = 4000$ componentes.
* **Custo de Regularização ($C$):** $C = 2.0$.
* **Regularização / Penalidade:** ElasticNet com razão $L_1 = 0.15$ e $L_2 = 0.85$.
* **Tratamento de Dimensionalidade (MEL):** PCA desativado (as 256 bandas Mel são a entrada direta do estimador).
* **Tratamento de Dimensionalidade (LOFAR):** PCA ativado (projeção linear prévia para $n_{components} = 64$).
"""),
    md(f"""
### Tabela 1: Métricas de Desempenho Geral no Conjunto de Teste (Sem Rejeição)

| Classificador / Arquitetura | Representação Espectral | Índice SP (%) | Acurácia Balanceada (ACC) (%) | F1-Score (Macro) (%) |
| :--- | :--- | :---: | :---: | :---: |
{art_rows}
| **SVM** *(m=4000, C=2.0, ElasticNet — re-execução local)* | MEL | {F(S_MEL['SP'])} | {F(S_MEL['ACC'])} | {F(S_MEL['F1'])} |
| **SVM** *(m=4000, C=2.0, PCA64, ElasticNet — re-execução local)* | LOFAR | {F(S_LOFAR['SP'])} | {F(S_LOFAR['ACC'])} | {F(S_LOFAR['F1'])} |

*Nota: para referência cruzada com análises baseadas em contagem micro de arquivos, a Acurácia Global/Micro (`UACC`) do SVM MEL ($m=4000$) é de ${F(S_MEL['ACC_micro'])}\\%$ e a do SVM LOFAR ($m=4000$, PCA64) é de ${F(S_LOFAR['ACC_micro'])}\\%$. As linhas do artigo não têm resultados por fold em disco; por isso, os testes de significância (Seção 3) usam as re-execuções locais dos baselines.*
"""),
    code(HELPERS),
    code(SETUP_PLOT),
    md(r"""
### 1. Definição do Conjunto de Dados
Os resultados obtidos sob o protocolo de **Validação Cruzada 5x2 (10 folds)** com **particionamento por gravação de áudio (`ID`)** (`exclusive_header='ID'`) são montados no DataFrame abaixo: as linhas RF/MLP/CNN vêm de `R.ARTICLE_TABLE7` (Tabela 7 do artigo) e as do SVM são recalculadas a partir das predições em disco.
"""),
    code(f'''
SVM_MEL = "{SVM_MEL}"
SVM_LOFAR = "{SVM_LOFAR}"

linhas = []
for modelo in ['RF', 'MLP', 'CNN']:
    for rep in ['MEL', 'LOFAR']:
        a = R.ARTICLE_TABLE7[(modelo, rep)]
        linhas.append({{'Modelo': f'{{modelo}} {{rep.capitalize()}}', 'Representacao': rep, 'Fonte': 'Tabela 7 do artigo',
                       'SP': a['SP'][0], 'SP_std': a['SP'][1], 'ACC': a['ACC'][0], 'ACC_std': a['ACC'][1]}})
for rep, exp in [('MEL', SVM_MEL), ('LOFAR', SVM_LOFAR)]:
    s = R.summary(exp)
    linhas.append({{'Modelo': f'SVM {{rep.capitalize()}}', 'Representacao': rep, 'Fonte': exp,
                   'SP': s['SP'][0], 'SP_std': s['SP'][1], 'ACC': s['ACC'][0], 'ACC_std': s['ACC'][1]}})

df = pd.DataFrame(linhas).round(2)
df
'''),
    code(f'''
# Detalhe das linhas do SVM (inclui F1 macro, acurácia micro e recall médio por classe)
R.table([('SVM MEL (m=4000)', SVM_MEL), ('SVM LOFAR (m=4000, PCA64)', SVM_LOFAR)],
        metrics=('SP', 'ACC', 'ACC_micro', 'F1'))
'''),
    md(r"""
### 2. Visualização das Métricas Globais (MEL vs LOFAR)
Um gráfico de barras comparando a **Acurácia Balanceada (ACC)** e o **Índice SP** com barras de desvio padrão é plotado a seguir.
"""),
    code(r"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

mel_mask = df['Representacao'] == 'MEL'
lofar_mask = df['Representacao'] == 'LOFAR'

x = np.arange(4)
width = 0.35

# Gráfico 1: Acurácia Balanceada
ax1.bar(x - width/2, df[mel_mask]['ACC'], width, yerr=df[mel_mask]['ACC_std'],
        label='MEL', color='#1abc9c', edgecolor='black', capsize=5, alpha=0.9)
ax1.bar(x + width/2, df[lofar_mask]['ACC'], width, yerr=df[lofar_mask]['ACC_std'],
        label='LOFAR', color='#34495e', edgecolor='black', capsize=5, alpha=0.9)
ax1.set_ylabel('Acurácia Balanceada (%)')
ax1.set_title('Acurácia Balanceada (ACC) por Modelo e Extrator')
ax1.set_xticks(x)
ax1.set_xticklabels(['RF (artigo)', 'MLP (artigo)', 'CNN (artigo)', 'SVM (m=4000, local)'])
ax1.set_ylim(50, 75)
ax1.legend()

# Gráfico 2: Índice SP
ax2.bar(x - width/2, df[mel_mask]['SP'], width, yerr=df[mel_mask]['SP_std'],
        label='MEL', color='#e67e22', edgecolor='black', capsize=5, alpha=0.9)
ax2.bar(x + width/2, df[lofar_mask]['SP'], width, yerr=df[lofar_mask]['SP_std'],
        label='LOFAR', color='#2c3e50', edgecolor='black', capsize=5, alpha=0.9)
ax2.set_ylabel('Índice SP (%)')
ax2.set_title('Índice SP por Modelo')
ax2.set_xticks(x)
ax2.set_xticklabels(['RF (artigo)', 'MLP (artigo)', 'CNN (artigo)', 'SVM (m=4000, local)'])
ax2.set_ylim(50, 75)
ax2.legend()

plt.tight_layout()
plt.show()
"""),
    md(r"""
### 3. Testes de Significância (F-test 5×2cv de Alpaydin, como no artigo)
O artigo compara modelos pelo F-test 5×2cv combinado. Como a Tabela 7 não traz resultados por fold, o SVM é comparado com as **re-execuções locais** dos baselines no mesmo protocolo e nos mesmos folds (`cnn` = CNN MEL, `mlp` = MLP MEL, `mlp_lofar` = MLP LOFAR), e as variantes com/sem PCA são comparadas entre si. $p < 0.05$ indicaria diferença significativa.
"""),
    code(f'''
print('Re-execuções locais dos baselines (mesmos folds do SVM):')
display(R.table([('CNN MEL (local)', "{CNN_MEL}"), ('MLP MEL (local)', "{MLP_MEL}"),
                 ('MLP LOFAR (local)', "{MLP_LOFAR}")], metrics=('SP', 'ACC')))

pares = [
    ('SVM MEL vs. CNN MEL (local)', SVM_MEL, "{CNN_MEL}"),
    ('SVM MEL vs. MLP MEL (local)', SVM_MEL, "{MLP_MEL}"),
    ('SVM LOFAR vs. MLP LOFAR (local)', SVM_LOFAR, "{MLP_LOFAR}"),
    ('SVM LOFAR m=4000: PCA64+ElasticNet vs. sem PCA', SVM_LOFAR, "{SVM_LOFAR_NOPCA}"),
    ('SVM LOFAR m=4000 PCA64 vs. m=6000 sem PCA', SVM_LOFAR, "{SVM_LOFAR_6000}"),
    ('SVM MEL m=4000: PCA64 vs. sem PCA', "{SVM_MEL_PCA}", SVM_MEL),
]
linhas = []
for rotulo, a, b in pares:
    sa, sb = R.summary(a), R.summary(b)
    f_sp, p_sp = R.f_test_5x2(a, b, metric='SP')
    f_acc, p_acc = R.f_test_5x2(a, b, metric='ACC')
    linhas.append({{'Comparação': rotulo, 'SP A': R.fmt(sa['SP']), 'SP B': R.fmt(sb['SP']), 'p (SP)': round(p_sp, 3),
                   'ACC A': R.fmt(sa['ACC']), 'ACC B': R.fmt(sb['ACC']), 'p (ACC)': round(p_acc, 3)}})
pd.DataFrame(linhas).set_index('Comparação')
'''),
    md(f"""
### 4. Discussão Científica e Conclusões:
1. **Desempenho equivalente ao dos baselines:** O **SVM Nyström (MEL)** alcançou **{F(S_MEL['ACC'])}% de Acurácia Balanceada (ACC)** (e **{F(S_MEL['SP'])}% de Índice SP**), valores próximos aos da CNN do artigo (**{F(A7[('CNN', 'MEL')]['ACC'])}%** de ACC e **{F(A7[('CNN', 'MEL')]['SP'])}%** de SP); a diferença das médias está dentro do desvio. Contra a re-execução local da CNN MEL nos mesmos folds (ACC {F(S_CNN['ACC'])}%, SP {F(S_CNN['SP'])}%), o F-test 5×2cv não indica diferença ({p_txt(FT['mel_vs_cnn'][1])} em SP; {p_txt(FT['mel_vs_cnn_acc'][1])} em ACC): os modelos são **estatisticamente equivalentes**. O desvio entre folds do SVM MEL é menor: $\\sigma_{{SVM}} = {S_MEL['ACC'][1]:.2f}\\%$ vs. $\\sigma_{{CNN}} = {A7[('CNN', 'MEL')]['ACC'][1]:.2f}\\%$ (artigo), isto é, cerca de **{RATIO_STD:.2f}× o desvio** (≈ {RATIO_STD**2:.2f}× a variância). A menor dispersão é compatível com a formulação convexa do SVM, mas isso é uma hipótese, não algo testado aqui.
2. **PCA sem efeito significativo:** Não há evidência de que o PCA seja essencial no LOFAR nem de que degrade o MEL. Em $m=4000$, o LOFAR com PCA64 + ElasticNet tem SP {F(S_LOFAR['SP'])}% contra {F(S_LOFAR_NOPCA['SP'])}% sem PCA ({p_txt(FT['lofar_pca_4000'][1])} em SP; {p_txt(FT['lofar_pca_4000_acc'][1])} em ACC), e o SVM LOFAR $m=6000$ **sem** PCA (`{SVM_LOFAR_6000}`) tem ACC balanceada **maior** ({F(S_LOFAR_6000['ACC'])}% vs. {F(S_LOFAR['ACC'])}%; {p_txt(FT['lofar_pca_vs_6000'][1])}). No MEL $m=4000$, o PCA64 dá SP {F(S_MEL_PCA['SP'])}% contra {F(S_MEL['SP'])}% sem PCA ({p_txt(FT['mel_pca_4000'][1])}), ou seja, ligeiramente maior e equivalente. A escolha "MEL sem PCA / LOFAR com PCA64" é, portanto, uma configuração adotada, não uma conclusão estatística.
"""),
]

# ==================== NOTEBOOK 2: OPÇÃO DE REJEIÇÃO ====================
REJ_MODELS = [("LOFAR (SVM - m=4000, C=2.0, PCA64)", SVM_LOFAR), ("MEL (SVM - m=4000, C=2.0)", SVM_MEL),
              ("CNN MEL (re-execução local)", CNN_MEL), ("MLP MEL (re-execução local)", MLP_MEL)]
CURVAS = {exp: curva_rejeicao(exp) for _, exp in REJ_MODELS}
CV = {exp: {l['t']: l for l in CURVAS[exp]} for exp in CURVAS}
NSEG = sorted({int(n) for d in R.file_predictions(CNN_MEL).values() for n in d.n_windows.unique()})
NWIN_SVM = float(np.median(np.concatenate([d.n_windows.values for d in R.file_predictions(SVM_LOFAR).values()])))
NSEG_TXT = ", ".join(map(str, NSEG))
T_NOMES = {0.0: "*(Sem Rejeição)*", 0.5: "*(Maioria Simples)*", 0.6: "*(Maioria Absoluta)*",
           0.7: "*(Consenso Forte)*", 0.8: "*(Consenso Seguro)*", 0.9: "*(Consenso Crítico)*"}

rej_rows = []
for t in LIMIARES:
    for lab, exp in REJ_MODELS:
        l = CV[exp][t]
        rej_rows.append(f"| **$t \\ge {t:.1f}$** {T_NOMES[t]} | {lab} | {F(l['Cobertura'])} | {F(l['ACC'])} | "
                        f"{F(l['ACC_micro'])} | {F(l['SP'])} |")
rej_rows = "\n".join(rej_rows)

L09, C09 = CV[SVM_LOFAR][0.9], CV[CNN_MEL][0.9]
# Comparação a cobertura semelhante: para cada modelo, o limiar cuja cobertura é mais próxima da CNN em t >= 0.9
COB_REF = C09['Cobertura'][0]
MATCH = {}
for lab, exp in REJ_MODELS:
    best = min(CURVAS[exp], key=lambda l: abs(l['Cobertura'][0] - COB_REF))
    MATCH[exp] = best
match_txt = "; ".join(f"{lab.split(' (')[0] if 'SVM' not in lab else 'SVM ' + lab.split(' (')[0]}: $t \\ge {MATCH[exp]['t']:.1f}$, "
                      f"cobertura {MATCH[exp]['Cobertura'][0]:.2f}%, ACC {MATCH[exp]['ACC'][0]:.2f}%"
                      for lab, exp in REJ_MODELS)
CLS09 = {r['Classe']: r for r in rejeicao_por_classe(SVM_LOFAR, 0.9)}
N_TOT = float(np.mean([len(d) for d in R.file_predictions(SVM_LOFAR).values()]))

cls_rows = "\n".join(
    f"| **{c}** | {CLS09[c]['Arquivos/fold']:.1f} | {F(CLS09[c]['Cobertura'])} | {F(CLS09[c]['Recall nos aceitos'])} |"
    for c in ("LARGE", "BACKGROUND", "MEDIUM", "SMALL"))

cells_2 = [
    md(f"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 2: Otimização com Opção de Rejeição (Filtro de Confiança)

Neste notebook, é avaliada a incorporação de um **operador de decisão com opção de rejeição** sobre os 10 folds do protocolo $5\\times2$cv (com **particionamento por gravação de áudio `ID`**, `exclusive_header='ID'`, garantindo que janelas de um mesmo arquivo de áudio não coexistam no treino e no teste). A curva de trade-off entre a taxa de cobertura (fração de áudios classificados) e as métricas de desempenho é analisada à medida que maior certeza é exigida do sistema de sonar.

> **Nota Metodológica Importante (Rejeição Local vs. Tabela 9 de *Silva et al., 2025* e Discretização da CNN):**
> 1. **Índice de confiança e métricas:** a confiança de um arquivo é a **taxa de concordância dos votos das janelas** (fração das janelas/segmentos do arquivo que votaram na classe vencedora do voto majoritário), o mesmo princípio da Tabela 9 do artigo; **não** são usadas probabilidades/softmax. Um arquivo é aceito se concordância $\\ge t$, para $t \\in \\{{0.5, 0.6, 0.7, 0.8, 0.9\\}}$. Sobre os arquivos aceitos de cada fold são calculadas a **Acurácia Balanceada (`Bal. Acc`, métrica principal, como no artigo)**, a **Acurácia Global/Micro (`UACC`)**, em coluna separada, e o **Índice SP**; reporta-se média ± desvio-padrão populacional nos 10 folds. Os números são recalculados neste notebook a partir de `results/trainings/tests/<experimento>/eval/fold_k/` (função `curva_rejeicao`). Em contraste, a **Tabela 9 de *Silva et al. (2025)*** buscou o ponto ótimo da razão de concordância (*Classification Ratio*) que maximiza o Índice SP no extrator MEL, reportando Acurácia Balanceada e `File Ratio` (cobertura): **RF** (`Classification Ratio = 88%` $\\rightarrow$ `SP = 72.29 ± 3.86%`, `Bal. Acc = 73.23 ± 3.62%`, `File Ratio = 38.12%`), **MLP** (`Classification Ratio = 85%` $\\rightarrow$ `SP = 74.18 ± 4.04%`, `Bal. Acc = 76.26 ± 2.99%`, `File Ratio = 39.36%`) e **CNN** (`Classification Ratio = 93%` $\\rightarrow$ `SP = 70.04 ± 2.86%`, `Bal. Acc = 72.07 ± 2.36%`, `File Ratio = 52.11%`) — valores citados da Tabela 9 do artigo.
> 2. **Por que `CNN` possui valores idênticos em $t \\ge 0.6$ e $t \\ge 0.7$?** Diferentemente dos modelos baseados em janelas individuais (`InputType.Window`), a CNN opera sobre sub-imagens espectrais (`InputType.Image`), e nas predições em disco cada arquivo tem exatamente **{NSEG_TXT} segmentos de imagem**. Como $4/7 \\approx 0.5714 < 0.6$ e $5/7 \\approx 0.7143 \\ge 0.7$, não existe nenhum número inteiro de segmentos $k$ tal que $0.6 \\le k/7 < 0.7$. Portanto, tanto $t \\ge 0.6$ quanto $t \\ge 0.7$ exigem $k \\ge 5$ segmentos concordantes em 7, produzindo a mesma cobertura (`{F(CV[CNN_MEL][0.6]['Cobertura'])}%`), acurácia balanceada (`{F(CV[CNN_MEL][0.6]['ACC'])}%`) e SP (`{F(CV[CNN_MEL][0.6]['SP'])}%`).
> 3. **Cuidado na leitura:** em cada limiar, as métricas são calculadas sobre um subconjunto diferente de arquivos (os aceitos), e modelos diferentes aceitam subconjuntos diferentes; comparações entre modelos devem ser feitas a **cobertura semelhante**, não ao mesmo $t$.
"""),
    md(r"""
### Parâmetros de Configuração e Treinamento do SVM Nyström:
As configurações de hiperparâmetros abaixo foram mantidas constantes durante a validação cruzada (experimentos `svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0` e `svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0`; baselines `cnn` e `mlp`, re-execuções locais do pipeline iara com MEL):
* **Aproximação de Kernel:** Kernel Gaussiano RBF aproximado pelo método de Nyström com $m = 4000$ componentes.
* **Custo de Regularização ($C$):** $C = 2.0$.
* **Regularização / Penalidade:** ElasticNet com razão $L_1 = 0.15$ e $L_2 = 0.85$.
* **Tratamento de Dimensionalidade (MEL):** PCA desativado.
* **Tratamento de Dimensionalidade (LOFAR):** PCA ativado ($n_{components} = 64$).
"""),
    md(f"""
### Tabela 2: Curva de Trade-off Cobertura-Desempenho (LOFAR vs. MEL vs. CNN vs. MLP)

| Limiar de Confiança ($t$) | Representação Espectral / Arquitetura | Taxa de Cobertura (%) | Acurácia Balanceada (%) | Acurácia Micro (`UACC`) (%) | Índice SP (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
{rej_rows}
"""),
    code(HELPERS),
    code(SETUP_PLOT),
    md(r"""
### 1. Estruturação dos Dados de Rejeição
As métricas obtidas sob os limiares críticos são recalculadas a partir das predições por janela e indexadas no DataFrame a seguir.
"""),
    code("MODELOS = {\n" + "".join(f"    '{lab}': '{exp}',\n" for lab, exp in REJ_MODELS) + "}\n" + r"""
linhas = []
for rotulo, exp in MODELOS.items():
    for l in curva_rejeicao(exp):
        linhas.append({'Modelo': rotulo, 'Threshold': l['t'],
                       'Cov': l['Cobertura'][0], 'Cov_std': l['Cobertura'][1],
                       'ACC': l['ACC'][0], 'ACC_std': l['ACC'][1],
                       'UACC': l['ACC_micro'][0], 'UACC_std': l['ACC_micro'][1],
                       'SP': l['SP'][0], 'SP_std': l['SP'][1]})
df_rej = pd.DataFrame(linhas).round(2)

tab = df_rej.copy()
for c in ['Cov', 'ACC', 'UACC', 'SP']:
    tab[c] = tab.apply(lambda r: f"{r[c]:.2f} ± {r[c + '_std']:.2f}", axis=1)
tab.pivot(index='Threshold', columns='Modelo', values=['Cov', 'ACC', 'UACC', 'SP'])
"""),
    md(r"""
### 2. Plotagem da Curva de Trade-off Cobertura-Desempenho
A curva de trade-off é gerada abaixo, com a taxa de cobertura (%) no eixo horizontal e, no eixo vertical, a **Acurácia Balanceada** (esquerda) e o **Índice SP** (direita) nos arquivos aceitos.
"""),
    code(r"""
estilos = {list(MODELOS)[0]: ('o-', '#e74c3c'), list(MODELOS)[1]: ('s-', '#2ecc71'),
           list(MODELOS)[2]: ('^--', '#3498db'), list(MODELOS)[3]: ('d-.', '#9b59b6')}
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, met, nome in [(axes[0], 'ACC', 'Acurácia Balanceada nos Arquivos Aceitos (%)'),
                      (axes[1], 'SP', 'Índice SP nos Arquivos Aceitos (%)')]:
    for rotulo, (fmt_, cor) in estilos.items():
        d = df_rej[df_rej.Modelo == rotulo]
        ax.plot(d['Cov'], d[met], fmt_, label=rotulo, color=cor, linewidth=2.2, markersize=8)
    d = df_rej[df_rej.Modelo == list(MODELOS)[0]]
    for _, row in d.iterrows():
        ax.annotate(f"t={row['Threshold']}", (row['Cov'], row[met]), textcoords="offset points",
                    xytext=(10, -10), ha='center', fontsize=9, color='#c0392b', weight='bold')
    ax.set_xlabel('Taxa de Cobertura de Classificação (%)', fontsize=12)
    ax.set_ylabel(nome, fontsize=12)
    ax.set_xlim(15, 105)
    ax.legend(fontsize=10, loc='upper right')
fig.suptitle('Curva de Trade-off Cobertura-Desempenho sob Opção de Rejeição', fontsize=14, weight='bold')
plt.tight_layout()
plt.show()
"""),
    md(f"""
### 3. Discussão do Gráfico:
1. **SVM LOFAR sob limiar severo:** Com $t \\ge 0.9$, o SVM LOFAR retém **{F(L09['Cobertura'])}%** dos áudios e alcança, nesses arquivos, **{F(L09['ACC'])}%** de Acurácia Balanceada (`UACC` = {F(L09['ACC_micro'])}%; SP = {F(L09['SP'])}%). A acurácia micro é bem maior que a balanceada porque os arquivos aceitos são dominados por LARGE e BACKGROUND (Seção 4). Entre $t \\ge 0.8$ e $t \\ge 0.9$ o ganho satura: a ACC balanceada passa de {CV[SVM_LOFAR][0.8]['ACC'][0]:.2f}% para {L09['ACC'][0]:.2f}% e o SP de {CV[SVM_LOFAR][0.8]['SP'][0]:.2f}% para {L09['SP'][0]:.2f}%, com desvios entre folds de {CV[SVM_LOFAR][0.9]['ACC'][1]:.2f} p.p. (ACC) e {CV[SVM_LOFAR][0.9]['SP'][1]:.2f} p.p. (SP) em $t \\ge 0.9$.
2. **CNN: índice de concordância mais grosseiro, não "superconfiança":** Sob o mesmo limiar ($t \\ge 0.9$), a CNN MEL mantém cobertura maior (**{F(C09['Cobertura'])}%**) e ACC balanceada de **{F(C09['ACC'])}%** (SP = {F(C09['SP'])}%). A comparação no mesmo $t$ não é justa: a concordância da CNN é calculada sobre apenas 7 segmentos sobrepostos por arquivo, enquanto os modelos por janela usam ≈ {NWIN_SVM:.0f} janelas por arquivo (mediana), e as coberturas são muito diferentes. Comparando a cobertura semelhante à da CNN em $t \\ge 0.9$ ({COB_REF:.2f}%): {match_txt}. Como nenhuma probabilidade é usada, estes dados **não** permitem conclusões sobre calibração das saídas da CNN ou do SVM; a afirmação anterior de que a CNN sofre de "calibração inadequada" e de que o SVM é "mais bem calibrado" não é sustentada por esta análise e foi removida.
"""),
    md(f"""
### 4. Distribuição de Classes no Filtro Crítico ($t \\ge 0.90$)
Para compreender onde a opção de rejeição é mais assertiva, analisamos a distribuição multiclasse dos áudios aceitos e rejeitados para o **LOFAR SVM** no limiar mais rigoroso ($t \\ge 0.90$), mostrando a média e o desvio padrão obtidos na validação cruzada. "Acurácia de classe" é o recall da classe calculado apenas sobre os arquivos aceitos.

| Classe | Áudios Totais/Fold | Taxa de Cobertura (%) | Acurácia de Classe (recall nos aceitos) (%) |
| :--- | :---: | :---: | :---: |
{cls_rows}
| **TOTAL GLOBAL** | **{N_TOT:.1f}** | **{F(L09['Cobertura'])}** | **{F(L09['ACC'])}** *(Bal. Acc)* |

*A linha total traz a Acurácia Balanceada nos aceitos; a acurácia micro correspondente é {F(L09['ACC_micro'])}%.*
"""),
    code(f'''
df_dist = pd.DataFrame(rejeicao_por_classe("{SVM_LOFAR}", 0.9))
df_dist['Cobertura (%)'] = df_dist['Cobertura'].apply(R.fmt)
df_dist['Acurácia de classe nos aceitos (%)'] = df_dist['Recall nos aceitos'].apply(R.fmt)
df_dist[['Classe', 'Arquivos/fold', 'Cobertura (%)', 'Acurácia de classe nos aceitos (%)']]
'''),
    md(f"""
### Discussão por Alvo Acústico:
1. **LARGE & BACKGROUND (alta confiança):** Nos arquivos aceitos, o SVM atinge recall de **{F(CLS09['LARGE']['Recall nos aceitos'])}%** para navios grandes e **{F(CLS09['BACKGROUND']['Recall nos aceitos'])}%** para ruído de fundo, com coberturas de {CLS09['LARGE']['Cobertura'][0]:.2f}% e {CLS09['BACKGROUND']['Cobertura'][0]:.2f}%. Uma explicação plausível (não testada aqui) é que navios grandes têm assinaturas tonais mais intensas e o ruído de fundo não apresenta modulação periódica.
2. **MEDIUM:** Com recall de **{F(CLS09['MEDIUM']['Recall nos aceitos'])}%** e taxa de cobertura de **{F(CLS09['MEDIUM']['Cobertura'])}%**, a maior parte dos arquivos de navios médios é rejeitada, e o desvio entre folds é alto.
3. **SMALL (a rejeição não resolve a classe):** Apenas **{F(CLS09['SMALL']['Cobertura'])}%** dos arquivos SMALL são aceitos (≈ {CLS09['SMALL']['Cobertura'][0] / 100 * CLS09['SMALL']['Arquivos/fold']:.0f} arquivos por fold) e, mesmo entre esses, o recall é de apenas **{F(CLS09['SMALL']['Recall nos aceitos'])}%**. Ou seja, o filtro de concordância praticamente exclui SMALL da decisão sem torná-la confiável; o alto valor de `UACC` global reflete sobretudo LARGE e BACKGROUND.
"""),
]

# ==================== NOTEBOOK 3: CPA PROXIMIDADE ====================
CPA_MODELS = [("Forest Mel", "exp_1", "forest_window", "Re-execução Local"),
              ("MLP Mel", "exp_1", "mlp_window", "Re-execução Local"),
              ("CNN Mel", "exp_1", "cnn_image", "Re-execução Local"),
              ("SVM Mel", "exp_1", "svm_window", "m=4000, C=2.0"),
              ("SVM LOFAR", "lofar_exp_1", "svm_window", "m=4000, C=2.0, PCA64")]
CPA = {}
for nome, sub, mod, _ in CPA_MODELS:
    for tr in "AC":
        for te in "AC":
            CPA[(nome, tr, te)] = cpa_metricas(sub, mod, tr, te)


def cs(nome, tr, te, met):
    return ms(CPA[(nome, tr, te)][met])


cpa_rows = []
for nome, sub, mod, tag in CPA_MODELS:
    for tr, lab in (("A", "Dataset A (Near)"), ("C", "Dataset C (Far)")):
        cpa_rows.append(f"| **{nome}** *({tag})* | {lab} | {F(cs(nome, tr, 'A', 'SP'))} | {F(cs(nome, tr, 'A', 'ACC'))} | "
                        f"{F(cs(nome, tr, 'C', 'SP'))} | {F(cs(nome, tr, 'C', 'ACC'))} |")
cpa_rows = "\n".join(cpa_rows)

QUEDA = {nome: (CPA[(nome, 'A', 'A')]['ACC'] - CPA[(nome, 'A', 'C')]['ACC']).values for nome, *_ in CPA_MODELS}
QUEDA_M = {n: float(np.mean(v)) for n, v in QUEDA.items()}
MAIOR_QUEDA = max(QUEDA_M, key=QUEDA_M.get)
MENOR_QUEDA = min(QUEDA_M, key=QUEDA_M.get)
FT3 = {
    "lofar_vs_mlp_sp": R.f_test_5x2(CPA[("SVM LOFAR", "A", "A")]["SP"].values, CPA[("MLP Mel", "A", "A")]["SP"].values),
    "lofar_vs_mlp_acc": R.f_test_5x2(CPA[("SVM LOFAR", "A", "A")]["ACC"].values, CPA[("MLP Mel", "A", "A")]["ACC"].values),
    "queda_svmmel_vs_mlp": R.f_test_5x2(QUEDA[MENOR_QUEDA], QUEDA["MLP Mel"]),
    "c_a_lofar_vs_cnn": R.f_test_5x2(CPA[("SVM LOFAR", "C", "A")]["ACC"].values, CPA[("CNN Mel", "C", "A")]["ACC"].values),
    "c_a_lofar_vs_mlp": R.f_test_5x2(CPA[("SVM LOFAR", "C", "A")]["ACC"].values, CPA[("MLP Mel", "C", "A")]["ACC"].values),
}
DESV_C = (min(cs(n, 'A', 'C', 'ACC')[1] for n, *_ in CPA_MODELS), max(cs(n, 'A', 'C', 'ACC')[1] for n, *_ in CPA_MODELS))
# Guardas: o texto do NB3 assume estas relações nos dados
assert MAIOR_QUEDA == 'SVM LOFAR' and MENOR_QUEDA == 'SVM Mel'
assert max(CPA_MODELS, key=lambda r: cs(r[0], 'A', 'A', 'ACC')[0])[0] == 'SVM LOFAR'
assert cs('MLP Mel', 'A', 'A', 'SP')[0] > cs('SVM LOFAR', 'A', 'A', 'SP')[0]
assert all(FT3[k][1] >= 0.05 for k in ('lofar_vs_mlp_sp', 'lofar_vs_mlp_acc', 'queda_svmmel_vs_mlp', 'c_a_lofar_vs_mlp')), FT3
queda_txt = ", ".join(f"{n} {QUEDA_M[n]:.2f} p.p." for n, *_ in CPA_MODELS)

cells_3 = [
    md(r"""
# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
### Notebook 3: Robustez de Generalização no Ponto Crítico de Aproximação (CPA)

Neste notebook, é avaliada a resiliência das fronteiras de decisão sob variação da Relação Sinal-Ruído (SNR) associada à distância do CPA da embarcação ao hidrofone, utilizando o protocolo $5\times2$cv com **particionamento por gravação de áudio (`ID`, `exclusive_header='ID'`)** (garantindo que janelas de um mesmo arquivo de áudio não coexistam no treino e no teste):
* **Dataset A (Near CPA):** CPA a menos de $250\text{ m}$ do hidrofone (maior SNR esperada).
* **Dataset C (Far CPA):** CPA entre $250$ e $500\text{ m}$ (menor SNR esperada).

A replicação do experimento da Tabela 10 do artigo original é efetuada a seguir, incorporando-se as abordagens propostas baseadas em **SVM Mel** e **SVM LOFAR**. Todas as métricas `ACC (%)` reportadas correspondem à **Acurácia Balanceada (`Bal. Acc`, média dos recalls por classe)**. Todos os números deste notebook (exceto os citados da Tabela 10 do artigo) são recalculados a partir das predições por janela em `results/trainings/cpa_exp/`.
"""),
    md(r"""
### Parâmetros de Configuração e Treinamento do SVM Nyström:
As configurações de hiperparâmetros abaixo são as definidas em `src/iara/default.py` para os treinadores `svm mel` e `svm lofar` usados por `training_scripts/cpa_presence.py` e `cpa_presence_lofar.py` (os arquivos de configuração dos experimentos CPA não gravam esses hiperparâmetros):
* **Aproximação de Kernel:** Kernel Gaussiano RBF aproximado pelo método de Nyström com $m = 4000$ componentes.
* **Custo de Regularização ($C$):** $C = 2.0$.
* **Regularização / Penalidade:** ElasticNet com razão $L_1 = 0.15$ e $L_2 = 0.85$.
* **Tratamento de Dimensionalidade (MEL):** PCA desativado.
* **Tratamento de Dimensionalidade (LOFAR):** PCA ativado ($n_{components} = 64$).
"""),
    md(f"""
### Tabela 3: Desempenho e Generalização no Experimento de CPA Proximity (Re-execução Local $5\\times2$cv vs. Tabela 10 do Artigo)

| Modelo / Classificador | Treinado em | SP A (%) | ACC Balanceada A (%) | SP C (%) | ACC Balanceada C (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
{cpa_rows}

> **Nota de Rastreabilidade (`cpa_exp` vs. Tabela 10 de *Silva et al., 2025*):**
> As linhas acima são recalculadas a partir das predições da **re-execução local $5\\times2$cv** em `results/trainings/cpa_exp/exp_1/` (Forest, MLP, CNN e SVM Mel) e `results/trainings/cpa_exp/lofar_exp_1/` (SVM LOFAR): `<Treino>_<modelo>/eval/fold_k/` contém o teste no próprio dataset de treino e `comparison/<Treino>_<modelo>/` o teste no outro dataset (`iara.ml.experiment.CrossComparator`). Os valores coincidem com `cpa_proximity.tex` e `lofar_cpa_proximity.tex`. Os folds de teste são os mesmos em todos os modelos, o que permite o F-test 5×2cv pareado. Para completude documental, os valores publicados originalmente na **Tabela 10 de *Silva et al. (2025)*** são:
> * **RF (MEL) — Artigo Tabela 10:** Treinado em A $\\rightarrow$ `SP A = 57.49 ± 3.10%`, `ACC A = 59.58 ± 2.88%`, `SP C = 47.63 ± 5.44%`, `ACC C = 51.39 ± 4.92%` | Treinado em C $\\rightarrow$ `SP A = 48.02 ± 3.99%`, `ACC A = 51.33 ± 2.66%`, `SP C = 46.08 ± 5.80%`, `ACC C = 50.77 ± 4.55%`.
> * **MLP (MEL) — Artigo Tabela 10:** Treinado em A $\\rightarrow$ `SP A = 67.25 ± 2.44%`, `ACC A = 67.64 ± 2.39%`, `SP C = 59.08 ± 6.51%`, `ACC C = 60.11 ± 5.68%` | Treinado em C $\\rightarrow$ `SP A = 59.59 ± 5.22%`, `ACC A = 59.96 ± 5.13%`, `SP C = 58.08 ± 3.70%`, `ACC C = 59.02 ± 3.31%`.
> * **CNN (MEL) — Artigo Tabela 10:** Treinado em A $\\rightarrow$ `SP A = 63.28 ± 1.95%`, `ACC A = 63.78 ± 1.79%`, `SP C = 57.85 ± 5.36%`, `ACC C = 59.13 ± 4.72%` | Treinado em C $\\rightarrow$ `SP A = 53.30 ± 5.44%`, `ACC A = 54.08 ± 5.01%`, `SP C = 56.23 ± 4.36%`, `ACC C = 57.17 ± 4.09%`.
"""),
    code(HELPERS),
    code(SETUP_PLOT),
    md(r"""
### 1. Definição das Métricas de Generalização Inter-dataset
Os resultados por fold para o cruzamento de conjuntos de treino e teste entre A e C são recalculados na célula abaixo (`cpa_metricas`).
"""),
    code("MODELOS_CPA = [\n" + "".join(f"    ('{n}', '{s}', '{m}'),\n" for n, s, m, _ in CPA_MODELS) + "]\n" + r"""
CPA = {(nome, tr, te): cpa_metricas(sub, mod, tr, te)
       for nome, sub, mod in MODELOS_CPA for tr in 'AC' for te in 'AC'}

linhas = []
for nome, _, _ in MODELOS_CPA:
    for tr in 'AC':
        linha = {'Modelo': nome, 'Treinado em': tr}
        for te in 'AC':
            for met in ('SP', 'ACC'):
                linha[f'{met} {te}'] = R.fmt(ms(CPA[(nome, tr, te)][met]))
        linhas.append(linha)
display(pd.DataFrame(linhas).set_index(['Modelo', 'Treinado em']))

df_cpa = pd.DataFrame({
    'Modelo': [n for n, _, _ in MODELOS_CPA],
    'Trained_A_ACC_A': [CPA[(n, 'A', 'A')]['ACC'].mean() for n, _, _ in MODELOS_CPA],
    'Trained_A_ACC_C': [CPA[(n, 'A', 'C')]['ACC'].mean() for n, _, _ in MODELOS_CPA],
    'Trained_C_ACC_C': [CPA[(n, 'C', 'C')]['ACC'].mean() for n, _, _ in MODELOS_CPA],
    'Trained_C_ACC_A': [CPA[(n, 'C', 'A')]['ACC'].mean() for n, _, _ in MODELOS_CPA],
}).round(2)
df_cpa
"""),
    md(r"""
### 2. Plotagem do Teste de Robustez de Distância (Treinado em A -> Testado em C)
A atenuação de acurácia balanceada decorrente do teste de generalização (treinamento no Dataset A e teste no Dataset C) é ilustrada graficamente abaixo.
"""),
    code(r"""
x = np.arange(len(df_cpa['Modelo']))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - width/2, df_cpa['Trained_A_ACC_A'], width, label='Testado em A (Near CPA)', color='#3498db', edgecolor='black')
ax.bar(x + width/2, df_cpa['Trained_A_ACC_C'], width, label='Testado em C (Far CPA)', color='#e74c3c', edgecolor='black')

ax.set_ylabel('Acurácia Balanceada (ACC) (%)')
ax.set_title('Generalização (Trained on A -> Tested on A vs C)')
ax.set_xticks(x)
ax.set_xticklabels(df_cpa['Modelo'])
ax.set_ylim(40, 75)
ax.legend()

plt.tight_layout()
plt.show()
"""),
    md(r"""
### 3. Quedas A→C e Testes de Significância (F-test 5×2cv)
A queda A→C é calculada por fold (ACC balanceada testada em A menos testada em C, modelo treinado em A). O F-test 5×2cv compara modelos nos mesmos folds.
"""),
    code(r"""
queda = {n: (CPA[(n, 'A', 'A')]['ACC'] - CPA[(n, 'A', 'C')]['ACC']).values for n, _, _ in MODELOS_CPA}
display(pd.DataFrame({'Queda A→C da ACC balanceada (p.p.)': {n: R.fmt(ms(v)) for n, v in queda.items()}}))

testes = [
    ('Treino A, teste A: SVM LOFAR vs. MLP Mel (SP)', CPA[('SVM LOFAR', 'A', 'A')]['SP'], CPA[('MLP Mel', 'A', 'A')]['SP']),
    ('Treino A, teste A: SVM LOFAR vs. MLP Mel (ACC)', CPA[('SVM LOFAR', 'A', 'A')]['ACC'], CPA[('MLP Mel', 'A', 'A')]['ACC']),
    ('Queda A→C: SVM Mel vs. MLP Mel', queda['SVM Mel'], queda['MLP Mel']),
    ('Treino C, teste A: SVM LOFAR vs. CNN Mel (ACC)', CPA[('SVM LOFAR', 'C', 'A')]['ACC'], CPA[('CNN Mel', 'C', 'A')]['ACC']),
    ('Treino C, teste A: SVM LOFAR vs. MLP Mel (ACC)', CPA[('SVM LOFAR', 'C', 'A')]['ACC'], CPA[('MLP Mel', 'C', 'A')]['ACC']),
]
pd.DataFrame([{'Comparação': r, 'média A': np.mean(a), 'média B': np.mean(b), 'F': R.f_test_5x2(np.asarray(a), np.asarray(b))[0],
               'p': R.f_test_5x2(np.asarray(a), np.asarray(b))[1]} for r, a, b in testes]).set_index('Comparação').round(3)
"""),
    md(f"""
### 4. Discussão sobre Generalização:
1. **Maior média em A, mas equivalente à MLP:** O **SVM LOFAR** treinado e testado em A obteve a maior acurácia balanceada média do estudo de proximidade: **{F(cs('SVM LOFAR', 'A', 'A', 'ACC'))}%** (`SP = {F(cs('SVM LOFAR', 'A', 'A', 'SP'))}%`), contra **{F(cs('MLP Mel', 'A', 'A', 'ACC'))}%** da MLP Mel local (`SP = {F(cs('MLP Mel', 'A', 'A', 'SP'))}%`; **67.64 ± 2.39%** de ACC na Tabela 10 do artigo). Em SP a MLP é ligeiramente maior. O F-test 5×2cv não indica diferença ({p_txt(FT3['lofar_vs_mlp_sp'][1])} em SP; {p_txt(FT3['lofar_vs_mlp_acc'][1])} em ACC): os dois modelos são **equivalentes**; não se pode afirmar que o SVM LOFAR superou a MLP.
2. **Queda ao generalizar para C:** Treinando em A e testando em C, as quedas médias de acurácia balanceada foram: {queda_txt} A **maior queda é a do {MAIOR_QUEDA}** ({cs(MAIOR_QUEDA, 'A', 'A', 'ACC')[0]:.2f}% → {cs(MAIOR_QUEDA, 'A', 'C', 'ACC')[0]:.2f}%), isto é, o modelo com maior média em A é o que mais perde ao ser testado em C. A menor queda é a do **{MENOR_QUEDA}** ({cs(MENOR_QUEDA, 'A', 'A', 'ACC')[0]:.2f}% → {cs(MENOR_QUEDA, 'A', 'C', 'ACC')[0]:.2f}%), contra {QUEDA_M['MLP Mel']:.2f} p.p. da MLP Mel; essa diferença de quedas {'é' if FT3['queda_svmmel_vs_mlp'][1] < 0.05 else 'não é'} significativa ({p_txt(FT3['queda_svmmel_vs_mlp'][1])}), e os desvios da ACC testada em C variam de {DESV_C[0]:.2f} a {DESV_C[1]:.2f} p.p.
3. **Treino em C, teste em A:** A CNN Mel local atingiu **{F(cs('CNN Mel', 'C', 'A', 'ACC'))}%** de acurácia balanceada em A (`54.08 ± 5.01%` no artigo), e o **SVM LOFAR**, **{F(cs('SVM LOFAR', 'C', 'A', 'ACC'))}%** ({cs('SVM LOFAR', 'C', 'A', 'ACC')[0] - cs('CNN Mel', 'C', 'A', 'ACC')[0]:+.2f} p.p.). Pelo F-test 5×2cv essa diferença {'é' if FT3['c_a_lofar_vs_cnn'][1] < 0.05 else 'não é'} significativa ({p_txt(FT3['c_a_lofar_vs_cnn'][1])}). A MLP Mel obteve {F(cs('MLP Mel', 'C', 'A', 'ACC'))}% no mesmo cenário, equivalente ao SVM LOFAR ({p_txt(FT3['c_a_lofar_vs_mlp'][1])}).
"""),
]

# ==================== NOTEBOOK 4: TRAJETÓRIA DOS EXPERIMENTOS ====================
# (ID, extrator, m, C, pré-processamento, rótulo do gráfico, experimento em results/trainings/tests, observação)
# C e penalidade ausentes no nome do experimento = padrão da classe SVMNystroem (C=1.0, penalty='l2').
EXPS4 = [
    (1, "MEL (256 bins)", 300, 1.0, "Sem PCA", "MEL", "svm_nystroem", "Fase exploratória inicial. Prova de conceito básica."),
    (2, "MEL (256 bins)", 1000, 1.0, "Sem PCA", "MEL", "svm_nystroem_1000", "Escalonamento preliminar."),
    (3, "MEL (256 bins)", 1000, 1.0, "Com PCA (Norm)", "MEL (PCA)", "svm_nystroem_1000_log_melgram_norm_pca64", "Efeito da compressão PCA sobre a representação MEL."),
    (4, "MEL (256 bins)", 2000, 1.0, "Sem PCA", "MEL", "svm_nystroem_2000_log_melgram", "Teste intermediário de média escala."),
    (5, "MEL (256 bins)", 3000, 1.0, "Sem PCA", "MEL", "svm_nystroem_3000_log_melgram", "Teste intermediário de alta escala."),
    (6, "MEL (256 bins)", 4000, 2.0, "Sem PCA (ElasticNet)", "MEL", SVM_MEL, "**Golden MEL**. Equivalente à CNN MEL local (F-test 5×2cv), com menor desvio entre folds."),
    (7, "MEL (256 bins)", 4000, 2.0, "PCA (64 comps, ElasticNet)", "MEL (PCA)", SVM_MEL_PCA, "MEL m=4000 com PCA (equivalente ao Exp #6; menor recall em SMALL)."),
    (8, "LOFAR (Frequência)", 1000, 1.0, "Sem PCA", "LOFAR (Sem PCA, C=1.0)", "svm_nystroem_1000_lofar", "Primeira integração espectral LOFAR."),
    (9, "LOFAR (Frequência)", 1000, 2.0, "Sem PCA", "LOFAR (Sem PCA, C=2.0)", "svm_nystroem_1000_lofar_C2.0", "LOFAR m=1000 sem PCA, C=2.0."),
    (10, "LOFAR (Frequência)", 1000, 2.0, "PCA (64 comps, L2)", "LOFAR (PCA)", "svm_nystroem_1000_lofar_pca64_C2.0", "LOFAR m=1000 com PCA64 padrão (L2)."),
    (11, "LOFAR (Frequência)", 1000, 2.0, "PCA (64 comps) + ElasticNet", "LOFAR (PCA - ElasticNet)", "svm_nystroem_1000_lofar_pca64_elasticnet_l1r0.15_C2.0", "LOFAR m=1000 com PCA64 + ElasticNet."),
    (12, "LOFAR (Frequência)", 2000, 2.0, "Sem PCA", "LOFAR (Sem PCA, C=2.0)", "svm_nystroem_2000_lofar_C2.0", "LOFAR intermediário em escala 2000."),
    (13, "LOFAR (Frequência)", 4000, 2.0, "Sem PCA", "LOFAR (Sem PCA, C=2.0)", SVM_LOFAR_NOPCA, "LOFAR em alta escala sem redução linear."),
    (14, "LOFAR (Frequência)", 4000, 2.0, "PCA (64 comps) + ElasticNet", "LOFAR (PCA)", SVM_LOFAR, "**Golden LOFAR**. Configuração adotada nos NBs 1–3."),
]
H = {e[0]: resumo_exp(e[6]) for e in EXPS4}
FM4 = {e[0]: fold_metrics_exp(e[6]) for e in EXPS4}


def ft4(a, b, met="SP"):
    return R.f_test_5x2(FM4[a][met].values, FM4[b][met].values)


FT4 = {"lofar1000": ft4(10, 9, "ACC"), "lofar4000": ft4(14, 13, "ACC"), "lofar4000_sp": ft4(14, 13),
       "mel1000": ft4(3, 2, "ACC"), "mel4000": ft4(7, 6, "ACC"), "mel4000_small": ft4(7, 6, "R_SMALL"),
       "mel6_vs_5": ft4(6, 5, "ACC")}
# Guarda: o texto do NB4 afirma que nenhuma destas comparações é significativa
assert all(v[1] >= 0.05 for v in FT4.values()), FT4
STD_RATIO4 = H[6]["ACC"][1] / S_CNN["ACC"][1]


def b(i, txt):
    return f"**{txt}**" if i in (6, 14) else txt


hist_rows = "\n".join(
    f"| **Exp #{i}** | {ext} | {m} | {c:.1f} | {pre} | {b(i, F(H[i]['ACC']) + '%')} | {b(i, F(H[i]['ACC_micro']) + '%')} | "
    f"{b(i, F(H[i]['SP']) + '%')} | `{exp}` | {obs} |"
    for i, ext, m, c, pre, _, exp, obs in EXPS4)

cells_4 = [
    md(r"""
# Evolução do Desempenho e Trajetória dos Experimentos SVM Nyström
### Notebook 4: Histórico Completo de Escalonamento e Aprendizado (m = 300 até m = 4000)

Neste notebook, é documentado o percurso empírico da otimização do modelo **SVM com Aproximação de Nyström** sob o protocolo $5\times2$cv (10 folds com **particionamento por gravação de áudio `ID`, `exclusive_header='ID'`**, garantindo que janelas de um mesmo arquivo de áudio não coexistam no treino e no teste), cobrindo o escalonamento do número de componentes de Nyström ($m$) tanto para o extrator **MEL** quanto para o extrator **LOFAR** com e sem PCA.

> **Nota Metodológica sobre Consistência de Métricas:**
> Para garantir comparação homogênea entre os 14 experimentos, reportamos lado a lado as três métricas no nível de áudio (votação majoritária das janelas), **recalculadas neste notebook** a partir das predições em `results/trainings/tests/<experimento>/eval/fold_k/` (média ± desvio-padrão populacional nos 10 folds):
> * **Acurácia Balanceada (`Bal. Acc`, %):** Média aritmética dos recalls das 4 classes (`GridCompiler`, métrica `ACC` das Tabelas 7–13 de *Silva et al., 2025*).
> * **Acurácia Global/Micro (`UACC`, %):** Proporção total de áudios classificados corretamente (`accuracy_score`).
> * **Índice SP (%):** $SP = \sqrt{\overline{r} \cdot \left(\prod_i r_i\right)^{1/4}}$, raiz do produto da média aritmética pela média geométrica dos recalls das 4 classes (`src/iara/ml/metrics.py`).
>
> $C$ e penalidade não presentes no nome do experimento correspondem ao padrão da classe `SVMNystroem` ($C=1.0$, penalidade L2).
"""),
    md(f"""
### Tabela Histórica Consolidada de Experimentos SVM (10 Folds $5\\times2$cv):

| ID | Extrator Espectral | Dimensão ($m$) | Reg. ($C$) | Pré-processamento / Redução | Acurácia Balanceada (`Bal. Acc`) (%) | Acurácia Micro (`UACC`) (%) | Índice SP (%) | Experimento | Observações / Fase Científica |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
{hist_rows}
"""),
    code(HELPERS),
    code(SETUP_PLOT),
    md(r"""
### 1. Modelagem da Trajetória Empírica de Aprendizado
Os 14 experimentos são mapeados para as pastas em `results/trainings/tests/` e suas métricas (**Acurácia Balanceada `Bal_ACC`**, **Acurácia Global/Micro `UACC`** e **Índice SP**) são recalculadas na célula abaixo.
"""),
    code("EXPERIMENTOS = [\n    # (ID, m, C, rótulo do grupo, experimento)\n" +
         "".join(f"    ({i}, {m}, {c}, '{lab}', '{exp}'),\n" for i, _, m, c, _, lab, exp, _ in EXPS4) + "]\n" + r"""
linhas = []
for i, m, c, extrator, exp in EXPERIMENTOS:
    s = resumo_exp(exp)
    linhas.append({'Exp': f'Exp #{i}', 'm': m, 'C': c, 'Extrator': extrator,
                   'Bal_ACC': s['ACC'][0], 'Bal_ACC_std': s['ACC'][1],
                   'UACC': s['ACC_micro'][0], 'UACC_std': s['ACC_micro'][1],
                   'SP': s['SP'][0], 'SP_std': s['SP'][1], 'R_SMALL': s['R_SMALL'][0], 'Experimento': exp})
df_hist = pd.DataFrame(linhas).round(2)
df_hist
"""),
    md(r"""
### 2. Plotagem das Curvas de Escalonamento (Acurácia Balanceada e Índice SP vs. Dimensão m)
Abaixo plotamos a evolução dos 14 experimentos sob a **mesma métrica** (**Acurácia Balanceada `Bal. Acc`** no painel esquerdo e **Índice SP** no painel direito), separando os pontos de $C=1.0$ e $C=2.0$ em `m=1000` para evitar sobreposição vertical. Atenção: a curva "MEL (Sem PCA)" mistura $C=1.0$/L2 ($m \le 3000$) com $C=2.0$/ElasticNet ($m=4000$).
"""),
    code(r"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

mel_data = df_hist[df_hist['Extrator'] == 'MEL'].sort_values('m')
lofar_pca_data = df_hist[df_hist['Extrator'] == 'LOFAR (PCA)'].sort_values('m')
mel_pca_data = df_hist[df_hist['Extrator'] == 'MEL (PCA)'].sort_values('m')
lofar_no_pca_c2 = df_hist[df_hist['Extrator'] == 'LOFAR (Sem PCA, C=2.0)'].sort_values('m')
lofar_no_pca_c1 = df_hist[df_hist['Extrator'] == 'LOFAR (Sem PCA, C=1.0)']
lofar_pca_enet = df_hist[df_hist['Extrator'] == 'LOFAR (PCA - ElasticNet)']
golden_mel = df_hist[df_hist['Exp'] == 'Exp #6'].iloc[0]
golden_lofar = df_hist[df_hist['Exp'] == 'Exp #14'].iloc[0]

for ax, met, titulo, ylab, ylim in [
        (ax1, 'Bal_ACC', 'Escalonamento: Acurácia Balanceada (Bal. Acc)', 'Acurácia Balanceada Média (%)', (56, 68)),
        (ax2, 'SP', 'Escalonamento: Índice SP Oficial (%)', 'Índice SP Médio (%)', (54, 67))]:
    ax.errorbar(mel_data['m'], mel_data[met], yerr=mel_data[met + '_std'], fmt='o-', color='#1abc9c',
                linewidth=2.5, elinewidth=1.3, capsize=4, label='MEL (Sem PCA)', markersize=8)
    ax.errorbar(lofar_pca_data['m'], lofar_pca_data[met], yerr=lofar_pca_data[met + '_std'], fmt='s--', color='#34495e',
                linewidth=2.0, elinewidth=1.3, capsize=4, label='LOFAR + PCA64 (C=2.0)', markersize=8)
    ax.errorbar(mel_pca_data['m'], mel_pca_data[met], yerr=mel_pca_data[met + '_std'], fmt='^-.', color='#e67e22',
                linewidth=2.0, elinewidth=1.3, capsize=4, label='MEL + PCA64', markersize=8)
    ax.errorbar(lofar_no_pca_c2['m'], lofar_no_pca_c2[met], yerr=lofar_no_pca_c2[met + '_std'], fmt='x:', color='#c0392b',
                linewidth=2.0, elinewidth=1.3, capsize=4, label='LOFAR Sem PCA (C=2.0)', markersize=8)
    ax.scatter(lofar_no_pca_c1['m'], lofar_no_pca_c1[met], color='#e74c3c', marker='D', s=90, zorder=5,
               label='LOFAR m=1000 (Sem PCA, C=1.0)')
    ax.scatter(lofar_pca_enet['m'], lofar_pca_enet[met], color='#9b59b6', marker='*', s=140, zorder=5,
               label='LOFAR m=1000 (PCA64 + ElasticNet)')
    ax.set_title(titulo, fontsize=13, weight='bold')
    ax.set_xlabel('Número de Componentes de Nyström (m)', fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_xscale('log')
    ax.set_xticks([300, 1000, 2000, 3000, 4000], ['300', '1000', '2000', '3000', '4000'])
    ax.set_ylim(*ylim)
    ax.legend(fontsize=9, loc='lower right')

ax1.annotate(f"Golden MEL ({golden_mel['Bal_ACC']:.2f}% ± {golden_mel['Bal_ACC_std']:.2f}%)", xy=(4000, golden_mel['Bal_ACC']),
             xytext=(1100, 66.2), arrowprops=dict(facecolor='#16a085', shrink=0.08, width=1.5, headwidth=6),
             fontsize=9.5, weight='bold', color='#16a085')
ax1.annotate(f"Golden LOFAR ({golden_lofar['Bal_ACC']:.2f}% ± {golden_lofar['Bal_ACC_std']:.2f}%)", xy=(4000, golden_lofar['Bal_ACC']),
             xytext=(1500, 58.2), arrowprops=dict(facecolor='#2c3e50', shrink=0.08, width=1.5, headwidth=6),
             fontsize=9.5, weight='bold', color='#2c3e50')

plt.suptitle('Histórico Completo de Escalonamento do SVM Nyström (IARA - 10 Folds 5x2cv)', fontsize=15, weight='bold', y=1.01)
plt.tight_layout()
plt.show()
"""),
    md(r"""
### 3. Testes de Significância das Comparações Citadas (F-test 5×2cv)
"""),
    code(r"""
fm = {i: fold_metrics_exp(exp) for i, _, _, _, exp in EXPERIMENTOS}
pares = [(10, 9, 'LOFAR m=1000 C=2.0: PCA64 vs. sem PCA'), (14, 13, 'LOFAR m=4000: PCA64+ElasticNet vs. sem PCA'),
         (3, 2, 'MEL m=1000: PCA vs. sem PCA'), (7, 6, 'MEL m=4000: PCA64 vs. sem PCA'), (6, 5, 'MEL: Exp #6 (m=4000) vs. Exp #5 (m=3000)')]
linhas = []
for a, b_, rot in pares:
    linha = {'Comparação': rot}
    for met in ('ACC', 'SP', 'R_SMALL'):
        f, p = R.f_test_5x2(fm[a][met].values, fm[b_][met].values)
        linha[f'{met}: A − B (p.p.)'] = round(fm[a][met].mean() - fm[b_][met].mean(), 2)
        linha[f'p ({met})'] = round(p, 3)
    linhas.append(linha)
pd.DataFrame(linhas).set_index('Comparação')
"""),
    md(f"""
### 4. Discussão Científica e Análise das Fases de Sintonia:
1. **PCA no MEL vs. LOFAR — diferenças sem significância estatística:** No **LOFAR**, o PCA64 eleva a Acurácia Balanceada de **{H[9]['ACC'][0]:.2f}%** (`UACC = {H[9]['ACC_micro'][0]:.2f}%`, `SP = {H[9]['SP'][0]:.2f}%`) para **{H[10]['ACC'][0]:.2f}%** (`UACC = {H[10]['ACC_micro'][0]:.2f}%`, `SP = {H[10]['SP'][0]:.2f}%`) em $m=1000$ ($C=2.0$; {p_txt(FT4['lofar1000'][1])}) e de **{H[13]['ACC'][0]:.2f}%** (`SP = {H[13]['SP'][0]:.2f}%`) para **{H[14]['ACC'][0]:.2f}%** (`SP = {H[14]['SP'][0]:.2f}%`) em $m=4000$ ({p_txt(FT4['lofar4000'][1])} em ACC; {p_txt(FT4['lofar4000_sp'][1])} em SP). No **MEL** em $m=1000$, o PCA (Exp #3) reduz a Acurácia Balanceada de **{H[2]['ACC'][0]:.2f}%** para **{H[3]['ACC'][0]:.2f}%** ({p_txt(FT4['mel1000'][1])}); em $m=4000$ a ACC balanceada fica praticamente igual ({H[6]['ACC'][0]:.2f}% sem PCA vs. {H[7]['ACC'][0]:.2f}% com PCA64; {p_txt(FT4['mel4000'][1])}) e o recall de `SMALL` passa de **{H[6]['R_SMALL'][0]:.2f}%** para **{H[7]['R_SMALL'][0]:.2f}%** ({p_txt(FT4['mel4000_small'][1])}). Nenhuma dessas diferenças é significativa pelo F-test 5×2cv; as tendências (PCA ajuda no LOFAR, não ajuda no MEL) são compatíveis com o banco de filtros Mel já comprimir o espectro, mas não estão demonstradas.
2. **Escalonamento de $m$:** No MEL sem PCA, a Acurácia Balanceada passa de {H[1]['ACC'][0]:.2f}% ($m=300$) para {H[5]['ACC'][0]:.2f}% ($m=3000$) e {H[6]['ACC'][0]:.2f}% ($m=4000$); de $m=2000$ em diante as médias ficam dentro do desvio (Exp #6 vs. Exp #5: {p_txt(FT4['mel6_vs_5'][1])}). O desvio fold-wise do Golden MEL é o menor da série: $\\pm {H[6]['ACC'][1]:.2f}\\%$ em `Bal. Acc` ($\\pm {H[6]['SP'][1]:.2f}\\%$ em `SP` e $\\pm {H[6]['ACC_micro'][1]:.2f}\\%$ em `UACC`), cerca de {STD_RATIO4:.2f}× o desvio da CNN MEL local ($\\pm {S_CNN['ACC'][1]:.2f}\\%$), mas essa redução coincide com a troca de $C=1.0$/L2 por $C=2.0$/ElasticNet, de modo que não pode ser atribuída apenas ao aumento de $m$.
"""),
]

# Gerando os arquivos .ipynb fisicamente
create_notebook(os.path.join(HERE, "1_comparacao_geral.ipynb"), cells_1)
create_notebook(os.path.join(HERE, "2_opcao_rejeicao.ipynb"), cells_2)
create_notebook(os.path.join(HERE, "3_cpa_proximidade.ipynb"), cells_3)
create_notebook(os.path.join(HERE, "4_trajetoria_experimentos.ipynb"), cells_4)
print("Todos os notebooks Jupyter foram criados na pasta 'notebooks/' com sucesso!")
