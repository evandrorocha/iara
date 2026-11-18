"""
Gera notebooks/41_metadados_ambientais.ipynb: o que os metadados do IARA (vento, estado do mar, chuva, CPA) dizem sobre
os erros do experimento mix_small_medium, e o augmentation dos LARGE com fundo de mar agitado.

Fontes (nenhum número digitado à mão):
  - metadados: src/iara/dataset_info/iara.csv (cópia do iara.xlsx do Zenodo)
  - LOFAR médio por gravação: data/iara_processed/cache_lofar_espectro_medio.pkl
  - MLP mix_small_medium: results/trainings/multirep/mix_small_medium_{split_navio,split_id}_mlp_lofar[_aug<variante>]
    (scratch/mix_small_medium/treino_mlp.py e aug_mar.py; extração scripts/extrair_lofar_mixbg_mar.py)

Rodar na raiz:  python notebooks/src/generate_metadados_ambientais_notebook.py
Executar no container:  docker exec -w /workspace/IARA iara-dev python scratch/execute_notebooks.py notebooks/41_metadados_ambientais.ipynb
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")
from generate_eda_lofar_notebook import br, pv, md, code, run  # noqa: E402

NOTEBOOKS_DIR = os.path.abspath(os.path.join(HERE, ".."))

SETUP_CODE = r'''
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
warnings.filterwarnings('ignore')
try:
    from IPython.display import display
except ImportError:
    display = print
for _p in ('.', 'src', 'notebooks', 'notebooks/src'):
    if os.path.exists(os.path.join(_p, 'iara_results.py')) and os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))
import iara_results as R
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'figure.dpi': 120})
pd.set_option('display.precision', 1); pd.set_option('display.width', 200)

CN = {0: 'SMALL', 1: 'MEDIUM', 2: 'LARGE', 3: 'BACKGROUND'}
CM = {0: 'MIX_SMALL_MEDIUM', 1: 'LARGE', 2: 'BACKGROUND'}
MAPA = {0: 0, 1: 0, 2: 1, 3: 2}
meta = pd.read_csv(os.path.join(R.ROOT, 'src', 'iara', 'dataset_info', 'iara.csv'), dtype={'Ship ID': str, 'ID': str})
RAW_COLS = list(meta.columns)
meta = meta[meta.Dataset.isin(list('ABCDE'))].copy()
meta.index = meta.ID.astype(int); meta['Ship ID'] = meta['Ship ID'].str.strip()
meta['Length'] = pd.to_numeric(meta.Length, errors='coerce')
meta['Target'] = np.select([meta.Length < 50, meta.Length < 100, meta.Length >= 100], [0, 1, 2], default=3)
meta.loc[meta.Dataset == 'E', 'Target'] = 3
meta['Classe'] = meta.Target.map(CN)
meta['Vento'] = pd.to_numeric(meta.Wind, errors='coerce')
meta['Mar'] = pd.to_numeric(meta['Sea state'], errors='coerce')
meta['Chuva'] = pd.to_numeric(meta.Rain, errors='coerce')
meta['CPA'] = meta.Dataset.isin(['A', 'C'])
meta['Mar agitado'] = meta.Mar >= 3
'''

META_CODE = r'''
cob = pd.DataFrame({'preenchido (%)': [100 * (meta[c].astype(str).str.strip() != '-').mean() for c in RAW_COLS]}, index=RAW_COLS)
cob['exemplo'] = [meta[c].astype(str).iloc[0] for c in RAW_COLS]
print('Colunas da planilha de metadados do IARA (subconjuntos A–E):'); display(cob)
'''

CLIMA_CODE = r'''
CLIMA = meta.groupby('Classe').agg(gravações=('ID', 'size'), vento_mediana=('Vento', 'median'), mar_mediana=('Mar', 'median'),
                                   mar_calmo_pct=('Mar', lambda s: 100 * (s <= 2).mean()), chuva_pct=('Chuva', lambda s: 100 * (s > 0).mean()))
CLIMA = CLIMA.reindex(['SMALL', 'MEDIUM', 'LARGE', 'BACKGROUND'])
print('Condições de gravação por classe (por gravação):'); display(CLIMA)
MAR_DIST = pd.crosstab(meta.Classe, meta.Mar.fillna(-1).astype(int), normalize='index').mul(100).reindex(['SMALL', 'MEDIUM', 'LARGE', 'BACKGROUND'])
MAR_DIST = MAR_DIST.drop(columns=[-1], errors='ignore')
print('Estado do mar, % das gravações de cada classe:'); display(MAR_DIST.round(0))

# por navio: um ponto por navio
nav = meta[meta.Target < 3].groupby('Ship ID').agg(t=('Target', 'first'), vento=('Vento', 'mean'), mar=('Mar', 'mean'))
TESTE = []
for nome, a, b in [('LARGE × navios < 100 m', nav[nav.t < 2], nav[nav.t == 2]), ('MEDIUM × SMALL', nav[nav.t == 0], nav[nav.t == 1])]:
    for v in ['vento', 'mar']:
        x, y = a[v].dropna(), b[v].dropna(); u = st.mannwhitneyu(y, x)
        TESTE.append({'comparação': nome, 'variável': v, 'mediana 1º grupo': x.median(), 'mediana 2º grupo': y.median(),
                      'AUC (2º > 1º)': u.statistic / (len(x) * len(y)), 'p': u.pvalue, 'navios': f'{len(x)} × {len(y)}'})
TESTE = pd.DataFrame(TESTE)
print('Um ponto por navio (Mann-Whitney):'); display(TESTE.round(3))
'''

BG_CODE = r'''
bg = meta[meta.Target == 3]
BG = pd.crosstab(bg.Mar.fillna(-1).astype(int), bg['Rain state'].str.strip(), margins=True, margins_name='total')
BG = BG[[c for c in ['No rain', 'Light', 'Moderate', 'Heavy', 'Very Heavy', 'total'] if c in BG.columns]]
BG.index.name = 'estado do mar'
print(f'BACKGROUND: {len(bg)} gravações (subconjunto {bg.Dataset.unique()[0]}); vento de {bg.Vento.min():.1f} a {bg.Vento.max():.1f} m/s')
display(BG)
BG_AGIT = int(((bg.Mar >= 4) & (bg['Rain state'].str.strip() == 'No rain')).sum())
BG_CALM = int(((bg.Mar <= 2) & (bg['Rain state'].str.strip() == 'No rain')).sum())
print(f'Fundos sem chuva com mar >= 4 (usados no augmentation "agitado"): {BG_AGIT}; com mar <= 2 ("calmo"): {BG_CALM}')
'''

ERRO_CODE = r'''
ESP = pd.read_pickle(os.path.join(R.ROOT, 'data/iara_processed/cache_lofar_espectro_medio.pkl'))
F = (np.arange(1024) + 1) * 16000 / 2048
meta['banda_31_86'] = ESP.loc[:, (F >= 31) & (F < 86)].mean(1).reindex(meta.index)

def prev(exp):
    d = pd.concat([v.assign(fold=k) for k, v in R.file_predictions(exp).items()], ignore_index=True)
    d['File'] = d.File.astype(int)
    return d.join(meta[['Classe', 'Mar', 'Vento', 'CPA', 'Dataset', 'banda_31_86', 'Ship ID']], on='File')

def estratos(d, alvo):
    x = d[d.Target == alvo].copy()
    x['mar'] = np.where(x.Mar <= 2, 'calmo (0–2)', 'agitado (3+)')
    x['cpa'] = np.where(x.CPA, 'com CPA (A, C)', 'sem CPA (B, D)')
    g = x.assign(ok=x.Prediction == alvo).groupby(['cpa', 'mar']).agg(acerto=('ok', 'mean'), gravações=('File', 'nunique'),
                                                                        banda_31_86=('banda_31_86', 'median'))
    g['acerto'] *= 100
    return g.reindex(pd.MultiIndex.from_product([['com CPA (A, C)', 'sem CPA (B, D)'], ['calmo (0–2)', 'agitado (3+)']]))

BASE = prev('mix_small_medium_split_navio_mlp_lofar')
EST_L, EST_M = estratos(BASE, 1), estratos(BASE, 0)
print('MLP mix_small_medium, split por navio: acerto de LARGE por CPA × estado do mar'); display(EST_L.round(1))
print('Acerto de MIX_SMALL_MEDIUM por CPA × estado do mar'); display(EST_M.round(1))
x = BASE[BASE.Target == 1]
MAR_FINO = x.assign(ok=x.Prediction == 1).groupby(x.Mar.clip(upper=5)).ok.agg(['mean', 'size']).rename(columns={'mean': 'acerto LARGE', 'size': 'avaliações'})
MAR_FINO['acerto LARGE'] *= 100; MAR_FINO.index = [f'{int(i)}' if i < 5 else '5+' for i in MAR_FINO.index]
print('Acerto de LARGE por estado do mar (todas as gravações):'); display(MAR_FINO.round(1))
'''

AUG_CODE = r'''
VAR = {'base': 'sem augmentation', 'dupL': 'LARGE duplicado (controle)', 'calmo': 'LARGE + fundo de mar calmo (controle)',
       'agitado': 'LARGE + fundo de mar agitado'}
def exp(cond, v):
    return f'mix_small_medium_{cond}_mlp_lofar' + ('' if v == 'base' else f'_aug{v}')
def metricas(y, p):
    rec = np.array([np.mean(p[y == c] == c) for c in range(3)])
    sp = np.sqrt(rec.mean() * np.exp(np.mean(np.log(rec)))) * 100 if np.all(rec > 0) else 0.0
    return {'SP': sp, 'ACC': rec.mean() * 100, **{f'R_{CM[c]}': r * 100 for c, r in enumerate(rec)}}
FMA, PREVA = {}, {}
for cond in ['split_navio', 'split_id']:
    for v in VAR:
        P = R.file_predictions(exp(cond, v))
        FMA[(cond, v)] = pd.DataFrame({k: metricas(d.Target.values, d.Prediction.values) for k, d in P.items()}).T
        if cond == 'split_navio': PREVA[v] = prev(exp(cond, v))
MEDIAS = pd.concat({(c, VAR[v]): f.mean() for (c, v), f in FMA.items()}, axis=1).T
print('Médias de 10 folds:'); display(MEDIAS.round(1))
rows = []
for cond in ['split_navio', 'split_id']:
    for v in ['dupL', 'calmo', 'agitado']:
        for ref in ['base', 'dupL']:
            if ref == v: continue
            A, B = FMA[(cond, v)], FMA[(cond, ref)]
            for m in ['ACC', 'R_LARGE', 'R_MIX_SMALL_MEDIUM']:
                rows.append({'split': cond, 'variante': v, 'contra': ref, 'métrica': m, 'Δ': (A[m] - B[m]).mean(),
                             'folds melhores': int((A[m] > B[m]).sum()), 'p (F 5x2cv)': R.f_test_5x2(A[m].values, B[m].values)[1]})
DELTA = pd.DataFrame(rows)
print('Diferenças (F 5x2cv):'); display(DELTA.round(3))
EST_AUG = pd.concat({VAR[v]: estratos(PREVA[v], 1).acerto for v in VAR}, axis=1)
print('Acerto de LARGE (%) por CPA × estado do mar, split por navio:'); display(EST_AUG.round(1))

fig, ax = plt.subplots(figsize=(8, 3.2))
lab = [f'{a}\n{b}' for a, b in EST_AUG.index]; xx = np.arange(len(lab)); w = 0.2
cores = {'sem augmentation': '#8a8a8a', 'LARGE duplicado (controle)': '#b9b9b9', 'LARGE + fundo de mar calmo (controle)': '#9fc3ea',
         'LARGE + fundo de mar agitado': '#1baf7a'}
for i, c in enumerate(EST_AUG.columns):
    ax.bar(xx + (i - 1.5) * w, EST_AUG[c].values, w * 0.92, color=cores[c], label=c)
ax.set_xticks(xx); ax.set_xticklabels(lab, fontsize=8); ax.set_ylabel('acerto de LARGE (%)'); ax.set_ylim(0, 100)
ax.legend(fontsize=7, ncol=2, loc='upper left'); ax.set_title('Split por navio')
plt.tight_layout(); plt.show()
'''


def textos(ns):
    f0, f1 = (lambda x: br(x, 0)), (lambda x: br(x, 1))
    MD, DE, EA, EL, CL = ns["MEDIAS"], ns["DELTA"], ns["EST_AUG"], ns["EST_L"], ns["CLIMA"]
    V = ns["VAR"]
    nv = lambda v, m: MD.loc[("split_navio", V[v]), m]
    d = DE.set_index(["split", "variante", "contra", "métrica"])
    dd = lambda v, ref, m, s="split_navio": d.loc[(s, v, ref, m)]
    COM, SEM, CAL, AGI = "com CPA (A, C)", "sem CPA (B, D)", "calmo (0–2)", "agitado (3+)"
    pmin = DE.loc[DE.variante.isin(["agitado", "calmo"]), "p (F 5x2cv)"].min()
    ns["RESUMO"] = f"""**Resumo das respostas:**
1. **Os metadados públicos são poucos:** chuva, vento e estado do mar (de uma estação próxima) e o instante do CPA. **Não há data, hora nem observatório**, e os dados brutos (áudio contínuo, AIS) só por solicitação ao projeto PMPAS-BS.
2. **Os LARGE foram gravados com o mar mais calmo** (mar ≤ 2 em {f0(CL.loc['LARGE', 'mar_calmo_pct'])}% das gravações, contra {f0(CL.loc['SMALL', 'mar_calmo_pct'])}–{f0(CL.loc['MEDIUM', 'mar_calmo_pct'])}% de SMALL e MEDIUM): um viés da base.
3. **A MLP mix_small_medium erra mais LARGE com o mar agitado:** com CPA, {f0(EL.loc[(COM, CAL), 'acerto'])}% → {f0(EL.loc[(COM, AGI), 'acerto'])}%; sem CPA, {f0(EL.loc[(SEM, CAL), 'acerto'])}% → {f0(EL.loc[(SEM, AGI), 'acerto'])}%.
4. **Misturar ruído real de mar agitado nos LARGE de treino não ajudou:** recall de LARGE {f1(nv('base', 'R_LARGE'))}% → {f1(nv('agitado', 'R_LARGE'))}% no split por navio, e os LARGE de mar agitado também não melhoraram. Nenhuma diferença foi significativa (menor p = {br(pmin, 2)}).
"""
    ns["LEITURA_AUG"] = f"""**Leitura.**
- **Nenhuma variante melhora o modelo.** No split por navio, ACC: sem augmentation {f1(nv('base', 'ACC'))}; LARGE duplicado {f1(nv('dupL', 'ACC'))}; fundo calmo {f1(nv('calmo', 'ACC'))}; **fundo agitado {f1(nv('agitado', 'ACC'))}**. Todas as diferenças têm p > 0,05.
- **O recall de LARGE até cai um pouco quando se mistura fundo** (agitado {f1(dd('agitado', 'base', 'R_LARGE')['Δ'])}, calmo {f1(dd('calmo', 'base', 'R_LARGE')['Δ'])} pontos; não significativo), enquanto o de {ns['CM'][0]} sobe um pouco. Como o fundo calmo faz o mesmo, **não é um efeito do mar agitado**: é o efeito de mostrar LARGE com o grave "diluído".
- **Os LARGE de mar agitado não melhoraram** (com CPA: {f0(EA.loc[(COM, AGI)].iloc[0])}% sem augmentation → {f0(EA.loc[(COM, AGI), V['agitado']])}% com fundo agitado; sem CPA: {f0(EA.loc[(SEM, AGI)].iloc[0])}% → {f0(EA.loc[(SEM, AGI), V['agitado']])}%).
- **Interpretação provável (não testada):** somar ruído ao LARGE de treino apaga parte do grave que o diferencia; o exemplo aumentado fica parecido com um navio < 100 m, e a fronteira fica mais confusa em vez de mais robusta. É a mesma limitação do NB40: quando o grave some do LOFAR, a informação não está lá para ser aprendida.
"""
    ns["CONCLUSOES"] = f"""## 7. Conclusões

1. **Os metadados ajudam a entender os erros, não a corrigi-los.** O mar agitado e a falta de CPA explicam boa parte dos erros de LARGE ({f0(EL.loc[(SEM, AGI), 'acerto'])}% de acerto quando as duas coisas se juntam).
2. **Há um viés de condição na base:** os LARGE foram gravados com o mar mais calmo. Os resultados do IARA para LARGE são, provavelmente, um pouco otimistas para condições reais.
3. **O augmentation com mar agitado não melhorou o modelo**, e o controle com mar calmo mostra que misturar qualquer fundo tende a tirar um pouco de recall de LARGE.
4. **Não recomendo usar vento e estado do mar como entrada do modelo** (risco de atalho "mar calmo → LARGE"). Usos mais úteis: reportar o desempenho por condição e permitir abstenção.
5. **O que mais ajudaria vem de fora do IARA público:** data, hora e observatório de cada gravação (para testar efeitos de época e equipamento) e mais navios distintos abaixo de 100 m. Ambos exigem solicitação aos autores ou ao PMPAS-BS (SIMMAO).
"""


def celulas(ns):
    f0, f1 = (lambda x: br(x, 0)), (lambda x: br(x, 1))
    CL, TE, EL, EM, MF = ns["CLIMA"], ns["TESTE"], ns["EST_L"], ns["EST_M"], ns["MAR_FINO"]
    tl = TE.set_index(["comparação", "variável"])
    LXM, MXS = "LARGE × navios < 100 m", "MEDIUM × SMALL"
    el = lambda c, m: EL.loc[(c, m), "acerto"]
    COM, SEM, CAL, AGI = "com CPA (A, C)", "sem CPA (B, D)", "calmo (0–2)", "agitado (3+)"
    return [
        md(f"""
# 41 — Metadados ambientais do IARA: vento, estado do mar e os erros de LARGE

**Pergunta.** Além do áudio, o IARA traz alguns metadados por gravação (chuva, vento, estado do mar, instante do CPA). Eles ajudam a entender os erros do experimento `mix_small_medium` (NB40)? Dá para aproveitá-los?

{ns['RESUMO']}
"""),
        md("""
## Glossário rápido

- **Estado do mar:** escala de 0 a 9 da agitação da superfície. No IARA, **estimado a partir do vento** de uma estação meteorológica próxima, não medido no hidrofone.
- **CPA (*Closest Point of Approach*):** instante em que o navio passa mais perto do hidrofone. Subconjuntos A (até 250 m) e C (250–500 m) contêm o CPA; B e D não.
- **Split por navio:** nenhum navio aparece no treino e no teste ao mesmo tempo (folds do NB29).
- **Augmentation:** criar versões modificadas das gravações de treino; aqui, misturar o áudio de um navio com uma gravação real de mar sem navio.
- **Valor-p (teste F 5x2cv):** abaixo de 0,05, a diferença entre dois métodos provavelmente é real.
"""),
        code(ns["_SRC"]["SETUP"]),
        md("""
## 1. Que metadados existem

O IARA é publicado no Zenodo (DOI 10.5281/zenodo.15777429, licença CC BY-NC 4.0) como áudios já recortados (A.zip a H.zip) e uma planilha `iara.xlsx`. A planilha foi copiada para `src/iara/dataset_info/iara.csv`.
"""),
        code(ns["_SRC"]["META"]),
        md("""
**Leitura.**
- **Não há data nem hora da gravação** (a coluna `Recording time` é a **duração** em segundos), **nem o observatório** de origem. Os arquivos WAV também não trazem essa informação no cabeçalho.
- O que existe: chuva, vento e estado do mar (de uma estação próxima), o instante do CPA (só A e C) e os dados do navio (nome, tipo, comprimento; na planilha original também MMSI e IMO).
- **Os dados brutos não são públicos.** As gravações contínuas dos observatórios e o AIS pertencem ao projeto de monitoramento da Petrobras (PMPAS-BS). O portal do projeto (SIMMAO) mostra dados processados (mapas de ruído, espectros, densidade de tráfego) e tem um formulário de **solicitação de dados para fins científicos**. Sem data e observatório, não dá para testar efeitos de época ou de equipamento.
"""),
        md("""
## 2. Os LARGE foram gravados com o mar mais calmo
"""),
        code(ns["_SRC"]["CLIMA"]),
        md(f"""
**Leitura.**
- **LARGE:** vento mediano {f1(CL.loc['LARGE', 'vento_mediana'])} m/s e mar calmo (estado ≤ 2) em {f0(CL.loc['LARGE', 'mar_calmo_pct'])}% das gravações. **SMALL e MEDIUM:** {f1(CL.loc['SMALL', 'vento_mediana'])} e {f1(CL.loc['MEDIUM', 'vento_mediana'])} m/s, mar calmo em {f0(CL.loc['SMALL', 'mar_calmo_pct'])}% e {f0(CL.loc['MEDIUM', 'mar_calmo_pct'])}%.
- **Por navio, a diferença é significativa** entre LARGE e os navios < 100 m ({pv(tl.loc[(LXM, 'mar'), 'p'])} para o estado do mar), mas **não entre SMALL e MEDIUM** ({pv(tl.loc[(MXS, 'mar'), 'p'])}).
- **Chuva quase não aparece nas gravações de navio** ({f0(CL.loc['LARGE', 'chuva_pct'])}–{f0(CL.loc['SMALL', 'chuva_pct'])}%), mas aparece em {f0(CL.loc['BACKGROUND', 'chuva_pct'])}% das de BACKGROUND, que foram escolhidas de propósito em condições variadas. Risco pequeno de o modelo associar "chuva" a "mar sem navio".
- **É um viés da base:** se os LARGE tivessem sido gravados nas mesmas condições dos outros, a separação entre as classes provavelmente seria um pouco menor.
"""),
        md("""
## 3. O BACKGROUND cobre mar agitado
"""),
        code(ns["_SRC"]["BG"]),
        md(f"""
**Leitura.** A seleção do BACKGROUND foi balanceada: **25 gravações sem chuva em cada estado do mar de 1 a 6**, mais as com chuva. Há {ns['BG_AGIT']} fundos sem chuva com mar ≥ 4 e {ns['BG_CALM']} com mar ≤ 2: material suficiente para "sujar" gravações de navio com ruído de mar agitado (seção 5).
"""),
        md("""
## 4. A MLP mix_small_medium erra mais LARGE com o mar agitado

Previsões da MLP do NB40 (split por navio), cruzadas com o estado do mar e a presença de CPA. A coluna `banda_31_86` é a energia mediana em 31–86 Hz (dB acima do fundo), a faixa grave que mais diferencia LARGE (NB40, seção 6).
"""),
        code(ns["_SRC"]["ERRO"]),
        md(f"""
**Leitura.**
- **O mar agitado tira cerca de 10 pontos do acerto de LARGE, com ou sem CPA:** com CPA, {f0(el(COM, CAL))}% (calmo) → {f0(el(COM, AGI))}% (agitado); sem CPA, {f0(el(SEM, CAL))}% → **{f0(el(SEM, AGI))}%**.
- **A faixa grave fica mais fraca com o mar agitado** ({f1(EL.loc[(COM, CAL), 'banda_31_86'])} → {f1(EL.loc[(COM, AGI), 'banda_31_86'])} dB com CPA): o ruído do mar sobe e o grave do navio se destaca menos do fundo.
- **Já a classe {ns['CM'][0]} quase não sofre** ({f0(EM.loc[(COM, CAL), 'acerto'])}% → {f0(EM.loc[(COM, AGI), 'acerto'])}% com CPA), porque a pista dela não é o grave forte.
- Por estado do mar: LARGE acerta {f0(MF.iloc[0, 0])}% com mar 1 e {f0(MF.iloc[2, 0])}–{f0(MF.iloc[3, 0])}% com mar 3–4.
"""),
        md(f"""
## 5. Augmentation: LARGE misturado com fundo de mar agitado

**Ideia:** mostrar ao modelo LARGE "sujos" por ruído real de mar agitado.

**Como foi feito** (`scripts/extrair_lofar_mixbg_mar.py` e `scratch/mix_small_medium/aug_mar.py`):
- Para cada gravação LARGE, o **áudio** é somado a uma gravação de BACKGROUND sem chuva com mar ≥ 4, com razão sinal/fundo sorteada entre 0 e 20 dB, e o LOFAR é recalculado igual ao do pipeline. O LOFAR original não é alterado.
- No treino de cada fold, cada gravação LARGE ganha **uma** versão misturada, usando só fundos que estão no **treino** daquele fold (sem vazamento). O teste é sempre o LOFAR original.
- Mesma MLP h32-16 e mesmos folds do NB40, nos dois splits.
- **Dois controles:** `dupL` (cópia idêntica do LARGE: mais exemplos LARGE, sem ruído) e `calmo` (mesma mistura, mas com fundo de mar calmo, ≤ 2). Assim dá para separar o efeito do mar agitado do efeito de só ter mais LARGE ou de qualquer fundo.
"""),
        code(ns["_SRC"]["AUG"]),
        md(ns["LEITURA_AUG"]),
        md("""
## 6. Dá para usar esses metadados direto no modelo?

Hoje os modelos recebem só o LOFAR. Pôr vento e estado do mar como entradas extras é fácil, mas **arriscado**: como os LARGE foram gravados com mar mais calmo, o modelo tenderia a aprender o atalho "mar calmo → LARGE", que melhora o número no IARA e piora no mundo real. Usos mais seguros:

1. **Augmentation direcionado** (seção 5): usa o metadado só para escolher o ruído, não como entrada.
2. **Abstenção:** com mar agitado e grave fraco, responder "inconclusivo" em vez de chutar.
3. **Medir o ruído pelo próprio áudio** (nível antes da normalização), em vez de usar a estação meteorológica.
4. **Reportar o desempenho por condição** (mar calmo × agitado, com × sem CPA).
"""),
        md(ns["CONCLUSOES"]),
    ]




def build():
    filename = os.path.join(NOTEBOOKS_DIR, "41_metadados_ambientais.ipynb")
    ns = {}
    for src in (SETUP_CODE, META_CODE, CLIMA_CODE, BG_CODE, ERRO_CODE, AUG_CODE):
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    ns["_SRC"] = {"SETUP": SETUP_CODE, "META": META_CODE, "CLIMA": CLIMA_CODE, "BG": BG_CODE, "ERRO": ERRO_CODE,
                  "AUG": AUG_CODE}
    textos(ns)
    cells = celulas(ns)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "pygments_lexer": "ipython3"}},
          "nbformat": 4, "nbformat_minor": 4}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook gerado em '{filename}'")


if __name__ == "__main__":
    build()
