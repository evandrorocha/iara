"""
Gera notebooks/42_demon_augmentation_mar.ipynb: o DEMON aproveita o augmentation dos LARGE com fundo de mar agitado?
Experimento separado dos de LOFAR (NB40/NB41), a pedido do usuário.

Fontes (nenhum número digitado à mão):
  - DEMON por janela: data/iara_processed/cache_fusion_mel_demon.pkl['demon'] (= demon_features, scratch/extract_all_demon.py)
  - cópias misturadas: data/iara_processed/cache_demon_mixbg_{agitado,calmo}.pkl (scripts/extrair_demon_mixbg_mar.py,
    mesmas misturas do LOFAR de scripts/extrair_lofar_mixbg_mar.py)
  - MLP DEMON 4 classes: results/trainings/multirep/demon_mar_{split_navio,split_id}_mlp_demon_4c[_aug<variante>]
    (scratch/demon_mar/aug_mar_demon.py)
  - MLP LOFAR 4 classes (referência): {split}_mlp_lofar_4c[_aug<variante>] (NB29 e scratch/mix_small_medium/aug_mar_4c.py)

Rodar na raiz:  python notebooks/src/generate_demon_mar_notebook.py
Executar no container:  docker exec -w /workspace/IARA iara-dev python scratch/execute_notebooks.py notebooks/42_demon_augmentation_mar.ipynb
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
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
COR = {'SMALL': '#2a78d6', 'MEDIUM': '#eb6834', 'LARGE': '#1baf7a', 'BACKGROUND': '#8a8a8a'}
meta = pd.read_csv(os.path.join(R.ROOT, 'src', 'iara', 'dataset_info', 'iara.csv'), dtype={'Ship ID': str, 'ID': str})
meta = meta[meta.Dataset.isin(list('ABCDE'))].copy()
meta.index = meta.ID.astype(int)
meta['Length'] = pd.to_numeric(meta.Length, errors='coerce')
meta['Target'] = np.select([meta.Length < 50, meta.Length < 100, meta.Length >= 100], [0, 1, 2], default=3)
meta.loc[meta.Dataset == 'E', 'Target'] = 3
meta['Mar'] = pd.to_numeric(meta['Sea state'], errors='coerce')
meta['CPA'] = meta.Dataset.isin(['A', 'C'])

with open(os.path.join(R.ROOT, 'data/iara_processed/cache_fusion_mel_demon.pkl'), 'rb') as fh:
    DEMON = pickle.load(fh)['demon']
FD = (np.arange(253) + 4) * 50 / 512   # frequências de modulação (Hz)
print(f'DEMON: {len(DEMON)} gravações; janela de 10,24 s a cada 5,12 s; {len(FD)} frequências de {FD[0]:.2f} a {FD[-1]:.1f} Hz')
'''

ESPEC_CODE = r'''
media = pd.DataFrame({f: w.mean(0) for f, w in DEMON.items()}).T
media['Target'] = meta.Target.reindex(media.index)
fig, ax = plt.subplots(figsize=(8, 3.2))
for c, n in CN.items():
    ax.plot(FD, media[media.Target == c].drop(columns='Target').median().values, color=COR[n], lw=1.8, label=n)
ax.set_xlabel('frequência de modulação (Hz)'); ax.set_ylabel('DEMON (amplitude, mediana)')
ax.set_title('DEMON médio por classe (mediana das gravações)'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

def contraste(w):
    s = w.mean(0)
    return 20 * np.log10(s.max() / np.median(s))
CONTR = pd.Series({f: contraste(w) for f, w in DEMON.items()}).to_frame('contraste_dB').join(meta.Target)
CONTR_CLASSE = CONTR.groupby(CONTR.Target.map(CN)).contraste_dB.median().reindex(list(CN.values()))
print('Contraste do DEMON médio da gravação (pico / mediana, dB), mediana por classe:'); display(CONTR_CLASSE.round(1).to_frame())
'''

REF_CODE = r'''
def metricas(y, p):
    rec = np.array([np.mean(p[y == c] == c) for c in range(4)])
    sp = np.sqrt(rec.mean() * np.exp(np.mean(np.log(rec)))) * 100 if np.all(rec > 0) else 0.0
    return {'SP': sp, 'ACC': rec.mean() * 100, **{f'R_{CN[c]}': r * 100 for c, r in enumerate(rec)}}

def por_fold(exp):
    P = R.file_predictions(exp)
    return pd.DataFrame({k: metricas(d.Target.values, d.Prediction.values) for k, d in P.items()}).T

REF = {}
for cond, nome in [('split_navio', 'por navio'), ('split_id', 'por gravação')]:
    REF[('LOFAR', nome)] = por_fold(f'{cond}_mlp_lofar_4c').mean()
    REF[('DEMON', nome)] = por_fold(f'demon_mar_{cond}_mlp_demon_4c').mean()
REF = pd.DataFrame(REF).T
print('MLP h32-16, 4 classes, mesmos folds, sem augmentation (média de 10 folds):'); display(REF.round(1))
'''

AUG_CODE = r'''
VAR = {'base': 'sem augmentation', 'dupL': 'LARGE duplicado (controle)', 'calmo': 'LARGE + fundo de mar calmo (controle)',
       'agitado': 'LARGE + fundo de mar agitado'}
def exp(cond, v, rep='demon'):
    b = f'demon_mar_{cond}_mlp_demon_4c' if rep == 'demon' else f'{cond}_mlp_lofar_4c'
    return b + ('' if v == 'base' else f'_aug{v}')
FMA = {(c, v): por_fold(exp(c, v)) for c in ['split_navio', 'split_id'] for v in VAR}
MEDIAS = pd.concat({(c, VAR[v]): f.mean() for (c, v), f in FMA.items()}, axis=1).T
print('DEMON — médias de 10 folds:'); display(MEDIAS.round(1))
rows = []
for cond in ['split_navio', 'split_id']:
    for v in ['dupL', 'calmo', 'agitado']:
        A, B = FMA[(cond, v)], FMA[(cond, 'base')]
        for m in ['SP', 'ACC', 'R_LARGE', 'R_SMALL', 'R_MEDIUM', 'R_BACKGROUND']:
            rows.append({'split': cond, 'variante': v, 'métrica': m, 'Δ vs base': (A[m] - B[m]).mean(),
                         'folds melhores': int((A[m] > B[m]).sum()), 'p (F 5x2cv)': R.f_test_5x2(A[m].values, B[m].values)[1]})
DELTA = pd.DataFrame(rows)
print('Diferenças contra "sem augmentation" (F 5x2cv):'); display(DELTA.round(3))

# LARGE por CPA × estado do mar, split por navio
def estratos(exp_name):
    d = pd.concat(R.file_predictions(exp_name).values(), ignore_index=True)
    d['File'] = d.File.astype(int); d = d[d.Target == 2].join(meta[['Mar', 'CPA']], on='File')
    d['cpa'] = np.where(d.CPA, 'com CPA (A, C)', 'sem CPA (B, D)'); d['mar'] = np.where(d.Mar <= 2, 'calmo (0–2)', 'agitado (3+)')
    return d.assign(ok=d.Prediction == 2).groupby(['cpa', 'mar']).ok.mean().mul(100)
EST = pd.concat({VAR[v]: estratos(exp('split_navio', v)) for v in VAR}, axis=1)
print('Acerto de LARGE (%) por CPA × estado do mar, split por navio:'); display(EST.round(0))

# comparação com o LOFAR (mesmas misturas, mesma MLP)
LOF = pd.DataFrame({VAR[v]: por_fold(exp('split_navio', v, 'lofar')).mean() for v in VAR}).T
COMP = pd.concat({'LOFAR': LOF[['ACC', 'R_LARGE']], 'DEMON': MEDIAS.loc['split_navio', ['ACC', 'R_LARGE']]}, axis=1)
print('LOFAR × DEMON, split por navio (ACC e recall de LARGE):'); display(COMP.round(1))

fig, ax = plt.subplots(figsize=(8, 3.2))
lab = [f'{a}\n{b}' for a, b in EST.index]; xx = np.arange(len(lab)); w = 0.2
cores = dict(zip(VAR.values(), ['#8a8a8a', '#b9b9b9', '#9fc3ea', '#1baf7a']))
for i, c in enumerate(EST.columns):
    ax.bar(xx + (i - 1.5) * w, EST[c].values, w * 0.92, color=cores[c], label=c)
ax.set_xticks(xx); ax.set_xticklabels(lab, fontsize=8); ax.set_ylabel('acerto de LARGE (%)'); ax.set_ylim(0, 100)
ax.legend(fontsize=7, ncol=2, loc='upper right'); ax.set_title('MLP DEMON, split por navio')
plt.tight_layout(); plt.show()
'''

CONTR_CODE = r'''
ref_mix = CONTR[CONTR.Target.isin([0, 1])].contraste_dB.median()
rows = []
for g in ['agitado', 'calmo']:
    with open(os.path.join(R.ROOT, f'data/iara_processed/cache_demon_mixbg_{g}.pkl'), 'rb') as fh:
        mix = pickle.load(fh)
    for r in mix['sorteios'].itertuples():
        rows.append({'fundo': g, 'ID': r.ID, 'rsf_db': r.rsf_db, 'c': contraste(mix['janelas'][(r.ID, r.copia)]),
                     'c_orig': CONTR.contraste_dB[r.ID]})
CP = pd.DataFrame(rows); CP['Δ'] = CP.c - CP.c_orig; CP['parece < 100 m'] = CP.c < ref_mix
CP['razão sinal/fundo'] = pd.cut(CP.rsf_db, [0, 5, 10, 15, 20], include_lowest=True,
                                 labels=['0–5 dB', '5–10 dB', '10–15 dB', '15–20 dB'])
CONTR_RSF = CP.groupby(['fundo', 'razão sinal/fundo'], observed=True).agg(
    cópias=('Δ', 'size'), contraste_original=('c_orig', 'median'), contraste_cópia=('c', 'median'),
    queda_mediana_dB=('Δ', 'median'), pct_parece_menor_100m=('parece < 100 m', 'mean'))
CONTR_RSF['pct_parece_menor_100m'] *= 100
L_ORIG_ABAIXO = (CONTR[CONTR.Target == 2].contraste_dB < ref_mix).mean() * 100
print(f'Mediana do contraste dos navios < 100 m: {ref_mix:.1f} dB; LARGE originais abaixo dela: {L_ORIG_ABAIXO:.0f}%')
display(CONTR_RSF.round(1))
CONTR_TOT = CP.groupby('fundo').agg(queda_mediana_dB=('Δ', 'median'), pct_parece_menor_100m=('parece < 100 m', 'mean'))
CONTR_TOT['pct_parece_menor_100m'] *= 100
display(CONTR_TOT.round(1))

fig, ax = plt.subplots(figsize=(7.5, 3.2))
for g, cor in [('agitado', '#1baf7a'), ('calmo', '#9fc3ea')]:
    t = CONTR_RSF.loc[g]
    ax.plot(range(len(t)), t.contraste_cópia.values, marker='o', color=cor, lw=2, label=f'cópias com fundo de mar {g}')
ax.axhline(CONTR_CLASSE['LARGE'], color=COR['LARGE'], ls=':', label='LARGE original (mediana)')
ax.axhline(ref_mix, color=COR['MEDIUM'], ls=':', label='navios < 100 m (mediana)')
ax.set_xticks(range(4)); ax.set_xticklabels(CONTR_RSF.loc['agitado'].index); ax.set_xlabel('razão sinal/fundo da mistura')
ax.set_ylabel('contraste do DEMON (dB)'); ax.legend(fontsize=8); ax.set_title('Quanto mais ruído, mais o LARGE aumentado parece um navio < 100 m')
plt.tight_layout(); plt.show()
'''


def textos(ns):
    f0, f1 = (lambda x: br(x, 0)), (lambda x: br(x, 1))
    REF, MD, DE, EST, CC = ns["REF"], ns["MEDIAS"], ns["DELTA"], ns["EST"], ns["CONTR_CLASSE"]
    V = ns["VAR"]
    nv = lambda v, m, s="split_navio": MD.loc[(s, V[v]), m]  # noqa: E731
    d = DE.set_index(["split", "variante", "métrica"])
    pmin = DE.loc[DE.variante.isin(["agitado", "calmo"]), "p (F 5x2cv)"].min()
    dacc = d.loc[("split_navio", "dupL", "ACC")]
    ntest = len(DE)
    CR, CT = ns["CONTR_RSF"], ns["CONTR_TOT"]
    COM, SEM, CAL, AGI = "com CPA (A, C)", "sem CPA (B, D)", "calmo (0–2)", "agitado (3+)"
    ns["RESUMO"] = f"""**Resumo das respostas:**
1. **O DEMON sozinho é bem mais fraco que o LOFAR** na mesma MLP: SP {f1(REF.loc[('DEMON', 'por navio'), 'SP'])} contra {f1(REF.loc[('LOFAR', 'por navio'), 'SP'])} no split por navio. Ele separa bem o BACKGROUND e razoavelmente o LARGE, mas SMALL fica perto do acaso.
2. **O augmentation com mar agitado também não ajuda no DEMON:** recall de LARGE {f1(nv('base', 'R_LARGE'))}% → {f1(nv('agitado', 'R_LARGE'))}% no split por navio; nenhuma diferença significativa (menor p = {br(pmin, 2)}).
3. **O mecanismo é o mesmo do LOFAR:** o ruído deixa as linhas de modulação mais rasas. Com muito ruído (0–5 dB), {f0(CR.loc[('agitado', '0–5 dB'), 'pct_parece_menor_100m'])}% das cópias de LARGE ficam com o contraste de um navio < 100 m.
"""
    ns["LEITURA_AUG"] = f"""**Leitura.**
- **Nenhuma variante melhora o DEMON.** Split por navio, SP: sem augmentation {f1(nv('base', 'SP'))}; LARGE duplicado {f1(nv('dupL', 'SP'))}; fundo calmo {f1(nv('calmo', 'SP'))}; **fundo agitado {f1(nv('agitado', 'SP'))}**. Menor p = {br(pmin, 2)}.
- **Repetir os LARGE não ajuda** (recall de LARGE {f1(nv('dupL', 'R_LARGE'))} contra {f1(nv('base', 'R_LARGE'))}). A única diferença com p < 0,05 é do controle: o LARGE duplicado **piora** a ACC em {f1(-dacc['Δ vs base'])} pontos no split por navio ({pv(dacc['p (F 5x2cv)'])}). Como foram feitas {ntest} comparações, um p desse tamanho é esperado por acaso; não muda a conclusão. **Misturar fundo tira recall de LARGE** (agitado {f1(d.loc[('split_navio', 'agitado', 'R_LARGE'), 'Δ vs base'])}, calmo {f1(d.loc[('split_navio', 'calmo', 'R_LARGE'), 'Δ vs base'])} pontos), e os erros vão para os navios menores — o mesmo padrão do LOFAR.
- **Os LARGE gravados com mar agitado não melhoraram:** com CPA, {f0(EST.loc[(COM, AGI), V['base']])}% → {f0(EST.loc[(COM, AGI), V['agitado']])}%; sem CPA, {f0(EST.loc[(SEM, AGI), V['base']])}% → {f0(EST.loc[(SEM, AGI), V['agitado']])}%.
- **A hipótese não se confirmou:** esperava-se que o DEMON aproveitasse melhor o augmentation porque o ruído não desloca as linhas de modulação. Na prática, o ruído deixa as linhas mais rasas, e boa parte das cópias de LARGE fica com o contraste típico de um navio menor (seção 5).
"""
    ns["LEITURA_CONTR"] = f"""**Leitura.**
- **Contraste típico (pico / mediana do DEMON):** LARGE {f1(CC['LARGE'])} dB, MEDIUM {f1(CC['MEDIUM'])} dB, SMALL {f1(CC['SMALL'])} dB, BACKGROUND {f1(CC['BACKGROUND'])} dB. Os LARGE têm as linhas de modulação mais marcadas.
- **O ruído reduz o contraste:** com mar agitado e razão 0–5 dB, a queda mediana é de {f1(-CR.loc[('agitado', '0–5 dB'), 'queda_mediana_dB'])} dB; com 15–20 dB, {f1(-CR.loc[('agitado', '15–20 dB'), 'queda_mediana_dB'])} dB.
- **Muitas cópias passam a parecer navio < 100 m:** {f0(CT.loc['agitado', 'pct_parece_menor_100m'])}% das cópias com mar agitado (e {f0(CR.loc[('agitado', '0–5 dB'), 'pct_parece_menor_100m'])}% com 0–5 dB), contra {f0(ns['L_ORIG_ABAIXO'])}% dos LARGE originais.
- **Mar calmo faz quase o mesmo** ({f0(CT.loc['calmo', 'pct_parece_menor_100m'])}% das cópias), provavelmente porque a banda do DEMON (1–2 kHz) recebe bastante ruído mesmo com mar calmo.
"""
    ns["CONCLUSOES"] = f"""## 6. Conclusões

1. **O DEMON, sozinho, é uma representação mais fraca que o LOFAR** para as 4 classes do IARA (SP {f1(REF.loc[('DEMON', 'por navio'), 'SP'])} × {f1(REF.loc[('LOFAR', 'por navio'), 'SP'])} no split por navio). Seu papel natural é complementar o LOFAR ou o MEL, não substituí-los.
2. **O augmentation com fundo de mar agitado não ajuda no DEMON**, assim como não ajudou no LOFAR. Os controles mostram o mesmo padrão: repetir os LARGE não muda nada; misturar qualquer fundo tira um pouco de recall de LARGE.
3. **O mecanismo é comum às duas representações:** somar ruído real ao áudio apaga parte da característica que identifica o LARGE (o grave no LOFAR, as linhas de modulação no DEMON), e a cópia aumentada passa a parecer um navio menor. O problema está no ruído somado ao áudio, não na representação.
4. **Consequência prática:** augmentation por mistura com fundo real, do jeito que foi feito (razão 0–20 dB), não é um caminho promissor no IARA. Se for retomado, limitar a razão sinal/fundo (por exemplo, ≥ 10 dB) para que a cópia continue sendo um LARGE reconhecível.
"""


def celulas(ns):
    return [
        md(f"""
# 42 — DEMON e augmentation com fundo de mar agitado

**Pergunta.** No LOFAR (NB41), misturar ruído real de mar agitado nos LARGE de treino não ajudou: o ruído apagava o grave que identifica o LARGE. O **DEMON** mede outra coisa — os ritmos de modulação da cavitação (rotação do eixo, passagem das pás) na banda de 1–2 kHz, justamente onde o ruído de vento e ondas mais cresce. Será que o DEMON aproveita esse augmentation?

Notebook separado dos experimentos de LOFAR para não misturar os resultados.

{ns['RESUMO']}
"""),
        md("""
## Glossário rápido

- **DEMON (*Demodulation of Envelope Modulation On Noise*):** filtra o áudio numa banda (aqui 1–2 kHz), tira o envelope e calcula o espectro desse envelope. Picos em poucos Hz indicam rotação do eixo e passagem das pás do hélice.
- **Janela do DEMON no IARA:** 10,24 s, a cada 5,12 s; 253 frequências de modulação de 0,4 a 25 Hz.
- **Contraste do DEMON:** quanto o pico do espectro DEMON sobressai da sua mediana (dB). Linhas de modulação marcadas = contraste alto.
- **Augmentation:** versão modificada de uma gravação de treino; aqui, o áudio do LARGE somado a uma gravação real de mar sem navio.
- **Split por navio:** nenhum navio aparece no treino e no teste ao mesmo tempo.
- **Valor-p (teste F 5x2cv):** abaixo de 0,05, a diferença entre dois métodos provavelmente é real.
"""),
        code(ns["_SRC"]["SETUP"]),
        md("""
## 1. Como é o DEMON de cada classe
"""),
        code(ns["_SRC"]["ESPEC"]),
        md(f"""
**Leitura.** Os LARGE têm o DEMON mais "marcado" (contraste {br(ns['CONTR_CLASSE']['LARGE'], 1)} dB), os navios menores um pouco menos, e o BACKGROUND quase nada ({br(ns['CONTR_CLASSE']['BACKGROUND'], 1)} dB): sem navio, o envelope do ruído não tem ritmo.
"""),
        md("""
## 2. Referência: DEMON × LOFAR na mesma MLP

Mesma MLP h32-16 do NB29, mesmos folds, 4 classes, sem augmentation. A previsão de cada gravação é o voto majoritário das janelas.
"""),
        code(ns["_SRC"]["REF"]),
        md(f"""
**Leitura.** O DEMON sozinho fica bem abaixo do LOFAR (SP {br(ns['REF'].loc[('DEMON', 'por navio'), 'SP'], 1)} × {br(ns['REF'].loc[('LOFAR', 'por navio'), 'SP'], 1)} no split por navio). SMALL fica perto do acaso ({br(ns['REF'].loc[('DEMON', 'por navio'), 'R_SMALL'], 0)}%). Hipótese (não testada): lanchas e rebocadores pequenos têm modulação menos marcada na banda de 1–2 kHz.
"""),
        md("""
## 3. Como o augmentation foi feito

- **As mesmas misturas do LOFAR** (`scripts/extrair_demon_mixbg_mar.py`): para cada cópia de LARGE, o mesmo fundo, o mesmo deslocamento e a mesma razão sinal/fundo (0 a 20 dB) usados no NB41; o DEMON é recalculado com os parâmetros do pipeline. Sem mistura, o script reproduz o DEMON original (diferença máxima ~10⁻⁹). O DEMON original não é alterado: pastas novas.
- **No treino** de cada fold, cada LARGE ganha uma versão extra, só com fundos do treino daquele fold. **Teste** sempre no DEMON original.
- **Variantes:** fundo de mar agitado (estado ≥ 4, sem chuva) e dois controles — fundo de mar calmo (≤ 2) e LARGE duplicado (só repetição).
"""),
        md("""
## 4. Resultado
"""),
        code(ns["_SRC"]["AUG"]),
        md(ns["LEITURA_AUG"]),
        md("""
## 5. O ruído apaga as linhas de modulação?
"""),
        code(ns["_SRC"]["CONTR"]),
        md(ns["LEITURA_CONTR"]),
        md(ns["CONCLUSOES"]),
    ]


def build():
    filename = os.path.join(NOTEBOOKS_DIR, "42_demon_augmentation_mar.ipynb")
    ns = {}
    srcs = {"SETUP": SETUP_CODE, "ESPEC": ESPEC_CODE, "REF": REF_CODE, "AUG": AUG_CODE, "CONTR": CONTR_CODE}
    for src in srcs.values():
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    ns["_SRC"] = srcs
    textos(ns)
    nb = {"cells": celulas(ns),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "pygments_lexer": "ipython3"}},
          "nbformat": 4, "nbformat_minor": 4}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook gerado em '{filename}'")
    for k in ["REF", "MEDIAS", "EST", "CONTR_RSF", "CONTR_TOT", "COMP"]:
        print(k); print(ns[k].round(2).to_string())
    print(ns["DELTA"].round(3).to_string())


if __name__ == "__main__":
    build()
