"""
Gera notebooks/43_small_subconjunto_d.ipynb: por que, no subconjunto D (sem CPA, 250–500 m), o MEL não separa SMALL de
LARGE, mas separa MEDIUM de LARGE?

Três passos:
  1. quem mudou no D: o SMALL ou o LARGE? (cada classe no D contra a mesma classe em A+C, por navio)
  2. o mesmo navio soa diferente no D? (navios gravados no D e em A/C, teste pareado)
  3. o D tem dois tipos de SMALL misturados? (subgrupos de SMALL no D contra LARGE do D; navios mais "LARGE")

Fontes (nenhum número digitado à mão): metadados (scratch/eval_ship_split_5x2cv.metadata) e os espectros médios por
gravação já em cache: cache_lofar_espectro_medio.pkl (LOFAR do pipeline), cache_mel_espectro_medio.pkl (MEL L2, NB37)
e cache_psd24k_resumo.pkl (psd24k).

Rodar na raiz:  python notebooks/src/generate_subconjunto_d_notebook.py
Executar no container (nbclient), em -w /workspace/IARA/notebooks.
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
from scipy.stats import mannwhitneyu, wilcoxon, false_discovery_control
warnings.filterwarnings('ignore')
try:
    from IPython.display import display
except ImportError:
    display = print
ROOT = next(os.path.abspath(p) for p in ('.', '..') if os.path.exists(os.path.join(p, 'scratch', 'eval_ship_split_5x2cv.py')))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, 'scratch'))
from eval_ship_split_5x2cv import metadata
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'figure.dpi': 110})
pd.set_option('display.precision', 3); pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
P = 'data/iara_processed/'
COR = {'SMALL': '#2a78d6', 'MEDIUM': '#eb6834', 'LARGE': '#1baf7a'}

m = metadata(); m.index = m.index.map(int)
m = m[m.Target < 3].copy()                                    # só gravações de navio (A–D)
m['classe'] = m.Target.map({0: 'SMALL', 1: 'MEDIUM', 2: 'LARGE'}); m['ship'] = m['Ship ID'].astype(str)
m['Length'] = pd.to_numeric(m['Length'], errors='coerce')
m['grupo'] = np.where(m.Dataset == 'D', 'D', np.where(m.Dataset == 'B', 'B', 'A+C'))

# espectro médio de cada gravação, em três representações
lofar = pd.read_pickle(P + 'cache_lofar_espectro_medio.pkl'); lofar.index = lofar.index.map(int)
FL = np.arange(1024) * 16000 / 2048                            # 1.024 faixas de 7,8 Hz (dB acima do fundo TPSW)
mel = pd.read_pickle(P + 'cache_mel_espectro_medio.pkl')['L2 (modelos)']
import librosa
FM = librosa.mel_frequencies(n_mels=258, fmin=0, fmax=8000)[1:-1]   # centro das 256 bandas mel
_ps = pd.read_pickle(P + 'cache_psd24k_resumo.pkl'); FP = _ps['freqs']
psd = _ps['psd_db'] if isinstance(_ps['psd_db'], pd.DataFrame) else pd.DataFrame(_ps['psd_db']); psd.index = psd.index.map(int)
_k = (FP >= 20) & (FP <= 8000); FP8 = FP[_k]
PF = psd.loc[:, _k]; PF = PF.sub(PF.mean(1), axis=0)           # forma do psd24k: dB menos a média da gravação (sem ganho)

def banda(X, F, lo, hi):
    return X.loc[:, (F >= lo) & (F < hi)].mean(1)

# três medidas-resumo de "grave forte" (todas independentes do ganho)
MED = pd.DataFrame({
    'LOFAR 31–86 Hz': banda(lofar, FL, 31, 86),
    'MEL grave − agudo': banda(mel, FM, 0, 1000) - banda(mel, FM, 1000, 6000),
    'psd24k grave relativo': banda(PF, FP8, 31, 125) - banda(PF, FP8, 1000, 4000)})
UNID = {'LOFAR 31–86 Hz': 'dB acima do fundo', 'MEL grave − agudo': 'L2 média < 1 kHz menos 1–6 kHz',
        'psd24k grave relativo': 'dB, 31–125 Hz menos 1–4 kHz'}
REPS = {'LOFAR (20 Hz–6 kHz)': lofar.loc[:, (FL >= 20) & (FL <= 6000)], 'MEL L2': mel, 'psd24k forma (20 Hz–8 kHz)': PF}

def nav(X, ids):
    """média por navio: cada navio pesa 1"""
    ids = [i for i in ids if i in X.index]
    return X.loc[ids].groupby(m.loc[ids, 'ship'].values).mean()

def sel(c, ds):
    return m.index[(m.classe == c) & m.Dataset.isin(list(ds))]

def faixas(A, B):
    """Mann-Whitney faixa a faixa entre navios + FDR: % de faixas com q < 0,05 e AUC por faixa"""
    U, p = mannwhitneyu(A.values, B.values, axis=0)
    return 100 * (false_discovery_control(p) < .05).mean(), U / (len(A) * len(B))

print('Navios por classe e subconjunto:')
display(m.groupby(['classe', 'Dataset']).ship.nunique().unstack()[list('ABCD')])
'''

EFEITO_CODE = r'''
# O efeito existe nas três representações? % de faixas com diferença (q < 0,05) por subconjunto
rows = []
for nome, X in REPS.items():
    for ds in 'ABCD':
        for a, b in (('SMALL', 'LARGE'), ('MEDIUM', 'LARGE'), ('SMALL', 'MEDIUM')):
            A, B = nav(X, sel(a, ds)), nav(X, sel(b, ds))
            pct, auc = faixas(A, B)
            rows.append({'representação': nome, 'comparação': f'{a} × {b}', 'subconjunto': ds, '% faixas': pct})
EFEITO = pd.DataFrame(rows).pivot_table(index=['representação', 'comparação'], columns='subconjunto', values='% faixas').round(0)
print('% das faixas com q < 0,05 (navio = unidade):'); display(EFEITO)
'''

PASSO1_CODE = r'''
# Passo 1: cada classe no D (e no B) contra a mesma classe em A+C
rows = []
for c in ('SMALL', 'MEDIUM', 'LARGE'):
    for med in MED:
        x, y, b = (nav(MED[[med]], sel(c, g))[med] for g in ('D', 'AC', 'B'))
        U, p = mannwhitneyu(x, y)
        rows.append({'classe': c, 'medida': med, 'mediana A+C': y.median(), 'mediana B': b.median(), 'mediana D': x.median(),
                     'AUC (D > A+C)': U / (len(x) * len(y)), 'p': p})
PASSO1 = pd.DataFrame(rows).set_index(['medida', 'classe'])
display(PASSO1.round(3))

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for j, med in enumerate(MED):
    pos = 0; ticks, labs = [], []
    for c in ('SMALL', 'MEDIUM', 'LARGE'):
        for g in ('A+C', 'B', 'D'):
            v = nav(MED[[med]], sel(c, g.replace('+', '')))[med]
            bp = ax[j].boxplot(v, positions=[pos], widths=.7, showfliers=False, patch_artist=True)
            bp['boxes'][0].set(facecolor=COR[c], alpha=.35 if g != 'D' else .8); bp['medians'][0].set(color='k')
            ticks.append(pos); labs.append(f'{g}\n{c}' if g == 'B' else f'{g}\n'); pos += 1
        pos += .7
    ax[j].set_xticks(ticks); ax[j].set_xticklabels(labs, fontsize=8); ax[j].set_title(f'{med}\n({UNID[med]})', fontsize=9)
plt.suptitle('Uma caixa por classe e subconjunto (navio = unidade); D em cor cheia', y=1.04, fontsize=10)
plt.tight_layout(); plt.show()
'''

PASSO2_CODE = r'''
# Passo 2: o mesmo navio, gravado no D e em A/C (teste de Wilcoxon pareado)
rows, PARES = [], {}
for c in ('SMALL', 'MEDIUM', 'LARGE'):
    d, ac = nav(MED, sel(c, 'D')), nav(MED, sel(c, 'AC'))
    com = d.index.intersection(ac.index); PARES[c] = (ac.loc[com], d.loc[com])
    for med in MED:
        dif = d.loc[com, med] - ac.loc[com, med]
        rows.append({'classe': c, 'medida': med, 'navios no D e em A/C': len(com), 'Δ mediano (D − A/C)': dif.median(),
                     '% navios com D maior': 100 * (dif > 0).mean(), 'p (Wilcoxon)': wilcoxon(dif).pvalue})
PASSO2 = pd.DataFrame(rows).set_index(['medida', 'classe'])
display(PASSO2.round(3))

med = 'LOFAR 31–86 Hz'
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)
for j, c in enumerate(('SMALL', 'MEDIUM', 'LARGE')):
    a, d = PARES[c]
    ax[j].scatter(a[med], d[med], color=COR[c], s=22)
    lo, hi = MED[med].quantile([.01, .99]); ax[j].plot([lo, hi], [lo, hi], color='gray', lw=.8)
    ax[j].set_title(f'{c}: {len(a)} navios', fontsize=10); ax[j].set_xlabel(f'{med} em A/C (dB)')
ax[0].set_ylabel(f'{med} no D (dB)')
plt.suptitle('Mesmo navio: acima da diagonal = mais grave no D', y=1.03, fontsize=10)
plt.tight_layout(); plt.show()
'''

PASSO3_CODE = r'''
# Passo 3a: SMALL do D por subgrupo contra LARGE do D (MEL L2, faixa a faixa)
LD, LAC = nav(mel, sel('LARGE', 'D')), nav(mel, sel('LARGE', 'AC'))
sD = m[(m.classe == 'SMALL') & (m.Dataset == 'D')]; sAC = m[(m.classe == 'SMALL') & m.Dataset.isin(['A', 'C'])]
em_ac = set(sAC.ship)
def subgrupos(s):
    t = s['DETAILED TYPE']
    return {'todas as SMALL': s.index, 'lanchas de prático': s.index[t == 'Pilot Vessel'], 'rebocadores': s.index[t == 'Tug'],
            'pesca': s.index[t == 'Fishing Vessel']}
rows = []
for onde, s, L in (('D', sD, LD), ('A+C', sAC, LAC)):
    g = subgrupos(s)
    if onde == 'D':
        g.update({'navios também gravados em A/C': s.index[s.ship.isin(em_ac)], 'navios só no D': s.index[~s.ship.isin(em_ac)]})
    g['MEDIUM (referência)'] = sel('MEDIUM', onde.replace('+', ''))
    for nome, ix in g.items():
        A = nav(mel, ix); pct, auc = faixas(A, L)
        rows.append({'subconjunto': onde, 'grupo contra LARGE': nome, 'navios': len(A), '% faixas q < 0,05': pct,
                     'AUC mediana < 1 kHz': np.median(auc[FM < 1000]), 'AUC mediana > 1 kHz': np.median(auc[FM > 1000])})
SUB = pd.DataFrame(rows).set_index(['subconjunto', 'grupo contra LARGE'])
display(SUB.round(2))

# Passo 3b: quantos navios SMALL ficam acima da mediana do LARGE do mesmo subconjunto?
rows = []
for med in MED:
    for g in ('A+C', 'D'):
        s, L = nav(MED[[med]], sel('SMALL', g.replace('+', '')))[med], nav(MED[[med]], sel('LARGE', g.replace('+', '')))[med]
        rows.append({'medida': med, 'subconjunto': g, 'mediana SMALL': s.median(), 'mediana LARGE': L.median(),
                     '% SMALL acima da mediana LARGE': 100 * (s > L.median()).mean()})
ACIMA = pd.DataFrame(rows).set_index(['medida', 'subconjunto'])
display(ACIMA.round(2))

# Passo 3c: os navios SMALL do D mais parecidos com LARGE (maior grave no LOFAR)
info = m.groupby('ship').agg(tipo=('DETAILED TYPE', 'first'), comprimento=('Length', 'first'))
nD = sD.groupby('ship').size().rename('gravações no D')
TOP = nav(MED, sD.index).join(info).join(nD).assign(**{'também em A/C': lambda t: t.index.isin(em_ac)})
TOP = TOP.sort_values('LOFAR 31–86 Hz', ascending=False)
print(f"Navios SMALL do D com mais grave (LOFAR 31–86 Hz); mediana do LARGE do D = {nav(MED, sel('LARGE', 'D'))['LOFAR 31–86 Hz'].median():.2f} dB")
display(TOP.head(10).round(2))
print('Mediana por tipo, SMALL do D:'); display(TOP.groupby('tipo')[list(MED)].median().join(TOP.tipo.value_counts().rename('navios')).sort_values('navios', ascending=False).head(4).round(2))
'''

LARGE_CODE = r'''
# O LARGE do D: são outros navios? Comprimento e grave dos navios LARGE por subconjunto
L = m[m.classe == 'LARGE']; ac_ships, d_ships = set(L[L.grupo == 'A+C'].ship), set(L[L.grupo == 'D'].ship)
rows = []
for nome, ix in (('A+C: navios só em A+C', L.index[(L.grupo == 'A+C') & ~L.ship.isin(d_ships)]),
                 ('A+C: navios também no D', L.index[(L.grupo == 'A+C') & L.ship.isin(d_ships)]),
                 ('D: navios também em A+C', L.index[(L.grupo == 'D') & L.ship.isin(ac_ships)]),
                 ('D: navios só no D', L.index[(L.grupo == 'D') & ~L.ship.isin(ac_ships)])):
    v = nav(MED, ix)
    rows.append({'grupo de LARGE': nome, 'navios': len(v), 'comprimento mediano (m)': L.loc[ix].drop_duplicates('ship').Length.median(),
                 **{med: v[med].median() for med in MED}})
LARGE_D = pd.DataFrame(rows).set_index('grupo de LARGE')
display(LARGE_D.round(2))
from scipy.stats import spearmanr
_v = nav(MED, L.index).join(L.drop_duplicates('ship').set_index('ship').Length)
RHO_L = spearmanr(_v.Length, _v['LOFAR 31–86 Hz'], nan_policy='omit')
print(f'Entre os LARGE, comprimento × LOFAR 31–86 Hz: ρ = {RHO_L.statistic:.2f} (p = {RHO_L.pvalue:.3f})')
_t = L.drop_duplicates(['ship', 'grupo']); TIPOS_L = pd.crosstab(_t['DETAILED TYPE'], _t.grupo)[['A+C', 'D']]
TIPOS_L = (100 * TIPOS_L / TIPOS_L.sum()).round(0).sort_values('A+C', ascending=False).head(7)
print('% dos navios LARGE por tipo:'); display(TIPOS_L)
'''


def textos(ns):
    f0, f2 = (lambda x: br(x, 0)), (lambda x: br(x, 2))
    E, P1, P2, S, A, LD = ns["EFEITO"], ns["PASSO1"], ns["PASSO2"], ns["SUB"], ns["ACIMA"], ns["LARGE_D"]
    MEL, LOF, PSD = "MEL L2", "LOFAR (20 Hz–6 kHz)", "psd24k forma (20 Hz–8 kHz)"
    SL, ML, SM = "SMALL × LARGE", "MEDIUM × LARGE", "SMALL × MEDIUM"
    e = lambda r, c, d: f0(E.loc[(r, c), d])
    p1 = lambda med, c, k: P1.loc[(med, c), k]
    p2 = lambda med, c, k: P2.loc[(med, c), k]
    s = lambda o, g, k: S.loc[(o, g), k]
    L0 = "LOFAR 31–86 Hz"
    ns["T_EFEITO"] = f"""**Leitura.**
- **O achado se confirma no MEL:** no D, {SL} tem {e(MEL, SL, 'D')}% das faixas com diferença, contra {e(MEL, SL, 'A')}% em A e {e(MEL, SL, 'C')}% em C; {ML} continua com {e(MEL, ML, 'D')}%.
- **Mas o D enfraquece tudo, não só o SMALL:** no psd24k, {ML} também cai a {e(PSD, ML, 'D')}% no D (contra {e(PSD, ML, 'A')}% em A); no LOFAR, a {e(LOF, ML, 'D')}% (contra {e(LOF, ML, 'A')}%). O MEL é só a representação em que o MEDIUM ainda passa do limiar.
- **No B nada separa nada** ({e(MEL, SL, 'B')}% e {e(MEL, ML, 'B')}%), em parte porque o B tem poucos navios (ver a primeira tabela).
- {SM}: {e(MEL, SM, 'D')}% no D, como em todos os subconjuntos.
"""
    ns["T_PASSO1"] = f"""**Leitura.**
- **Quem mudou foi o LARGE.** O LARGE do D tem menos grave que o de A+C nas três medidas (LOFAR 31–86 Hz: {f2(p1(L0, 'LARGE', 'mediana A+C'))} → {f2(p1(L0, 'LARGE', 'mediana D'))} dB, AUC {f2(p1(L0, 'LARGE', 'AUC (D > A+C)'))}, {pv(p1(L0, 'LARGE', 'p'))}); no B, ainda menos ({f2(p1(L0, 'LARGE', 'mediana B'))} dB).
- **O SMALL não mudou** (LOFAR {f2(p1(L0, 'SMALL', 'mediana A+C'))} → {f2(p1(L0, 'SMALL', 'mediana D'))} dB, {pv(p1(L0, 'SMALL', 'p'))}), **nem o MEDIUM** ({f2(p1(L0, 'MEDIUM', 'mediana A+C'))} → {f2(p1(L0, 'MEDIUM', 'mediana D'))} dB, {pv(p1(L0, 'MEDIUM', 'p'))}).
- Com o LARGE descendo e o SMALL parado, a distância entre os dois encolhe. O MEDIUM do D fica um pouco mais baixo que o SMALL, e por isso ainda passa do limiar no MEL.
"""
    ns["T_PASSO2"] = f"""**Leitura.**
- **O mesmo navio não muda entre A/C e D**, em nenhuma classe e em nenhuma medida (menor p = {br(P2['p (Wilcoxon)'].min(), 2)}).
- LARGE: {f0(p2(L0, 'LARGE', 'navios no D e em A/C'))} navios, Δ mediano {f2(p2(L0, 'LARGE', 'Δ mediano (D − A/C)'))} dB ({pv(p2(L0, 'LARGE', 'p (Wilcoxon)'))}). SMALL: {f0(p2(L0, 'SMALL', 'navios no D e em A/C'))} navios, Δ {f2(p2(L0, 'SMALL', 'Δ mediano (D − A/C)'))} dB ({pv(p2(L0, 'SMALL', 'p (Wilcoxon)'))}).
- Os pontos ficam longe da diagonal nos dois sentidos: **o mesmo navio varia muito de uma gravação para outra**, então o teste só detectaria um efeito grande.
- Então a queda do grave do LARGE no D **não parece efeito da distância nem da falta do CPA** sobre o mesmo navio: é que os LARGE gravados no D são, em boa parte, **outros navios** (próxima seção).
"""
    ns["T_LARGE"] = f"""**Leitura.**
- **{f0(LD.loc['D: navios só no D', 'navios'])} dos {f0(LD.loc['D: navios só no D', 'navios'] + LD.loc['D: navios também em A+C', 'navios'])} navios LARGE do D só aparecem no D**, e são menores: comprimento mediano de {f0(LD.loc['D: navios só no D', 'comprimento mediano (m)'])} m, contra {f0(LD.loc['A+C: navios só em A+C', 'comprimento mediano (m)'])} m dos LARGE que só aparecem em A+C.
- Os LARGE que só aparecem em A+C têm o grave mais forte ({f2(LD.loc['A+C: navios só em A+C', L0])} dB no LOFAR); os LARGE do D, só no D, {f2(LD.loc['D: navios só no D', L0])} dB.
- Os {f0(LD.loc['D: navios também em A+C', 'navios'])} navios gravados nos dois têm grave intermediário e igual nos dois lugares ({f2(LD.loc['A+C: navios também no D', L0])} e {f2(LD.loc['D: navios também em A+C', L0])} dB): coerente com o passo 2.
- **Não é só o comprimento:** entre todos os LARGE, comprimento e grave quase não se relacionam (ρ = {f2(ns['RHO_L'].statistic)}, {pv(ns['RHO_L'].pvalue)}). O comprimento só mostra que são navios diferentes; a causa pode ser tipo, carga, velocidade ou propulsão.
- Nos tipos, o D tem relativamente mais carga geral e menos graneleiros.
"""
    ns["T_PASSO3"] = f"""**Leitura.**
- **Nenhum subgrupo de SMALL do D se separa do LARGE do D** no MEL: lanchas {br(s('D', 'lanchas de prático', '% faixas q < 0,05'), 1)}%, rebocadores {f0(s('D', 'rebocadores', '% faixas q < 0,05'))}%, pesca {f0(s('D', 'pesca', '% faixas q < 0,05'))}%, navios também gravados em A/C {f0(s('D', 'navios também gravados em A/C', '% faixas q < 0,05'))}%, só no D {f0(s('D', 'navios só no D', '% faixas q < 0,05'))}%. Não é um grupo "estranho" de SMALL que apaga a diferença.
- Em A+C, ao contrário, rebocadores ({f0(s('A+C', 'rebocadores', '% faixas q < 0,05'))}%) e lanchas ({f0(s('A+C', 'lanchas de prático', '% faixas q < 0,05'))}%) se separam do LARGE; a **pesca não se separa nem em A+C** ({f0(s('A+C', 'pesca', '% faixas q < 0,05'))}%): os pesqueiros têm grave parecido com o do LARGE em todos os subconjuntos.
- **A diferença entre SMALL e MEDIUM no D é pequena:** AUC mediana abaixo de 1 kHz {f2(s('D', 'todas as SMALL', 'AUC mediana < 1 kHz'))} (SMALL × LARGE) contra {f2(s('D', 'MEDIUM (referência)', 'AUC mediana < 1 kHz'))} (MEDIUM × LARGE). É uma questão de limiar, não um comportamento oposto.
- **Proporção de SMALL acima da mediana do LARGE** (LOFAR 31–86 Hz): {f0(A.loc[(L0, 'A+C'), '% SMALL acima da mediana LARGE'])}% em A+C, {f0(A.loc[(L0, 'D'), '% SMALL acima da mediana LARGE'])}% no D, porque a mediana do LARGE desceu ({f2(A.loc[(L0, 'A+C'), 'mediana LARGE'])} → {f2(A.loc[(L0, 'D'), 'mediana LARGE'])} dB) e a do SMALL não.
- Os SMALL do D com mais grave são sobretudo **pesqueiros e rebocadores com poucas gravações**; as lanchas de prático quase não aparecem no topo.
"""
    ns["CONCLUSOES"] = f"""## 5. Conclusões

1. **O SMALL do D é igual ao SMALL de A e C.** Ele não ganha grave longe do hidrofone, nem como classe (passo 1) nem dentro do mesmo navio (passo 2).
2. **Quem muda é o LARGE:** o LARGE do D tem menos grave porque é formado, em boa parte, por **outros navios** (também menores: {f0(LD.loc['D: navios só no D', 'comprimento mediano (m)'])} m contra {f0(LD.loc['A+C: navios só em A+C', 'comprimento mediano (m)'])} m), e não porque o mesmo navio soa diferente a 250–500 m sem CPA.
3. **O "D não separa SMALL de LARGE, mas separa MEDIUM" é um efeito de limiar:** no D, todas as separações ficam mais fracas (no psd24k e no LOFAR, nem MEDIUM × LARGE passa). No MEL, o MEDIUM do D fica um pouco mais longe do LARGE e passa; o SMALL fica um pouco mais perto e não passa.
4. **Nenhum subgrupo explica sozinho** (lanchas, rebocadores, pesca, navios só no D). Os pesqueiros não se separam do LARGE em lugar nenhum, o que reforça que "grave forte" não é exclusividade de navio grande.
5. **Consequência para o modelo:** a pista do LARGE (grave em 31–86 Hz) depende de **quais** LARGE foram gravados: os LARGE do D se parecem mais com os navios < 100 m. Isso é composição da base, não geometria da gravação, e não cria uma assinatura nova de SMALL.
6. **Limite:** o teste pareado do passo 2 tem só {f0(p2(L0, 'LARGE', 'navios no D e em A/C'))} navios LARGE e {f0(p2(L0, 'SMALL', 'navios no D e em A/C'))} SMALL; um efeito pequeno de distância pode existir sem ser detectado.
"""


def celulas(ns):
    return [
        md("""
# 43 — SMALL no subconjunto D: por que o MEL não separa SMALL de LARGE?

**Pergunta.** Na análise de MEL e HHT da classe SMALL (apresentação `20261116-classe-small`), apareceu algo estranho: no **subconjunto D** (sem CPA, navio a 250–500 m), o MEL **não separa SMALL de LARGE em nenhuma faixa**, mas separa MEDIUM de LARGE. Em A e C, os dois se separam do LARGE.

**Três passos para entender:**
1. **Quem mudou no D: o SMALL ou o LARGE?** Cada classe no D contra a mesma classe em A+C.
2. **O mesmo navio soa diferente no D?** Navios gravados no D e em A/C, comparados com eles mesmos.
3. **O D tem algum grupo de SMALL que apaga a diferença?** Subgrupos (tipo, navios só no D) contra o LARGE do D.

**Unidade de análise:** o navio (média das suas gravações). **Medidas de "grave forte"** (todas independentes do ganho do gravador):
- **LOFAR 31–86 Hz:** dB acima do fundo TPSW na banda que mais identifica LARGE (NB35);
- **MEL grave − agudo:** média do MEL L2 abaixo de 1 kHz menos a média de 1 a 6 kHz;
- **psd24k grave relativo:** dB em 31–125 Hz menos dB em 1–4 kHz.
"""),
        code(SETUP_CODE),
        md("## 0. O efeito existe nas três representações?"),
        code(EFEITO_CODE),
        md(ns["T_EFEITO"]),
        md("## 1. Quem mudou no D: o SMALL ou o LARGE?"),
        code(PASSO1_CODE),
        md(ns["T_PASSO1"]),
        md("## 2. O mesmo navio soa diferente no D?"),
        code(PASSO2_CODE),
        md(ns["T_PASSO2"]),
        md("### 2.1 Os LARGE do D são outros navios?"),
        code(LARGE_CODE),
        md(ns["T_LARGE"]),
        md("""
## 3. O D tem algum grupo de SMALL que apaga a diferença?

Cada subgrupo de SMALL contra o LARGE **do mesmo subconjunto**, faixa a faixa no MEL L2. AUC abaixo de 0,5 = o LARGE tem mais.
"""),
        code(PASSO3_CODE),
        md(ns["T_PASSO3"]),
        md(ns["CONCLUSOES"]),
    ]


def build():
    filename = os.path.join(NOTEBOOKS_DIR, "43_small_subconjunto_d.ipynb")
    ns = {}
    for src in (SETUP_CODE, EFEITO_CODE, PASSO1_CODE, PASSO2_CODE, LARGE_CODE, PASSO3_CODE):
        ns["display"] = lambda *a, **k: None
        run(ns, src.replace("plt.show()", "plt.close('all')"))
    textos(ns)
    nb = {"cells": celulas(ns),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "pygments_lexer": "ipython3"}},
          "nbformat": 4, "nbformat_minor": 4}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook gerado em '{filename}'")


if __name__ == "__main__":
    build()
