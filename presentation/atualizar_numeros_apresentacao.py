"""
Números da apresentação `presentation/apresentacao-resultados.tex`, recalculados a partir das
predições por janela em disco (results/trainings/{tests,multirep,cpa_exp}/...).

Uso (na raiz do repositório):  python presentation/atualizar_numeros_apresentacao.py

Convenções (iguais às de notebooks/iara_results.py e do pipeline iara):
  * predição por arquivo = voto majoritário das janelas (collections.Counter.most_common);
  * SP = sqrt(média(recall) * média_geom(recall)); ACC = acurácia balanceada (média dos recalls);
  * média ± desvio populacional (ddof=0) sobre os 10 folds do 5x2cv;
  * F-test 5x2cv combinado de Alpaydin (R.f_test_5x2), como no artigo.
Os valores impressos aqui são os que estão escritos no .tex (com vírgula decimal).
Valores do artigo (Silva et al., 2025) vêm de R.ARTICLE_TABLE7 ou são citados das Tabelas 9/10.
"""
import csv
import glob
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
os.chdir(ROOT)
import iara_results as R  # noqa: E402

TESTS = os.path.join(ROOT, "results", "trainings", "tests")
CPA_DIR = os.path.join(ROOT, "results", "trainings", "cpa_exp")
A7 = R.ARTICLE_TABLE7


def pt(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def pm(ms, nd=2):
    return f"{pt(ms[0], nd)} ± {pt(ms[1], nd)}"


def ms(v):
    v = np.asarray(v, dtype=float)
    return float(np.mean(v)), float(np.std(v, ddof=0))


def header(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


# ---------------------------------------------------------------------------------------------
# Leitura tolerante: 'svm_nystroem' grava 'svm_nystroem_300_multiclass_test.csv'
# ---------------------------------------------------------------------------------------------
def votos_csv(path):
    df = pd.read_csv(path, usecols=["File", "Target", "Prediction"])
    rows = []
    for fid, g in df.groupby("File", sort=True):
        c = Counter(g["Prediction"].tolist())
        pred = c.most_common(1)[0][0]
        rows.append((fid, Counter(g["Target"]).most_common(1)[0][0], pred, len(g), c[pred] / len(g)))
    return pd.DataFrame(rows, columns=["File", "Target", "Prediction", "n_windows", "agreement"])


def file_preds(exp, subset="test"):
    try:
        return R.file_predictions(exp, subset)
    except FileNotFoundError:
        out = {}
        for k in range(10):
            found = glob.glob(os.path.join(TESTS, exp, "eval", f"fold_{k}", f"*_multiclass_{subset}.csv"))
            assert len(found) == 1, (exp, k, found)
            out[k] = votos_csv(found[0])
        return out


def fold_metrics(exp):
    try:
        return R.fold_metrics(exp)
    except FileNotFoundError:
        return pd.DataFrame([R._metrics(d.Target.values, d.Prediction.values) for d in file_preds(exp).values()])


def summ(exp):
    fm = fold_metrics(exp)
    return {c: ms(fm[c]) for c in fm.columns}


def ftest(a, b, metric="SP"):
    sa = fold_metrics(a)[metric].values if isinstance(a, str) else np.asarray(a)
    sb = fold_metrics(b)[metric].values if isinstance(b, str) else np.asarray(b)
    return R.f_test_5x2(sa, sb)


def line(label, exp):
    s = summ(exp)
    rec = "/".join(pt(s["R_" + c][0], 2) for c in R.CLASSES)
    print(f"  {label:<55} SP {pm(s['SP'])} | ACC(bal) {pm(s['ACC'])} | ACC(micro) {pm(s['ACC_micro'])} "
          f"| recalls S/M/L/BG {rec}   [{exp}]")
    return s


# ---------------------------------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------------------------------
MLP_MEL, MLP_LOFAR, CNN_MEL = "mlp", "mlp_lofar", "cnn"
SVM_MEL = "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0"
SVM_MEL_C05 = "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C0.5"
SVM_L6000 = "svm_nystroem_6000_lofar_elasticnet_l1r0.15"
MLP_HYB = "mlp_hybrid_pca64"
PFX = "svm_nystroem_4000_hybrid_cascade_pretrained_sm6000pca64_smallalso_fbmc_large_sl6000pca8_"
CASC_JAN = PFX + "lofarmc6000_nosm_elasticnet_l1r0.15"          # "Cascata LOFAR 6000 + SL" (por janela)
CASC_JAN_SMONLY = PFX + "lofarmc6000_smonly_elasticnet_l1r0.15"
CASC_JAN_HYB = PFX + "hybmc4000C2.0_elasticnet_l1r0.15"
CASC_JAN_HYB_SLRO = PFX + "hybmc4000C2.0_slro_elasticnet_l1r0.15"
CASC_ARQ = "cascade_file_lofarmc6000_sl6000pca8_4c"               # cascata por arquivo (NB5, seção 5.1)
SL = "svm_nystroem_6000_hybrid_binary_small_large_pca8_elasticnet_l1r0.15"

# =============================================================================================
header("Slides 2-3: protocolo e distribuição das classes no teste")
iara = pd.read_csv(os.path.join(ROOT, "src", "iara", "dataset_info", "iara.csv"), dtype={"ID": int})
iara = iara.set_index("ID")
tst = file_preds(SVM_L6000, "test")
trn = file_preds(SVM_L6000, "trn")
val = file_preds(SVM_L6000, "val")
n_tst = [len(tst[k]) for k in range(10)]
n_trn = [len(trn[k]) for k in range(10)]
n_val = [len(val[k]) for k in range(10)]
print(f"  arquivos por fold: treino {sorted(set(n_trn))}, validação {sorted(set(n_val))}, teste {sorted(set(n_tst))}")
print(f"  total de avaliações de teste (10 folds): {sum(n_tst)}")
cls_counts = pd.DataFrame([tst[k].Target.value_counts().sort_index() for k in range(10)])
print("  arquivos de teste por classe e fold (S,M,L,BG):")
print(cls_counts.to_string())
tot_cls = cls_counts.sum()
print("  totais 10 folds:", dict(zip(R.CLASSES, tot_cls.values)),
      " proporção (%):", [pt(v / tot_cls.sum() * 100, 1) for v in tot_cls.values])
vezes = Counter(f for k in range(10) for f in tst[k].File)
print(f"  arquivos distintos que aparecem no teste: {len(vezes)}; aparições por arquivo: "
      f"média {pt(np.mean(list(vezes.values())), 2)} (mín {min(vezes.values())}, máx {max(vezes.values())})")
# Horas de teste por fold (Recording time em segundos, iara.csv)
horas = [iara.loc[tst[k].File, "Recording time"].sum() / 3600 for k in range(10)]
print(f"  horas de gravação no teste por fold: {pt(min(horas), 1)} a {pt(max(horas), 1)} h "
      f"(média {pt(np.mean(horas), 1)} h)")
uo = iara[iara.Dataset.isin(list("ABCDE"))]
print(f"  gravações UO (A-E): {len(uo)}; horas UO: {pt(uo['Recording time'].sum() / 3600, 1)} h; "
      f"horas IARA total: {pt(iara['Recording time'].sum() / 3600, 1)} h")
# Navio de teste com o mesmo Ship ID no treino
tot = comp = 0
for k in range(10):
    t = tst[k][tst[k].Target != 3]
    ship_trn = set(iara.loc[trn[k].File, "Ship ID"].dropna())
    s = iara.loc[t.File, "Ship ID"]
    tot += len(t)
    comp += int(s.isin(ship_trn).sum())
print(f"  gravações de navio no teste cujo Ship ID também está no treino: {comp}/{tot} = {pt(comp / tot * 100, 1)}%")

# =============================================================================================
header("Slide 'Reprodução local dos baselines' (local = pipeline iara oficial, results/trainings/tests)")
for (m, f), exp in (((("MLP", "MEL")), MLP_MEL), (("MLP", "LOFAR"), MLP_LOFAR), (("CNN", "MEL"), CNN_MEL)):
    s = line(f"{m} {f} local", exp)
    a = A7[(m, f)]
    print(f"     artigo SP {pm(a['SP'])} ACC {pm(a['ACC'])} | diferença local - artigo: SP {s['SP'][0] - a['SP'][0]:+.2f} "
          f"ACC {s['ACC'][0] - a['ACC'][0]:+.2f}")
print("  RF MEL/LOFAR e CNN LOFAR: sem re-execução no pipeline oficial (0,512 s). Existem apenas versões ad hoc com")
print("  cache de 5,12 s (não comparáveis ao artigo):")
line("RF MEL (cache 5,12 s, 100 árvores)", "rf_mel5s_n100_balanced_4c")
line("RF LOFAR (cache 5,12 s, 100 árvores)", "rf_lofar5s_n100_balanced_4c")
line("CNN 2D LOFAR ad hoc (cache 5,12 s)", "cnn2d_lofar5s_w32_b64_e15_4c")

# =============================================================================================
header("Slide 'Evolução do SVM com MEL'")
MEL_PROG = [("SVM MEL m=300 (C=1, L2)", "svm_nystroem"), ("SVM MEL m=1000 (C=1, L2)", "svm_nystroem_1000"),
            ("SVM MEL m=2000 (C=1, L2)", "svm_nystroem_2000_log_melgram"),
            ("SVM MEL m=3000 (C=1, L2)", "svm_nystroem_3000_log_melgram"),
            ("SVM MEL m=4000 (C=2, ElasticNet)", SVM_MEL), ("SVM MEL m=4000 (C=0,5, ElasticNet)", SVM_MEL_C05)]
for lab, e in MEL_PROG:
    line(lab, e)
F_MEL_CNN = ftest(SVM_MEL, CNN_MEL)
F_MEL_MLP = ftest(SVM_MEL, MLP_MEL)
F_MEL_3000 = ftest(SVM_MEL, "svm_nystroem_3000_log_melgram")
print(f"  F-test SVM MEL m=4000 vs CNN MEL local: F={F_MEL_CNN[0]:.2f} p={pt(F_MEL_CNN[1], 3)}")
print(f"  F-test SVM MEL m=4000 vs MLP MEL local: F={F_MEL_MLP[0]:.2f} p={pt(F_MEL_MLP[1], 3)}")
print(f"  F-test SVM MEL m=4000 vs m=3000:       F={F_MEL_3000[0]:.2f} p={pt(F_MEL_3000[1], 3)}")
# menor desvio de SP entre os modelos locais do benchmark
LOCAIS = {"MLP MEL": MLP_MEL, "MLP LOFAR": MLP_LOFAR, "CNN MEL": CNN_MEL, "SVM MEL m=4000": SVM_MEL,
          "SVM LOFAR m=6000": SVM_L6000}
print("  desvio do SP:", {k: pt(summ(v)["SP"][1]) for k, v in LOCAIS.items()},
      "| artigo:", {f"{m} {f}": pt(A7[(m, f)]["SP"][1]) for (m, f) in A7})

# =============================================================================================
header("Slide 'Hiperparâmetros e limites do SVM'")
HP = [("LOFAR m=1000 C=1 (L2)", "svm_nystroem_1000_lofar"), ("LOFAR m=2000 C=0,1 (L2)", "svm_nystroem_2000_lofar_C0.1"),
      ("LOFAR m=4000 C=0,5 E-Net", "svm_nystroem_4000_lofar_elasticnet_l1r0.15_C0.5"),
      ("LOFAR m=6000 C=1 E-Net", SVM_L6000), ("MEL m=4000 C=0,5 E-Net", SVM_MEL_C05),
      ("MEL m=2000 C=10 (L2)", "svm_nystroem_2000_log_melgram_C10.0"), ("MEL m=4000 C=2 E-Net", SVM_MEL)]
for lab, e in HP:
    line(lab, e)
F_L6_MLPL = ftest(SVM_L6000, MLP_LOFAR)
F_L6_MEL = ftest(SVM_L6000, SVM_MEL)
print(f"  F-test SVM LOFAR m=6000 vs MLP LOFAR local: F={F_L6_MLPL[0]:.2f} p={pt(F_L6_MLPL[1], 3)}")
print(f"  F-test SVM LOFAR m=6000 vs SVM MEL m=4000:  F={F_L6_MEL[0]:.2f} p={pt(F_L6_MEL[1], 3)}")

# =============================================================================================
header("Slide 'Opção de rejeição' (concordância de janelas por arquivo; ACC = balanceada nos aceitos)")
REJ = [("SVM MEL m=4000", SVM_MEL), ("SVM LOFAR m=6000", SVM_L6000), ("MLP MEL (local)", MLP_MEL),
       ("MLP LOFAR (local)", MLP_LOFAR), ("CNN MEL (local)", CNN_MEL)]
for lab, e in REJ:
    out = []
    for t in (0.0, 0.6, 0.9):
        cob, acc, mic = [], [], []
        for k, df in R.file_predictions(e).items():
            a = df[df.agreement >= t]
            m = R._metrics(a.Target.values, a.Prediction.values)
            cob.append(len(a) / len(df) * 100)
            acc.append(m["ACC"])
            mic.append(m["ACC_micro"])
        out.append(f"t>={t:.1f}: cob {pt(np.mean(cob), 1)} ACCbal {pt(np.mean(acc), 1)} (micro {pt(np.mean(mic), 1)})")
    print(f"  {lab:<22} " + " | ".join(out))

# =============================================================================================
header("Slide 'Robustez ao CPA (Tabela 10)': results/trainings/cpa_exp (re-execução local)")


def cpa_paths(sub, modelo, treino, teste):
    paths = []
    for k in range(10):
        if treino == teste:
            pat = os.path.join(CPA_DIR, sub, f"{treino}_{modelo}", "eval", f"fold_{k}", "*_multiclass_test.csv")
        else:
            pat = os.path.join(CPA_DIR, sub, "comparison", f"{treino}_{modelo}", f"*_multiclass_test_{k}.csv")
        found = glob.glob(pat)
        assert len(found) == 1, (pat, found)
        paths.append(found[0])
    return paths


def cpa_fm(sub, modelo, tr, te):
    return pd.DataFrame([R._metrics(d.Target.values, d.Prediction.values)
                         for d in map(votos_csv, cpa_paths(sub, modelo, tr, te))])


CPA_M = {"MLP MEL": ("exp_1", "mlp_window"), "SVM MEL": ("exp_1", "svm_window"), "CNN MEL": ("exp_1", "cnn_image"),
         "SVM LOFAR (m=4000, PCA64)": ("lofar_exp_1", "svm_window")}
CPA_R = {}
for nome, (sub, mod) in CPA_M.items():
    for tr, te in (("A", "A"), ("A", "C"), ("C", "C"), ("C", "A")):
        fm = cpa_fm(sub, mod, tr, te)
        CPA_R[(nome, tr, te)] = fm
        print(f"  {nome:<28} treino {tr} -> teste {te}: SP {pm(ms(fm.SP))} | ACC {pm(ms(fm.ACC))}")
for tr, te in (("A", "A"), ("A", "C"), ("C", "C"), ("C", "A")):
    f, p = R.f_test_5x2(CPA_R[("SVM LOFAR (m=4000, PCA64)", tr, te)].SP.values, CPA_R[("MLP MEL", tr, te)].SP.values)
    print(f"  F-test SP SVM LOFAR vs MLP MEL ({tr}->{te}): F={f:.2f} p={pt(p, 3)}")
print("  Tabela 10 do artigo, MLP (MEL): A->A SP 67,25±2,44 | A->C SP 59,08±6,51 | C->C SP 58,08±3,70 | C->A SP 59,59±5,22")

# =============================================================================================
header("Slide 'Síntese' e 'Ranking geral'")
RANK = [("Cascata por janela LOFAR 6000 + SL (nosm)", CASC_JAN), ("Cascata por janela híbrida SM+SL (slro)", CASC_JAN_HYB_SLRO),
        ("Cascata por janela híbrida SM+SL", CASC_JAN_HYB), ("Cascata por arquivo LOFAR 6000 + SL", CASC_ARQ),
        ("Cascata por janela SM only", CASC_JAN_SMONLY),
        ("MLP LOFAR (local)", MLP_LOFAR), ("MLP MEL (local)", MLP_MEL), ("CNN MEL (local)", CNN_MEL),
        ("MLP híbrido MEL+LOFAR-PCA64 (local)", MLP_HYB),
        ("SVM LOFAR m=6000", SVM_L6000), ("SVM MEL m=4000", SVM_MEL)]
for lab, e in RANK:
    s = line(lab, e)
    if e != MLP_LOFAR:
        f, p = ftest(e, MLP_LOFAR)
        f2, p2 = ftest(e, MLP_LOFAR, "ACC")
        print(f"     F-test vs MLP LOFAR local: SP F={f:.2f} p={pt(p, 3)} | ACC p={pt(p2, 3)}")
    if e.startswith(PFX) or e == CASC_ARQ:
        f, p = ftest(e, SVM_L6000)
        print(f"     F-test vs estágio 1 (SVM LOFAR m=6000): F={f:.2f} p={pt(p, 3)}")
print("  artigo (Tabela 7):", {f"{m} {f}": (pm(A7[(m, f)]['SP']), pm(A7[(m, f)]['ACC'])) for (m, f) in A7})

# =============================================================================================
header("Matrizes de confusão por arquivo (acumuladas nos 10 folds; % por linha; precisão por coluna)")


def confusao(exp):
    cm = np.zeros((4, 4), dtype=int)
    for k, df in R.file_predictions(exp).items():
        for t, p in zip(df.Target, df.Prediction):
            cm[t, p] += 1
    return cm


CM = {}
for lab, e in (("MLP MEL", MLP_MEL), ("MLP LOFAR", MLP_LOFAR), ("SVM MEL m=4000", SVM_MEL),
               ("SVM LOFAR m=6000 (base)", SVM_L6000), ("Cascata LOFAR 6000 + SL (janela, nosm)", CASC_JAN),
               ("Cascata por arquivo", CASC_ARQ)):
    cm = confusao(e)
    CM[e] = cm
    rowp = cm / cm.sum(1, keepdims=True) * 100
    prec = np.diag(cm) / cm.sum(0) * 100
    s = summ(e)
    print(f"  {lab} [{e}]")
    for i, c in enumerate(R.CLASSES):
        print(f"     {c:<11} " + " ".join(f"{pt(v, 1):>6}" for v in rowp[i]) + "   | " + " ".join(f"{v:>5}" for v in cm[i])
              + f"  (total {cm[i].sum()})")
    print("     precisão   " + " ".join(f"{pt(v, 1):>6}" for v in prec))
    print(f"     corretos {np.trace(cm)}/{cm.sum()} = ACC micro acumulada {pt(np.trace(cm) / cm.sum() * 100, 1)}% | "
          f"SP (média folds) {pt(s['SP'][0], 2)} | ACC bal (média folds) {pt(s['ACC'][0], 2)} | "
          f"ACC micro (média folds) {pt(s['ACC_micro'][0], 2)}")
d = CM[CASC_JAN] - CM[SVM_L6000]
print(f"  cascata - base: corretos {np.trace(CM[CASC_JAN]) - np.trace(CM[SVM_L6000]):+d}; "
      f"diagonal {np.diag(d).tolist()}; S->L {CM[SVM_L6000][0, 2]} -> {CM[CASC_JAN][0, 2]}")
# Recall médio por fold (tabela de efeito colateral)
sb, sc = summ(SVM_L6000), summ(CASC_JAN)
print("  recalls médios por fold base -> cascata:",
      {c: f"{pt(sb['R_' + c][0], 2)} -> {pt(sc['R_' + c][0], 2)} ({sc['R_' + c][0] - sb['R_' + c][0]:+.2f})" for c in R.CLASSES},
      f"SP {pt(sb['SP'][0])} -> {pt(sc['SP'][0])} ({sc['SP'][0] - sb['SP'][0]:+.2f})")
# Arquivos SMALL resgatados (base votou LARGE, cascata votou SMALL) e MEDIUM perdidos para SMALL
for e in (CASC_JAN, CASC_ARQ):
    resg = med = 0
    for k in range(10):
        m = R.file_predictions(SVM_L6000)[k].merge(R.file_predictions(e)[k], on=["File", "Target"], suffixes=("_b", "_c"))
        resg += int(((m.Target == 0) & (m.Prediction_b == 2) & (m.Prediction_c == 0)).sum())
        med += int(((m.Target == 1) & (m.Prediction_b == 1) & (m.Prediction_c == 0)).sum())
    print(f"  [{e}] SMALL resgatados de LARGE: {resg} (10 folds; {pt(resg / 10, 1)}/fold); "
          f"MEDIUM correto -> SMALL: {med}")

# =============================================================================================
header("Especialistas binários SMALL x LARGE (arquivo; classe 1 = LARGE); verificação de sobreposição treino/teste")
SL_EXPS = [("Híbrido MEL + LOFAR-PCA4 (jun/2026)", "svm_nystroem_6000_hybrid_binary_small_large_pca4_elasticnet_l1r0.15"),
           ("Híbrido MEL + LOFAR-PCA8 (retreinado 20/10/2026)", SL),
           ("Híbrido MEL + LOFAR-PCA16 (jun/2026)", "svm_nystroem_6000_hybrid_binary_small_large_pca16_elasticnet_l1r0.15"),
           ("Híbrido MEL + LOFAR-PCA64 (jun/2026)", "svm_nystroem_6000_hybrid_binary_small_large_pca64_elasticnet_l1r0.15")]
mc_tst = R.file_predictions(SVM_L6000)
for lab, e in SL_EXPS:
    prec, recL, recS, bal, own, leak = [], [], [], [], 0, 0
    nt = 0
    for k in range(10):
        t = votos_csv(os.path.join(TESTS, e, "eval", f"fold_{k}", f"{e}_multiclass_test.csv"))
        tr = set(pd.read_csv(os.path.join(TESTS, e, "eval", f"fold_{k}", f"{e}_multiclass_trn.csv"), usecols=["File"]).File)
        own += len(set(t.File) & tr)
        sl4 = mc_tst[k][mc_tst[k].Target.isin([0, 2])]
        leak += int(sl4.File.isin(list(tr)).sum())
        nt += len(sl4)
        tp = ((t.Prediction == 1) & (t.Target == 1)).sum()
        prec.append(tp / (t.Prediction == 1).sum() * 100)
        recL.append(tp / (t.Target == 1).sum() * 100)
        recS.append(((t.Prediction == 0) & (t.Target == 0)).sum() / (t.Target == 0).sum() * 100)
        bal.append((recL[-1] + recS[-1]) / 2)
    print(f"  {lab:<50} precisão LARGE {pm(ms(prec), 1)} | recall LARGE {pm(ms(recL), 1)} | recall SMALL {pm(ms(recS), 1)} "
          f"| ACC bal {pm(ms(bal), 1)} | teste∩treino próprio {own} | teste SMALL/LARGE do 4c no treino {leak}/{nt}")

# =============================================================================================
header("Slide 'Robustez ao CPA da cascata': desempenho por coleção do arquivo de teste (modelos 4c)")
DS = iara["Dataset"]
GRUPOS = {"A": ["A"], "C": ["C"], "A+C": ["A", "C"], "B+D": ["B", "D"], "Geral": list("ABCDE")}
for lab, e in (("MLP LOFAR (local)", MLP_LOFAR), ("MLP MEL (local)", MLP_MEL), ("CNN MEL (local)", CNN_MEL),
               ("MLP híbrido (local)", MLP_HYB), ("SVM LOFAR 6000", SVM_L6000), ("Cascata LOFAR+SL (janela)", CASC_JAN),
               ("Cascata por arquivo", CASC_ARQ)):
    out = []
    for g, cols in GRUPOS.items():
        sp, acc = [], []
        for k, df in R.file_predictions(e).items():
            d_ = df[DS.loc[df.File].isin(cols).values]
            m = R._metrics(d_.Target.values, d_.Prediction.values)
            sp.append(m["SP"])
            acc.append(m["ACC"])
        out.append(f"{g}: {pt(np.mean(sp), 1)} ({pt(np.mean(acc), 1)})")
    print(f"  {lab:<28} " + " | ".join(out))

# =============================================================================================
header("Complementos: acervo, janelas de treino, F-tests adicionais")
print("  gravações por coleção:", iara.Dataset.value_counts().sort_index().to_dict())
_cls = pd.cut(iara.Length.apply(pd.to_numeric, errors="coerce"), [-np.inf, 50, 100, np.inf], right=False,
              labels=["SMALL(<50)", "MEDIUM(50-100)", "LARGE(>=100)"])
for g, cols in (("Gliders F+G", ["F", "G"]), ("UO A-D", list("ABCD"))):
    print(f"  classes por comprimento em {g} (regra aproximada; ver iara.default.Target):",
          _cls[iara.Dataset.isin(cols)].value_counts().to_dict())
nw = {}
for e in (SVM_MEL, SVM_L6000):
    nw[e] = [len(pd.read_csv(os.path.join(TESTS, e, "eval", f"fold_{k}", f"{e}_multiclass_trn.csv"), usecols=["File"]))
             for k in range(10)]
    print(f"  janelas de treino por fold [{e}]: {min(nw[e])} a {max(nw[e])}")
N = max(nw[SVM_L6000])
print(f"  memória Gram N x N (float32): {N * N * 4 / 1e9:.0f} GB; (float64): {N * N * 8 / 1e9:.0f} GB")
for m in (4000, 6000):
    print(f"  memória base projetada N x m (m={m}, float32): {N * m * 4 / 1e9:.1f} GB")
for a, b in (("svm_nystroem", "svm_nystroem_1000"), ("svm_nystroem_1000", "svm_nystroem_2000_log_melgram"),
             ("svm_nystroem_2000_log_melgram", "svm_nystroem_3000_log_melgram"),
             ("svm_nystroem_3000_log_melgram", SVM_MEL), (SVM_MEL, SVM_MEL_C05)):
    f, p = ftest(a, b)
    f2, p2 = ftest(a, b, "ACC")
    print(f"  F-test {a} vs {b}: SP p={pt(p, 3)} | ACC p={pt(p2, 3)}")
f, p = ftest(SVM_MEL, SVM_MEL_C05, "R_SMALL")
print(f"  F-test recall SMALL SVM MEL C=2 vs C=0,5: p={pt(p, 3)}")
f, p = ftest(SVM_L6000, "svm_nystroem_4000_lofar_elasticnet_l1r0.15_C0.5")
print(f"  F-test SVM LOFAR m=6000 vs m=4000 C=0,5: p={pt(p, 3)}")
# F-test por coleção: cascata por janela vs MLP LOFAR e MLP híbrido
for g, cols in GRUPOS.items():
    sps = {}
    for e in (CASC_JAN, MLP_LOFAR, MLP_HYB, SVM_L6000):
        v = []
        for k, df in R.file_predictions(e).items():
            d_ = df[DS.loc[df.File].isin(cols).values]
            v.append(R._metrics(d_.Target.values, d_.Prediction.values)["SP"])
        sps[e] = np.array(v)
    print(f"  coleção {g}: p cascata vs MLP LOFAR {pt(R.f_test_5x2(sps[CASC_JAN], sps[MLP_LOFAR])[1], 3)} | "
          f"vs MLP híbrido {pt(R.f_test_5x2(sps[CASC_JAN], sps[MLP_HYB])[1], 3)} | "
          f"vs SVM base {pt(R.f_test_5x2(sps[CASC_JAN], sps[SVM_L6000])[1], 3)}")
