"""Gera reports/comparacao_modelos.md com todos os números recalculados a partir dos resultados em disco.

Nenhum número do relatório é digitado: tudo sai das predições por janela em
results/trainings/{tests,multirep,cpa_exp,_cascatas_com_sl_vazado_pre20261020}/ via notebooks/iara_results.py
(voto majoritário por arquivo com Counter.most_common, SP de src/iara/ml/metrics.py, ACC = acurácia
balanceada, média ± desvio populacional ddof=0 sobre os 10 folds, F-test 5x2cv de Alpaydin).
Valores do artigo vêm de R.ARTICLE_TABLE7 (Tabela 7) ou são citados explicitamente como "Tabela X do artigo".

Cada valor usado no relatório é impresso no stdout (seção "VALORES USADOS").

Uso (cwd = raiz do repositório):  python reports/atualizar_numeros_comparacao_modelos.py
"""
import glob
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
os.chdir(ROOT)
import iara_results as R  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

OUT_MD = os.path.join(ROOT, "reports", "comparacao_modelos.md")
TESTS = os.path.join(ROOT, "results", "trainings", "tests")
CPA_DIR = os.path.join(ROOT, "results", "trainings", "cpa_exp")
OLD_CASC_DIR = os.path.join(ROOT, "results", "trainings", "_cascatas_com_sl_vazado_pre20261020")
A7 = R.ARTICLE_TABLE7

USED = []  # (rótulo, valor) — impresso no final


def use(label, value):
    USED.append((label, value))
    return value


# ----------------------------------------------------------------------------------------------
# Leitura genérica (para CSVs fora do padrão de nome de R.file_predictions)
# ----------------------------------------------------------------------------------------------
def votos_csv(path):
    df = pd.read_csv(path, usecols=["File", "Target", "Prediction"])
    rows = []
    for fid, g in df.groupby("File", sort=True):
        c = Counter(g["Prediction"].tolist())
        pred = c.most_common(1)[0][0]
        rows.append((fid, Counter(g["Target"]).most_common(1)[0][0], pred, len(g), c[pred] / len(g)))
    return pd.DataFrame(rows, columns=["File", "Target", "Prediction", "n_windows", "agreement"])


_CACHE = {}


def files(exp, suffix="", base=None):
    """{fold: DataFrame por arquivo}. suffix: ex. '_b0.4_-0.19_0_0' (CSVs com bias); base: pasta alternativa."""
    key = (exp, suffix, base)
    if key in _CACHE:
        return _CACHE[key]
    if not suffix and base is None:
        try:
            out = R.file_predictions(exp)
            _CACHE[key] = out
            return out
        except FileNotFoundError:
            pass
    d = os.path.join(base or TESTS, exp)
    out = {}
    for k in range(10):
        p = os.path.join(d, "eval", f"fold_{k}", f"{exp}_multiclass_test{suffix}.csv")
        if not os.path.exists(p):
            found = glob.glob(os.path.join(d, "eval", f"fold_{k}", f"*_multiclass_test{suffix}.csv"))
            if len(found) != 1:
                raise FileNotFoundError(f"{exp}{suffix}: fold {k} ausente")
            p = found[0]
        out[k] = votos_csv(p)
    _CACHE[key] = out
    return out


def folds(exp, suffix="", base=None, mode="4c"):
    """DataFrame por fold (SP, ACC, ACC_micro, F1, R_*)."""
    if not suffix and base is None:
        try:
            return R.fold_metrics(exp, mode)
        except FileNotFoundError:
            pass
    rows = []
    for k, df in files(exp, suffix, base).items():
        if mode == "4c_sem_bg":
            df = df[df.Target != 3]
        rows.append({"fold": k, **R._metrics(df.Target.values, df.Prediction.values)})
    return pd.DataFrame(rows).set_index("fold")


def n_folds(exp, base=None):
    d = os.path.join(base or TESTS, exp, "eval")
    return len(glob.glob(os.path.join(d, "fold_*", "*_multiclass_test.csv")))


def ms(v):
    v = np.asarray(v, dtype=float)
    return float(np.mean(v)), float(np.std(v, ddof=0))


def S(exp, metric, **kw):
    return use(f"{exp}{kw.get('suffix', '')}{'@' + kw['mode'] if 'mode' in kw else ''} {metric}",
               ms(folds(exp, **kw)[metric]))


def F(x, nd=2):
    return f"{x[0]:.{nd}f} ± {x[1]:.{nd}f}"


def P(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def ftest(a, b, metric="SP", ka=None, kb=None):
    va = folds(a, **(ka or {}))[metric].values
    vb = folds(b, **(kb or {}))[metric].values
    f, p = R.f_test_5x2(va, vb)
    use(f"F-test {metric}: {a} vs {b}", (round(f, 3), round(p, 4)))
    return p


# ==============================================================================================
# Experimentos (nomes em results/trainings/{tests,multirep})
# ==============================================================================================
SVM_MEL = "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0"
SVM_MEL_PCA = "svm_nystroem_4000_log_melgram_pca64_elasticnet_l1r0.15_C2.0"
SVM_LOFAR = "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0"
SVM_LOFAR_C05 = "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C0.5"
SVM_LOFAR_NOPCA = "svm_nystroem_4000_lofar_C2.0"
SVM_LOFAR_6000 = "svm_nystroem_6000_lofar_elasticnet_l1r0.15"
HYB_C05 = "svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5"
HYB_C2 = "svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C2.0"
MLP_MEL, MLP_LOFAR, MLP_HYB, CNN_MEL = "mlp", "mlp_lofar", "mlp_hybrid_pca64", "cnn"

# Cascatas
PFX = "svm_nystroem_4000_hybrid_cascade_pretrained_sm6000pca64_smallalso_fbmc_large_sl6000pca8_"
PFX_OLD = ("svm_nystroem_4000_hybrid_cascade_pretrained_svm_nystroem_6000_hybrid_binary_small_medium_pca64_"
           "elasticnet_l1r0.15_smallalso_fbmc_large_svm_nystroem_6000_hybrid_binary_small_large_pca8_elasticnet_l1r0.15_")
SM = "svm_nystroem_6000_hybrid_binary_small_medium_pca64_elasticnet_l1r0.15"
CASC_SL_INVALID = f"svm_nystroem_4000_hybrid_cascade_pretrained_{SM}_smallalso_large_svm_nystroem_6000_hybrid_binary_small_large_pca8_elasticnet_l1r0.15"
CASC_FILE = "cascade_file_lofarmc6000_sl6000pca8_4c"
SL = "svm_nystroem_6000_hybrid_binary_small_large_pca8_elasticnet_l1r0.15"
# (rótulo, experimento novo, experimento antigo c/ SL vazado (pasta movida), estágio 1, roteamento)
CASC_JANELA = [
    ("HYBRID-MC(C=2.0) → SM(PCA64)+SL(PCA8) — sempre", PFX + "hybmc4000C2.0_elasticnet_l1r0.15", PFX_OLD + "hybmc4000C2.0", HYB_C2, "SM + SL"),
    ("HYBRID-MC(C=2.0) → SM(PCA64)+SL(PCA8) — runner-up", PFX + "hybmc4000C2.0_slro_elasticnet_l1r0.15", PFX_OLD + "hybmc4000C2.0_slro", HYB_C2, "SM + SL só se a 2ª classe for SMALL"),
    ("LOFAR-MC(m=6000) → SL(PCA8)", PFX + "lofarmc6000_nosm_elasticnet_l1r0.15", PFX + "lofarmc6000_nosm_elasticnet_l1r0.15", SVM_LOFAR_6000, "só SL"),
    ("LOFAR-MC(m=6000) → SM(PCA64) só MEDIUM + SL(PCA8)", PFX + "lofarmc6000_smonly_elasticnet_l1r0.15", PFX + "lofarmc6000_smonly_elasticnet_l1r0.15", SVM_LOFAR_6000, "SM só p/ MEDIUM + SL"),
]
# Cascatas que NÃO usam o especialista SMALL×LARGE vazado (continuam válidas). Estágio 1 não registrado no config.
CASC_VALIDAS = [
    ("MEL → SM(PCA64)/LOFAR (SMALL+MEDIUM)", f"svm_nystroem_4000_log_melgram_cascade_pretrained_{SM}_smallalso_large_{SVM_LOFAR_6000}"),
    ("HÍBRIDO → SM(PCA64)/LOFAR (só MEDIUM)", f"svm_nystroem_4000_hybrid_cascade_pretrained_{SM}_large_{SVM_LOFAR_6000}"),
    ("HÍBRIDO → SM(PCA64)/LOFAR (SMALL+MEDIUM)", f"svm_nystroem_4000_hybrid_cascade_pretrained_{SM}_smallalso_large_{SVM_LOFAR_6000}"),
    ("MEL → cascata 1000 (k-means)", "svm_nystroem_4000_log_melgram_cascade_1000_kmeans_elasticnet_l1r0.15"),
]

DEMON = {"SVM m=1000": "svm_nystroem_1000_demon_C2.0_elasticnet_l1r0.15_it1000",
         "SVM m=4000": "svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_it1000",
         "SVM m=6000": "svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000",
         "RF": "rf_demon_n100_balanced", "MLP": "mlp_demon_h32-16_e25", "CNN": "cnn2d_demon_w32_b32_e20"}
HHT = {"RF": "rf_hht_n100_balanced", "MLP": "mlp_hht_h32-16_e30_noseed",
       "SVM": "svm_nystroem_6000_hht_C2.0_elasticnet_l1r0.15", "CNN": "cnn2d_hht_w32_b32_e25"}
# Grade da fusão (mesma do NB8): (exp 4c, exp 3c retreinado ou None, observação)
GRID = {
    ("MLP", "DEMON"): ("mlp_demon_h32-16_e35_4c", "mlp_demon_h32-16_e35_3c", "[32,16], 35 ép., 5,12 s"),
    ("MLP", "MEL"): ("mlp_mel5s_h32-16_e35_4c", "mlp_mel5s_h32-16_e35_3c", "[32,16], 35 ép., MEL 5,12 s"),
    ("MLP", "LOFAR"): (MLP_LOFAR, None, "pipeline iara, 0,512 s"),
    ("MLP", "FUSÃO"): ("mlp_mel5s-demon_h64-32_e35_4c", "mlp_mel5s-demon_h64-32_e35_3c", "[64,32] (rede maior), 35 ép., 5,12 s"),
    ("SVM", "DEMON"): ("svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000_4c", "svm_nystroem_6000_demon_C2.0_elasticnet_l1r0.15_it1000_3c", "m=6000"),
    ("SVM", "MEL"): (SVM_MEL, None, "m=4000 (não 6000), 0,512 s"),
    ("SVM", "LOFAR"): (SVM_LOFAR_6000, None, "m=6000, 0,512 s"),
    ("SVM", "FUSÃO"): ("svm_nystroem_6000_mel5s-demon_C2.0_elasticnet_l1r0.15_4c", "svm_nystroem_6000_mel5s-demon_C2.0_elasticnet_l1r0.15_3c", "m=6000, 5,12 s"),
    ("RF", "DEMON"): ("rf_demon_n100_balanced_4c", "rf_demon_n100_balanced_3c", "100 árv."),
    ("RF", "MEL"): ("rf_mel5s_n100_balanced_4c", None, "100 árv., MEL 5,12 s"),
    ("RF", "LOFAR"): ("rf_lofar5s_n100_balanced_4c", None, "100 árv., LOFAR 5,12 s"),
    ("RF", "FUSÃO"): ("rf_mel5s-demon_n100_balanced_4c", "rf_mel5s-demon_n100_balanced_3c", "100 árv., 5,12 s"),
    ("CNN", "DEMON"): ("cnn2d_demon_w32_b32_e20_4c", "cnn2d_demon_w32_b32_e20_3c", "CNN ad hoc [COLAPSADA]"),
    ("CNN", "MEL"): (CNN_MEL, None, "CNN oficial, 0,512 s"),
    ("CNN", "LOFAR"): ("cnn2d_lofar5s_w32_b64_e15_4c", None, "CNN ad hoc, LOFAR 5,12 s [COLAPSADA]"),
    ("CNN", "FUSÃO"): ("cnn2d_mel5s-demon_w32_b32_e25_4c", "cnn2d_mel5s-demon_w32_b32_e25_3c", "CNN ad hoc [COLAPSADA]"),
}

# ==============================================================================================
# Tabela 1 — (rótulo, representação, fonte, linha do .md antigo, SP antigo)
#   fonte: ("art", chave da Tabela 7) | nome do experimento | (nome, sufixo do CSV)
# ==============================================================================================
T1 = [
    [("RF", "MEL", ("art", ("RF", "MEL")), 23, 62.22), ("RF", "LOFAR", ("art", ("RF", "LOFAR")), 24, 56.92),
     ("MLP", "MEL", ("art", ("MLP", "MEL")), 25, 63.38), ("MLP", "LOFAR", ("art", ("MLP", "LOFAR")), 26, 66.51),
     ("CNN", "MEL", ("art", ("CNN", "MEL")), 30, 63.52), ("CNN", "LOFAR", ("art", ("CNN", "LOFAR")), 31, 66.05),
     ("MLP (re-execução local)", "MEL", MLP_MEL, 27, 64.66), ("MLP (re-execução local)", "LOFAR", MLP_LOFAR, 28, 66.25),
     ("CNN (re-execução local)", "MEL", CNN_MEL, None, None),
     ("MLP (pca64)", "HÍBRIDO", MLP_HYB, 29, 66.13)],
    [("SVM (m=300)", "MEL", "svm_nystroem", 33, 59.96), ("SVM (m=1000)", "MEL", "svm_nystroem_1000", 34, 62.96),
     ("SVM (m=1000) (norm + pca64)", "MEL", "svm_nystroem_1000_log_melgram_norm_pca64", 35, 60.83),
     ("SVM (m=2000)", "MEL", "svm_nystroem_2000_log_melgram", 36, 63.04),
     ("SVM (m=2000) (C=10)", "MEL", "svm_nystroem_2000_log_melgram_C10.0", 37, 62.18),
     ("SVM (m=3000)", "MEL", "svm_nystroem_3000_log_melgram", 38, 63.65),
     ("SVM (m=4000) (C=2.0) (pca64) (elasticnet)", "MEL", SVM_MEL_PCA, 39, 63.82),
     ("SVM (m=4000) (C=2.0) (elasticnet) — \"Golden MEL\"", "MEL", SVM_MEL, 40, 63.78),
     ("SVM (m=4000) (C=0.5) (elasticnet)", "MEL", "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C0.5", 41, 57.51)],
    [("SVM (m=1000)", "LOFAR", "svm_nystroem_1000_lofar", 43, 57.86), ("SVM (m=2000)", "LOFAR", "svm_nystroem_2000_lofar", 44, 60.10),
     ("SVM (m=2000) (C=0.1)", "LOFAR", "svm_nystroem_2000_lofar_C0.1", 45, 54.84),
     ("SVM (m=2000) (C=2.0)", "LOFAR", "svm_nystroem_2000_lofar_C2.0", 46, 60.15),
     ("SVM (m=4000) (C=2.0)", "LOFAR", SVM_LOFAR_NOPCA, 47, 61.70),
     ("SVM (m=1000) (C=2.0) (pca64)", "LOFAR", "svm_nystroem_1000_lofar_pca64_C2.0", 48, 61.96),
     ("SVM (m=4000) (C=2.0) (pca64) (elasticnet)", "LOFAR", SVM_LOFAR, 49, 63.10),
     ("SVM (m=4000) (C=0.5) (pca64) (elasticnet)", "LOFAR", SVM_LOFAR_C05, 50, 63.81),
     ("SVM (m=4000) (C=0.5) (elasticnet)", "LOFAR", "svm_nystroem_4000_lofar_elasticnet_l1r0.15_C0.5", 51, 60.66),
     ("SVM (m=6000) (elasticnet)", "LOFAR", SVM_LOFAR_6000, 52, 63.83)],
    [("SVM (m=1000), max_iter=1000", "DEMON", DEMON["SVM m=1000"] + "_4c", 54, 38.75),
     ("SVM (m=4000), max_iter=1000", "DEMON", DEMON["SVM m=4000"] + "_4c", 55, 43.51),
     ("SVM (m=6000), max_iter=1000", "DEMON", DEMON["SVM m=6000"] + "_4c", 56, 46.57),
     ("RF (100 árvores)", "DEMON", DEMON["RF"] + "_4c", 57, 44.30),
     ("MLP [32,16] (25 ép.)", "DEMON", DEMON["MLP"] + "_4c", 58, 52.12),
     ("CNN 2D ad hoc [COLAPSADA]", "DEMON", DEMON["CNN"] + "_4c", 59, 35.72)],
    [("RF (100 árvores)", "HHT", HHT["RF"] + "_4c", 61, 58.14), ("MLP [32,16]", "HHT", HHT["MLP"] + "_4c", 62, 56.14),
     ("SVM (m=6000) (elasticnet)", "HHT", HHT["SVM"] + "_4c", 63, 53.63),
     ("CNN 2D ad hoc [COLAPSADA]", "HHT", HHT["CNN"] + "_4c", 64, 46.10)],
    [("MLP [64,32] Fusão (MEL 5,12 s + DEMON)", "MULTIMODAL", GRID[("MLP", "FUSÃO")][0], 66, 66.99),
     ("SVM (m=6000) Fusão (MEL 5,12 s + DEMON)", "MULTIMODAL", GRID[("SVM", "FUSÃO")][0], 67, 64.85),
     ("RF Fusão (MEL 5,12 s + DEMON)", "MULTIMODAL", GRID[("RF", "FUSÃO")][0], 68, 62.05),
     ("CNN 2D ad hoc Fusão [COLAPSADA]", "MULTIMODAL", GRID[("CNN", "FUSÃO")][0], 69, 23.96),
     ("SVM (m=4000) (C=2.0) (elasticnet, norm)", "HÍBRIDO", HYB_C2, 70, 64.67),
     ("SVM (m=4000) (C=0.5) (elasticnet, norm)", "HÍBRIDO", HYB_C05, 71, 65.16)],
    [("SVM Cascata: estágio 1 → SM(PCA64)/LOFAR (só MEDIUM)", "HÍBRIDO", CASC_VALIDAS[1][1], 72, 65.11),
     ("SVM Cascata: estágio 1 → SM(PCA64)/LOFAR (SMALL+MEDIUM)", "HÍBRIDO", CASC_VALIDAS[2][1], 73, 65.35),
     ("SVM Cascata: MEL → SM(PCA64)/LOFAR (SMALL+MEDIUM)", "HÍBRIDO", CASC_VALIDAS[0][1], None, None),
     ("SVM Cascata: HYBRID-MC → SM(PCA64)+SL(PCA8) (sempre), SL retreinado", "HÍBRIDO", CASC_JANELA[0][1], 75, 68.05),
     ("SVM Cascata: HYBRID-MC → SM(PCA64)+SL(PCA8) (runner-up), SL retreinado", "HÍBRIDO", CASC_JANELA[1][1], 76, 67.59),
     ("SVM Cascata: LOFAR-MC → SL(PCA8), SL retreinado", "HÍBRIDO", CASC_JANELA[2][1], None, None),
     ("SVM Cascata: LOFAR-MC → SM só MEDIUM + SL(PCA8), SL retreinado", "HÍBRIDO", CASC_JANELA[3][1], None, None),
     ("Cascata por arquivo: LOFAR-MC → SL(PCA8), SL retreinado", "HÍBRIDO", CASC_FILE, None, None),
     ("SVM Cascata Dupla com especialista LOFAR (t=0.55)", "HÍBRIDO", (f"svm_nystroem_4000_hybrid_cascade_pretrained_{SVM_LOFAR_6000}_elasticnet_l1r0.15", "_ct0.55"), 77, 64.33)],
]


def src_args(src):
    if isinstance(src, tuple):
        return src[0], {"suffix": src[1]}
    return src, {}


def t1_line(lab, rep, src):
    if isinstance(src, tuple) and src[0] == "art":
        a = A7[src[1]]
        use(f"Tabela 7 {src[1]}", a)
        return f"| **{lab}** | {rep} | {F(a['SP'])} | {F(a['ACC'])} | {F(a['F1'])} | Tabela 7 do artigo (Silva et al., 2025) |", a["SP"][0]
    exp, kw = src_args(src)
    sp, acc, f1 = S(exp, "SP", **kw), S(exp, "ACC", **kw), S(exp, "F1", **kw)
    name = exp + kw.get("suffix", "")
    return f"| **{lab}** | {rep} | {F(sp)} | {F(acc)} | {F(f1)} | `{name}` |", sp[0]


DIFFS = []  # (linha antiga, rótulo, SP antigo, SP novo)
t1_rows = []
for gi, group in enumerate(T1):
    if gi:
        t1_rows.append("| *---* | *---* | *---* | *---* | *---* | *---* |")
    for lab, rep, src, old_line, old_sp in group:
        line, sp_new = t1_line(lab, rep, src)
        t1_rows.append(line)
        if old_sp is not None:
            DIFFS.append((old_line, f"{lab} / {rep}", old_sp, sp_new))
t1_rows.append(f"| **SVM Cascata: MEL→SM(PCA64)+SL(PCA8) (SMALL+MEDIUM)** | HÍBRIDO | **INVALIDADA** | — | — | "
               f"especialista SL com vazamento; ver Seção 9 |")
T1_MD = "\n".join(t1_rows)

# ==============================================================================================
# Seção 3 — variância e PCA (F-test 5x2cv)
# ==============================================================================================
s_mel_sp, s_mel_acc = S(SVM_MEL, "SP"), S(SVM_MEL, "ACC")
s_cnn_sp, s_cnn_acc = S(CNN_MEL, "SP"), S(CNN_MEL, "ACC")
p_mel_cnn = ftest(SVM_MEL, CNN_MEL)
p_mel_cnn_acc = ftest(SVM_MEL, CNN_MEL, "ACC")
ratio_std = use("razão desvio SP SVM MEL / CNN MEL artigo", s_mel_sp[1] / A7[("CNN", "MEL")]["SP"][1])
s_melpca_sp = S(SVM_MEL_PCA, "SP")
p_mel_pca = ftest(SVM_MEL_PCA, SVM_MEL)
s_lofar_sp, s_lofar_acc = S(SVM_LOFAR, "SP"), S(SVM_LOFAR, "ACC")
s_lofar_nopca_sp = S(SVM_LOFAR_NOPCA, "SP")
p_lofar_pca = ftest(SVM_LOFAR, SVM_LOFAR_NOPCA)
p_lofar_pca_acc = ftest(SVM_LOFAR, SVM_LOFAR_NOPCA, "ACC")
s_l6_sp, s_l6_acc = S(SVM_LOFAR_6000, "SP"), S(SVM_LOFAR_6000, "ACC")
p_lofar_pca_6000 = ftest(SVM_LOFAR, SVM_LOFAR_6000, "ACC")
s_mlpl_sp, s_mlpl_acc = S(MLP_LOFAR, "SP"), S(MLP_LOFAR, "ACC")
s_mlpm_sp = S(MLP_MEL, "SP")

# ==============================================================================================
# Tabela 2 — rejeição por concordância das janelas
# ==============================================================================================
LIM = [0.0, 0.5, 0.6, 0.9]
REJ = [("SVM MEL (m=4000, C=2.0)", SVM_MEL, ""),
       ("SVM LOFAR (m=4000, C=0.5, PCA64)", SVM_LOFAR_C05, ""),
       ("SVM LOFAR (m=4000, C=2.0, PCA64)", SVM_LOFAR, ""),
       ("MLP MEL (local)", MLP_MEL, ""), ("MLP LOFAR (local)", MLP_LOFAR, ""),
       ("MLP Híbrido (pca64)", MLP_HYB, ""), ("SVM Híbrido (C=0.5)", HYB_C05, ""),
       ("SVM Híbrido (C=0.5) + bias [0.40, −0.19, 0, 0]", HYB_C05, "_b0.4_-0.19_0_0"),
       ("SVM LOFAR (C=0.5, PCA64) + bias [0.32, 0.27, 0, 0]", SVM_LOFAR_C05, "_b0.32_0.27_0_0"),
       ("CNN MEL (local; 7 segmentos/arquivo)", CNN_MEL, "")]


def curva(exp, suffix=""):
    out = {}
    for t in LIM:
        acc = {"Cob": [], "ACC": [], "ACC_micro": [], "SP": []}
        for k, df in files(exp, suffix).items():
            a = df[df.agreement >= t]
            m = R._metrics(a.Target.values, a.Prediction.values)
            acc["Cob"].append(len(a) / len(df) * 100)
            for key in ("ACC", "ACC_micro", "SP"):
                acc[key].append(m[key])
        out[t] = {key: use(f"rejeição {exp}{suffix} t>={t} {key}", ms(v)) for key, v in acc.items()}
    return out


CUR = {(e, s): curva(e, s) for _, e, s in REJ}
T_NOME = {0.0: "*(sem rejeição)*", 0.5: "*(maioria simples)*", 0.6: "*(maioria absoluta)*", 0.9: "*(consenso crítico)*"}
t2_rows = []
for t in LIM:
    for lab, e, s in REJ:
        c = CUR[(e, s)][t]
        t2_rows.append(f"| $t \\ge {t:.1f}$ {T_NOME[t]} | {lab} | {F(c['Cob'], 1)} | {F(c['ACC'], 1)} | {F(c['ACC_micro'], 1)} | {F(c['SP'], 1)} |")
T2_MD = "\n".join(t2_rows)


def rj(exp, t, key, suffix=""):
    return CUR[(exp, suffix)][t][key]


# arquivos por fold e janelas por arquivo
NWIN_SVM = use("mediana janelas/arquivo SVM LOFAR", float(np.median(np.concatenate([d.n_windows.values for d in files(SVM_LOFAR).values()]))))
NSEG_CNN = use("segmentos/arquivo CNN", sorted({int(n) for d in files(CNN_MEL).values() for n in d.n_windows.unique()}))
N_FILES_UO = use("nº de arquivos distintos nos testes (mlp_lofar)", len(set().union(*[set(d.File) for d in files(MLP_LOFAR).values()])))

# ==============================================================================================
# Seção 5.6 — DEMON
# ==============================================================================================
def conf_frac(exp, tgt, pred, mode_suffix=""):
    """Fração média (sobre os folds) dos arquivos da classe tgt preditos como pred."""
    v = []
    for k, df in files(exp).items():
        d = df[df.Target == tgt]
        v.append((d.Prediction == pred).mean() * 100)
    return use(f"{exp}: % classe {tgt} predita {pred}", ms(v))


D4 = {k: e + "_4c" for k, e in DEMON.items()}
D3 = {k: e + "_3c" for k, e in DEMON.items()}
dm = {k: {m: S(e, m) for m in ("SP", "ACC", "R_SMALL", "R_MEDIUM", "R_LARGE", "R_BACKGROUND")} for k, e in D4.items()}
dm3 = {k: {m: S(e, m, mode="3c") for m in ("SP", "R_SMALL", "R_MEDIUM", "R_LARGE")} for k, e in D3.items()
       if os.path.isdir(os.path.join(ROOT, "results", "trainings", "multirep", e))}
sm_to_l_svm = conf_frac(D4["SVM m=6000"], 0, 2)
sm_to_l_mlp = conf_frac(D4["MLP"], 0, 2)
bg_to_s_rf = conf_frac(D4["RF"], 3, 0)
p_mlp_vs_svm_demon = ftest(D4["MLP"], D4["SVM m=6000"])
p_mlp_vs_rf_demon = ftest(D4["MLP"], D4["RF"])
fam4 = {"MLP": D4["MLP"], "SVM m=6000": D4["SVM m=6000"], "RF": D4["RF"], "CNN": D4["CNN"]}
gap_art = {k: A7[("MLP", "LOFAR")]["SP"][0] - dm[k]["SP"][0] for k in fam4}
gap_mel = {k: s_mel_sp[0] - dm[k]["SP"][0] for k in fam4}
use("gap MLP LOFAR artigo - DEMON", gap_art)
use("gap SVM MEL local - DEMON", gap_mel)
best_demon = max(fam4, key=lambda k: dm[k]["SP"][0])

# ==============================================================================================
# Seção 5.7 / 10 — Fusão MEL + DEMON (grade do NB8)
# ==============================================================================================
G4 = {k: {m: S(v[0], m) for m in ("SP", "ACC", "R_SMALL", "R_LARGE", "R_BACKGROUND")} for k, v in GRID.items()}
G3 = {k: {m: S(v[1], m, mode="3c") for m in ("SP", "ACC", "R_SMALL", "R_LARGE")} for k, v in GRID.items() if v[1]}
FT_FUS = {}
for fam in ("MLP", "SVM", "RF", "CNN"):
    for rep in ("LOFAR", "MEL", "DEMON"):
        FT_FUS[(fam, rep)] = ftest(GRID[(fam, "FUSÃO")][0], GRID[(fam, rep)][0])
        FT_FUS[(fam, rep, "R_SMALL")] = ftest(GRID[(fam, "FUSÃO")][0], GRID[(fam, rep)][0], "R_SMALL")
p_rf_lofar_vs_fus = FT_FUS[("RF", "LOFAR")]


def t5_line(fam, rep, label_fam, label_rep):
    e4, e3, obs = GRID[(fam, rep)]
    g = G4[(fam, rep)]
    if e3:
        g3 = G3[(fam, rep)]
        c3 = f"{F(g3['ACC'])} | {F(g3['SP'])}"
        rs = f"{g['R_SMALL'][0]:.2f}% / {g3['R_SMALL'][0]:.2f}%"
    else:
        c3 = "— | —"
        rs = f"{g['R_SMALL'][0]:.2f}% / —"
    b = "**" if rep == "FUSÃO" else ""
    return f"| {b}{label_fam}{b} | {b}{label_rep}{b} | {F(g['ACC'])} | {F(g['SP'])} | {c3} | {rs} | `{e4}` ({obs}) |"


FAM_LAB = {"MLP": "MLP (PyTorch)", "SVM": "SVM Nyström", "RF": "Random Forest (100 árv.)", "CNN": "CNN 2D"}
REP_LAB = {"DEMON": "DEMON isolado", "MEL": "MEL isolado", "LOFAR": "LOFAR isolado (local)", "FUSÃO": "FUSÃO MULTIMODAL"}
t5_rows = []
for i, fam in enumerate(("MLP", "SVM", "RF", "CNN")):
    if i:
        t5_rows.append("| *---* | *---* | *---* | *---* | *---* | *---* | *---* | *---* |")
    for rep in ("DEMON", "MEL", "LOFAR", "FUSÃO"):
        t5_rows.append(t5_line(fam, rep, FAM_LAB[fam], REP_LAB[rep]))
T5_MD = "\n".join(t5_rows)
p_mlpfus_vs_mlplofar = FT_FUS[("MLP", "LOFAR")]
p_mlpfus_vs_mlpmel = FT_FUS[("MLP", "MEL")]
p_mlpfus_vs_mlpmel_small = FT_FUS[("MLP", "MEL", "R_SMALL")]
sm_lofar_range = (min(G4[(f, "LOFAR")]["R_SMALL"][0] for f in ("MLP", "SVM", "RF")),
                  max(G4[(f, "LOFAR")]["R_SMALL"][0] for f in ("MLP", "SVM", "RF")))
use("faixa recall SMALL LOFAR (MLP/SVM/RF)", sm_lofar_range)

# ==============================================================================================
# Tabela 3 — CPA (results/trainings/cpa_exp, mesma leitura do NB3)
# ==============================================================================================
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


CPA_MODELS = [("Forest Mel", "exp_1", "forest_window"), ("MLP Mel", "exp_1", "mlp_window"),
              ("CNN Mel", "exp_1", "cnn_image"), ("SVM Mel (m=4000, C=2.0)", "exp_1", "svm_window"),
              ("SVM LOFAR (m=4000, C=2.0, PCA64)", "lofar_exp_1", "svm_window")]
CPA = {}
CPA_NCLS = set()
for nome, sub, mod in CPA_MODELS:
    for tr in "AC":
        for te in "AC":
            rows = []
            for p in cpa_paths(sub, mod, tr, te):
                d = votos_csv(p)
                CPA_NCLS.add(d.Target.nunique())
                rows.append(R._metrics(d.Target.values, d.Prediction.values))
            CPA[(nome, tr, te)] = pd.DataFrame(rows)
use("nº de classes no CPA", sorted(CPA_NCLS))


def cs(nome, tr, te, met):
    return use(f"CPA {nome} treino {tr} teste {te} {met}", ms(CPA[(nome, tr, te)][met]))


t3 = []
for nome, *_ in CPA_MODELS:
    for tr, lab in (("A", "Dataset A (Near)"), ("C", "Dataset C (Far)")):
        t3.append(f"| **{nome}** | {lab} | {F(cs(nome, tr, 'A', 'SP'))} | {F(cs(nome, tr, 'A', 'ACC'))} | "
                  f"{F(cs(nome, tr, 'C', 'SP'))} | {F(cs(nome, tr, 'C', 'ACC'))} |")
T3_MD = "\n".join(t3)
QUEDA = {n: (CPA[(n, "A", "A")]["ACC"] - CPA[(n, "A", "C")]["ACC"]).values for n, *_ in CPA_MODELS}
QUEDA_M = {n: use(f"CPA queda ACC A->C {n}", float(np.mean(v))) for n, v in QUEDA.items()}
SVMM, SVML = "SVM Mel (m=4000, C=2.0)", "SVM LOFAR (m=4000, C=2.0, PCA64)"


def ft_arr(label, a, b):
    return use(label, R.f_test_5x2(np.asarray(a), np.asarray(b)))[1]


p_cpa_queda = ft_arr("F-test queda A->C SVM Mel vs MLP Mel", QUEDA[SVMM], QUEDA["MLP Mel"])
p_cpa_lofar_mlp_acc = ft_arr("F-test CPA A->A ACC SVM LOFAR vs MLP", CPA[(SVML, "A", "A")]["ACC"], CPA[("MLP Mel", "A", "A")]["ACC"])
p_cpa_lofar_mlp_sp = ft_arr("F-test CPA A->A SP SVM LOFAR vs MLP", CPA[(SVML, "A", "A")]["SP"], CPA[("MLP Mel", "A", "A")]["SP"])
p_cpa_ca_lofar_cnn = ft_arr("F-test CPA C->A ACC SVM LOFAR vs CNN", CPA[(SVML, "C", "A")]["ACC"], CPA[("CNN Mel", "C", "A")]["ACC"])
p_cpa_cc_svm_cnn = ft_arr("F-test CPA C->C ACC SVM Mel vs CNN", CPA[(SVMM, "C", "C")]["ACC"], CPA[("CNN Mel", "C", "C")]["ACC"])
p_cpa_ca_svm_cnn = ft_arr("F-test CPA C->A ACC SVM Mel vs CNN", CPA[(SVMM, "C", "A")]["ACC"], CPA[("CNN Mel", "C", "A")]["ACC"])

# ==============================================================================================
# Seção 9 / Tabela 4 — cascatas
# ==============================================================================================
def votos_sl(d, subset):
    out = {}
    for k in range(10):
        df = pd.read_csv(os.path.join(d, "eval", f"fold_{k}", f"{SL}_multiclass_{subset}.csv"), usecols=["File", "Prediction"])
        out[k] = {f: Counter(g).most_common(1)[0][0] for f, g in df.groupby("File")["Prediction"]}
    return out


def teste_no_treino_sl(d):
    trn = votos_sl(d, "trn")
    n = t = 0
    for k, df in files(SVM_LOFAR_6000).items():
        alvo = df[df.Target.isin([0, 2])]
        t += len(alvo)
        n += int(alvo.File.isin(list(trn[k])).sum())
    return n, t


SL_DIR_NEW = R.exp_dir(SL)
SL_DIR_OLD = os.path.join(SL_DIR_NEW, "20261020-174746")
VZ_OLD = use("arquivos SMALL/LARGE de teste no treino do SL antigo", teste_no_treino_sl(SL_DIR_OLD))
VZ_NEW = use("arquivos SMALL/LARGE de teste no treino do SL retreinado", teste_no_treino_sl(SL_DIR_NEW))
assert VZ_NEW[0] == 0


def cascata_arquivo_historica():
    """Mesma reconstrução do NB6 (cascata por arquivo com o especialista SL de jun/2026, vazado)."""
    sl = {s: votos_sl(SL_DIR_OLD, s) for s in ("trn", "val", "test")}
    rows = []
    for k, df in files(SVM_LOFAR_6000).items():
        pred = []
        for fid, p in zip(df.File, df.Prediction):
            if p == 2:
                v = next((sl[s][k][fid] for s in sl if fid in sl[s][k]), None)
                p = p if v is None else (0 if v == 0 else 2)
            pred.append(p)
        rows.append(R._metrics(df.Target.values, np.array(pred)))
    return pd.DataFrame(rows)


CASC_FILE_OLD_SP = use("cascata por arquivo (SL vazado) SP", ms(cascata_arquivo_historica()["SP"]))


def old_sp(name):
    if n_folds(name, OLD_CASC_DIR) < 10:
        return None
    return use(f"SP antigo (SL vazado) {name}", ms(folds(name, base=OLD_CASC_DIR)["SP"]))


def t4_line(label, exp, kw=None, s1=None, old=None, ref=True, bold=False):
    kw = kw or {}
    f = folds(exp, **kw)
    r = {m: use(f"{exp}{kw.get('suffix', '')} {m}", ms(f[m])) for m in ("R_SMALL", "R_MEDIUM", "R_LARGE", "R_BACKGROUND", "ACC", "SP")}
    p1 = "—" if s1 is None else P(use(f"F-test {exp} vs {s1}", R.f_test_5x2(f["SP"].values, folds(s1)["SP"].values))[1])
    pm = P(use(f"F-test {exp} vs mlp_lofar", R.f_test_5x2(f["SP"].values, folds(MLP_LOFAR)["SP"].values))[1]) if ref else "—"
    o = "—" if old is None else ("incompleto" if old == "inc" else F(old))
    b = "**" if bold else ""
    cells = " | ".join(F(r[m], 1) for m in ("R_SMALL", "R_MEDIUM", "R_LARGE", "R_BACKGROUND", "ACC")) + f" | {b}{F(r['SP'])}{b}"
    return f"| {b}{label}{b} | {cells} | {o} | {p1} | {pm} |", r


CASC_R = {}
t4 = []
for lab, e in (("Estágio 1: MEL (m=4000, C=2.0)", SVM_MEL), ("Estágio 1: Híbrido (m=4000, C=2.0, norm)", HYB_C2),
               ("Estágio 1: LOFAR (m=6000)", SVM_LOFAR_6000), ("MLP LOFAR (local, referência)", MLP_LOFAR)):
    line, CASC_R[e] = t4_line(lab, e, ref=(e != MLP_LOFAR))
    t4.append(line)
t4.append("| *Cascatas com o especialista SMALL×LARGE retreinado (20/10/2026)* | | | | | | | | | |")
for lab, new, old, s1, rota in CASC_JANELA:
    o = old_sp(old)
    line, CASC_R[new] = t4_line(f"{lab} (por janela)", new, s1=s1, old=o if o else "inc")
    t4.append(line)
line, CASC_R[CASC_FILE] = t4_line("LOFAR-MC(m=6000) → SL(PCA8) (por arquivo, `eval_cascade_by_ship.py`)", CASC_FILE,
                                  s1=SVM_LOFAR_6000, old=CASC_FILE_OLD_SP)
t4.append(line)
t4.append("| *Cascatas que não usam o especialista SMALL×LARGE (válidas; estágio 1 não registrado no config)* | | | | | | | | | |")
for lab, e in CASC_VALIDAS:
    line, CASC_R[e] = t4_line(lab, e)
    t4.append(line)
CT055 = f"svm_nystroem_4000_hybrid_cascade_pretrained_{SVM_LOFAR_6000}_elasticnet_l1r0.15"
line, CASC_R["ct0.55"] = t4_line("Cascata dupla com especialista LOFAR (t=0.55)", CT055, kw={"suffix": "_ct0.55"})
t4.append(line)
INV_OLD = old_sp(CASC_SL_INVALID)
t4.append(f"| ~~MEL→SM(PCA64)+SL(PCA8) (SMALL+MEDIUM)~~ **INVALIDADA** | — | — | — | — | — | — | {F(INV_OLD)} | — | — |")
T4_MD = "\n".join(t4)
BEST_J = max([c[1] for c in CASC_JANELA], key=lambda e: CASC_R[e]["SP"][0])
use("melhor cascata por janela (SL retreinado)", BEST_J)
P_CASC = {}
for _, e, _, s1, _ in CASC_JANELA:
    P_CASC[e] = (R.f_test_5x2(folds(e)["SP"].values, folds(s1)["SP"].values)[1],
                 R.f_test_5x2(folds(e)["SP"].values, folds(MLP_LOFAR)["SP"].values)[1])
P_CASC[CASC_FILE] = (R.f_test_5x2(folds(CASC_FILE)["SP"].values, folds(SVM_LOFAR_6000)["SP"].values)[1],
                     R.f_test_5x2(folds(CASC_FILE)["SP"].values, folds(MLP_LOFAR)["SP"].values)[1])
use("p (vs estágio 1, vs mlp_lofar) das cascatas", P_CASC)
P_MLP_MIN = min(v[1] for v in P_CASC.values())

# ==============================================================================================
# Seção 8 — landmarks k-means (experimentos que já existem em disco)
# ==============================================================================================
KM = [("MEL m=4000, landmarks k-means", "svm_nystroem_4000_log_melgram_kmeans_elasticnet_l1r0.15"),
      ("MEL m=4000, k-means estratificado balanceado", "svm_nystroem_4000_log_melgram_stratified_kmeans_balanced_elasticnet_l1r0.15")]
KM_ROWS = []
for lab, e in KM:
    try:
        sp = S(e, "SP")
    except FileNotFoundError:
        KM_ROWS.append(f"| {lab} | `{e}` | incompleto (< 10 folds) | — |")
        continue
    KM_ROWS.append(f"| {lab} | `{e}` | {F(sp)} | {P(ftest(e, SVM_MEL))} |")
KM_MD = "\n".join(KM_ROWS)


# ==============================================================================================
# Texto do relatório
# ==============================================================================================
def sig(p):
    return "diferença **significativa**" if p < 0.05 else "**sem diferença significativa**"


def f1n(x):
    return f"{x:.2f}"


a_mlpl, a_cnnl, a_cnnm, a_mlpm = A7[("MLP", "LOFAR")], A7[("CNN", "LOFAR")], A7[("CNN", "MEL")], A7[("MLP", "MEL")]
d_mlpl_art = use("|mlp_lofar SP - artigo|", abs(s_mlpl_sp[0] - a_mlpl["SP"][0]))
d_mlpl_art_acc = use("|mlp_lofar ACC - artigo|", abs(s_mlpl_acc[0] - a_mlpl["ACC"][0]))
L05 = SVM_LOFAR_C05
c_l09, c_m06, c_m09 = CUR[(L05, "")][0.9], CUR[(SVM_MEL, "")][0.6], CUR[(SVM_MEL, "")][0.9]
c_cnn09, c_mlp09, c_mlph09, c_mlpm06 = CUR[(CNN_MEL, "")][0.9], CUR[(MLP_MEL, "")][0.9], CUR[(MLP_HYB, "")][0.9], CUR[(MLP_MEL, "")][0.6]
c_l2_09 = CUR[(SVM_LOFAR, "")][0.9]
c_mlpl09 = CUR[(MLP_LOFAR, "")][0.9]
s_mlph_sp, s_mlph_acc = S(MLP_HYB, "SP"), S(MLP_HYB, "ACC")
s_hyb05_sp, s_hyb2_sp = S(HYB_C05, "SP"), S(HYB_C2, "SP")
p_mlph_hyb = ftest(MLP_HYB, HYB_C05)
p_mlph_mlpl = ftest(MLP_HYB, MLP_LOFAR)
p_hyb05_hyb2 = ftest(HYB_C05, HYB_C2)
mlph_vs_mlpm09 = {k: c_mlph09[k][0] > c_mlp09[k][0] for k in ("ACC", "ACC_micro", "SP")}
use("MLP híbrido > MLP MEL em t>=0.9 (ACC, micro, SP)", mlph_vs_mlpm09)
# Cobertura semelhante à da CNN em t>=0.9: limiar (entre os da Tabela 2) mais próximo para o SVM MEL
COB_CNN09 = c_cnn09["Cob"][0]
t_match_mel = min(LIM, key=lambda t: abs(CUR[(SVM_MEL, "")][t]["Cob"][0] - COB_CNN09))
c_mel_match = CUR[(SVM_MEL, "")][t_match_mel]
use("limiar do SVM MEL com cobertura mais próxima da CNN t>=0.9", t_match_mel)

g_mf, g_mm, g_ml, g_md = G4[("MLP", "FUSÃO")], G4[("MLP", "MEL")], G4[("MLP", "LOFAR")], G4[("MLP", "DEMON")]
g3_mf, g3_mm = G3[("MLP", "FUSÃO")], G3[("MLP", "MEL")]
g_sf, g3_sf, g_rf, g3_rf, g_cf = G4[("SVM", "FUSÃO")], G3[("SVM", "FUSÃO")], G4[("RF", "FUSÃO")], G3[("RF", "FUSÃO")], G4[("CNN", "FUSÃO")]
g_rl = G4[("RF", "LOFAR")]
best_small_fus = max(("MLP", "SVM", "RF"), key=lambda f: G4[(f, "FUSÃO")]["R_SMALL"][0])
CHANCE_CPA = 100 / max(CPA_NCLS)

cs_ = cs
mlp_aa, mlp_ac = cs_("MLP Mel", "A", "A", "ACC"), cs_("MLP Mel", "A", "C", "ACC")
svm_aa, svm_ac = cs_(SVMM, "A", "A", "ACC"), cs_(SVMM, "A", "C", "ACC")
cnn_cc, cnn_ca = cs_("CNN Mel", "C", "C", "ACC"), cs_("CNN Mel", "C", "A", "ACC")
svm_cc, svm_ca = cs_(SVMM, "C", "C", "ACC"), cs_(SVMM, "C", "A", "ACC")
lof_aa, lof_ca = cs_(SVML, "A", "A", "ACC"), cs_(SVML, "C", "A", "ACC")
lof_aa_sp, mlp_aa_sp = cs_(SVML, "A", "A", "SP"), cs_("MLP Mel", "A", "A", "SP")
MENOR_Q = min(QUEDA_M, key=QUEDA_M.get)
MAIOR_Q = max(QUEDA_M, key=QUEDA_M.get)
BEST_AA = max([n for n, *_ in CPA_MODELS], key=lambda n: cs_(n, "A", "A", "ACC")[0])
STD_C = {n: (cs_(n, "A", "C", "ACC")[1], cs_(n, "C", "C", "ACC")[1]) for n, *_ in CPA_MODELS}
queda_txt = "; ".join(f"{n}: {QUEDA_M[n]:.2f} p.p." for n, *_ in CPA_MODELS)
std_c_txt = "; ".join(f"{n}: {v[0]:.2f} (treino A) / {v[1]:.2f} (treino C)" for n, v in STD_C.items())

hy2 = CASC_R[HYB_C2]
rJ = {e: CASC_R[e] for _, e, *_ in CASC_JANELA}
JS, JR = CASC_JANELA[0][1], CASC_JANELA[1][1]
P_REF_MIN = min([R.f_test_5x2(folds(e)["SP"].values, folds(MLP_LOFAR)["SP"].values)[1]
                 for e in (SVM_MEL, HYB_C2, HYB_C05, SVM_LOFAR, SVM_LOFAR_6000, GRID[("MLP", "FUSÃO")][0])] + [P_MLP_MIN])
use("menor p vs mlp_lofar (SVMs, fusão MLP, cascatas)", P_REF_MIN)
best_lab = next(l for l, e, *_ in CASC_JANELA if e == BEST_J)
dem = dm
diffs_txt = ", ".join(f"{k} {gap_art[k]:.1f}" for k in fam4)
diffs_mel_txt = ", ".join(f"{k} {gap_mel[k]:.1f}" for k in fam4)
mlp3 = dm3.get("MLP")

MD = f"""# Estudo Comparativo de Classificação Acústica Submarina — Dataset IARA
**Abordagem Proposta: SVM Nyström Gaussiano com Penalização ElasticNet e Opção de Rejeição**

> **Atualização de 20/10/2026.** Todos os números deste relatório são gerados por
> `reports/atualizar_numeros_comparacao_modelos.py` a partir das predições por janela em disco
> (`results/trainings/{{tests,multirep,cpa_exp}}/`), com a mesma convenção do pipeline iara e de `notebooks/iara_results.py`:
> voto majoritário por arquivo (`Counter.most_common`), **ACC = acurácia balanceada** (média dos recalls), F1 macro,
> média ± desvio-padrão populacional (ddof=0) nos 10 folds e **F-test 5×2cv de Alpaydin** (o teste do artigo) para toda
> afirmação de "melhor/supera". Valores do artigo vêm da Tabela 7 (*Silva et al., 2025*). Versões anteriores deste
> relatório tinham números de cascatas com **vazamento** (especialista SMALL×LARGE; ver Seção 9), mistura de acurácia
> micro e balanceada, um "SP" calculado só como média geométrica (Tabela 2) e valores 3C sem experimento correspondente
> (Tabela 5); isso foi corrigido abaixo.

---

## 1. Introdução e Metodologia Experimental

Este documento apresenta uma análise comparativa entre o modelo proposto de **Support Vector Machines com Aproximação de Nyström (RBF-SVM)** e os modelos baselines do artigo de referência do dataset IARA (*Silva et al., 2025*): *Random Forest (RF)*, *Multi-Layer Perceptron (MLP)* e *Convolutional Neural Network (CNN)*.

### Metodologia de Validação Cruzada (5x2cv):
Adotou-se o protocolo de **Validação Cruzada 5x2 (5x2cv)**, totalizando 10 folds, sobre os {N_FILES_UO} áudios da coleção usada. A partição é estratificada e restrita no nível de **gravação de áudio (`ID`)** (`exclusive_header='ID'` em `src/iara/ml/experiment.py`), como no artigo: janelas de um mesmo arquivo nunca ficam simultaneamente em treino e teste. O isolamento **não** é por navio: gravações diferentes do mesmo navio podem cair em treino e teste (a auditoria `reports/auditoria_claude_iara.md` mediu que 63,5% das gravações de navio no teste têm o mesmo navio no treino). A antiga menção a "*Exclusive Ships on Test*" referia-se ao nome do parâmetro do código, não a um isolamento por navio.

---

## 2. Desempenho de Classificação Global (Cobertura de 100%)

A Tabela 1 consolida as métricas globais na partição de teste para as classes `SMALL`, `MEDIUM`, `LARGE` e `BACKGROUND`.

### Tabela 1: Métricas de Desempenho Geral no Conjunto de Teste (Sem Opção de Rejeição)

| Classificador / Arquitetura | Representação | Índice SP (%) | Acurácia Balanceada (ACC) (%) | F1-Score (Macro) (%) | Fonte |
| :--- | :--- | :---: | :---: | :---: | :--- |
{T1_MD}

> [!NOTE]
> *As incertezas são o desvio-padrão populacional entre os 10 folds. As linhas "Tabela 7 do artigo" são os valores publicados (sem resultados por fold em disco); todas as demais são recalculadas das predições em disco. O MLP LOFAR re-executado localmente (`{MLP_LOFAR}`: SP {F(s_mlpl_sp)}, ACC {F(s_mlpl_acc)}) difere do artigo em {d_mlpl_art:.2f} p.p. de SP e {d_mlpl_art_acc:.2f} p.p. de ACC. As linhas DEMON, HHT e de fusão vêm de scripts ad hoc (`results/trainings/multirep`), com vetores a cada 5,12 s (o MEL dessas linhas é o cache `mel5s`, com resolução temporal 10× menor que a das features oficiais de 0,512 s). As linhas marcadas **[COLAPSADA]** são CNNs ad hoc com folds em que uma classe tem recall ≈ 0 (SP baixo e desvio alto); não servem como baseline. Os SVM DEMON usam `max_iter=1000` sem early stopping. A coluna "F1" da versão anterior misturava F1 micro e acurácia; agora é F1 macro em todas as linhas.*

---

## 3. Análise da Variância Estatística e Sensibilidade Multiclasse

1. **Equivalência com a CNN, com menor desvio:** O **SVM (m=4000, "Golden MEL")** alcançou **{F(s_mel_sp)}%** de SP, próximo da CNN MEL do artigo (**{F(a_cnnm['SP'])}%**). Contra a re-execução local da CNN MEL nos mesmos folds (`{CNN_MEL}`: SP {F(s_cnn_sp)}), o F-test 5×2cv dá {P(p_mel_cnn)} em SP e {P(p_mel_cnn_acc)} em ACC: {sig(p_mel_cnn)}. O desvio entre folds do SVM é ≈ {ratio_std:.2f}× o da CNN do artigo (≈ {ratio_std ** 2:.2f}× a variância); a versão anterior dizia "metade da variância", mas a razão de ~0,5 é a dos **desvios**.
2. **PCA sem efeito significativo (MEL e LOFAR):** No MEL m=4000, o PCA64 dá SP {F(s_melpca_sp)} contra {F(s_mel_sp)} sem PCA ({P(p_mel_pca)}); no LOFAR m=4000, PCA64+ElasticNet dá SP {F(s_lofar_sp)} contra {F(s_lofar_nopca_sp)} sem PCA ({P(p_lofar_pca)} em SP; {P(p_lofar_pca_acc)} em ACC). O SVM LOFAR m=6000 **sem** PCA tem ACC {F(s_l6_acc)} contra {F(s_lofar_acc)} do m=4000 com PCA64 ({P(p_lofar_pca_6000)}). Portanto, as afirmações anteriores de que o PCA "degrada" o MEL ("dupla compressão") e é "essencial" no LOFAR ("filtro passa-faixa estatístico") **não são sustentadas pelos dados**; a escolha "MEL sem PCA / LOFAR com PCA64" é uma configuração adotada, não uma conclusão estatística.

---

## 4. Opção de Rejeição (Filtro de Confiança)

Propõe-se um **operador de decisão com opção de rejeição** baseado na concordância temporal das janelas. Seja um arquivo fatiado em $W$ janelas e $n_k$ o número de janelas votadas na classe $k$. A confiança do arquivo é $c(x) = \\max_k n_k / W$ (fração de janelas que votaram na classe vencedora); o arquivo é aceito se $c(x) \\ge t$. Nenhuma probabilidade/softmax é usada. Sobre os arquivos aceitos de cada fold calculam-se cobertura, ACC balanceada, acurácia micro e SP (fórmula do artigo); reporta-se média ± desvio (ddof=0).

> Diferenças em relação à versão anterior desta tabela (`scratch/scratch_rejection_10folds.py`): lá a coluna "ACC" era a acurácia **micro**, o "SP" era só a média geométrica dos recalls (não o SP do artigo) e o desvio usava ddof=1. As linhas "+ bias" (antes "Calibrado") somam um vetor fixo de bias aos escores de SMALL/MEDIUM (`scratch/scratch_generate_calibrated_csvs*.py`); **não** é calibração de probabilidade, e não está documentado neste relatório em que conjunto (validação ou teste) esses vetores foram escolhidos.

### Tabela 2: Curva de Trade-off Cobertura-Acurácia

| Limiar ($t$) | Modelo | Cobertura (%) | ACC balanceada (%) | Acurácia micro (%) | Índice SP (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
{T2_MD}

---

## 5. Análise dos Resultados de Rejeição e Discussão

As interpretações físicas abaixo (propagação, SNR, harmônicas) são **hipóteses** compatíveis com os números, não foram testadas diretamente.

### 5.1 LOFAR (SVM m=4000, C=0.5, PCA64)
* Sob $t \\ge 0.9$, a ACC balanceada nos arquivos aceitos é **{F(c_l09['ACC'])}%** (acurácia micro {F(c_l09['ACC_micro'])}%), com cobertura de **{F(c_l09['Cob'])}%**. Com C=2.0 (`{SVM_LOFAR}`) os valores são ACC balanceada {F(c_l2_09['ACC'])}%, acurácia micro {F(c_l2_09['ACC_micro'])}% e cobertura {F(c_l2_09['Cob'])}%. Os valores citados antes ("90,43%" e "22,27%") eram a acurácia **micro** e a cobertura deste modelo C=2.0, não a ACC balanceada.
* Hipótese: as harmônicas discretas do LOFAR se degradam com a distância/SNR, o que reduziria a concordância temporal e, portanto, a cobertura.

### 5.2 MEL (SVM m=4000, C=2.0)
* Sob maioria absoluta ($t \\ge 0.6$), o SVM MEL retém **{F(c_m06['Cob'])}%** dos arquivos com ACC balanceada de {F(c_m06['ACC'])}%. Em $t \\ge 0.9$, a ACC balanceada é {F(c_m09['ACC'])}% (cobertura {F(c_m09['Cob'])}%), abaixo do LOFAR no mesmo limiar, mas com cobertura maior.
* Hipótese: a integração em bandas Mel suaviza flutuações (mais cobertura) e funde harmônicas próximas (menos acurácia no topo de certeza).

### 5.3 CNN: índice de concordância mais grosseiro (não "superconfiança")
* Sob $t \\ge 0.9$, a CNN MEL local mantém cobertura de **{F(c_cnn09['Cob'])}%** com ACC balanceada de **{F(c_cnn09['ACC'])}%**. A comparação no mesmo $t$ não é justa: a concordância da CNN é calculada sobre {', '.join(map(str, NSEG_CNN))} segmentos por arquivo, enquanto os modelos por janela usam ≈ {NWIN_SVM:.0f} janelas (mediana). Entre os limiares da Tabela 2, o do SVM MEL com cobertura mais próxima da CNN em $t \\ge 0.9$ é $t \\ge {t_match_mel:.1f}$ (cobertura {F(c_mel_match['Cob'])}%, ACC {F(c_mel_match['ACC'])}%); as coberturas continuam diferentes, e a afirmação antiga de "desvantagem de 16% / 9%" da CNN comparava acurácias micro em coberturas distintas.
* Como nenhuma probabilidade é usada, estes dados **não** permitem conclusões sobre calibração ou "superconfiança" da Softmax; as afirmações anteriores nesse sentido (incl. "estimador de incerteza infinitamente mais calibrado") foram removidas.

### 5.4 MLP MEL (re-execução local)
* Sob $t \\ge 0.9$, a ACC balanceada do MLP MEL é **{F(c_mlp09['ACC'])}%** (cobertura {F(c_mlp09['Cob'])}%; SVM MEL: {F(c_m09['ACC'])}% com cobertura {F(c_m09['Cob'])}%). Os modelos aceitam conjuntos de arquivos diferentes, então a diferença de acurácia residual não é comparável diretamente nem foi testada. A afirmação anterior de "calibração superior" do MLP não é sustentada (nenhuma probabilidade é usada).

### 5.5 Fusão espectral híbrida no MLP (Mel + LOFAR-PCA64)
* **Desempenho geral:** O MLP Híbrido (`{MLP_HYB}`) atingiu **{F(s_mlph_sp)}%** de SP e **{F(s_mlph_acc)}%** de ACC. Contra o SVM Híbrido C=0.5 (SP {F(s_hyb05_sp)}): {P(p_mlph_hyb)}, {sig(p_mlph_hyb)}; contra o MLP LOFAR local: {P(p_mlph_mlpl)}, {sig(p_mlph_mlpl)}.
* **Rejeição:** Sob $t \\ge 0.9$, o MLP Híbrido tem ACC balanceada de {F(c_mlph09['ACC'])}%, cobertura de {F(c_mlph09['Cob'])}% e SP de {F(c_mlph09['SP'])}%. Em relação ao MLP MEL no mesmo limiar, a média é {'maior' if mlph_vs_mlpm09['ACC'] else 'menor'} em ACC, {'maior' if mlph_vs_mlpm09['ACC_micro'] else 'menor'} em acurácia micro e {'maior' if mlph_vs_mlpm09['SP'] else 'menor'} em SP (sem teste de significância).

### 5.6 O Extrator DEMON (SVM, RF, MLP e CNN)
O DEMON (253 bins de 0,39 a 25 Hz, um vetor a cada 5,12 s) foi avaliado nos 10 folds oficiais em quatro famílias:
* **MLP [32,16] (25 ép.):** SP {F(dem['MLP']['SP'])}%, ACC {F(dem['MLP']['ACC'])}%. Contra o SVM m=6000 (SP {F(dem['SVM m=6000']['SP'])}): {P(p_mlp_vs_svm_demon)}, {sig(p_mlp_vs_svm_demon)}; contra o RF (SP {F(dem['RF']['SP'])}): {P(p_mlp_vs_rf_demon)}, {sig(p_mlp_vs_rf_demon)}. Recall de BACKGROUND {dem['MLP']['R_BACKGROUND'][0]:.2f}%; fração de SMALL predita como LARGE: {sm_to_l_svm[0]:.1f}% no SVM m=6000 e {sm_to_l_mlp[0]:.1f}% no MLP.{f" Com o modelo retreinado em 3 classes (`{D3['MLP']}`), SP {F(mlp3['SP'])}% (SMALL {mlp3['R_SMALL'][0]:.1f}%, MEDIUM {mlp3['R_MEDIUM'][0]:.1f}%, LARGE {mlp3['R_LARGE'][0]:.1f}%)." if mlp3 else ""}
* **Random Forest:** SP {F(dem['RF']['SP'])}%; recall de SMALL {dem['RF']['R_SMALL'][0]:.2f}% (4 classes){f" e {dm3['RF']['R_SMALL'][0]:.2f}% (3 classes, retreinado)" if 'RF' in dm3 else ""}; {bg_to_s_rf[0]:.1f}% dos arquivos de BACKGROUND são preditos SMALL e o recall de BACKGROUND é {dem['RF']['R_BACKGROUND'][0]:.2f}%.
* **CNN 2D ad hoc:** SP {F(dem['CNN']['SP'])}% — **colapsada** (desvio alto, folds com classe zerada). O resultado indica falha de treino dessa CNN ad hoc; não permite concluir que "convoluções espaciais são inadequadas" para o DEMON.
* **Comparação com MEL/LOFAR:** o DEMON isolado fica abaixo do MLP LOFAR do artigo ({a_mlpl['SP'][0]:.2f}%) em {diffs_txt} p.p. e abaixo do SVM MEL local ({s_mel_sp[0]:.2f}%) em {diffs_mel_txt} p.p. Isso indica que o DEMON sozinho é uma representação mais fraca nestes modelos; não "comprova" um teto teórico. Detalhes no Notebook 5, em `reports/decisao_metodo_demon.md` e em `reports/comparacao_modelos_demon.md`.

### 5.7 Fusão Multimodal Direta (MEL + DEMON)
A combinação do MEL (256 filtros, cache de 5,12 s) com o DEMON (253 bins) foi testada no protocolo $5\\times2$cv (particionamento por gravação, {N_FILES_UO} áudios):
* **MLP Fusão** (`{GRID[('MLP', 'FUSÃO')][0]}`, rede [64,32], **maior** que a [32,16] das entradas isoladas): SP **{F(g_mf['SP'])}%**, ACC **{F(g_mf['ACC'])}%**. Contra o MLP LOFAR local (SP {F(g_ml['SP'])}): {P(p_mlpfus_vs_mlplofar)}, {sig(p_mlpfus_vs_mlplofar)}; contra o MLP MEL 5,12 s: {P(p_mlpfus_vs_mlpmel)}. O MLP LOFAR do artigo é {F(a_mlpl['SP'])}% e a CNN LOFAR do artigo {F(a_cnnl['SP'])}%; como esses valores não têm folds em disco, **não** é possível afirmar que a fusão os "supera" nem que estabelece "recorde".
* **Recall de SMALL:** MLP MEL 5,12 s: {g_mm['R_SMALL'][0]:.2f}% (4 classes) / {g3_mm['R_SMALL'][0]:.2f}% (3 classes); MLP Fusão: {g_mf['R_SMALL'][0]:.2f}% / {g3_mf['R_SMALL'][0]:.2f}% ({g_mf['R_SMALL'][0] - g_mm['R_SMALL'][0]:+.2f} p.p. / {g3_mf['R_SMALL'][0] - g3_mm['R_SMALL'][0]:+.2f} p.p.; F-test do recall SMALL 4c: {P(p_mlpfus_vs_mlpmel_small)}). BACKGROUND na fusão: {g_mf['R_BACKGROUND'][0]:.2f}%; LARGE: {g_mf['R_LARGE'][0]:.2f}% (4c) / {g3_mf['R_LARGE'][0]:.2f}% (3c).
* **Desvio entre folds:** SP do MLP MEL 5,12 s ±{g_mm['SP'][1]:.2f} vs. ±{g_mf['SP'][1]:.2f} na fusão. Uma diferença de desvio em 10 folds não "comprova" complementaridade ortogonal.

---

## 6. Conclusões e Delineamento Operacional

Com os números atuais, o desenho em duas camadas continua sendo uma **proposta operacional**, não um resultado estatisticamente demonstrado:

1. **Camada de cobertura (SVM MEL com $t \\ge 0.6$):** aceita {F(c_m06['Cob'])}% dos arquivos com ACC balanceada de {F(c_m06['ACC'])}% (MLP MEL no mesmo limiar: cobertura {F(c_mlpm06['Cob'])}%, ACC {F(c_mlpm06['ACC'])}%).
2. **Camada de alta certeza ($t \\ge 0.9$):** SVM LOFAR (C=0.5, PCA64) com ACC balanceada de {F(c_l09['ACC'])}% sobre {F(c_l09['Cob'])}% dos arquivos; MLP MEL com {F(c_mlp09['ACC'])}% sobre {F(c_mlp09['Cob'])}%; MLP LOFAR com {F(c_mlpl09['ACC'])}% sobre {F(c_mlpl09['Cob'])}%. A afirmação anterior de "25,6% de falsos alarmes" da CNN e de "alarmes falsos catastróficos" não tem base nestes dados (ver 5.3).
3. **Sem rejeição**, nenhum dos SVM MEL/LOFAR/Híbrido, a fusão MLP ou as cascatas re-executadas tem SP significativamente diferente do MLP LOFAR local (maior SP sem vazamento entre as cascatas por janela: {F(rJ[BEST_J]['SP'])}, {P(P_CASC[BEST_J][1])}; menor p desse conjunto: {P(P_REF_MIN)}). O melhor baseline do artigo (MLP LOFAR, {F(a_mlpl['SP'])}) não foi superado com significância.

---

## 7. Robustez ao CPA e Experimentos de Proximidade (Tabela 10 do Artigo)

Replicou-se o experimento de **CPA Proximity** da Tabela 10 do artigo (*Silva et al., 2025*):
* **Dataset A (Near CPA):** trechos próximos do ponto de máxima aproximação (maior SNR).
* **Dataset C (Far CPA):** trechos com a embarcação distante (menor SNR).

1. **Trained on A:** treino em A, avaliação em A e C.
2. **Trained on C:** treino em C, avaliação em A e C.

### Tabela 3: Desempenho no Experimento de CPA Proximity (re-execução local, `results/trainings/cpa_exp`)

| Modelo / Classificador | Treinado em | SP A (%) | ACC A (%) | SP C (%) | ACC C (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
{T3_MD}

*Todas as linhas são re-execuções locais (Forest/MLP/CNN com os modelos do artigo; não são os valores publicados na Tabela 10). ACC = acurácia balanceada. O problema tem {max(CPA_NCLS)} classes (acaso da ACC balanceada = {CHANCE_CPA:.1f}%).*

### Conclusões e Análise das Métricas de Proximidade:

1. **Queda de ACC ao passar de A para C (treino em A):** {queda_txt}. A menor queda é a do **{MENOR_Q}** e a maior a do **{MAIOR_Q}**. A diferença de queda entre o SVM Mel e o MLP Mel tem {P(p_cpa_queda)} ({sig(p_cpa_queda)}); a afirmação anterior de que o SVM "deforma-se de maneira muito mais resiliente" não é sustentada como diferença estatística.
2. **Treino em C (Far):** a CNN Mel atinge {F(cnn_cc)}% em C e {F(cnn_ca)}% em A; o SVM Mel atinge {F(svm_cc)}% em C e {F(svm_ca)}% em A (F-test do ACC: {P(p_cpa_cc_svm_cnn)} em C, {P(p_cpa_ca_svm_cnn)} em A). A frase anterior de que {cnn_ca[0]:.2f}% estaria "próximo do limiar aleatório" era exagerada: o acaso da ACC balanceada é {CHANCE_CPA:.1f}% ({max(CPA_NCLS)} classes), e a CNN fica {cnn_ca[0] - CHANCE_CPA:.1f} p.p. acima dele. A vantagem antiga "de mais de 4% em ambas as frentes" do SVM sobre a CNN não é significativa.
3. **Desvio entre folds no Dataset C (ACC):** {std_c_txt}.
4. **SVM LOFAR:** em A→A, o SVM LOFAR tem ACC {F(lof_aa)}% contra {F(mlp_aa)}% do MLP Mel ({P(p_cpa_lofar_mlp_acc)}); em SP, {F(lof_aa_sp)}% contra {F(mlp_aa_sp)}% ({P(p_cpa_lofar_mlp_sp)}). O modelo com maior ACC A→A é o {BEST_AA}, mas sem diferença significativa para o MLP; o "recorde absoluto" anterior não é sustentado. Em C→A, o SVM LOFAR tem {F(lof_ca)}% contra {F(cnn_ca)}% da CNN Mel ({P(p_cpa_ca_lofar_cnn)}).

---

## 8. Landmarks via MiniBatchKMeans

A proposta de substituir a amostragem aleatória de landmarks do Nyström por centroides de `MiniBatchKMeans` já tem experimentos em disco (MEL, m=4000):

| Variante | Experimento | SP (%) | F-test vs. SVM MEL aleatório (SP {F(s_mel_sp)}) |
| :--- | :--- | :---: | :---: |
{KM_MD}

As expectativas listadas antes (mesma acurácia com $m$ menor, melhor cobertura de `SMALL`) eram hipóteses sem referência bibliográfica nem experimento; ficam como hipóteses.

---

## 9. Arquitetura em Cascata (Duplo Especialista)

Sistema em dois estágios:

1. **Estágio 1 (multiclasse):** SVM Nyström em 4 classes — Híbrido MEL+LOFAR (m=4000, C=2.0, norm; recall de LARGE {hy2['R_LARGE'][0]:.1f}%), MEL (m=4000) ou LOFAR (m=6000), conforme a variante.
2. **Estágio 2 (especialistas binários m=6000):**
   * predições `SMALL`/`MEDIUM` → **Especialista SM** (MEL-256 + LOFAR-PCA64; `{SM}`);
   * predições `LARGE` → **Especialista SL** (MEL-256 + LOFAR-PCA8; `{SL}`).
   * Roteamento "runner-up": o SL só é acionado quando a 2ª classe mais votada é SMALL.

> ⚠️ **Vazamento corrigido.** O especialista SL de junho/2026 usava um `StratifiedKFold` só sobre {{SMALL, LARGE}}: **{VZ_OLD[0]}/{VZ_OLD[1]} = {100 * VZ_OLD[0] / VZ_OLD[1]:.1f}%** dos arquivos SMALL/LARGE de teste estavam no **treino** do especialista do mesmo fold. Ele foi **retreinado em 20/10/2026** (agora {VZ_NEW[0]}/{VZ_NEW[1]}) e todas as cascatas que o usam foram re-executadas. As versões antigas estão em `results/trainings/_cascatas_com_sl_vazado_pre20261020/` e aparecem só na coluna "SP antes (SL vazado)". A cascata `..._smallalso_large_{SL}` (sem `fbmc`, versão antiga do `svm.py`, SP antes {F(INV_OLD)}) **não foi re-executada e está invalidada**. O especialista SM não tem vazamento.

#### Tabela 4: Cascatas (recalls, ACC balanceada e SP; p = F-test 5×2cv do SP)

| Variante | Recall SMALL | Recall MEDIUM | Recall LARGE | Recall BG | ACC | SP | SP antes (SL vazado) | p vs. estágio 1 | p vs. MLP LOFAR local |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{T4_MD}

### Discussão dos Resultados da Cascata:
1. **Sem vazamento, os ganhos encolhem:** a variante "sempre" com estágio 1 híbrido caiu de {F(old_sp(CASC_JANELA[0][2]))} para **{F(rJ[JS]['SP'])}**; a "runner-up", de {F(old_sp(CASC_JANELA[1][2]))} para **{F(rJ[JR]['SP'])}**; a cascata por arquivo, de {F(CASC_FILE_OLD_SP)} para **{F(CASC_R[CASC_FILE]['SP'])}**.
2. **Melhor cascata por janela:** {best_lab}, SP **{F(rJ[BEST_J]['SP'])}** (recall SMALL {rJ[BEST_J]['R_SMALL'][0]:.1f}%, LARGE {rJ[BEST_J]['R_LARGE'][0]:.1f}%). Contra o próprio estágio 1: {P(P_CASC[BEST_J][0])} ({sig(P_CASC[BEST_J][0])}); contra o MLP LOFAR local ({F(s_mlpl_sp)}): {P(P_CASC[BEST_J][1])}.
3. **Nenhuma** cascata re-executada difere significativamente do MLP LOFAR local (menor p: {P(P_MLP_MIN)}); os SP vão de {min(min(rJ[e]['SP'][0] for e in rJ), CASC_R[CASC_FILE]['SP'][0]):.2f} a {max(max(rJ[e]['SP'][0] for e in rJ), CASC_R[CASC_FILE]['SP'][0]):.2f}, contra {F(a_mlpl['SP'])} do MLP LOFAR do artigo. A "melhor" cascata foi escolhida olhando o teste (entre 4 variantes), o que torna otimista sua vantagem sobre o estágio 1. As afirmações anteriores "melhor arquitetura geral" (68,05%) e "empata com a cascata MEL pura (67,59 vs 67,58)" vinham do especialista vazado e foram removidas.
4. **"Sempre" vs. "runner-up":** com o SL retreinado, "sempre" tem SMALL {rJ[JS]['R_SMALL'][0]:.1f}% e LARGE {rJ[JS]['R_LARGE'][0]:.1f}%; "runner-up" tem SMALL {rJ[JR]['R_SMALL'][0]:.1f}% e LARGE {rJ[JR]['R_LARGE'][0]:.1f}%.
5. **SP fold-wise:** $SP = \\sqrt{{\\bar{{r}} \\cdot \\tilde{{r}}}} \\times 100$ ($\\bar{{r}}$ = média aritmética dos recalls, $\\tilde{{r}}$ = média geométrica), calculado por fold e reportado como média ± desvio (ddof=0) sobre os 10 folds.

---

## 10. Fusão Multimodal Direta (MEL + DEMON) entre Famílias de Modelos

Avaliou-se o espaço de **509 dimensões** (256 Mel + 253 DEMON, janelas de 5,12 s sincronizadas) nas famílias do benchmark, sob o protocolo $5\\times2$cv com particionamento por gravação (`exclusive_header='ID'`), mesma grade do Notebook 8.

### Tabela 5: Classificadores na Fusão Multimodal (5x2cv)

| Arquitetura / Modelo | Representação | ACC 4C (%) | SP 4C (%) | ACC 3C (%) | SP 3C (%) | Recall SMALL (4C / 3C) | Experimento |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
{T5_MD}

*"3C" = modelo **retreinado** sem BACKGROUND (`*_3c`). "—" = não há experimento 3C em disco para essa linha (os valores 3C que a versão anterior trazia para MEL/LOFAR — p.ex. 65,50 / 64,20 no MLP LOFAR — não correspondiam a nenhum experimento e foram removidos). As linhas "LOFAR isolado" agora são re-execuções locais (não a Tabela 7 do artigo). O RF e a CNN LOFAR usam o cache LOFAR de 5,12 s; a CNN LOFAR/DEMON/FUSÃO ad hoc está colapsada.*

### 10.1 A fusão MEL + DEMON empata com o LOFAR?

1. **F-test 5×2cv (SP), FUSÃO vs. LOFAR isolado:** MLP {P(FT_FUS[('MLP', 'LOFAR')])}; SVM {P(FT_FUS[('SVM', 'LOFAR')])}; RF {P(FT_FUS[('RF', 'LOFAR')])}. No RF, o LOFAR local (SP {F(g_rl['SP'])}) é {'maior' if g_rl['SP'][0] > g_rf['SP'][0] else 'menor'} que a fusão (SP {F(g_rf['SP'])}). Os modelos são, portanto, comparáveis, mas a afirmação de que a fusão "supera" o LOFAR não se sustenta. A interpretação de que MEL + DEMON "recompõe exatamente os mesmos fenômenos físicos que o LOFAR" é uma hipótese, não algo medido aqui. (O MEL e o LOFAR cobrem 0–8 kHz após a reamostragem a 16 kHz; o DEMON cobre 0,39–25 Hz.)
2. **Recall de SMALL:** no LOFAR isolado (MLP/SVM/RF) fica entre {sm_lofar_range[0]:.1f}% e {sm_lofar_range[1]:.1f}%; na fusão é {g_mf['R_SMALL'][0]:.2f}% (MLP), {g_sf['R_SMALL'][0]:.2f}% (SVM, 4C) / {g3_sf['R_SMALL'][0]:.2f}% (SVM, 3C) e {g_rf['R_SMALL'][0]:.2f}% (RF). O maior recall de SMALL na fusão em 4C é o do {best_small_fus}. A explicação pela frequência de pás ("15 a 25 Hz") é uma hipótese.
3. **CNN 2D:** a CNN ad hoc na fusão tem SP {F(g_cf['SP'])} — colapso de treino (folds com recall nulo). A hipótese de que a descontinuidade entre as bandas Mel e DEMON prejudica as convoluções não foi testada.
"""


if __name__ == "__main__":
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(MD)
    print(f"Escrito: {OUT_MD}")
    print("\n=== SP antigo (linha do .md anterior) -> SP recalculado ===")
    for line, lab, old, new in DIFFS:
        flag = "  <-- muda > 0,1 p.p." if abs(old - new) > 0.1 else ""
        print(f"L{line}: {lab}: {old:.2f} -> {new:.2f}{flag}")
    print("\n=== VALORES USADOS ===")
    for lab, v in USED:
        print(f"{lab}: {v}")
