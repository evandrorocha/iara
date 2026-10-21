"""
Funções comuns aos notebooks 24 (Gating Front-End) e 25 (falhas do Super-Oráculo).

Reproduz exatamente o roteador de scratch/test_competence_gating_all_models.py,
scratch/test_competence_gating_svm.py e scratch/run_heterogeneous_gating_experiment.py
(mesmas predições por arquivo de data/iara_processed/{rf,mlp,svm,cnn}_all_representations_predictions.pkl,
mesmo RF de competência treinado só no fold complementar `fold ^ 1`, mesmo prior de SP), mas devolve o
SP POR FOLD, necessário para o F-test 5x2cv e para a análise de sensibilidade aos hiperparâmetros.

Observação: as predições por arquivo destes pickles usam voto majoritário com desempate pela MENOR classe
(np.argmax(np.bincount)); notebooks/iara_results.py usa Counter.most_common (empate -> 1ª janela), como o
pipeline iara. Os dois diferem só nos arquivos empatados (ver a tabela de especialistas do NB24).
"""
import itertools
import numbers
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def _root():
    for p in [".", ".."]:
        if os.path.isdir(os.path.join(p, "data", "iara_processed")):
            return p
    raise FileNotFoundError("data/iara_processed não encontrado (rode na raiz do repositório ou em notebooks/)")


ROOT = _root()
PROC = os.path.join(ROOT, "data", "iara_processed")
REPS = ["mel", "lofar", "hht", "demon"]
FAMS = {"RF": "rf", "MLP": "mlp", "SVM": "svm", "CNN": "cnn"}
FEATS = ["crest_factor", "spectral_flatness", "mod_index", "flux_std", "high_low_ratio", "rms_db", "zcr"]
CLASS_NAMES = ["Small", "Medium", "Large", "Background"]

# Experimento em disco (results/trainings/...) que originou cada especialista dos pickles.
EXPERTS_EXP = {
    ("RF", "mel"): "rf_mel5s_n100_balanced_4c",
    ("RF", "lofar"): "rf_lofar5s_n100_balanced_4c",
    ("RF", "hht"): "rf_hht_n100_balanced_4c",
    ("RF", "demon"): "rf_demon_n100_balanced_4c",
    ("MLP", "mel"): "mlp",
    ("MLP", "lofar"): "mlp_lofar",
    ("MLP", "hht"): "mlp_hht_h32-16_e30_4c",
    ("MLP", "demon"): "mlp_demon_h32-16_e30_4c",
    ("SVM", "mel"): "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0",
    ("SVM", "lofar"): "svm_nystroem_6000_lofar_elasticnet_l1r0.15",
    ("SVM", "hht"): "svm_nystroem_2000_hht_C2.0_elasticnet_l1r0.15_4c",
    ("SVM", "demon"): "svm_nystroem_2000_demon_C2.0_elasticnet_l1r0.15_4c",
    ("CNN", "mel"): "cnn",
    ("CNN", "lofar"): "cnn2d_lofar5s_w32_b64_e15_4c",
    ("CNN", "hht"): "cnn2d_hht_w32_b64_e20_4c",
    ("CNN", "demon"): "cnn2d_demon_w32_b64_e20_4c",
}
EXPERTS_NOTE = {
    "RF": "scratch/generate_rf_predictions_and_analysis.py: 100 árvores, class_weight='balanced', sem max_depth "
          "(o artigo usa 50/30 e 200/20); MEL e LOFAR vêm dos caches com média em 5,12 s (resolução 10× menor que a do artigo)",
    "MLP": "MEL e LOFAR: pipeline iara oficial (results/trainings/tests/mlp e mlp_lofar, com early stopping); "
           "HHT e DEMON: MLP ad hoc [32,16], 30 épocas fixas, sem validação (scratch/generate_mlp_predictions_and_analysis.py)",
    "SVM": "MEL: Nyström m=4000; LOFAR: m=6000 (pipeline oficial); HHT e DEMON: m=2000 (exportação em nível de arquivo)",
    "CNN": "MEL: CNN do pipeline oficial (results/trainings/tests/cnn); LOFAR/HHT/DEMON: CNN 2D ad hoc "
           "(15–20 épocas, sem early stopping; LOFAR do cache de 5,12 s)",
}
COLLAPSE_STD = 5.0  # desvio do SP entre folds acima do qual o especialista é sinalizado como instável/colapsado

PREDS = {k: pickle.load(open(os.path.join(PROC, f"{v}_all_representations_predictions.pkl"), "rb")) for k, v in FAMS.items()}
META = pickle.load(open(os.path.join(PROC, "cache_gating_metafeatures.pkl"), "rb"))
GRID = list(itertools.product([3, 5, None], [1, 5], [0.0, 0.1, 0.3]))  # max_depth x min_samples_leaf x peso do prior
REPORTED_CFG = (3, 5, 0.1)
FIXED_HMOE = [("RF", "mel"), ("MLP", "lofar"), ("RF", "hht"), ("MLP", "demon")]
ALL16 = [(m, r) for m in ["RF", "SVM", "MLP", "CNN"] for r in REPS]


# ----------------------------------------------------------------------------------------- formatação
def br(x, nd=2):
    """Número no formato brasileiro: 1.234,56 (inteiros: 1.234)."""
    if isinstance(x, numbers.Integral) or nd == 0:
        return f"{int(round(x)):,}".replace(",", ".")
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def p_txt(p):
    return "p < 0,001" if p < 0.001 else f"p = {br(p, 2)}"


def pm(a, nd=2):
    return f"{br(float(np.mean(a)), nd)} ± {br(float(np.std(a)), nd)}"


# ----------------------------------------------------------------------------------------- métricas
def sp_rec(y, p):
    """SP (como src/iara/ml/metrics.py e os scripts do gate) e recall por classe 0..3, em %."""
    rec = np.array([np.mean(p[y == c] == c) for c in range(4)])
    sp = 0.0 if np.any(rec == 0) else float(np.sqrt(rec.mean() * scipy.stats.gmean(rec)) * 100)
    return sp, rec * 100


def f_test_5x2(a, b):
    """F-test 5x2cv combinado (Alpaydin) sobre o SP por fold (folds 2r e 2r+1 = repetição r)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d1, d2 = a[0::2] - b[0::2], a[1::2] - b[1::2]
    m = (d1 + d2) / 2
    s2 = (d1 - m) ** 2 + (d2 - m) ** 2
    num = np.sum(d1 ** 2) + np.sum(d2 ** 2)
    if num == 0:
        return 0.0, 1.0
    f = num / (2 * np.sum(s2))
    return float(f), float(1 - scipy.stats.f.cdf(f, 10, 5))


def fold_arrays(fold):
    t = PREDS["RF"]["targets"][fold]
    fids = sorted(t)
    return fids, np.array([t[f] for f in fids])


def experts_preds(experts, fold, fids):
    """experts: lista de pares (família, representação). Devolve matriz n_especialistas x n_arquivos."""
    return np.array([[PREDS[m][r][fold][f] for f in fids] for m, r in experts])


def single_folds(fam, rep):
    """SP e recalls por fold de um especialista isolado (predições por arquivo do pickle)."""
    sps, recs = [], []
    for k in range(10):
        fids, y = fold_arrays(k)
        s, r = sp_rec(y, experts_preds([(fam, rep)], k, fids)[0])
        sps.append(s)
        recs.append(r)
    return np.array(sps), np.mean(recs, axis=0)


# ----------------------------------------------------------------------------------------- roteador
def route(experts, fold, depth=3, leaf=5, w_prior=0.10, seed=42):
    """Roteia o fold de teste `fold` com o roteador treinado SÓ no fold complementar fold ^ 1."""
    fids, y = fold_arrays(fold)
    trn = fold ^ 1
    tf, _ = fold_arrays(trn)
    assert not set(fids) & set(tf), "vazamento entre fold e fold ^ 1"
    tf = [f for f in tf if f in META]
    ty = np.array([PREDS["RF"]["targets"][trn][f] for f in tf])
    TP = experts_preds(experts, trn, tf)
    X = np.array([[META[f][c] for c in FEATS] for f in tf])
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    prior = np.array([sp_rec(ty, TP[m])[0] / 100 for m in range(len(experts))])
    Xt = sc.transform(np.array([[META[f][c] if f in META else 0.0 for c in FEATS] for f in fids]))
    pr = np.column_stack([
        RandomForestClassifier(n_estimators=50, max_depth=depth, min_samples_leaf=leaf, random_state=seed)
        .fit(Xs, (TP[m] == ty).astype(int)).predict_proba(Xt)[:, 1] for m in range(len(experts))])
    r = np.argmax((1 - w_prior) * pr + w_prior * prior[None, :], axis=1)
    P = experts_preds(experts, fold, fids)
    return y, P[r, np.arange(len(fids))], r


def gate_folds(experts, **kw):
    """SP por fold, recalls médios e contagem de rotas do gate."""
    sps, recs, routes = [], [], np.zeros(len(experts), int)
    for k in range(10):
        y, p, r = route(experts, k, **kw)
        s, rc = sp_rec(y, p)
        sps.append(s)
        recs.append(rc)
        routes += np.bincount(r, minlength=len(experts))
    return np.array(sps), np.mean(recs, axis=0), routes


def homog(fam):
    return [(fam, r) for r in REPS]


def nested_auto_experts(fold, candidates=("RF", "MLP", "SVM")):
    """Especialista de maior SP por representação no fold ^ 1 (como run_heterogeneous_gating_experiment.py)."""
    trn = fold ^ 1
    tf, _ = fold_arrays(trn)
    tf = [f for f in tf if f in META]
    ty = np.array([PREDS["RF"]["targets"][trn][f] for f in tf])
    return [(max(candidates, key=lambda m: sp_rec(ty, experts_preds([(m, r)], trn, tf)[0])[0]), r) for r in REPS]


def hmoe_auto_folds(**kw):
    sps = []
    for k in range(10):
        y, p, _ = route(nested_auto_experts(k), k, **kw)
        sps.append(sp_rec(y, p)[0])
    return np.array(sps)


# ----------------------------------------------------------------------------------------- oráculos
def oracle(experts):
    """Oráculo (acerta se ALGUM especialista acerta; usa o rótulo, não é realizável) e o esperado se os
    erros fossem independentes, recall_c = 1 - prod_i(1 - r_ic). Médias sobre os folds."""
    sps, obs, exp = [], [], []
    for k in range(10):
        fids, y = fold_arrays(k)
        H = experts_preds(experts, k, fids) == y[None, :]
        sps.append(sp_rec(y, np.where(H.any(0), y, -1))[0])
        obs.append([H[:, y == c].any(0).mean() * 100 for c in range(4)])
        exp.append([(1 - np.prod(1 - H[:, y == c].mean(1))) * 100 for c in range(4)])
    obs, exp = np.mean(obs, 0), np.mean(exp, 0)
    e = exp / 100
    sp_exp = float(np.sqrt(e.mean() * scipy.stats.gmean(e)) * 100)
    return {"sp": np.array(sps), "rec_obs": obs, "rec_exp": exp, "sp_exp": sp_exp}


def random_oracle_sp(n):
    """SP do oráculo de n preditores uniformemente aleatórios em 4 classes: recall = 1 - (3/4)^n em todas."""
    return (1 - 0.75 ** n) * 100


# ----------------------------------------------------------------------------------------- dataset
def dataset_counts():
    t = PREDS["RF"]["targets"]
    c = Counter(f for k in range(10) for f in t[k])
    others = []
    for k in range(10):
        o = set().union(*[set(t[j]) for j in range(10) if j // 2 != k // 2])
        others.append(len(set(t[k]) & o) / len(t[k]) * 100)
    return {"n_uo": len(c), "n_fold": len(t[0]), "n_par": len(set(t[0]) | set(t[1])),
            "n_aval": sum(c.values()), "aval_por_arq": float(np.mean(list(c.values()))),
            "overlap_outras_rep": float(np.mean(others)), "n_meta": len(META)}


def load_info():
    for p in [os.path.join(ROOT, "src", "iara", "dataset_info", "iara.csv")]:
        df = pd.read_csv(p)
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    for c in ["Length", "Sea state", "CPA time", "Wind"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("ID")


def unanimous_failures():
    """Instâncias (arquivo x fold) em que NENHUM dos 16 especialistas acerta. Predição de consenso =
    Counter.most_common (empate -> o primeiro especialista na ordem RF, SVM, MLP, CNN x MEL, LOFAR, HHT, DEMON)."""
    info = load_info()
    rows, total = [], 0
    for k in range(10):
        fids, y = fold_arrays(k)
        P = experts_preds(ALL16, k, fids)
        total += len(fids)
        for i, f in enumerate(fids):
            if (P[:, i] == y[i]).any():
                continue
            c = Counter(P[:, i].tolist())
            dom, nv = c.most_common(1)[0]
            inf = info.loc[f] if f in info.index else {}
            m = META.get(f, {})
            rows.append({
                "file_id": f, "fold": k, "y_true_id": int(y[i]), "classe_real": CLASS_NAMES[y[i]],
                "predicao_dominante": CLASS_NAMES[dom], "votos_dominantes": nv,
                "empate_no_consenso": sum(v == nv for v in c.values()) > 1,
                "nome_navio": inf.get("Name", "Desconhecido"), "ship_id": inf.get("Ship ID", None),
                "tipo_embarcacao": inf.get("DETAILED TYPE", "Desconhecido"), "tipo_geral": inf.get("SHIPTYPE", "Desconhecido"),
                "comprimento_m": inf.get("Length", np.nan), "estado_mar": inf.get("Sea state", np.nan),
                "vento": inf.get("Wind", np.nan), "chuva": inf.get("Rain state", None), "cpa_time": inf.get("CPA time", np.nan),
                **{c_: m.get(c_, np.nan) for c_ in FEATS},
            })
    return pd.DataFrame(rows), total


def class_medians():
    """Mediana de cada meta-feature por classe real, sobre as gravações do UO (uma vez cada)."""
    t = {}
    for k in range(10):
        t.update(PREDS["RF"]["targets"][k])
    df = pd.DataFrame([{**{c: META[f][c] for c in FEATS}, "classe_real": CLASS_NAMES[y]} for f, y in t.items()])
    return df.groupby("classe_real")[FEATS].median()
