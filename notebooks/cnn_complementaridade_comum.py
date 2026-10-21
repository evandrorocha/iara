"""
Análise de complementaridade da CNN 2D (NBs 20-22), lida das predições em disco via iara_results.

Experimentos:
  MEL   -> results/trainings/tests/cnn                         (pipeline iara oficial, features de 0,512 s)
  LOFAR -> results/trainings/multirep/cnn2d_lofar5s_w32_b64_e15_4c  (CNN ad hoc, scratch/generate_cnn_predictions_and_analysis.py)
  HHT   -> results/trainings/multirep/cnn2d_hht_w32_b64_e20_4c
  DEMON -> results/trainings/multirep/cnn2d_demon_w32_b64_e20_4c

Não modifica iara_results; só funções locais sobre R.aligned / R.oracle_recall / R.majority_vote.
"""
import os
from collections import Counter

import numpy as np
import pandas as pd

import iara_results as R

CLS = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}
CLS_ID = {v: k for k, v in CLS.items()}

CNN = {
    "MEL": "cnn",
    "LOFAR": "cnn2d_lofar5s_w32_b64_e15_4c",
    "HHT": "cnn2d_hht_w32_b64_e20_4c",
    "DEMON": "cnn2d_demon_w32_b64_e20_4c",
}
REPS = list(CNN)

# Famílias da Seção 8 (mesmos experimentos dos NBs 10-19)
FAMILIES = {
    "SVM Nyström": {
        "MEL": "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0",
        "LOFAR": "svm_nystroem_6000_lofar_elasticnet_l1r0.15",
        "HHT": "svm_nystroem_4000_hht_C2.0_elasticnet_l1r0.15_4c",
        "DEMON": "svm_nystroem_4000_demon_C2.0_elasticnet_l1r0.15_4c",
    },
    "MLP": {
        "MEL": "mlp",
        "LOFAR": "mlp_lofar",
        "HHT": "mlp_hht_h32-16_e30_4c",
        "DEMON": "mlp_demon_h32-16_e30_4c",
    },
    "Random Forest": {
        "MEL": "rf_mel5s_n100_balanced_4c",
        "LOFAR": "rf_lofar5s_n100_balanced_4c",
        "HHT": "rf_hht_n100_balanced_4c",
        "DEMON": "rf_demon_n100_balanced_4c",
    },
    "CNN 2D": CNN,
}
FAMILY_NB = {"SVM Nyström": {0: 10, 1: 12, 2: 13}, "MLP": {0: 14, 1: 15, 2: 16},
             "Random Forest": {0: 17, 1: 18, 2: 19}, "CNN 2D": {0: 20, 1: 21, 2: 22}}


def stacked(models):
    """Todas as instâncias (fold, arquivo) alinhadas entre os modelos; colunas renomeadas para as representações."""
    al = R.aligned(list(models.values()))
    df = pd.concat([d.assign(fold=k) for k, d in al.items()], ignore_index=True)
    return df.rename(columns={v: k for k, v in models.items()})


def analyze(models, c):
    """Análise de complementaridade da classe c (instâncias de fold = arquivo x fold de teste)."""
    reps = list(models)
    df = stacked(models)
    d = df[df.Target == c].reset_index(drop=True)
    H = pd.DataFrame({r: (d[r].values == c) for r in reps})
    nh = H.sum(axis=1).values
    n = len(d)
    out = {"n": n, "n_files": d.File.nunique(), "reps": reps}
    out["hits"] = {r: int(H[r].sum()) for r in reps}
    out["recall"] = {r: H[r].mean() * 100 for r in reps}
    out["consensus"] = {k: int((nh == k).sum()) for k in range(len(reps), -1, -1)}
    out["oracle_hits"] = int((nh >= 1).sum())
    excl = {r: (H[r].values & (nh == 1)) for r in reps}
    out["exclusive"] = {r: int(excl[r].sum()) for r in reps}
    any_excl = nh == 1
    out["exclusive_files"] = int(d.File[any_excl].nunique())
    # gravações resgatadas exclusivamente (por algum modelo) em TODAS as suas aparições no teste
    g = pd.DataFrame({"File": d.File, "ex": any_excl}).groupby("File").ex.all()
    out["exclusive_files_always"] = int(g.sum())
    # confusão
    conf = {}
    for r in reps:
        wrong = d[r][~H[r].values]
        vc = wrong.map(CLS).value_counts()
        conf[r] = {"n_err": int(len(wrong)), **{CLS[k]: vc.get(CLS[k], 0) / max(len(wrong), 1) * 100
                                                 for k in CLS if k != c}}
    out["confusion"] = conf
    # efeito reverso a partir do melhor modelo isolado na classe
    base = max(reps, key=lambda r: out["recall"][r])
    out["base"] = base
    hb = H[base].values
    rev = {}
    for r in reps:
        if r == base:
            continue
        ho = H[r].values
        rev[r] = {"keep": int((hb & ho).sum()), "lose": int((hb & ~ho).sum()), "rescue": int((~hb & ho).sum())}
        rev[r]["lose_pct"] = rev[r]["lose"] / hb.sum() * 100
        rev[r]["rescue_pct"] = rev[r]["rescue"] / (~hb).sum() * 100
        rev[r]["net"] = rev[r]["rescue"] - rev[r]["lose"]
    out["reverse"] = rev
    # oráculo observado x esperado com erros independentes (média dos folds)
    orc = R.oracle_recall(list(models.values()))
    out["oracle_obs"] = orc.loc[CLS[c], "oráculo observado"]
    out["oracle_exp"] = orc.loc[CLS[c], "esperado c/ erros independentes"]
    # votos
    out["ge2"] = int((nh >= 2).sum())
    out["ge3"] = int((nh >= 3).sum())
    vote = R.majority_vote(list(models.values()))
    out["vote_recall"] = vote[f"R_{CLS[c]}"].mean()
    out["vote_sp"] = (vote["SP"].mean(), vote["SP"].std(ddof=0))
    best_sp = max(reps, key=lambda r: R.summary(models[r])["SP"][0])
    out["best_sp_rep"] = best_sp
    out["best_sp"] = R.summary(models[best_sp])["SP"]
    out["vote_vs_best_sp"] = R.f_test_5x2(vote["SP"].values, models[best_sp])
    out["vote_vs_base_recall"] = R.f_test_5x2(vote[f"R_{CLS[c]}"].values,
                                              R.fold_metrics(models[base])[f"R_{CLS[c]}"].values)
    out["vote"] = vote
    return out


def tables(A, c):
    """DataFrames prontos para o notebook a partir de analyze()."""
    n, reps = A["n"], A["reps"]
    order = sorted(reps, key=lambda r: A["recall"][r])
    df_hits = pd.DataFrame({
        "Representacao": order + ["ORÁCULO CNN (≥1/4; não realizável)", "Esperado c/ erros independentes",
                                  "Voto real (pluralidade)"],
        "Acertos": [A["hits"][r] for r in order] + [A["oracle_hits"], np.nan, np.nan],
        "Recall_Pct": [A["recall"][r] for r in order] + [A["oracle_hits"] / n * 100, A["oracle_exp"], A["vote_recall"]],
    })
    k = len(reps)
    names = {k: f"{k} de {k} (Unanimidade)", k - 1: f"{k-1} de {k} (Alto Consenso)", k - 2: f"{k-2} de {k} (Dividido)",
             1: f"1 de {k} (Resgate Exclusivo)", 0: f"0 de {k} (Erro Comum)"}
    df_consensus = pd.DataFrame({
        "Nivel_Consenso": [names[j] for j in A["consensus"]],
        "Instancias": list(A["consensus"].values()),
        "Pct_Total": [v / n * 100 for v in A["consensus"].values()],
        "Pct_Dentro_Oraculo": [v / A["oracle_hits"] * 100 if j > 0 else np.nan for j, v in A["consensus"].items()],
    })
    df_exclusive = pd.DataFrame({
        "Representacao": [f"{r} Exclusivo" for r in sorted(reps, key=lambda r: A["exclusive"][r])],
        "Instancias_Salvas": [A["exclusive"][r] for r in sorted(reps, key=lambda r: A["exclusive"][r])],
        "Percentual": [A["exclusive"][r] / n * 100 for r in sorted(reps, key=lambda r: A["exclusive"][r])],
    })
    b = A["base"]
    df_reverse = pd.DataFrame([{"Destino_Troca": f"Para {r}", f"Mantem_do_{b}": v["keep"],
                                f"Perde_do_{b} (Efeito Reverso)": v["lose"], "Pct_Perda_Acertos (%)": v["lose_pct"],
                                f"Resgata_Erros_{b}": v["rescue"], "Pct_Resgate_Erros (%)": v["rescue_pct"],
                                "Saldo_Liquido": v["net"]} for r, v in A["reverse"].items()])
    others = [CLS[j] for j in CLS if j != c]
    df_conf = pd.DataFrame([{"Representacao": f"{r} ({A['confusion'][r]['n_err']} erros)",
                             **{f"Confundido_{o}_Pct": A["confusion"][r][o] for o in others}} for r in reps])
    return df_hits, df_consensus, df_exclusive, df_reverse, df_conf


def collapse_table(models=CNN):
    """SP/ACC por modelo e o recall mínimo/máximo por classe ao longo dos 10 folds."""
    rows = []
    for r, exp in models.items():
        fm = R.fold_metrics(exp)
        s = R.summary(exp)
        rec = fm[[f"R_{CLS[k]}" for k in CLS]]
        rows.append({"Representação": r, "Experimento": exp, "SP": R.fmt(s["SP"]), "ACC (bal.)": R.fmt(s["ACC"]),
                     "SP mín–máx": f"{fm.SP.min():.1f}–{fm.SP.max():.1f}",
                     **{f"{CLS[k]} mín–máx": f"{rec[f'R_{CLS[k]}'].min():.1f}–{rec[f'R_{CLS[k]}'].max():.1f}" for k in CLS},
                     "folds c/ recall < 10% em alguma classe": int((rec.min(axis=1) < 10).sum())})
    return pd.DataFrame(rows).set_index("Representação")


def sp_lowest_index_tiebreak(exp):
    """SP por fold com o desempate do script de scratch (argmax(bincount): empate -> menor índice de classe)."""
    d = R.exp_dir(exp)
    sps = []
    for k in range(10):
        df = pd.read_csv(os.path.join(d, "eval", f"fold_{k}", f"{exp}_multiclass_test.csv"),
                         usecols=["File", "Target", "Prediction"])
        y, p = [], []
        for fid, g in df.groupby("File", sort=True):
            y.append(Counter(g["Target"]).most_common(1)[0][0])
            p.append(int(np.argmax(np.bincount(g["Prediction"].values, minlength=4))))
        sps.append(R._metrics(np.array(y), np.array(p)))
    return pd.DataFrame(sps)


def images_per_file(exp="cnn2d_lofar5s_w32_b64_e15_4c"):
    """Distribuição do número de imagens (32 frames) por arquivo de teste."""
    nw = pd.concat(R.file_predictions(exp).values()).n_windows
    return nw


def family_row(models, c):
    A = analyze(models, c)
    n = A["n"]
    best = A["base"]
    champ = max(A["exclusive"], key=A["exclusive"].get)
    k = len(models)
    return {
        "Melhor modelo isolado (recall)": f"{A['recall'][best]:.2f}% ({best})",
        "Oráculo observado (≥1/4)": f"{A['oracle_hits'] / n * 100:.2f}% ({A['oracle_hits']})",
        "Oráculo esperado c/ erros independentes": f"{A['oracle_exp']:.2f}%",
        "Unanimidade (4/4)": f"{A['consensus'][k] / n * 100:.2f}% ({A['consensus'][k]})",
        "Erro comum (0/4)": f"{A['consensus'][0] / n * 100:.2f}% ({A['consensus'][0]})",
        "Campeão de resgates exclusivos": f"{champ} ({A['exclusive'][champ]})",
        "≥2/4 (empates contam como acerto)": f"{A['ge2'] / n * 100:.2f}%",
        "Voto real (recall da classe)": f"{A['vote_recall']:.2f}%",
        "Voto real: SP global": f"{A['vote_sp'][0]:.2f} ± {A['vote_sp'][1]:.2f}",
        "Melhor SP isolado": f"{A['best_sp'][0]:.2f} ± {A['best_sp'][1]:.2f} ({A['best_sp_rep']})",
        "F-test voto × melhor SP (p)": f"{A['vote_vs_best_sp'][1]:.3f}",
    }, A


def family_table(c):
    rows, res = {}, {}
    for fam, models in FAMILIES.items():
        rows[f"{fam} (NB {FAMILY_NB[fam][c]})"], res[fam] = family_row(models, c)
    return pd.DataFrame(rows), res


_FRAMES_CACHE = None


def lofar_cache_frames():
    """Número de frames de 5,12 s por gravação no cache LOFAR usado pela CNN ad hoc (só na geração; ~350 MB)."""
    global _FRAMES_CACHE
    if _FRAMES_CACHE is None:
        import pickle
        with open(os.path.join(R.ROOT, "data", "iara_processed", "cache_lofar.pkl"), "rb") as f:
            cache = pickle.load(f)
        _FRAMES_CACHE = {fid: len(v) for fid, v in cache.items()}
    return np.array(list(_FRAMES_CACHE.values()))


def lofar_cache_frames_by_file():
    lofar_cache_frames()
    return _FRAMES_CACHE
