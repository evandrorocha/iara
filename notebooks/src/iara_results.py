"""
Leitura única dos resultados do projeto IARA para os notebooks.

Todos os números dos notebooks devem sair daqui (e não de literais digitados), a partir das
predições por janela gravadas em disco:

    results/trainings/tests/<exp>/eval/fold_k/<exp>_multiclass_test.csv     (pipeline iara oficial)
    results/trainings/multirep/<exp>/eval/fold_k/<exp>_multiclass_test.csv  (scripts de scratch/)

Convenções (iguais às de src/iara/ml/metrics.py e do artigo):
  * predição por arquivo = voto majoritário das janelas, com collections.Counter.most_common como em
    src/iara/ml/models/trainer.py (empate -> a classe que aparece primeiro na sequência de janelas);
  * SP = sqrt(mean(recall) * gmean(recall)) sobre as classes presentes no alvo;
  * ACC = acurácia balanceada (média dos recalls), como a coluna "ACC" da Tabela 7 do artigo;
  * média ± desvio-padrão populacional (np.std, ddof=0) sobre os 10 folds do 5x2cv;
  * F-test 5x2cv combinado de Alpaydin sobre o SP por fold (igual a iara.ml.metrics.Test.f_test_5x2).

Modos de avaliação:
  * "4c"        -> problema de 4 classes (SMALL, MEDIUM, LARGE, BACKGROUND);
  * "3c"        -> modelo treinado sem BACKGROUND (experimentos *_3c);
  * "4c_sem_bg" -> modelo 4c avaliado só nos arquivos de navio; prever BACKGROUND conta como erro
                   (nenhum arquivo é descartado pela predição).
"""
import os
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import f1_score

CLASSES = ["SMALL", "MEDIUM", "LARGE", "BACKGROUND"]


def _find_root():
    here = os.path.abspath(os.getcwd())
    file_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = (
        here,
        os.path.dirname(here),
        os.path.abspath(os.path.join(here, "..", "..")),
        file_dir,
        os.path.abspath(os.path.join(file_dir, "..")),
        os.path.abspath(os.path.join(file_dir, "..", "..")),
    )
    for cand in candidates:
        if any(os.path.isdir(os.path.join(cand, "results", d)) for d in ("trainings", "predicoes")):
            return os.path.abspath(cand)
    raise FileNotFoundError("Não encontrei results/trainings nem results/predicoes a partir de " + here)


ROOT = _find_root()

# Clone novo do repositório: results/trainings/ (88 GB, fora do git) não existe, mas results/predicoes/
# (versionado) tem os CSVs de predição em gzip. Recria-os para que todo o código leia os caminhos de sempre.
if not os.path.isdir(os.path.join(ROOT, "results", "trainings")):
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("predicoes", os.path.join(ROOT, "scripts", "predicoes.py"))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _mod.restaurar()
SEARCH_DIRS = [os.path.join(ROOT, "results", "trainings", "multirep"),
               os.path.join(ROOT, "results", "trainings", "tests")]

# Tabela 7 do artigo (IEEE Access, vol. 13, 2025, p. 120301): (SP, ACC, F1) como (média, desvio).
ARTICLE_TABLE7 = {
    ("RF", "MEL"): {"SP": (62.22, 1.86), "ACC": (62.64, 1.84), "F1": (63.79, 1.73)},
    ("RF", "LOFAR"): {"SP": (56.92, 1.69), "ACC": (58.87, 1.68), "F1": (58.00, 1.41)},
    ("MLP", "MEL"): {"SP": (63.38, 1.81), "ACC": (64.51, 1.75), "F1": (62.89, 1.68)},
    ("MLP", "LOFAR"): {"SP": (66.51, 1.39), "ACC": (67.48, 1.24), "F1": (66.72, 1.17)},
    ("CNN", "MEL"): {"SP": (63.52, 2.26), "ACC": (64.99, 2.09), "F1": (63.04, 2.02)},
    ("CNN", "LOFAR"): {"SP": (66.05, 1.90), "ACC": (67.02, 1.78), "F1": (66.29, 2.13)},
}


def exp_dir(name):
    # 'pasta/<exp>' = caminho relativo a results/trainings (ex.: backups como _backup_hht_iter_misto_20261015/<exp>)
    if "/" in name:
        d = os.path.join(ROOT, "results", "trainings", *name.split("/"))
        if os.path.isdir(os.path.join(d, "eval")):
            return d
    for base in SEARCH_DIRS:
        d = os.path.join(base, name)
        if os.path.isdir(os.path.join(d, "eval")):
            return d
    raise FileNotFoundError(f"Experimento '{name}' não encontrado em results/trainings/{{multirep,tests}}")


@lru_cache(maxsize=None)
def file_predictions(name, subset="test"):
    """{fold: DataFrame(File, Target, Prediction, n_windows, agreement)} com uma linha por arquivo."""
    d = exp_dir(name)
    out = {}
    for k in range(10):
        path = os.path.join(d, "eval", f"fold_{k}", f"{name.split('/')[-1]}_multiclass_{subset}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name}: falta o fold {k} ({path})")
        df = pd.read_csv(path, usecols=["File", "Target", "Prediction"])
        rows = []
        for fid, g in df.groupby("File", sort=True):
            preds = g["Prediction"].tolist()
            c = Counter(preds)
            pred = c.most_common(1)[0][0]  # igual a src/iara/ml/models/trainer.py (empate -> 1ª janela)
            rows.append((fid, Counter(g["Target"]).most_common(1)[0][0], pred, len(preds), c[pred] / len(preds)))
        out[k] = pd.DataFrame(rows, columns=["File", "Target", "Prediction", "n_windows", "agreement"])
    return out


def _metrics(y, p):
    labels = sorted(set(y))
    y, p = np.asarray(y), np.asarray(p)
    rec = np.array([np.mean(p[y == c] == c) for c in labels])
    sp = np.sqrt(np.mean(rec) * scipy.stats.gmean(rec)) * 100 if np.all(rec > 0) else 0.0
    f1 = f1_score(y, p, labels=labels, average="macro", zero_division=0) * 100
    return {"SP": sp, "ACC": rec.mean() * 100, "ACC_micro": np.mean(y == p) * 100, "F1": f1,
            **{f"R_{CLASSES[c]}": r * 100 for c, r in zip(labels, rec)}}


@lru_cache(maxsize=None)
def fold_metrics(name, mode="4c"):
    """DataFrame com uma linha por fold: SP, ACC (balanceada), ACC_micro, F1 (macro) e recall por classe."""
    rows = []
    for k, df in file_predictions(name).items():
        if mode == "4c_sem_bg":
            df = df[df.Target != 3]
        rows.append({"fold": k, **_metrics(df.Target.values, df.Prediction.values)})
    return pd.DataFrame(rows).set_index("fold")


def summary(name, mode="4c"):
    """{métrica: (média, desvio)} sobre os 10 folds (desvio populacional, como o pipeline iara)."""
    fm = fold_metrics(name, mode)
    return {c: (fm[c].mean(), fm[c].std(ddof=0)) for c in fm.columns}


def fmt(ms, nd=2):
    return f"{ms[0]:.{nd}f} ± {ms[1]:.{nd}f}"


def table(rows, mode="4c", metrics=("SP", "ACC"), recalls=True):
    """rows: lista de (rótulo, nome_do_experimento) ou (rótulo, nome, modo).
    Devolve DataFrame com 'média ± desvio' das métricas e a média dos recalls por classe."""
    out = []
    for row in rows:
        label, name = row[0], row[1]
        m = row[2] if len(row) > 2 else mode
        s = summary(name, m)
        line = {"Modelo": label}
        line.update({k: fmt(s[k]) for k in metrics})
        if recalls:
            line.update({k[2:]: round(v[0], 2) for k, v in s.items() if k.startswith("R_")})
        line["Experimento"] = name
        out.append(line)
    return pd.DataFrame(out).set_index("Modelo")


def f_test_5x2(a, b, metric="SP", mode_a="4c", mode_b="4c"):
    """F-test 5x2cv combinado (Alpaydin, 1999) entre dois experimentos. Devolve (F, p)."""
    sa = fold_metrics(a, mode_a)[metric].values if isinstance(a, str) else np.asarray(a, dtype=float)
    sb = fold_metrics(b, mode_b)[metric].values if isinstance(b, str) else np.asarray(b, dtype=float)
    d1, d2 = sa[0::2] - sb[0::2], sa[1::2] - sb[1::2]
    mean = (d1 + d2) / 2
    s2 = (d1 - mean) ** 2 + (d2 - mean) ** 2
    num = np.sum(d1 ** 2) + np.sum(d2 ** 2)
    if num == 0:
        return 0.0, 1.0
    f = num / (2 * np.sum(s2))
    return f, 1 - scipy.stats.f.cdf(f, 10, 5)


def aligned(names, mode="4c"):
    """{fold: DataFrame(File, Target, <nome1>, <nome2>, ...)} só com arquivos presentes em todos."""
    out = {}
    for k in range(10):
        merged = None
        for n in names:
            df = file_predictions(n)[k][["File", "Target", "Prediction"]].rename(columns={"Prediction": n})
            merged = df if merged is None else merged.merge(df, on=["File", "Target"], how="inner")
        if mode == "4c_sem_bg":
            merged = merged[merged.Target != 3]
        out[k] = merged
    return out


def oracle_recall(names, mode="4c"):
    """Recall por classe do oráculo (acerta se ALGUM modelo acerta) e o esperado se os erros fossem
    independentes, 1 - prod(1 - r_i). Médias sobre os folds."""
    obs, exp = [], []
    for k, df in aligned(names, mode).items():
        hit = np.column_stack([df[n].values == df.Target.values for n in names])
        r_obs, r_exp = {}, {}
        for c in sorted(df.Target.unique()):
            m = df.Target.values == c
            r_obs[CLASSES[c]] = hit[m].any(axis=1).mean() * 100
            r_exp[CLASSES[c]] = (1 - np.prod(1 - hit[m].mean(axis=0))) * 100
        obs.append(r_obs)
        exp.append(r_exp)
    return pd.DataFrame({"oráculo observado": pd.DataFrame(obs).mean(),
                         "esperado c/ erros independentes": pd.DataFrame(exp).mean()})


def majority_vote(names, mode="4c", seed=0):
    """Voto de pluralidade entre modelos (empate -> sorteio entre as classes empatadas).
    Devolve DataFrame por fold com as métricas do voto."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, df in aligned(names, mode).items():
        P = np.column_stack([df[n].values for n in names])
        pred = []
        for row in P:
            c = np.bincount(row, minlength=4)
            pred.append(rng.choice(np.flatnonzero(c == c.max())))
        rows.append({"fold": k, **_metrics(df.Target.values, np.array(pred))})
    return pd.DataFrame(rows).set_index("fold")


def config(name):
    import json
    p = os.path.join(exp_dir(name), "config.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)
