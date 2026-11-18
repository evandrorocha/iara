"""
A condição de gravação (ambiente/equipamento) é uma pista para SMALL × MEDIUM? O modelo a usa?

Motivação: avaliar se misturar BACKGROUND no áudio (data augmentation) teria efeito além de robustez a SNR,
apagando uma "impressão digital" de condição que o modelo usaria como atalho.

O IARA não registra data, sessão nem hidrofone. Assinatura de ambiente usada aqui (psd24k, fora da faixa do LOFAR):
  * AMB = espectro do bloco de 10 s MAIS SILENCIOSO de cada gravação (mínimo por faixa entre blocos, ~piso de
    ambiente/equipamento), de 8 a 23 kHz, em 16 bandas log; versão "nivel" (dB absolutos: ganho + ambiente) e
    versão "forma" (média da gravação removida).
Testes:
  T1  AMB prevê o subconjunto (A–D) / SMALL × MEDIUM em navios novos? (logística, CV agrupada por navio, peso 1 por navio)
  T2  As previsões do MLP LOFAR (split por navio, augnavio_mlp_lofar_base_4c) seguem a classe das gravações de
      TREINO com ambiente mais parecido (10 vizinhos em AMB), dentro da mesma classe verdadeira?
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "notebooks", "src"))
from eval_ship_split_5x2cv import metadata, FOLDS_CSV  # noqa: E402
import iara_results as R  # noqa: E402

CACHE = "data/iara_processed/cache_amb_min_psd24k.pkl"
PSD_DIR = "data/iara_processed/psd24k"


def amb_features():
    if os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    fr = pickle.load(open("data/iara_processed/cache_psd24k_resumo.pkl", "rb"))["freqs"]
    edges = np.geomspace(8000, 23000, 17)
    idx = [np.where((fr >= a) & (fr < b))[0] for a, b in zip(edges[:-1], edges[1:])]
    rows = {}
    for f in sorted(os.listdir(PSD_DIR)):
        if not f.endswith(".pkl"):
            continue
        d = pd.read_pickle(os.path.join(PSD_DIR, f))
        b = np.asarray(d["blocos_db"], dtype=np.float64)
        # bloco mais silencioso = menor nível médio em 8-23 kHz (o mesmo bloco para todas as faixas)
        q = b[np.argmin(b[:, idx[0][0]:idx[-1][-1]].mean(1))]
        rows[int(f[:-4])] = [q[i].mean() for i in idx]
    X = pd.DataFrame(rows).T.sort_index()
    X.columns = [f"{int(a)}-{int(b)}" for a, b in zip(edges[:-1], edges[1:])]
    X.to_pickle(CACHE)
    return X


def cv_auc(X, y, groups, w, reps=10, binary=True):
    out = []
    for r in range(reps):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r)
        p = np.zeros((len(y), len(np.unique(y))))
        for tr, te in sgkf.split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
            clf.fit(X[tr], y[tr], logisticregression__sample_weight=w[tr])
            p[te] = clf.predict_proba(X[te])
        out.append(roc_auc_score(y, p[:, 1], sample_weight=w) if binary
                   else balanced_accuracy_score(y, p.argmax(1)))
    return np.mean(out), np.min(out), np.max(out)


def ship_weights(ships, y):
    s = pd.Series(ships)
    w = 1.0 / s.map(s.value_counts()).values
    for c in np.unique(y):  # classes com peso total igual
        w[y == c] /= w[y == c].sum()
    return w * len(y) / w.sum()


def main():
    m = metadata()
    A = amb_features()
    m = m.join(A, how="inner")
    cols = list(A.columns)
    nivel = m[cols].values
    forma = nivel - nivel.mean(1, keepdims=True)
    print(f"{len(m)} gravações com psd24k; AMB = bloco mais silencioso, 8-23 kHz, {len(cols)} bandas")

    print("\n=== T1: a assinatura de ambiente prevê ... (CV agrupada por navio, peso 1 por navio; média [mín, máx] de 10 rep.)")
    navio = m[m.Target < 3]
    g = navio["Ship ID"].values
    for nome, X in [("nivel", nivel), ("forma", forma)]:
        Xn = X[m.Target.values < 3]
        ysub = navio.Dataset.map({"A": 0, "B": 1, "C": 2, "D": 3}).values
        r = cv_auc(Xn, ysub, g, ship_weights(g, ysub), binary=False)
        print(f"  [{nome}] subconjunto A-D (acc. balanceada, acaso 0,25): {r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]")
        sm = navio.Target.values < 2
        r = cv_auc(Xn[sm], navio.Target.values[sm], g[sm], ship_weights(g[sm], navio.Target.values[sm]))
        print(f"  [{nome}] SMALL x MEDIUM (AUC): {r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]")
        for sub in "ABCD":
            mm = sm & (navio.Dataset.values == sub)
            r = cv_auc(Xn[mm], navio.Target.values[mm], g[mm], ship_weights(g[mm], navio.Target.values[mm]), reps=5)
            print(f"      dentro de {sub}: AUC {r[0]:.2f}")

    print("\n=== T2: a previsão do MLP (split por navio) segue o ambiente das gravações de treino?")
    folds = pd.read_csv(FOLDS_CSV, dtype={"Ship ID": str})
    folds = folds[folds.condicao == "split_navio"]
    preds = R.file_predictions("augnavio_mlp_lofar_base_4c")
    rows = []
    for nome, X in [("nivel", nivel), ("forma", forma)]:
        Xd = pd.DataFrame(X, index=m.index)
        for k in range(10):
            d = folds[folds.fold == k]
            tr = d[(d.Subset == "trn") & (d.Target < 3)].File.values
            tr = tr[np.isin(tr, Xd.index)]
            sc = StandardScaler().fit(Xd.loc[tr].values)
            nn = NearestNeighbors(n_neighbors=10).fit(sc.transform(Xd.loc[tr].values))
            pk = preds[k].copy()
            pk["File"] = pk.File.astype(int)
            pk = pk[(pk.Target < 2) & pk.File.isin(Xd.index)]
            _, ind = nn.kneighbors(sc.transform(Xd.loc[pk.File].values))
            ytr = m.loc[tr, "Target"].values
            for (_, r), nb in zip(pk.iterrows(), ind):
                v = np.bincount(ytr[nb], minlength=3) / 10
                rows.append({"rep": nome, "fold": k, "File": r.File, "y": r.Target, "pred": r.Prediction,
                             "amb_small": v[0], "amb_medium": v[1], "amb_large": v[2]})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(HERE, "teste_impressao_T2.csv"), index=False)
    for nome in ["nivel", "forma"]:
        t = T[T.rep == nome]
        for y, cy, co in [(0, "SMALL", "MEDIUM"), (1, "MEDIUM", "SMALL")]:
            ty = t[t.y == y].copy()
            # fração de vizinhos de ambiente da OUTRA classe (SMALL<->MEDIUM), em tercis
            col = "amb_medium" if y == 0 else "amb_small"
            ty["terc"] = pd.qcut(ty[col].rank(method="first"), 3, labels=["baixo", "médio", "alto"])
            tab = ty.groupby("terc", observed=True).agg(
                viz_outra=(col, "mean"), acerto=("pred", lambda s: (s == y).mean() * 100),
                prev_outra=("pred", lambda s: (s == (1 - y)).mean() * 100), n=("pred", "size"))
            rho = ty[[col]].assign(o=(ty.pred == 1 - y).astype(float)).corr(method="spearman").iloc[0, 1]
            print(f"  [{nome}] {cy} verdadeiras: % vizinhos de ambiente {co} -> % previstas {co} "
                  f"(Spearman {rho:+.2f})")
            print(tab.round(2).to_string())


if __name__ == "__main__":
    main()
