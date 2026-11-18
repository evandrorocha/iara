"""Teste de circularidade dos slides "LOFAR × DEMON dentro de cada lancha": os grupos (acertada / errada → MEDIUM /
errada → LARGE) vinham da MLP LOFAR. Aqui os grupos vêm de OUTROS modelos, com o mesmo protocolo (split por gravação,
split_id_*, cada gravação testada 5 vezes; acertada = SMALL em >= 60% dos testes; errada → classe mais votada).
Para cada modelo, compara DENTRO de cada lancha (valor menos a mediana da lancha):
  LOFAR 31–86 Hz (dB acima do TPSW), MEL grave − agudo (31–86 Hz menos 1–6 kHz), DEMON força e frequência da linha.
Também a CNN MEL do artigo (results/trainings/tests/cnn_mel_artigo; voto das janelas por teste).
Rodar na raiz, no container iara-dev:  python scratch/apres_small_medium/lanchas_grupos_outros_modelos.py
Saídas: reports/small_medium/lanchas_grupos_outros_modelos.csv (AUCs) e _grupos.csv (grupo de cada gravação por modelo)"""
import glob
import sys
import numpy as np, pandas as pd
import librosa
from scipy import stats as st
sys.path.insert(0, "notebooks/src"); sys.path.insert(0, "scratch")
import iara_results as R  # noqa: E402
from eval_ship_split_5x2cv import metadata  # noqa: E402

m = metadata(); m.index = m.index.map(int)
D = pd.read_csv("reports/small_medium/lanchas_lofar_demon.csv", index_col=0)   # LOFAR/DEMON por gravação de lancha
D.index = D.index.map(int)
mel = pd.read_pickle("data/iara_processed/cache_mel_espectro_medio.pkl")["sem normalização"]; mel.index = mel.index.map(int)
FM = librosa.mel_frequencies(n_mels=258, fmin=0, fmax=8000)[1:-1]
D["mel_grave_agudo"] = (mel.loc[D.index, (FM >= 31) & (FM < 86)].mean(1) - mel.loc[D.index, (FM >= 1000) & (FM < 6000)].mean(1))
SMALL_ALL = m.index[m.Target == 0]


def fracoes_split_id(exp):
    P = pd.concat(R.file_predictions(exp).values()); P["File"] = P.File.astype(int)
    g = P.groupby("File").Prediction
    return pd.DataFrame({c: g.apply(lambda s, c=c: (s == c).mean()) for c in range(4)})


def fracoes_cnn_mel():
    votos = []
    for f in sorted(glob.glob("results/trainings/tests/cnn_mel_artigo/eval/fold_*/cnn_mel_artigo_multiclass_test.csv")):
        x = pd.read_csv(f)
        votos.append(x.groupby("File").Prediction.agg(lambda s: s.value_counts().idxmax()))
    V = pd.concat(votos, axis=1); n = V.notna().sum(1)
    out = pd.DataFrame({c: (V == c).sum(1) / n for c in range(4)}); out.index = out.index.map(int)
    return out


def grupos(fr):
    fr = fr.reindex(D.index)
    outra = fr[[1, 2, 3]].idxmax(1).map({1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"})
    return pd.Series(np.where(fr[0] >= 0.6, "acertada", "errada → " + outra), index=D.index)


VARS = {"lofar_31_86": "LOFAR 31–86 Hz", "mel_grave_agudo": "MEL grave−agudo",
        "demon_pico_dB": "DEMON força", "demon_freq_Hz": "DEMON freq."}


def dentro(gr, a, b):
    x = D.assign(g=gr)[gr.isin([a, b])]
    ok = x.groupby("ship").g.nunique() == 2
    x = x[x.ship.isin(ok[ok].index)]
    out = {"lanchas": int(ok.sum()), "gravações": len(x)}
    for v, nome in VARS.items():
        z = x[v] - x.groupby("ship")[v].transform("median")
        u = st.mannwhitneyu(z[x.g == b], z[x.g == a])
        out[nome] = u.statistic / ((x.g == b).sum() * (x.g == a).sum()); out["p " + nome] = u.pvalue
    return out


MODELOS = {"MLP LOFAR": "split_id_mlp_lofar_4c", "RF LOFAR": "split_id_rf_lofar_4c", "SVM LOFAR": "split_id_svm2000_lofar_4c",
           "BiGRU LOFAR": "split_id_bigru_lofar_4c", "MLP MEL": "split_id_mlp_mel_4c", "RF MEL": "split_id_rf_mel_4c",
           "SVM MEL": "split_id_svm2000_mel_4c", "MLP HHT": "split_id_mlp_hht_4c", "RF HHT": "split_id_rf_hht_4c",
           "SVM HHT": "split_id_svm2000_hht_4c", "CNN MEL (artigo)": None}
rows, G = [], {}
for nome, exp in MODELOS.items():
    try:
        fr = fracoes_cnn_mel() if exp is None else fracoes_split_id(exp)
    except Exception as e:  # noqa: BLE001
        print("sem previsões:", nome, e); continue
    gr = grupos(fr); G[nome] = gr
    s_all = fr.reindex(SMALL_ALL).dropna()
    base = {"modelo": nome, "recall SMALL (todas)": 100 * (s_all[0] >= 0.5).mean(), "recall lanchas": 100 * (gr == "acertada").mean(),
            "n → LARGE": int((gr == "errada → LARGE").sum()), "n → MEDIUM": int((gr == "errada → MEDIUM").sum()),
            "concordância c/ MLP LOFAR": np.nan}
    for a, b, rot in [("acertada", "errada → LARGE", "L×ok"), ("acertada", "errada → MEDIUM", "M×ok"), ("errada → MEDIUM", "errada → LARGE", "L×M")]:
        r = dentro(gr, a, b)
        rows.append({**base, "comparação": rot, **r})
T = pd.DataFrame(rows)
G = pd.DataFrame(G)
conc = {c: 100 * (G[c] == G["MLP LOFAR"]).mean() for c in G}
T["concordância c/ MLP LOFAR"] = T.modelo.map(conc)
pd.set_option("display.width", 260)
print("Recall SMALL e tamanho dos grupos (lanchas):")
print(T.drop_duplicates("modelo")[["modelo", "recall SMALL (todas)", "recall lanchas", "n → LARGE", "n → MEDIUM",
                                    "concordância c/ MLP LOFAR"]].round(1).to_string(index=False))
for rot in ["L×ok", "M×ok", "L×M"]:
    print(f"\nDentro de cada lancha, {rot} (AUC; * p<0,05):")
    t = T[T.comparação == rot].copy()
    for v in VARS.values():
        t[v] = t[v].round(2).astype(str) + np.where(t["p " + v] < .05, "*", " ")
    print(t[["modelo", "lanchas", "gravações", *VARS.values()]].to_string(index=False))
T.to_csv("reports/small_medium/lanchas_grupos_outros_modelos.csv", index=False)
G.to_csv("reports/small_medium/lanchas_grupos_outros_modelos_grupos.csv")
