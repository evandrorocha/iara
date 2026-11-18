"""Para cada navio MEDIUM: energia média em 37-58 Hz (dB acima do fundo TPSW, LOFAR médio da gravação inteira)
comparada com TODAS as gravações LARGE do IARA (A, B, C, D). Conta quantas gravações LARGE têm mais e quantas menos."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "scratch"); from eval_ship_split_5x2cv import metadata
m = metadata(); S = pd.read_pickle("data/iara_processed/cache_lofar_espectro_medio.pkl")
F = (np.arange(1024) + 1) * 16000 / 2048; band = (F >= 37) & (F < 58)
e = S.loc[:, band].mean(1).rename("E").to_frame().join(m[["Dataset", "Ship ID", "Name", "SHIPTYPE", "DETAILED TYPE", "Length", "Target"]])
L = e[e.Target == 2]
med = e[e.Target == 1].groupby("Ship ID").agg(Nome=("Name", "first"), Tipo=("DETAILED TYPE", "first"), Comprimento=("Length", "first"),
                                              Gravacoes=("E", "size"), Energia_dB=("E", "mean"))
# faixa de maior energia do navio: espectro médio do navio (média das gravações), pico nas 14 bandas log e na faixa de 7,8 Hz
EDGES = np.geomspace(15, 8000, 15)
ids1 = e.index[e.Target == 1]; spec = S.loc[ids1].groupby(e.loc[ids1, "Ship ID"]).mean()
bandas = pd.DataFrame({f"{a:.0f}-{b:.0f} Hz": spec.loc[:, (F >= a) & (F < b)].mean(1) for a, b in zip(EDGES[:-1], EDGES[1:])})
med["Faixa_pico"] = bandas.idxmax(1).reindex(med.index); med["Energia_pico_dB"] = bandas.max(1).reindex(med.index)
ok = F >= 15  # a 1ª faixa (7,8 Hz) fica fora das bandas e é instável
med["Freq_pico_Hz"] = F[ok][spec.loc[:, ok].values.argmax(1)][pd.Index(spec.index).get_indexer(med.index)]
med["LARGE_maior"] = [(L.E > v).sum() for v in med.Energia_dB]
med["LARGE_menor"] = [(L.E < v).sum() for v in med.Energia_dB]
med["pct_LARGE_maior"] = 100 * med.LARGE_maior / len(L)
for ds in "ABCD":
    Ld = L[L.Dataset == ds].E; med[f"pct_maior_{ds}"] = [100 * (Ld > v).mean() for v in med.Energia_dB]
med = med.sort_values("Energia_dB", ascending=False).round(2)
med.to_csv("reports/small_medium/medium_vs_large_3758.csv")
print(f"bins: {F[band]}; gravações LARGE: {len(L)} (por subconjunto: {L.Dataset.value_counts().sort_index().to_dict()})")
print(f"energia LARGE: mediana {L.E.median():.2f} dB; MEDIUM (navio) mediana {med.Energia_dB.median():.2f} dB")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 22)
print(med.drop(columns=["Nome"]).to_string())
print("\nresumo: navios MEDIUM em que a maioria das gravações LARGE tem MAIS energia:", (med.pct_LARGE_maior > 50).sum(), "de", len(med))
print("média de % LARGE com mais energia:", med.pct_LARGE_maior.mean().round(1))
