"""
Figura didática do processo de extração do LOFAR do pipeline iara (src/iara/processing/analysis.py), passo a passo,
com uma gravação real. Usa as próprias funções do pipeline e confere o resultado com o LOFAR já processado.

Parâmetros do LOFAR atual (data/iara_processed/lofar_bc15…/config.json): n_pts = 1024 (FFT de 2048 pontos),
decimação 3, sem sobreposição, integração de 0,512 s, sem normalização.

Rodar na raiz do repositório:  python notebooks/src/figura_extracao_lofar.py [ID]
Saída: notebooks/figuras/extracao_lofar.png
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import scipy.signal as sci

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from iara.processing import analysis as A  # noqa: E402

ID = int(sys.argv[1]) if len(sys.argv) > 1 else 208
OUT = os.path.join(ROOT, "notebooks", "figuras", "extracao_lofar.png")
LOFAR_DIR = os.path.join(ROOT, "data", "iara_processed", "lofar_bc1559516c522e2b993895bf5cb2ac94")
DEC, NPTS, INTEG = 3, 1024, 0.512

meta = pd.read_csv(os.path.join(ROOT, "src", "iara", "dataset_info", "iara.csv"), dtype=str).set_index("ID")
m = meta.loc[f"{ID:04d}"]
arq = glob.glob(os.path.join(ROOT, "data", "iara", "*", f"*-{ID:04d}.wav"))[0]
fs0, x = wav.read(arq)
x = (x[:, 0] if x.ndim > 1 else x).astype(float)

# ---------- pipeline, etapa por etapa (mesmas funções e parâmetros do iara) ----------
x0 = x - x.mean()
xd = A.decimate(x0, DEC); fs = fs0 / DEC                       # Chebyshev I ordem 8 (corte 0,8·Nyquist) + reamostragem
N = 2 * NPTS                                                    # 2048 amostras por FFT
f, t, Z = sci.spectrogram(xd, nfft=N, fs=fs, window=np.hanning(N), noverlap=0, detrend=False,
                          scaling="spectrum", mode="complex")
P = (np.abs(Z) * N / 2)[1:]; f = f[1:]                          # magnitude; descarta o bin DC -> 1024 faixas
P[P < 1e-9] = 1e-9
LOG = 20 * np.log10(P)                                          # log-espectrograma (dB)
BG = A.tpsw(LOG)                                                # fundo estimado pelo TPSW, faixa a faixa
LOF = LOG - BG; LOF[LOF < -0.2] = 0                             # LOFAR por FFT
nm = int(round(INTEG / (t[-1] - t[-2])))                        # 4 FFTs por linha
LOFI = np.array([LOF[:, i:i + nm].mean(1) for i in range(0, LOF.shape[1], nm)])   # integração -> linhas de 0,512 s

ref = pd.read_pickle(os.path.join(LOFAR_DIR, f"{ID}.pkl"))["df"].values
err = np.abs(ref - LOFI[:len(ref)]).max() if ref.shape == LOFI.shape else np.nan
print(f"ID {ID}: {LOFI.shape} vs pkl {ref.shape}; maior diferença = {err:.2e} dB")

# ---------- figura ----------
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
C1, C2, C3, C4 = "#1f4e79", "#c0504d", "#4f8f3a", "#7f6000"
fig = plt.figure(figsize=(16, 21))
gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1.25], hspace=0.62, wspace=0.18,
                      left=0.06, right=0.97, top=0.915, bottom=0.035)
fig.suptitle(f"Como o LOFAR do IARA é extraído — gravação {ID} ({m['DETAILED TYPE']}, {m['Length']} m, subconjunto {m['Dataset']})",
             fontsize=15, fontweight="bold")
fig.text(0.5, 0.952, "Mesmas funções do pipeline (src/iara/processing/analysis.py); o resultado final é idêntico ao LOFAR usado pelos modelos"
         + (f" (diferença máxima {err:.0e} dB)" if err == err else ""), ha="center", fontsize=10, color="0.3")

def br_(v, nd=3):
    return f"{v:.{nd}f}".replace(".", ",")

def titulo(ax, n, txt, sub):
    ax.set_title(f"{n}. {txt}", loc="left", fontsize=12, fontweight="bold", color=C1, pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color="0.25", va="bottom")

w0 = int(60 * fs)                     # trecho de exemplo: começa em 60 s
iw = w0 // N                          # índice da FFT correspondente

# 1. sinal bruto
ax = fig.add_subplot(gs[0, 0])
tt = np.arange(len(x)) / fs0
ax.plot(tt[::50], x[::50], color=C1, lw=0.3)
ax.set_xlabel("tempo (s)"); ax.set_ylabel("amplitude"); ax.set_ylim(-1.15 * np.abs(x).max(), 2.6 * np.abs(x).max())
# zoom no trecho de 0,512 s (4 FFTs): na escala de 300 s ele teria ~1 pixel de largura
t0, t1 = iw * N / fs, (iw + nm) * N / fs
s0, s1 = int(t0 * fs0), int(t1 * fs0)
ins = ax.inset_axes([0.52, 0.6, 0.45, 0.3]); ins.set_yticks([])
ins.plot(tt[s0:s1], x[s0:s1], color=C2, lw=0.4)
ins.set_xlim(t0, t1); ins.tick_params(labelsize=7)
ins.set_title(f"zoom: {br_(t0)} a {br_(t1)} s = 0,512 s ({s1 - s0} amostras a 48 kHz)", fontsize=7.5)
ax.indicate_inset_zoom(ins, edgecolor=C2, alpha=0.9)
titulo(ax, 1, "Sinal bruto do hidrofone",
       f"{fs0 / 1000:.0f} kHz ({fs0} amostras por segundo), {len(x) / fs0:.0f} s. No zoom, o trecho de 0,512 s usado nos passos 3–7.")

# 2. decimação
ax = fig.add_subplot(gs[0, 1])
fw, pw = sci.welch(x0, fs0, nperseg=8192)
fd, pdd = sci.welch(xd, fs, nperseg=8192 // DEC)
b, a = sci.cheby1(8, 0.05, 0.8 / DEC, btype="low")
fh, h = sci.freqz(b, a, worN=4096, fs=fs0)
ax.semilogy(fw / 1000, pw, color="0.6", lw=0.8, label=f"original ({fs0 / 1000:.0f} kHz)")
ax.semilogy(fd / 1000, pdd, color=C1, lw=1, label=f"decimado ({fs / 1000:.0f} kHz)")
ax2 = ax.twinx(); ax2.plot(fh / 1000, np.abs(h), color=C2, lw=1.2, ls="--", label="filtro passa-baixas")
ax2.set_ylim(0, 1.6); ax2.set_ylabel("ganho do filtro", color=C2); ax2.tick_params(axis="y", colors=C2)
ax.axvline(fs / 2000, color="k", ls=":", lw=1); ax.text(fs / 2000 + 0.3, 1e-4, "novo limite\n(Nyquist) 8 kHz", fontsize=8)
ax.set_xlabel("frequência (kHz)"); ax.set_ylabel("densidade espectral")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
titulo(ax, 2, "Decimação por 3: 48 kHz → 16 kHz",
       "Passa-baixas (Chebyshev, corte em 6,4 kHz) e reamostragem por FFT para 1/3 das amostras. O LOFAR quase não sente: o TPSW compensa.")

# 3. janelamento
ax = fig.add_subplot(gs[1, 0])
seg = xd[iw * N:(iw + nm) * N]; ts = np.arange(len(seg)) / fs
ax.plot(ts, seg, color="0.55", lw=0.4)
amp = np.abs(seg).max()
for j in range(nm):
    s = slice(j * N, (j + 1) * N)
    ax.plot(ts[s], np.hanning(N) * amp, color=C2, lw=1.2)
    ax.axvline(j * N / fs, color="k", lw=0.6, ls=":")
    ax.text((j + 0.5) * N / fs, -amp * 1.12, f"janela {j + 1}\n2048 amostras\n0,128 s", ha="center", va="top", fontsize=8)
ax.set_ylim(-amp * 1.6, amp * 1.15); ax.set_xlabel("tempo (s)")
titulo(ax, 3, "Corte em janelas de 2048 amostras (0,128 s)",
       "Cada janela é multiplicada pela janela de Hanning (vermelho), que suaviza as bordas. Sem sobreposição.")

# 4. FFT de uma janela
ax = fig.add_subplot(gs[1, 1])
k1 = f <= 1000
ax.plot(f[k1], LOG[k1, iw], color=C1, lw=0.9)
ax.set_xlabel("frequência (Hz)"); ax.set_ylabel("20·log10(magnitude) (dB)")
ins = ax.inset_axes([0.55, 0.5, 0.42, 0.38]); kz = (f >= 20) & (f <= 110)
ins.stem(f[kz], LOG[kz, iw], basefmt=" ", linefmt=C1, markerfmt="o")
ins.set_ylim(LOG[kz, iw].min() - 3, LOG[kz, iw].max() + 3)
ins.set_title("zoom 20–110 Hz: cada ponto é uma faixa de 7,8 Hz", fontsize=7.5); ins.tick_params(labelsize=7)
titulo(ax, 4, "FFT de cada janela e escala em dB",
       "16000 / 2048 = 7,8125 Hz por faixa: 1024 faixas até 8 kHz (a de 0 Hz é descartada). Mostrado só até 1 kHz.")

# 5. TPSW
ax = fig.add_subplot(gs[2, 0])
ax.plot(f[k1], LOG[k1, iw], color="0.6", lw=0.8, label="espectro da janela (dB)")
ax.plot(f[k1], BG[k1, iw], color=C2, lw=2, label="fundo estimado (TPSW)")
ax.set_xlabel("frequência (Hz)"); ax.set_ylabel("dB"); ax.legend(fontsize=8, loc="lower left")
n = int(round(1024 * .04 / 2 + 1)); p = int(round(n / 8 + 1))
hk = np.concatenate((np.ones(n - p + 1), np.zeros(2 * p - 1), np.ones(n - p + 1)))
ins = ax.inset_axes([0.6, 0.52, 0.37, 0.3])
ins.bar(np.arange(-n, n + 1) * 7.8125, hk, width=6, color=C2)
ins.set_title(f"pesos da média móvel (±{n} faixas = ±{n * 7.8125:.0f} Hz)", fontsize=7.5)
ins.set_xlabel("Hz em torno da faixa", fontsize=7); ins.tick_params(labelsize=7); ins.set_yticks([])
titulo(ax, 5, "TPSW: estimativa do fundo de banda larga",
       f"Média das vizinhas (±164 Hz) sem a própria faixa e as {p - 1} de cada lado; picos altos são cortados e a média, refeita.")

# 6. subtração
ax = fig.add_subplot(gs[2, 1])
d = LOG[k1, iw] - BG[k1, iw]
ax.plot(f[k1], d, color="0.7", lw=0.8, label="espectro − fundo")
ax.fill_between(f[k1], 0, LOF[k1, iw], color=C3, alpha=0.8, label="LOFAR (valores < −0,2 dB viram 0)")
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("frequência (Hz)"); ax.set_ylabel("dB acima do fundo")
ax.legend(fontsize=8, loc="upper right")
titulo(ax, 6, "Subtração do fundo e corte dos negativos",
       "O que sobra é quanto cada faixa sobe acima do fundo local: raias e bandas estreitas. O ruído de fundo vira ~0.")

# 7. integração
ax = fig.add_subplot(gs[3, 0])
for j in range(nm):
    ax.plot(f[k1], LOF[k1, iw + j], lw=0.7, alpha=0.6, label=f"janela {j + 1}")
ax.plot(f[k1], LOF[k1, iw:iw + nm].mean(1), color="k", lw=1.8, label="média = 1 linha do LOFAR")
ax.set_xlabel("frequência (Hz)"); ax.set_ylabel("dB acima do fundo"); ax.legend(fontsize=8, ncol=5, loc="upper right")
titulo(ax, 7, "Integração: média de 4 janelas → uma linha de 0,512 s",
       "Reduz o ruído entre janelas. Não muda a resolução em frequência, que continua 7,8 Hz.")

# 8. resumo dos números
ax = fig.add_subplot(gs[3, 1]); ax.axis("off")
linhas = [("Taxa original", f"{fs0 / 1000:.0f} kHz"),
          ("Decimação", "÷ 3 → 16 kHz (espectro até 8 kHz)"),
          ("Amostras por FFT (nfft)", "2048 → janela de 0,128 s"),
          ("Largura da faixa", "16000 / 2048 = 7,8125 Hz"),
          ("Faixas", "1024 (7,8 Hz a 8 kHz)"),
          ("Fundo (TPSW)", f"±{n} faixas (±164 Hz), sem a central e ±{p - 1} vizinhas"),
          ("Corte", "valores < −0,2 dB → 0"),
          ("Integração", "4 FFTs → 1 linha a cada 0,512 s"),
          ("Saída por gravação", f"{LOFI.shape[0]} linhas × 1024 faixas (esta gravação)"),
          ("Para a CNN LOFAR", "+ normalização min-max e L2 por linha")]
ax.set_title("8. Resumo dos parâmetros", loc="left", fontsize=12, fontweight="bold", color=C1, pad=22)
for i, (k, v) in enumerate(linhas):
    y = 0.95 - i * 0.095
    ax.add_patch(plt.Rectangle((0, y - 0.04), 1, 0.085, color="#eef3f8" if i % 2 == 0 else "white", transform=ax.transAxes))
    ax.text(0.02, y, k, fontsize=10, fontweight="bold", va="center", transform=ax.transAxes)
    ax.text(0.42, y, v, fontsize=10, va="center", transform=ax.transAxes)

# 9. resultado
ax = fig.add_subplot(gs[4, 0])
tl = np.arange(LOFI.shape[0]) * INTEG
im = ax.imshow(LOFI[:, k1].T, aspect="auto", origin="lower", extent=[0, tl[-1] + INTEG, f[0], 1000], cmap="viridis", vmin=0, vmax=15)
ax.annotate("banda 31–86 Hz", xy=(tl[-1] * 0.5, 60), xytext=(tl[-1] * 0.55, 200), color="w", fontsize=10,
            fontweight="bold", arrowprops=dict(arrowstyle="->", color="w", lw=1.5))
ax.set_xlabel("tempo (s)"); ax.set_ylabel("frequência (Hz)")
plt.colorbar(im, ax=ax, label="dB acima do fundo", pad=0.01)
titulo(ax, 9, "Resultado: o LOFARgrama (0–1 kHz)",
       "Cada coluna é uma linha de 0,512 s. Linhas horizontais = componentes que persistem no tempo.")

ax = fig.add_subplot(gs[4, 1])
ax.plot(f[k1], LOFI[:, k1].mean(0), color=C1, lw=1.2)
ax.axvspan(31, 86, color=C2, alpha=0.15, label="31–86 Hz")
ax.set_xlabel("frequência (Hz)"); ax.set_ylabel("dB acima do fundo"); ax.legend(fontsize=8)
titulo(ax, 10, "Espectro médio da gravação (média de todas as linhas)",
       "É o que a análise exploratória (NB35) compara entre gravações acertadas e erradas.")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=110)
print("figura salva em", OUT)
