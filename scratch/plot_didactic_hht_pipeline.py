import os
import sys
import numpy as np
import scipy.signal as scipy_sig
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PyEMD import EMD

def generate_didactic_hht_image():
    # 1. Carregar trecho de áudio real do IARA (A-0002.wav - SMALL / Pilot 10)
    wav_path = "data/iara/A/A-0002.wav"
    fs, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)

    # Decimação para fs = 2000 Hz
    fs_target = 2000
    decim = int(round(fs / fs_target))
    sig_dec = scipy_sig.decimate(data, decim, ftype='fir', zero_phase=True)
    actual_fs = fs / decim

    # Normalização
    sig_dec = sig_dec - np.mean(sig_dec)
    std_val = np.std(sig_dec)
    if std_val > 0:
        sig_dec /= std_val

    # Extrair janela de 2.56 s no trecho central (CPA)
    win_len = int(2.56 * actual_fs) # 5120 amostras
    start_idx = int(len(sig_dec) // 2 - win_len // 2)
    win_sig = sig_dec[start_idx : start_idx + win_len]
    t_win = np.linspace(0, 2.56, win_len)

    # Decomposição EMD
    emd = EMD()
    emd.MAX_ITERATION = 50
    imfs = emd(win_sig, max_imf=6)
    n_imfs = min(6, imfs.shape[0])

    # Transformada de Hilbert e Cálculo de HMS
    dt = 1.0 / actual_fs
    max_freq_hz = 1000.0
    n_bins = 256
    freq_grid = np.linspace(1.0, max_freq_hz, n_bins)
    df = freq_grid[1] - freq_grid[0]

    hms = np.zeros(n_bins, dtype=np.float32)
    amps = []
    freqs = []

    for imf in imfs[:n_imfs]:
        analytic = scipy_sig.hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.gradient(phase, dt) / (2.0 * np.pi)

        amps.append(amp)
        freqs.append(inst_freq)

        mask = (inst_freq >= 1.0) & (inst_freq <= max_freq_hz)
        bin_idx = np.clip(np.floor((inst_freq[mask] - 1.0) / df).astype(int), 0, n_bins - 1)
        energy = (amp[mask] ** 2).astype(np.float32)
        np.add.at(hms, bin_idx, energy)

    if np.max(hms) > 0:
        hms /= np.max(hms)

    # -------------------------------------------------------------
    # CONFIGURAÇÃO VISUAL DA FIGURA (Infográfico Científico Premium)
    # -------------------------------------------------------------
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(20, 11.5), dpi=220, facecolor='#f1f5f9')

    c_slate_900 = '#0f172a'
    c_slate_700 = '#334155'
    c_slate_500 = '#64748b'
    c_blue = '#0284c7'
    c_emerald = '#059669'
    c_amber = '#d97706'
    c_violet = '#7c3aed'
    c_card_bg = '#ffffff'
    c_card_edge = '#cbd5e1'

    # Título Principal
    fig.text(0.5, 0.966, "FLUXO DE EXTRAÇÃO DA TRANSFORMADA DE HILBERT-HUANG PURA (HHT)", 
             ha='center', va='center', fontsize=20, fontweight='heavy', color=c_slate_900)
    fig.text(0.5, 0.940, "Do Sinal Hidroacústico Bruto (WAV) ao Vetor Espectral de 256 Dimensões para Machine Learning — Projeto IARA",
             ha='center', va='center', fontsize=11.5, fontweight='medium', color=c_slate_500)

    gs = gridspec.GridSpec(2, 4, figure=fig, 
                           left=0.045, right=0.955, bottom=0.06, top=0.91, 
                           wspace=0.28, hspace=0.30,
                           height_ratios=[1.0, 1.35])

    def setup_card_axes(ax, title, badge_text, badge_color):
        ax.set_facecolor(c_card_bg)
        for spine in ax.spines.values():
            spine.set_edgecolor(c_card_edge)
            spine.set_linewidth(1.3)
        # Cabeçalho limpo sem sobreposição
        header = f"[{badge_text}]  {title}"
        ax.set_title(header, loc='left', fontsize=10.8, fontweight='bold', color=badge_color, pad=12)

    # -------------------------------------------------------------
    # COLUNA 1: ETAPA 1 (ÁUDIO & PRÉ-PROCESSAMENTO)
    # -------------------------------------------------------------
    ax_raw = fig.add_subplot(gs[0, 0])
    setup_card_axes(ax_raw, "Áudio Hidroacústico Bruto", "ETAPA 1", c_blue)
    t_full = np.linspace(0, len(sig_dec)/actual_fs, len(sig_dec))
    ax_raw.plot(t_full, sig_dec, color='#94a3b8', lw=0.6, alpha=0.85)
    ax_raw.axvspan(start_idx/actual_fs, (start_idx+win_len)/actual_fs, color=c_blue, alpha=0.4, label='Janela Central (2,56 s)')
    ax_raw.set_xlabel("Tempo da Gravação (s) [~282 s]", fontsize=8.5, color=c_slate_700)
    ax_raw.set_ylabel("Amplitude Normalizada (z-score)", fontsize=8.5, color=c_slate_700)
    ax_raw.tick_params(labelsize=8)
    ax_raw.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax_raw.grid(True, alpha=0.35, ls='--')

    ax_info1 = fig.add_subplot(gs[1, 0])
    setup_card_axes(ax_info1, "Parâmetros de Decimação", "CONFIG", c_blue)
    ax_info1.set_xticks([])
    ax_info1.set_yticks([])
    
    info_text = (
        "1. TAXA DE AMOSTRAGEM:\n"
        "   • Entrada: 48 kHz (ou 44,1 kHz) mono\n"
        "   • Decimação FIR Fase-Zero: fs = 2000 Hz\n"
        "   • Nyquist útil: 1000 Hz\n\n"
        "2. NORMALIZAÇÃO Z-SCORE:\n"
        "   • x_norm = (x - média) / desvio_pad\n\n"
        "3. JANELAMENTO TEMPORAL:\n"
        "   • Tamanho da Janela: 2,56 s (5120 pts)\n"
        "   • Deslocamento (Hop): 5,12 s\n"
        "   • Total: ~55 janelas por gravação\n\n"
        "4. JUSTIFICATIVA FÍSICA:\n"
        "   • Captura ruídos de propulsão e casco\n"
        "     na mesma banda do LOFAR (1 a 1000 Hz),\n"
        "     porém sem restrições de FFT."
    )
    ax_info1.text(0.06, 0.50, info_text, transform=ax_info1.transAxes, fontsize=8.8, 
                  color=c_slate_700, va='center', family='monospace', linespacing=1.35)

    # -------------------------------------------------------------
    # COLUNA 2: ETAPA 2 (DECOMPOSIÇÃO EMD - IMFs)
    # -------------------------------------------------------------
    ax_win = fig.add_subplot(gs[0, 1])
    setup_card_axes(ax_win, "Janela de Sinal (2,56 s)", "ETAPA 2A", c_emerald)
    ax_win.plot(t_win, win_sig, color=c_blue, lw=1.05)
    ax_win.set_xlabel("Tempo da Janela (s)", fontsize=8.5, color=c_slate_700)
    ax_win.set_ylabel("Pressão Acústica (u.a.)", fontsize=8.5, color=c_slate_700)
    ax_win.tick_params(labelsize=8)
    ax_win.grid(True, alpha=0.35, ls='--')

    gs_imfs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1, 1], hspace=0.45)
    
    imf_indices = [0, 1, 3, min(5, n_imfs-1)]
    imf_cards = [
        ("IMF 1 (> 400 Hz)", "Cavitação rápida / água", c_emerald),
        ("IMF 2 (~200–400 Hz)", "Ressonância estrutural", '#0284c7'),
        ("IMF 4 (~50–150 Hz)", "Harmônicos de combustão", c_amber),
        (f"IMF {imf_indices[3]+1} (< 30 Hz)", "Eixo / passagem de pás", c_violet),
    ]

    for idx, (imf_i, (t_txt, desc_txt, c_imf)) in enumerate(zip(imf_indices, imf_cards)):
        ax_i = fig.add_subplot(gs_imfs[idx, 0])
        ax_i.set_facecolor('#ffffff')
        for sp in ax_i.spines.values():
            sp.set_edgecolor(c_card_edge)
            sp.set_linewidth(0.9)
        
        ax_i.plot(t_win, imfs[imf_i], color=c_imf, lw=0.95)
        an = scipy_sig.hilbert(imfs[imf_i])
        ax_i.plot(t_win, np.abs(an), color='#f43f5e', lw=0.8, ls='--', alpha=0.7)
        ax_i.plot(t_win, -np.abs(an), color='#f43f5e', lw=0.8, ls='--', alpha=0.7)

        ax_i.text(0.02, 0.82, f"{t_txt} — {desc_txt}", transform=ax_i.transAxes, 
                  fontsize=7.8, fontweight='bold', color=c_slate_900)
        ax_i.tick_params(labelsize=7)
        ax_i.grid(True, alpha=0.25, ls=':')
        if idx < 3:
            ax_i.set_xticks([])
        else:
            ax_i.set_xlabel("Tempo (s)", fontsize=8, color=c_slate_700)

    # -------------------------------------------------------------
    # COLUNA 3: ETAPA 3 (HILBERT & PLANO TEMPO-FREQUÊNCIA)
    # -------------------------------------------------------------
    ax_math = fig.add_subplot(gs[0, 2])
    setup_card_axes(ax_math, "Transformada de Hilbert", "ETAPA 3A", c_amber)
    ax_math.set_xticks([])
    ax_math.set_yticks([])
    
    math_text = (
        "SINAL ANALÍTICO POR IMF:\n"
        "  z_k(t) = IMF_k(t) + j·H{IMF_k(t)}\n"
        "         = a_k(t) · exp( j·θ_k(t) )\n\n"
        "PROPRIEDADES INSTANTÂNEAS:\n"
        "• Amplitude: a_k(t) = |z_k(t)| (envelope)\n"
        "• Fase:      θ_k(t) = unwrap( arg(z_k) )\n"
        "• Freq.:     f_k(t) = (1/2π) · (dθ_k / dt)\n\n"
        "VANTAGEM FUNDAMENTAL:\n"
        "Não sofre do limite de incerteza da FFT.\n"
        "Fornece energia e frequência a cada milissegundo."
    )
    ax_math.text(0.06, 0.50, math_text, transform=ax_math.transAxes, fontsize=8.6,
                 color=c_slate_700, va='center', family='monospace', linespacing=1.35)

    ax_tf = fig.add_subplot(gs[1, 2])
    setup_card_axes(ax_tf, "Espectrograma de Hilbert (2D)", "ETAPA 3B", c_amber)
    
    step = 4
    for amp, freq in zip(amps, freqs):
        v_mask = (freq[::step] >= 1.0) & (freq[::step] <= 1000.0)
        t_pts = t_win[::step][v_mask]
        f_pts = freq[::step][v_mask]
        a_pts = amp[::step][v_mask]
        
        a_norm = a_pts / (np.max(amp) + 1e-6)
        scatter_colors = plt.cm.magma(a_norm * 0.85 + 0.15)
        ax_tf.scatter(t_pts, f_pts, s=a_norm*13 + 1.2, color=scatter_colors, alpha=0.6, edgecolors='none')

    ax_tf.set_ylim(0, 1000)
    ax_tf.set_xlim(0, 2.56)
    ax_tf.set_xlabel("Tempo da Janela (s)", fontsize=8.5, color=c_slate_700)
    ax_tf.set_ylabel("Frequência Instantânea (Hz)", fontsize=8.5, color=c_slate_700)
    ax_tf.tick_params(labelsize=8)
    ax_tf.grid(True, alpha=0.35, ls='--')

    # Faixas explicativas com fundo limpo
    ax_tf.axhspan(400, 950, color='#f59e0b', alpha=0.07)
    ax_tf.text(0.04, 0.93, "Banda Larga de Cavitação (> 400 Hz)", transform=ax_tf.transAxes, 
               fontsize=8.0, fontweight='bold', color='#b45309',
               bbox=dict(fc='white', alpha=0.85, ec='none', pad=2))
    
    ax_tf.axhline(140, color='#0284c7', ls=':', lw=1.3)
    ax_tf.text(0.04, 0.16, "Linha Tônica de Máquinas (~140 Hz)", transform=ax_tf.transAxes, 
               fontsize=8.0, fontweight='bold', color='#0284c7',
               bbox=dict(fc='white', alpha=0.85, ec='none', pad=2))

    # -------------------------------------------------------------
    # COLUNA 4: ETAPA 4 (HMS 256 BINS) & ETAPA 5 (ML)
    # -------------------------------------------------------------
    ax_hms = fig.add_subplot(gs[0, 3])
    setup_card_axes(ax_hms, "Espectro Marginal (256 Bins)", "ETAPA 4", c_violet)

    ax_hms.plot(freq_grid, hms, color=c_violet, lw=1.5, label='HMS')
    ax_hms.fill_between(freq_grid, 0, hms, color=c_violet, alpha=0.22)
    
    peak_i = np.argmax(hms)
    ax_hms.plot(freq_grid[peak_i], hms[peak_i], 'o', color='#ef4444', markersize=5.5)
    ax_hms.annotate(f"Pico Dominante\n{freq_grid[peak_i]:.1f} Hz",
                    xy=(freq_grid[peak_i], hms[peak_i]),
                    xytext=(freq_grid[peak_i]+120, hms[peak_i]*0.88),
                    arrowprops=dict(arrowstyle="->", color='#ef4444', lw=1.1),
                    fontsize=8, fontweight='bold', color='#ef4444')

    ax_hms.set_xlim(1, 1000)
    ax_hms.set_ylim(0, 1.08)
    ax_hms.set_xlabel("Frequência (Hz) [1 a 1000 Hz]", fontsize=8.5, color=c_slate_700)
    ax_hms.set_ylabel("Energia Marginal (HMS)", fontsize=8.5, color=c_slate_700)
    ax_hms.tick_params(labelsize=8)
    ax_hms.grid(True, alpha=0.35, ls='--')

    ax_ml = fig.add_subplot(gs[1, 3])
    setup_card_axes(ax_ml, "Vetor Final para Machine Learning", "ETAPA 5", '#059669')
    ax_ml.set_xticks([])
    ax_ml.set_yticks([])

    ml_text = (
        "ESTRUTURA DE DADOS FINAL:\n"
        "───────────────────────────────\n"
        "• Vetor por Janela (1 x 256):\n"
        "  [hht_0, hht_1, ..., hht_255]\n"
        "• Arquivo Salvo (.pkl):\n"
        "  Matriz (~55 janelas x 256 bins)\n"
        "  -> data/iara_processed/hht_features/\n\n"
        "MODELOS AVALIADOS (5x2cv):\n"
        "1. Random Forest (100 árvores) -> SP: 58,14%\n"
        "2. MLP PyTorch [32, 16]        -> SP: 56,14%\n"
        "3. SVM Nyström (m=6000, RBF)   -> SP: 53,63%\n"
        "4. CNN 2D PyTorch [64, 32]\n\n"
        "ROTEAMENTO INTELIGENTE (MoE):\n"
        "O Sistema de Gating (Notebook 24)\n"
        "seleciona HHT quando há alta energia\n"
        "não-estacionária de máquinas."
    )
    ax_ml.text(0.06, 0.50, ml_text, transform=ax_ml.transAxes, fontsize=8.5,
               color=c_slate_700, va='center', family='monospace', linespacing=1.35)

    # Setas conectoras limpas entre colunas
    fig.text(0.268, 0.69, "➔", fontsize=16, color='#94a3b8', ha='center', va='center')
    fig.text(0.495, 0.69, "➔", fontsize=16, color='#94a3b8', ha='center', va='center')
    fig.text(0.722, 0.69, "➔", fontsize=16, color='#94a3b8', ha='center', va='center')

    # Salvar em reports e artifacts
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/extracao_hht_pura_didatica.png"
    plt.savefig(out_path, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Imagem didática otimizada salva em: {out_path}")

    artifact_dir = r"C:\Users\ev_ro\.gemini\antigravity-ide\brain\2415ff3c-9185-41f5-b445-dc42478468f8"
    if os.path.exists(artifact_dir):
        dst = os.path.join(artifact_dir, "extracao_hht_pura_didatica.png")
        import shutil
        shutil.copyfile(out_path, dst)
        print(f"[OK] Copiado para artifacts: {dst}")

if __name__ == "__main__":
    generate_didactic_hht_image()
