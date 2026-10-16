"""
Blind Test (Teste Cego) of Pure HHT (Hilbert-Huang Transform) on the 4 Canonical IARA Audios.

Audios:
- ÁUDIO A: data/iara/A/A-0002.wav (SMALL: Pilot 10, 11m, lancha rápida)
- ÁUDIO B: data/iara/A/A-0001.wav (MEDIUM: Maersk Pacer, 74m, rebocador/supridor offshore)
- ÁUDIO C: data/iara/A/A-0004.wav (LARGE: Sinoglory, 322m, graneleiro de grande porte)
- ÁUDIO D: data/iara/E/E-1446.wav (AMBIENTAL: Ruído de fundo marinho)

Evaluates:
1. Decomposição EMD completa (IMFs e resíduo)
2. Espectro Marginal de Hilbert (HMS) na faixa de 0 a 1000 Hz (banda canônica do LOFAR)
3. Espectrograma de Hilbert Tempo-Frequência (Hilbert Spectrum)
4. Tabela de Métricas Físicas e Espectrais (Centróide, IMFs dominantes, Dispersão de Energia)
"""

import os
import time
import numpy as np
import scipy.signal as scipy_sig
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from PyEMD import EMD

def analyze_audio_hht_pure(filepath, duration_sec=10.0, fs_target=2000, max_freq_hz=1000.0, n_bins=500):
    """
    Performs full HHT analysis on a representative segment of an audio file:
    - Decimation to fs_target (Nyquist = fs_target / 2 = 1000 Hz)
    - Multiframe EMD decomposition (windows of 2.56s)
    - Hilbert Transform -> Instantaneous Frequency & Amplitude
    - Time-Frequency Hilbert Spectrum and Average Marginal Spectrum
    """
    fs, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    
    total_sec = len(data) / fs
    start_sec = max(0, (total_sec - duration_sec) / 2)
    start_idx = int(start_sec * fs)
    end_idx = start_idx + int(duration_sec * fs)
    segment = data[start_idx:end_idx]
    
    # Decimate to fs_target = 2000 Hz
    decim_factor = int(round(fs / fs_target))
    sig_dec = scipy_sig.decimate(segment, decim_factor, ftype='fir', zero_phase=True)
    actual_fs = fs / decim_factor
    
    # Normalize
    sig_dec = sig_dec - np.mean(sig_dec)
    std_val = np.std(sig_dec)
    if std_val > 0:
        sig_dec = sig_dec / std_val
        
    # Analyze in windows of 2.56s (5120 samples) with 50% overlap
    win_len = int(2.56 * actual_fs)
    hop_len = int(win_len // 2)
    n_windows = max(1, (len(sig_dec) - win_len) // hop_len + 1)
    
    freq_grid = np.linspace(1.0, max_freq_hz, n_bins)
    df = freq_grid[1] - freq_grid[0]
    
    accumulated_hms = np.zeros(n_bins)
    window_hms_list = []
    window_times = []
    
    imf_energies_all = []
    imf_mean_freqs_all = []
    total_emd_time = 0.0
    
    emd = EMD()
    emd.MAX_ITERATION = 100
    
    sample_imfs = None
    sample_sig = None
    
    for w in range(n_windows):
        w_start = w * hop_len
        w_end = w_start + win_len
        if w_end > len(sig_dec):
            break
            
        w_sig = sig_dec[w_start:w_end]
        t_center = (w_start + win_len / 2.0) / actual_fs
        window_times.append(t_center)
        
        t0 = time.time()
        imfs = emd(w_sig)
        total_emd_time += (time.time() - t0)
        
        if sample_imfs is None:
            sample_imfs = imfs
            sample_sig = w_sig
            
        dt = 1.0 / actual_fs
        w_hms = np.zeros(n_bins)
        
        n_imfs = imfs.shape[0]
        energies = []
        mean_freqs = []
        
        for k in range(n_imfs):
            imf = imfs[k]
            analytic = scipy_sig.hilbert(imf)
            amp = np.abs(analytic)
            phase = np.unwrap(np.angle(analytic))
            inst_freq = np.gradient(phase, dt) / (2.0 * np.pi)
            
            # IMF energy
            imf_en = np.sum(amp ** 2)
            energies.append(imf_en)
            
            # Mask valid frequencies
            mask = (inst_freq >= 1.0) & (inst_freq <= max_freq_hz)
            if np.any(mask):
                weighted_freq = np.sum(inst_freq[mask] * (amp[mask] ** 2)) / np.sum(amp[mask] ** 2)
                mean_freqs.append(weighted_freq)
                
                bin_idx = np.clip(np.floor((inst_freq[mask] - 1.0) / df).astype(int), 0, n_bins - 1)
                for b, en in zip(bin_idx, amp[mask] ** 2):
                    w_hms[b] += en
            else:
                mean_freqs.append(0.0)
                
        accumulated_hms += w_hms
        window_hms_list.append(w_hms)
        imf_energies_all.append(energies)
        imf_mean_freqs_all.append(mean_freqs)
        
    # Average HMS
    if np.max(accumulated_hms) > 0:
        hms_normalized = accumulated_hms / np.max(accumulated_hms)
    else:
        hms_normalized = accumulated_hms
        
    # Spectral Centroid of HHT
    centroid_hz = np.sum(freq_grid * hms_normalized) / max(1e-9, np.sum(hms_normalized))
    
    # 85% Rolloff frequency
    cum_energy = np.cumsum(hms_normalized)
    cum_energy /= max(1e-9, cum_energy[-1])
    rolloff_idx = np.where(cum_energy >= 0.85)[0]
    rolloff_hz = freq_grid[rolloff_idx[0]] if len(rolloff_idx) > 0 else max_freq_hz
    
    # Dominant peak in HMS
    peak_idx = np.argmax(hms_normalized)
    peak_freq_hz = freq_grid[peak_idx]
    
    # Time-Frequency Matrix
    tf_matrix = np.array(window_hms_list).T  # (n_bins, n_windows)
    
    return {
        'filepath': filepath,
        'actual_fs': actual_fs,
        'duration_analyzed': duration_sec,
        'n_windows': n_windows,
        'total_emd_time': total_emd_time,
        'avg_emd_time_per_win': total_emd_time / max(1, n_windows),
        'freq_grid': freq_grid,
        'hms_normalized': hms_normalized,
        'centroid_hz': centroid_hz,
        'rolloff_hz': rolloff_hz,
        'peak_freq_hz': peak_freq_hz,
        'tf_matrix': tf_matrix,
        'window_times': np.array(window_times),
        'sample_imfs': sample_imfs,
        'sample_sig': sample_sig
    }

def main():
    print("=" * 80)
    print("EXTRAÇÃO DE HHT PURA (TESTE CEGO NOS 4 ÁUDIOS CANÔNICOS DO IARA)")
    print("=" * 80)
    
    targets = [
        ('ÁUDIO A (SMALL - Pilot 10, 11m)', 'data/iara/A/A-0002.wav', '#1f77b4'),
        ('ÁUDIO B (MEDIUM - Maersk Pacer, 74m)', 'data/iara/A/A-0001.wav', '#ff7f0e'),
        ('ÁUDIO C (LARGE - Sinoglory, 322m)', 'data/iara/A/A-0004.wav', '#2ca02c'),
        ('ÁUDIO D (AMBIENTAL - Fundo do Mar)', 'data/iara/E/E-1446.wav', '#7f7f7f'),
    ]
    
    results = {}
    
    for label, path, color in targets:
        print(f"\n--> Analisando HHT Pura: {label}")
        res = analyze_audio_hht_pure(path, duration_sec=10.0, fs_target=2000, max_freq_hz=1000.0, n_bins=500)
        results[label] = (res, color)
        print(f"    Tempo total EMD: {res['total_emd_time']:.2f}s ({res['n_windows']} janelas de 2.56s)")
        print(f"    Pico Dominante: {res['peak_freq_hz']:.1f} Hz")
        print(f"    Centróide Espectral: {res['centroid_hz']:.1f} Hz")
        print(f"    Frequência de Rolloff (85%): {res['rolloff_hz']:.1f} Hz")
        print(f"    Número de IMFs detectadas: {res['sample_imfs'].shape[0]}")
        
    # =========================================================================
    # FIGURA 1: PAINEL COMPARATIVO DO ESPECTRO MARGINAL DE HILBERT (HMS)
    # =========================================================================
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("Teste Cego: Espectro Marginal de Hilbert (HHT Pura - Banda 0 a 1000 Hz)", fontsize=14, fontweight='bold', y=0.99)
    
    for i, (label, (res, color)) in enumerate(results.items()):
        ax = axes[i]
        fg = res['freq_grid']
        hms = res['hms_normalized']
        
        ax.plot(fg, hms, color=color, lw=1.5, label=label)
        
        # Mark peak
        p_idx = np.argmax(hms)
        ax.plot(fg[p_idx], hms[p_idx], 'ro', markersize=5)
        ax.annotate(f"Pico: {res['peak_freq_hz']:.1f} Hz\nCentróide: {res['centroid_hz']:.1f} Hz",
                    (fg[p_idx], hms[p_idx]),
                    textcoords="offset points", xytext=(15, -15),
                    fontsize=8.5, fontweight='bold', color='darkred',
                    bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", color='darkred'))
                    
        # Mark Rolloff 85%
        ax.axvline(res['rolloff_hz'], color='black', ls=':', lw=1.2, alpha=0.7)
        ax.text(res['rolloff_hz'] + 5, 0.7, f"Rolloff 85%:\n{res['rolloff_hz']:.0f} Hz", fontsize=8, color='black', alpha=0.8)
        
        ax.set_ylabel("Energia Normalizada", fontsize=9)
        ax.set_title(f"{label} | EMD: {res['sample_imfs'].shape[0]} IMFs ({res['total_emd_time']:.2f}s)", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, ls='--')
        ax.set_xlim(0, 1000)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(loc='upper right', fontsize=8.5)
        
    axes[-1].set_xlabel("Frequência Instantânea Hilbert (Hz)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig1_path = "reports/teste_cego_hht_marginal_spectrum.png"
    plt.savefig(fig1_path, dpi=160)
    plt.close()
    print(f"\n[OK] Gráfico salvo: {fig1_path}")
    
    # =========================================================================
    # FIGURA 2: ESPECTROGRAMA DE HILBERT NO TEMPO (HILBERT-HUANG SPECTRUM 2D)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    fig.suptitle("Distribuição Tempo-Frequência de Hilbert (Espectrograma HHT 2D)", fontsize=14, fontweight='bold', y=0.99)
    
    axes_list = axes.flatten()
    for i, (label, (res, color)) in enumerate(results.items()):
        ax = axes_list[i]
        tf = res['tf_matrix']
        tf_db = 10 * np.log10(tf + 1e-6)
        tf_db -= np.max(tf_db)
        tf_db = np.maximum(tf_db, -40.0)
        
        im = ax.imshow(
            tf_db, origin='lower', aspect='auto',
            extent=[res['window_times'][0], res['window_times'][-1], res['freq_grid'][0], res['freq_grid'][-1]],
            cmap='inferno', vmin=-40, vmax=0
        )
        ax.set_title(label, fontsize=10.5, fontweight='bold')
        ax.set_ylabel("Frequência Instantânea (Hz)", fontsize=9.5)
        ax.set_xlabel("Tempo (s)", fontsize=9.5)
        fig.colorbar(im, ax=ax, label="Energia (dB)", pad=0.02)
        
    plt.tight_layout()
    fig2_path = "reports/teste_cego_hht_espectrograma_2d.png"
    plt.savefig(fig2_path, dpi=160)
    plt.close()
    print(f"[OK] Gráfico salvo: {fig2_path}")

    # =========================================================================
    # FIGURA 3: DECOMPOSIÇÃO DE IMFS DE CADA ÁUDIO
    # =========================================================================
    fig, axes = plt.subplots(6, 4, figsize=(18, 11), sharex=True)
    fig.suptitle("Decomposição EMD: Comparativo das Primeiras 6 IMFs (Teste Cego)", fontsize=13, fontweight='bold', y=0.99)
    
    col_names = ["ÁUDIO A (SMALL)", "ÁUDIO B (MEDIUM)", "ÁUDIO C (LARGE)", "ÁUDIO D (AMBIENTAL)"]
    t_axis = np.arange(len(list(results.values())[0][0]['sample_sig'])) / list(results.values())[0][0]['actual_fs']
    
    for col, (label, (res, color)) in enumerate(results.items()):
        imfs = res['sample_imfs']
        axes[0, col].set_title(col_names[col], fontsize=10.5, fontweight='bold', color=color)
        for row in range(6):
            ax = axes[row, col]
            if row < imfs.shape[0]:
                ax.plot(t_axis, imfs[row], color=color, lw=0.9)
                ax.grid(True, alpha=0.25)
            if col == 0:
                ax.set_ylabel(f"IMF {row+1}", fontsize=9, fontweight='bold')
            if row == 5:
                ax.set_xlabel("Tempo (s)", fontsize=8.5)
                
    plt.tight_layout()
    fig3_path = "reports/teste_cego_hht_imfs_comparacao.png"
    plt.savefig(fig3_path, dpi=160)
    plt.close()
    print(f"[OK] Gráfico salvo: {fig3_path}")

if __name__ == '__main__':
    main()
