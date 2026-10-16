"""
Visual comparison: HHT Pura (como categoria independente) vs. LOFAR nos 4 áudios do IARA.

Mostra claramente:
- LOFAR: STFT (Fourier) + TPSW diretamente no sinal acústico (0 a 500 Hz).
- HHT Pura: EMD + Transformada de Hilbert diretamente no sinal acústico (0 a 500 Hz), SEM nenhuma demodulação DEMON.
"""

import os
import numpy as np
import scipy.signal as scipy_sig
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from PyEMD import EMD

def compute_lofar(signal, fs, n_fft=1024):
    """LOFAR standard processing (STFT magnitude averaged over time)."""
    f, t, zxx = scipy_sig.stft(signal, fs=fs, window='hann', nperseg=n_fft, noverlap=n_fft//2)
    psd = np.mean(np.abs(zxx)**2, axis=1)
    mask = f <= 500.0
    f = f[mask]
    psd = psd[mask]
    if np.max(psd) > 0:
        psd = psd / np.max(psd)
    return f, psd

def compute_hht_pure(signal, fs, max_freq=500.0, n_bins=250):
    """HHT Pura: EMD direto no sinal acústico -> Hilbert -> Espectro Marginal."""
    emd = EMD()
    emd.MAX_ITERATION = 100
    imfs = emd(signal)
    
    dt = 1.0 / fs
    freq_grid = np.linspace(1.0, max_freq, n_bins)
    df = freq_grid[1] - freq_grid[0]
    marginal_spec = np.zeros(n_bins)
    
    for imf in imfs[:8]:
        analytic = scipy_sig.hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        freq = np.gradient(phase, dt) / (2.0 * np.pi)
        
        mask = (freq >= 1.0) & (freq <= max_freq)
        bin_idx = np.clip(np.floor((freq[mask] - 1.0) / df).astype(int), 0, n_bins - 1)
        energy = amp[mask] ** 2
        for b, e in zip(bin_idx, energy):
            marginal_spec[b] += e
            
    if np.max(marginal_spec) > 0:
        marginal_spec = marginal_spec / np.max(marginal_spec)
        
    return freq_grid, marginal_spec, imfs

def main():
    samples = [
        ('SMALL (Pilot 10, 11m)', 'data/iara/A/A-0002.wav', '#1f77b4'),
        ('MEDIUM (Maersk Pacer, 74m)', 'data/iara/A/A-0001.wav', '#ff7f0e'),
        ('LARGE (Sinoglory, 322m)', 'data/iara/A/A-0004.wav', '#2ca02c'),
        ('AMBIENTAL (Ruído Fundo)', 'data/iara/E/E-1446.wav', '#7f7f7f'),
    ]
    
    duration_sec = 2.0  # 2.0s segment decimated to 1000 Hz = 2000 points (fast EMD)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    fig.suptitle("HHT como Nova Categoria: LOFAR (Fourier) vs. HHT Pura (EMD + Hilbert) no Sinal Acústico Direto", fontsize=14, fontweight='bold', y=0.99)
    
    for i, (label, filepath, color) in enumerate(samples):
        fs, data = wavfile.read(filepath)
        if data.ndim > 1: data = data[:, 0]
        data = data.astype(np.float32)
        
        # 2s segment
        total_sec = len(data) / fs
        start_sec = max(0, (total_sec - duration_sec) / 2)
        s_idx = int(start_sec * fs)
        e_idx = s_idx + int(duration_sec * fs)
        segment = data[s_idx:e_idx]
        
        # Decimate to fs = 1000 Hz (baseband 0-500 Hz)
        decim = int(fs / 1000)
        sig_dec = scipy_sig.decimate(segment, decim, ftype='fir', zero_phase=True)
        fs_base = fs / decim
        
        # 1. LOFAR (Fourier direto)
        f_lofar, psd_lofar = compute_lofar(sig_dec, fs_base, n_fft=512)
        
        # 2. HHT Pura (EMD direto)
        f_hht, spec_hht, imfs = compute_hht_pure(sig_dec, fs_base, max_freq=500.0, n_bins=250)
        
        # Left: LOFAR
        ax_l = axes[i, 0]
        ax_l.plot(f_lofar, psd_lofar, color=color, lw=1.6, label=label)
        ax_l.set_title(f"LOFAR (Fourier do Sinal): {label}", fontsize=10, fontweight='bold')
        ax_l.set_ylabel("PSD Normalizada", fontsize=9)
        ax_l.grid(True, alpha=0.3, ls='--')
        ax_l.set_xlim(0, 500)
        ax_l.legend(loc='upper right', fontsize=8)
        
        # Right: HHT Pura
        ax_r = axes[i, 1]
        ax_r.plot(f_hht, spec_hht, color=color, lw=1.6, label=f"HHT Pura ({imfs.shape[0]} IMFs)")
        ax_r.set_title(f"HHT Pura (EMD + Hilbert do Sinal): {label}", fontsize=10, fontweight='bold')
        ax_r.set_ylabel("Energia Marginal HHT", fontsize=9)
        ax_r.grid(True, alpha=0.3, ls='--')
        ax_r.set_xlim(0, 500)
        ax_r.legend(loc='upper right', fontsize=8)
        
    axes[-1, 0].set_xlabel("Frequência (Hz)", fontsize=11, fontweight='bold')
    axes[-1, 1].set_xlabel("Frequência Instantânea (Hz)", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    out_path = "reports/hht_pura_vs_lofar.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[OK] Salvo em: {out_path}")

if __name__ == '__main__':
    main()
