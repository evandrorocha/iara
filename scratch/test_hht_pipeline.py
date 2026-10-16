"""
Prototype and test of HHT (Hilbert-Huang Transform) on the 4 canonical IARA audios.
Audios:
- SMALL: data/iara/A/A-0002.wav (Pilot 10, 11m)
- MEDIUM: data/iara/A/A-0001.wav (Maersk Pacer, 74m)
- LARGE: data/iara/A/A-0004.wav (Sinoglory, 322m)
- AMBIENTAL: data/iara/E/E-1446.wav (Ruído ambiente)
"""

import os
import time
import numpy as np
import scipy.signal as scipy_sig
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from PyEMD import EMD

def compute_hht_features(signal, fs, n_imfs=6, max_freq_hz=500.0, n_freq_bins=256):
    """
    Computes HHT: EMD decomposition -> Hilbert transform -> Instantaneous Freqs -> Hilbert Marginal Spectrum.
    """
    # 1. EMD
    t_start = time.time()
    emd = EMD()
    emd.MAX_ITERATION = 100
    
    # Run EMD
    imfs = emd(signal)
    emd_time = time.time() - t_start
    
    # We take top n_imfs (or pad if fewer)
    n_detected = imfs.shape[0]
    
    # 2. Hilbert Transform for each IMF
    dt = 1.0 / fs
    instantaneous_amps = []
    instantaneous_freqs = []
    
    for i in range(min(n_detected, n_imfs)):
        imf = imfs[i]
        analytic = scipy_sig.hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        # Instantaneous frequency: d(phase)/dt / (2*pi)
        freq = np.gradient(phase, dt) / (2.0 * np.pi)
        
        instantaneous_amps.append(amp)
        instantaneous_freqs.append(freq)
        
    # 3. Compute Hilbert Marginal Spectrum (HMS)
    freq_grid = np.linspace(0, max_freq_hz, n_freq_bins)
    df = freq_grid[1] - freq_grid[0]
    marginal_spectrum = np.zeros(n_freq_bins)
    
    for amp, freq in zip(instantaneous_amps, instantaneous_freqs):
        # Accumulate energy where frequency falls within [0, max_freq_hz]
        valid_mask = (freq >= 0) & (freq <= max_freq_hz)
        valid_freqs = freq[valid_mask]
        valid_energy = (amp[valid_mask] ** 2)
        
        # Bin accumulation
        bin_indices = np.clip(np.floor(valid_freqs / df).astype(int), 0, n_freq_bins - 1)
        for b_idx, en in zip(bin_indices, valid_energy):
            marginal_spectrum[b_idx] += en
            
    # Normalize marginal spectrum
    if np.max(marginal_spectrum) > 0:
        marginal_spectrum_norm = marginal_spectrum / np.sum(marginal_spectrum)
    else:
        marginal_spectrum_norm = marginal_spectrum
        
    return {
        'imfs': imfs,
        'n_imfs': n_detected,
        'emd_time': emd_time,
        'freq_grid': freq_grid,
        'marginal_spectrum': marginal_spectrum,
        'marginal_spectrum_norm': marginal_spectrum_norm,
        'instantaneous_amps': instantaneous_amps,
        'instantaneous_freqs': instantaneous_freqs
    }

def main():
    print("=" * 70)
    print("TESTE DE HHT (HILBERT-HUANG TRANSFORM) NOS 4 ÁUDIOS CANÔNICOS")
    print("=" * 70)
    
    samples = [
        ('SMALL (Pilot 10, 11m)', 'data/iara/A/A-0002.wav', '#1f77b4'),
        ('MEDIUM (Maersk Pacer, 74m)', 'data/iara/A/A-0001.wav', '#ff7f0e'),
        ('LARGE (Sinoglory, 322m)', 'data/iara/A/A-0004.wav', '#2ca02c'),
        ('AMBIENTAL (Ruído Fundo)', 'data/iara/E/E-1446.wav', '#7f7f7f'),
    ]
    
    # Let's test with a 2-second and 5-second window decimated to fs=1000 Hz or fs=2000 Hz
    # Target fs for passive sonar low-frequency band (0-500 Hz): fs_target = 1000 Hz
    fs_target = 1000
    duration_sec = 2.0  # 2 seconds = 2000 points
    
    results = []
    
    for label, filepath, color in samples:
        print(f"\nCarregando: {label} -> {filepath}")
        if not os.path.exists(filepath):
            print(f"  [ERRO] Arquivo não encontrado: {filepath}")
            continue
            
        fs, data = wavfile.read(filepath)
        if data.ndim > 1:
            data = data[:, 0]
            
        # Segment around CPA
        total_sec = len(data) / fs
        start_sec = max(0, (total_sec - duration_sec) / 2)
        start_idx = int(start_sec * fs)
        end_idx = start_idx + int(duration_sec * fs)
        segment = data[start_idx:end_idx].astype(np.float32)
        
        # Decimate to fs_target = 1000 Hz
        decim_factor = int(round(fs / fs_target))
        sig_decimated = scipy_sig.decimate(segment, decim_factor, ftype='fir', zero_phase=True)
        actual_fs = fs / decim_factor
        
        print(f"  fs original = {fs} Hz | Fator decimação = {decim_factor} -> fs final = {actual_fs:.1f} Hz")
        print(f"  Pontos no sinal: {len(sig_decimated)} ({len(sig_decimated)/actual_fs:.2f} s)")
        
        res = compute_hht_features(sig_decimated, actual_fs, n_imfs=6, max_freq_hz=500.0, n_freq_bins=256)
        print(f"  IMFs detectadas: {res['n_imfs']} | Tempo EMD: {res['emd_time']:.3f} s")
        results.append((label, color, sig_decimated, actual_fs, res))
        
    # Generate Visualizations
    # Figure 1: Hilbert Marginal Spectrum Comparison
    plt.figure(figsize=(12, 6))
    for label, color, _, _, res in results:
        plt.plot(res['freq_grid'], res['marginal_spectrum_norm'], label=f"{label} (EMD: {res['emd_time']:.2f}s)", color=color, lw=1.8)
    plt.title("Espectro Marginal de Hilbert (HHT) - Banda 0 a 500 Hz", fontsize=13, fontweight='bold')
    plt.xlabel("Frequência Instantânea (Hz)", fontsize=11)
    plt.ylabel("Densidade de Energia Marginal Normalizada", fontsize=11)
    plt.grid(True, alpha=0.3, ls='--')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("scratch/hht_marginal_spectrum_comparison.png", dpi=160)
    plt.close()
    print("\n[OK] Gráfico salvo: scratch/hht_marginal_spectrum_comparison.png")
    
    # Figure 2: Detailed IMFs breakdown for SMALL vs LARGE
    fig, axes = plt.subplots(6, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Decomposição EMD: Comparativo de IMFs (SMALL vs. LARGE)", fontsize=13, fontweight='bold')
    
    # small is results[0], large is results[2]
    small_imfs = results[0][4]['imfs']
    large_imfs = results[2][4]['imfs']
    t_axis = np.arange(len(results[0][2])) / results[0][3]
    
    for i in range(min(6, small_imfs.shape[0])):
        axes[i, 0].plot(t_axis, small_imfs[i], color='#1f77b4', lw=1)
        axes[i, 0].set_ylabel(f"IMF {i+1}", fontsize=9, fontweight='bold')
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title("SMALL (Pilot 10)", fontsize=11, fontweight='bold')
            
    for i in range(min(6, large_imfs.shape[0])):
        axes[i, 1].plot(t_axis, large_imfs[i], color='#2ca02c', lw=1)
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title("LARGE (Sinoglory)", fontsize=11, fontweight='bold')
            
    axes[-1, 0].set_xlabel("Tempo (s)", fontsize=10)
    axes[-1, 1].set_xlabel("Tempo (s)", fontsize=10)
    plt.tight_layout()
    plt.savefig("scratch/hht_imf_decomposition.png", dpi=160)
    plt.close()
    print("[OK] Gráfico salvo: scratch/hht_imf_decomposition.png")

if __name__ == '__main__':
    main()
