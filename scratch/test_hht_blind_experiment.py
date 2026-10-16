"""
Comprehensive Blind Experiment: HHT vs DEMON on the 4 Canonical IARA Audios.

Audios:
- SMALL: data/iara/A/A-0002.wav (Pilot 10, 11m)
- MEDIUM: data/iara/A/A-0001.wav (Maersk Pacer, 74m)
- LARGE: data/iara/A/A-0004.wav (Sinoglory, 322m)
- AMBIENTAL: data/iara/E/E-1446.wav (Ruído ambiente marinho)

Evaluates:
1. Classical DEMON (Chebyshev Bandpass 1-2 kHz -> Envelope Abs -> Decimate to 50 Hz -> STFT / FFT)
2. HHT-DEMON (Chebyshev Bandpass 1-2 kHz -> Envelope Abs -> Decimate to 50 Hz -> EMD -> Hilbert Spectrum)
3. HHT Direct Baseband (Decimate to 1 kHz -> EMD -> Hilbert Marginal Spectrum 0-100 Hz)
"""

import os
import time
import math
import numpy as np
import scipy.signal as scipy_sig
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from PyEMD import EMD

def get_factors(n: int):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def get_decimation_steps(fs_in: int, fs_out: int = 50):
    if fs_in % fs_out != 0:
        ratio = fs_in / fs_out
        r1 = int(math.floor(math.sqrt(ratio)))
        while ratio % r1 != 0 and r1 > 1:
            r1 -= 1
        r2 = int(ratio // r1)
        return r1, r2

    factor_list = get_factors(int(fs_in // fs_out))
    r1, r2 = 1, 1
    add_one = True
    while factor_list:
        if len(factor_list) == 1:
            p1, p2 = factor_list.pop(), 1
        elif len(factor_list) == 2:
            p1, p2 = factor_list.pop(0), factor_list.pop()
        else:
            p1, p2 = factor_list.pop(0) * factor_list.pop(), 1

        if add_one:
            r1 *= p1
            r2 *= p2
        else:
            r1 *= p2
            r2 *= p1
        add_one = not add_one
    return r1, r2

def extract_demodulated_envelope(signal, fs, passband=(1000, 2000), fs_out=50):
    """Isolates cavitation band and extracts envelope, decimating to fs_out."""
    nyq = fs / 2.0
    wp = [passband[0] / nyq, passband[1] / nyq]
    ws = [max(1e-4, (passband[0] - 300) / nyq), min(0.9999, (passband[1] + 300) / nyq)]
    N, wc = scipy_sig.cheb2ord(wp, ws, gpass=0.5, gstop=40.0)
    sos = scipy_sig.cheby2(N, rs=40, Wn=wc, btype='bandpass', output='sos', analog=False)
    filtered = scipy_sig.sosfilt(sos, signal)
    
    # Envelope extraction
    envelope = np.abs(filtered)
    
    # Decimation to fs_out
    r1, r2 = get_decimation_steps(fs, fs_out)
    env_dec = scipy_sig.decimate(envelope, r1, ftype='fir', zero_phase=True)
    env_dec = scipy_sig.decimate(env_dec, r2, ftype='fir', zero_phase=True)
    
    final_fs = (fs // r1) // r2
    # Normalize & remove DC
    env_dec = env_dec - np.mean(env_dec)
    max_val = np.max(np.abs(env_dec))
    if max_val > 0:
        env_dec = env_dec / max_val
    return env_dec, final_fs

def compute_classical_demon(env_dec, fs_dem, n_fft=512, overlap_ratio=0.5):
    """Classical DEMON via STFT on envelope."""
    n_overlap = int(n_fft * overlap_ratio)
    hop = n_fft - n_overlap
    freqs, times, zxx = scipy_sig.stft(
        env_dec, fs=fs_dem, window='hann',
        nperseg=n_fft, noverlap=n_overlap, nfft=n_fft
    )
    mag = np.abs(zxx)
    # Discard DC (< 0.5 Hz)
    valid_idx = freqs >= 0.39
    freqs = freqs[valid_idx]
    mag = mag[valid_idx, :]
    avg_spec = np.mean(mag, axis=1)
    if np.max(avg_spec) > 0:
        avg_spec = avg_spec / np.max(avg_spec)
    return freqs, avg_spec, times, mag

def compute_hht_marginal_spectrum(signal, fs, n_imfs=5, max_freq_hz=25.0, n_bins=200):
    """
    Computes HHT on a signal (e.g. envelope or decimated baseband):
    EMD -> Hilbert transform of each IMF -> Instantaneous frequency & energy -> Marginal spectrum.
    """
    t0 = time.time()
    emd = EMD()
    emd.MAX_ITERATION = 100
    imfs = emd(signal)
    emd_time = time.time() - t0
    
    dt = 1.0 / fs
    freq_grid = np.linspace(0.1, max_freq_hz, n_bins)
    df = freq_grid[1] - freq_grid[0]
    marginal_spec = np.zeros(n_bins)
    
    n_detected = imfs.shape[0]
    imf_details = []
    
    for i in range(min(n_detected, n_imfs)):
        imf = imfs[i]
        analytic = scipy_sig.hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        freq = np.gradient(phase, dt) / (2.0 * np.pi)
        
        # Valid frequency mask
        mask = (freq >= 0.1) & (freq <= max_freq_hz)
        bin_idx = np.clip(np.floor((freq[mask] - 0.1) / df).astype(int), 0, n_bins - 1)
        energy = amp[mask] ** 2
        for b, e in zip(bin_idx, energy):
            marginal_spec[b] += e
            
        imf_details.append({
            'imf': imf,
            'mean_freq': np.mean(freq[(freq >= 0) & (freq <= max_freq_hz)]) if np.any((freq >= 0) & (freq <= max_freq_hz)) else 0,
            'energy': np.sum(amp ** 2)
        })
        
    if np.max(marginal_spec) > 0:
        marginal_spec = marginal_spec / np.max(marginal_spec)
        
    return {
        'freq_grid': freq_grid,
        'marginal_spec': marginal_spec,
        'emd_time': emd_time,
        'n_imfs': n_detected,
        'imfs': imfs,
        'imf_details': imf_details
    }

def main():
    print("=" * 80)
    print("EXPERIMENTO CEGO: HHT vs. DEMON NOS 4 ÁUDIOS CANÔNICOS DO IARA")
    print("=" * 80)
    
    samples = [
        ('SMALL (Pilot 10, 11m)', 'data/iara/A/A-0002.wav', '#1f77b4'),
        ('MEDIUM (Maersk Pacer, 74m)', 'data/iara/A/A-0001.wav', '#ff7f0e'),
        ('LARGE (Sinoglory, 322m)', 'data/iara/A/A-0004.wav', '#2ca02c'),
        ('AMBIENTAL (Ruído Fundo)', 'data/iara/E/E-1446.wav', '#7f7f7f'),
    ]
    
    duration_sec = 30.0  # 30 seconds segment around CPA
    
    results = []
    
    for label, filepath, color in samples:
        print(f"\n---> Processando {label}: {filepath}")
        fs, data = wavfile.read(filepath)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float32)
        
        total_sec = len(data) / fs
        start_sec = max(0, (total_sec - duration_sec) / 2)
        start_idx = int(start_sec * fs)
        end_idx = start_idx + int(duration_sec * fs)
        segment = data[start_idx:end_idx]
        
        # 1. Classical DEMON Envelope
        env_dec, fs_dem = extract_demodulated_envelope(segment, fs, passband=(1000, 2000), fs_out=50)
        f_demon, spec_demon, times_demon, zxx_demon = compute_classical_demon(env_dec, fs_dem, n_fft=512)
        
        # Dominant peak in classical DEMON
        p_idx = np.argmax(spec_demon)
        f_peak_demon = f_demon[p_idx]
        rpm_demon = f_peak_demon * 60.0
        
        # 2. HHT-DEMON (HHT applied to the modulation envelope)
        res_hht_demon = compute_hht_marginal_spectrum(env_dec, fs_dem, n_imfs=5, max_freq_hz=25.0, n_bins=200)
        p_hht_idx = np.argmax(res_hht_demon['marginal_spec'])
        f_peak_hht = res_hht_demon['freq_grid'][p_hht_idx]
        rpm_hht = f_peak_hht * 60.0
        
        print(f"  [DEMON Clássico] Pico: {f_peak_demon:.2f} Hz ({rpm_demon:.0f} RPM)")
        print(f"  [HHT-DEMON]       Pico: {f_peak_hht:.2f} Hz ({rpm_hht:.0f} RPM) | Tempo EMD: {res_hht_demon['emd_time']:.3f} s")
        
        results.append({
            'label': label,
            'color': color,
            'f_demon': f_demon,
            'spec_demon': spec_demon,
            'f_peak_demon': f_peak_demon,
            'rpm_demon': rpm_demon,
            'hht_demon': res_hht_demon,
            'f_peak_hht': f_peak_hht,
            'rpm_hht': rpm_hht
        })
        
    # Generate 4-panel Comparative Figure: DEMON Clássico vs HHT-DEMON
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    fig.suptitle("Teste Cego: DEMON Clássico (FFT) vs. HHT-DEMON (Transformada de Hilbert-Huang no Envelope)", fontsize=14, fontweight='bold', y=0.99)
    
    for i, r in enumerate(results):
        ax_dem = axes[i, 0]
        ax_hht = axes[i, 1]
        
        # Left: DEMON Clássico
        ax_dem.plot(r['f_demon'], r['spec_demon'], color=r['color'], lw=1.8, label=r['label'])
        ax_dem.plot(r['f_peak_demon'], r['spec_demon'][np.argmax(r['spec_demon'])], 'ro', markersize=5)
        ax_dem.annotate(f"Pico: {r['f_peak_demon']:.2f} Hz\n({r['rpm_demon']:.0f} RPM)",
                        (r['f_peak_demon'], 1.0),
                        textcoords="offset points", xytext=(0, -20),
                        ha='center', fontsize=8.5, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6))
        ax_dem.set_ylabel("Intensidade Normalizada", fontsize=9)
        ax_dem.set_title(f"DEMON Clássico (FFT de Envelope): {r['label']}", fontsize=10, fontweight='bold')
        ax_dem.grid(True, alpha=0.3, ls='--')
        ax_dem.set_xlim(0, 25)
        ax_dem.legend(loc='upper right', fontsize=8)
        
        # Right: HHT-DEMON
        hht = r['hht_demon']
        ax_hht.plot(hht['freq_grid'], hht['marginal_spec'], color=r['color'], lw=1.8, label=f"HHT (EMD: {hht['emd_time']:.2f}s)")
        p_idx = np.argmax(hht['marginal_spec'])
        ax_hht.plot(r['f_peak_hht'], hht['marginal_spec'][p_idx], 'ro', markersize=5)
        ax_hht.annotate(f"Pico HHT: {r['f_peak_hht']:.2f} Hz\n({r['rpm_hht']:.0f} RPM)",
                        (r['f_peak_hht'], 1.0),
                        textcoords="offset points", xytext=(0, -20),
                        ha='center', fontsize=8.5, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6))
        ax_hht.set_ylabel("Energia Marginal Normalizada", fontsize=9)
        ax_hht.set_title(f"HHT-DEMON (Espectro Marginal EMD): {r['label']}", fontsize=10, fontweight='bold')
        ax_hht.grid(True, alpha=0.3, ls='--')
        ax_hht.set_xlim(0, 25)
        ax_hht.legend(loc='upper right', fontsize=8)
        
    axes[-1, 0].set_xlabel("Frequência de Modulação (Hz)", fontsize=11, fontweight='bold')
    axes[-1, 1].set_xlabel("Frequência de Modulação (Hz)", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    out_fig = "reports/comparacao_demon_vs_hht.png"
    plt.savefig(out_fig, dpi=160)
    plt.savefig("scratch/comparacao_demon_vs_hht.png", dpi=160)
    plt.close()
    print(f"\n[OK] Gráfico comparativo salvo em: {out_fig}")

if __name__ == '__main__':
    main()
