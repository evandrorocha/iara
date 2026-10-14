"""
Test and validation of DEMON analysis pipeline based on Fábio Silva's implementation
(labsonar/signal_processing apps/demon.py).

Authors: Adapted for IARA research pipeline
"""

import os
import math
import numpy as np
import scipy.signal as scipy_sig
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

def get_factors(n: int):
    """Prime factorization in pure python (no sympy needed)."""
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

def get_demon_decimation_steps(fs_in: int, fs_out: int = 50):
    """
    Computes two-stage decimation ratios as in Fábio Silva's apps/demon.py.
    """
    if fs_in % fs_out != 0:
        # If not exact multiple, find closest factorable fs_out or handle gracefully
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

def compute_demon(signal: np.ndarray, fs: int,
                  n_fft: int = 512,
                  max_freq: int = 50,
                  overlap_ratio: float = 0.5,
                  passband: tuple = (1000, 2000),
                  method: str = 'abs',
                  discard_dc_bins: int = 4):
    """
    Core DEMON extraction algorithm (Fábio Silva's method).
    
    Parameters:
    -----------
    signal : np.ndarray
        Raw 1D acoustic audio signal.
    fs : int
        Original sample rate (e.g. 48000 or 16000 Hz).
    n_fft : int
        FFT window length for STFT.
    max_freq : int
        Target post-decimation sample rate (determines Nyquist max freq = max_freq/2).
    overlap_ratio : float
        Overlap fraction between consecutive windows.
    passband : tuple (f_low, f_high)
        Bandpass filter limits to isolate cavitation noise.
    method : str
        Demodulation method: 'abs' (full-wave) or 'hilbert'.
    discard_dc_bins : int
        Number of low-frequency bins to discard (removes DC drift / sub-hertz noise).
    """
    # 1. Bandpass filter to isolate high-frequency cavitation modulation
    nyq = fs / 2.0
    f_low, f_high = passband
    wp = [f_low / nyq, f_high / nyq]
    ws = [(f_low * 0.7) / nyq, min(0.99, (f_high * 1.3) / nyq)]
    
    # Chebyshev type 2 filter with SOS for numerical stability
    N, wc = scipy_sig.cheb2ord(wp, ws, gpass=0.5, gstop=40)
    sos = scipy_sig.cheby2(N, rs=40, Wn=wc, btype='bandpass', output='sos', analog=False)
    x_filtered = scipy_sig.sosfilt(sos, signal.astype(np.float64))

    # 2. Envelope extraction (demodulation)
    if method == 'abs':
        envelope = np.abs(x_filtered)
    elif method == 'hilbert':
        envelope = np.abs(scipy_sig.hilbert(x_filtered))
    else:
        raise ValueError(f"Invalid method '{method}'. Choose 'abs' or 'hilbert'.")

    # 3. Two-stage decimation down to baseband
    r1, r2 = get_demon_decimation_steps(fs, max_freq)
    x_dec1 = scipy_sig.decimate(envelope, r1, ftype='fir', zero_phase=False)
    x_base = scipy_sig.decimate(x_dec1, r2, ftype='fir', zero_phase=False)
    final_fs = (fs // r1) // r2

    # 4. Centralization and normalization
    x_base = x_base - np.mean(x_base)
    peak = np.max(np.abs(x_base))
    if peak > 0:
        x_base = x_base / peak

    # 5. Spectral analysis (STFT)
    hop_length = int(n_fft * (1.0 - overlap_ratio))
    f, t, Zxx = scipy_sig.stft(x_base, fs=final_fs, window='hann',
                               nperseg=n_fft, noverlap=n_fft - hop_length,
                               boundary=None)
    
    magnitude = np.abs(Zxx)
    
    # Discard DC and near-zero bins
    if discard_dc_bins > 0 and len(f) > discard_dc_bins:
        magnitude = magnitude[discard_dc_bins:, :]
        f = f[discard_dc_bins:]

    # Average modulation spectrum across time
    avg_spectrum = np.mean(magnitude, axis=1)

    return {
        'avg_spectrum': avg_spectrum,
        'waterfall': magnitude,
        'frequencies': f,
        'times': t,
        'final_fs': final_fs,
        'ratios': (r1, r2)
    }

def main():
    print("=" * 70)
    print("TESTE DE DEMON (Fábio Silva / LabSonar)")
    print("=" * 70)

    samples = [
        ('SMALL (Pilot 10, 11m)', 'data/iara/A/A-0002.wav', '#1f77b4'),
        ('MEDIUM (Maersk Pacer, 74m)', 'data/iara/A/A-0001.wav', '#ff7f0e'),
        ('LARGE (Sinoglory, 322m)', 'data/iara/A/A-0004.wav', '#2ca02c'),
        ('AMBIENTAL (Ruído Puro)', 'data/iara/E/E-1446.wav', '#7f7f7f'),
    ]

    results = []

    # Process 60 seconds around CPA (or beginning if CPA not defined)
    duration_sec = 60

    for label, filepath, color in samples:
        print(f"\nCarregando: {label} -> {filepath}")
        if not os.path.exists(filepath):
            print(f"  [AVISO] Arquivo não encontrado: {filepath}")
            continue

        fs, data = wavfile.read(filepath)
        if data.ndim > 1:
            data = data[:, 0]

        # Use middle segment (around CPA)
        total_sec = len(data) / fs
        start_sec = max(0, (total_sec - duration_sec) / 2)
        start_idx = int(start_sec * fs)
        end_idx = start_idx + int(duration_sec * fs)
        segment = data[start_idx:end_idx]

        print(f"  fs = {fs} Hz | Duração total = {total_sec:.1f}s | Segmento analisado = {len(segment)/fs:.1f}s")

        res = compute_demon(
            segment, fs=fs,
            n_fft=512,
            max_freq=50,       # final_fs = 50 Hz -> frequências até 25 Hz
            overlap_ratio=0.5,
            passband=(1000, 2000),
            method='abs',
            discard_dc_bins=4
        )

        f_rpm = res['frequencies'] * 60.0  # Hz -> RPM
        peak_idx = np.argmax(res['avg_spectrum'])
        peak_freq = res['frequencies'][peak_idx]
        peak_rpm = f_rpm[peak_idx]

        print(f"  Decimação: {res['ratios'][0]} x {res['ratios'][1]} | Fs base = {res['final_fs']} Hz")
        print(f"  Espectro DEMON: {len(res['frequencies'])} bins de {res['frequencies'][0]:.2f} Hz a {res['frequencies'][-1]:.2f} Hz")
        print(f"  Pico Dominante: {peak_freq:.2f} Hz ({peak_rpm:.0f} RPM)")

        results.append((label, color, res))

    # Generate Visualization Figure
    fig, axes = plt.subplots(len(results), 2, figsize=(15, 3.2 * len(results)), gridspec_kw={'width_ratios': [1.3, 1]})
    fig.suptitle("Análise Espectral DEMON (Detecção de Modulação do Envelope de Ruído)", fontsize=14, fontweight='bold', y=0.99)

    for i, (label, color, res) in enumerate(results):
        ax_spec = axes[i, 0]
        ax_water = axes[i, 1]

        f_hz = res['frequencies']
        f_rpm = f_hz * 60.0
        spec = res['avg_spectrum']

        # 1. Espectro Médio (Curva DEMON)
        ax_spec.plot(f_hz, spec, color=color, lw=1.6, label=f"{label}")
        # Top 3 peaks
        peaks, _ = scipy_sig.find_peaks(spec, height=np.mean(spec), distance=5)
        if len(peaks) > 0:
            top_peaks = peaks[np.argsort(spec[peaks])[-3:]]
            for p in top_peaks:
                ax_spec.plot(f_hz[p], spec[p], 'ro', markersize=4)
                ax_spec.annotate(f"{f_hz[p]:.1f} Hz\n({f_rpm[p]:.0f} RPM)",
                                (f_hz[p], spec[p]),
                                textcoords="offset points", xytext=(0, 7),
                                ha='center', fontsize=7.5, color='darkred', fontweight='bold')

        ax_spec.set_title(f"Espectro DEMON Integrado: {label}", fontsize=10, fontweight='bold')
        ax_spec.set_xlabel("Frequência de Modulação (Hz)", fontsize=8.5)
        ax_spec.set_ylabel("Intensidade Relativa", fontsize=8.5)
        ax_spec.set_xlim(0, 25)
        ax_spec.grid(True, alpha=0.3, ls='--')
        ax_spec.legend(loc='upper right', fontsize=8)

        # 2. Espectrograma DEMON no Tempo (Waterfall)
        water = res['waterfall']
        t = res['times']
        im = ax_water.imshow(
            20 * np.log10(water + 1e-6),
            origin='lower', aspect='auto',
            extent=[t[0], t[-1], f_hz[0], f_hz[-1]],
            cmap='viridis'
        )
        ax_water.set_title(f"Evolução Temporal (Waterfall): {label}", fontsize=10)
        ax_water.set_xlabel("Tempo (s)", fontsize=8.5)
        ax_water.set_ylabel("Frequência (Hz)", fontsize=8.5)
        fig.colorbar(im, ax=ax_water, pad=0.02, label="Magnitude (dB)")

    plt.tight_layout()
    output_path = "scratch/demon_comparison.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"\n[OK] Gráfico comparativo salvo em: {output_path}")

if __name__ == "__main__":
    main()
