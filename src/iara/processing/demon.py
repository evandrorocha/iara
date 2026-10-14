"""
DEMON (Detection of Envelope Modulation on Noise) Processing Module

Based on the original implementation by Fábio Silva (labsonar/signal_processing apps/demon.py),
with corrections for digital filter stability (SOS representation), pure Python integer factorization,
and direct SciPy-based STFT.
"""

import math
import typing
import numpy as np
import scipy.signal as scipy_sig


def get_prime_factors(n: int) -> typing.List[int]:
    """Prime factorization in pure Python (no external dependencies)."""
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


def get_demon_decimation_steps(fs_in: int, fs_out: int = 50) -> typing.Tuple[int, int]:
    """
    Computes two-stage decimation ratios to reduce sample rate from fs_in to fs_out.
    Preserves Fábio Silva's multi-stage factor balancing logic.
    """
    if fs_in % fs_out != 0:
        ratio = fs_in / fs_out
        r1 = int(math.floor(math.sqrt(ratio)))
        while ratio % r1 != 0 and r1 > 1:
            r1 -= 1
        r2 = int(ratio // r1)
        return r1, r2

    factor_list = get_prime_factors(int(fs_in // fs_out))
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


def compute_demon(
    signal: np.ndarray,
    fs: int,
    n_fft: int = 512,
    max_freq: int = 50,
    overlap_ratio: float = 0.5,
    passband: typing.Tuple[float, float] = (1000.0, 2000.0),
    method: str = "abs",
    discard_dc_bins: int = 4,
) -> typing.Dict[str, typing.Any]:
    """
    Computes DEMON modulation spectrogram and integrated modulation spectrum.

    Parameters
    ----------
    signal : np.ndarray
        Raw 1D acoustic audio time series.
    fs : int
        Sampling frequency in Hz (e.g. 48000 or 16000).
    n_fft : int, default=512
        FFT window length for STFT.
    max_freq : int, default=50
        Target post-decimation sample rate in Hz. The maximum observable frequency
        is Nyquist = max_freq / 2 (e.g., 25 Hz for max_freq=50).
    overlap_ratio : float, default=0.5
        Overlap ratio between consecutive STFT windows.
    passband : tuple (float, float), default=(1000.0, 2000.0)
        Bandpass filter cutoff frequencies (Hz) isolating cavitation noise.
    method : str, default='abs'
        Envelope demodulation method: 'abs' (full-wave) or 'hilbert'.
    discard_dc_bins : int, default=4
        Number of low-frequency bins to discard to eliminate sub-hertz DC drift.

    Returns
    -------
    dict:
        - 'avg_spectrum': 1D np.ndarray with time-averaged DEMON spectrum.
        - 'waterfall': 2D np.ndarray (frequencies x time frames).
        - 'frequencies': 1D np.ndarray with modulation frequencies (Hz).
        - 'times': 1D np.ndarray with time stamps (seconds).
        - 'final_fs': int with baseband sample rate (Hz).
        - 'ratios': tuple with (decimate_ratio1, decimate_ratio2).
    """
    if signal.ndim > 1:
        signal = signal[:, 0]

    nyq = fs / 2.0
    f_low, f_high = passband
    wp = [f_low / nyq, f_high / nyq]
    ws = [(f_low * 0.7) / nyq, min(0.99, (f_high * 1.3) / nyq)]

    # Digital Chebyshev Type 2 filter in SOS format for numerical stability
    N, wc = scipy_sig.cheb2ord(wp, ws, gpass=0.5, gstop=40)
    sos = scipy_sig.cheby2(N, rs=40, Wn=wc, btype="bandpass", output="sos", analog=False)
    x_filtered = scipy_sig.sosfilt(sos, signal.astype(np.float64))

    # Envelope extraction (demodulation)
    if method == "abs":
        envelope = np.abs(x_filtered)
    elif method == "hilbert":
        envelope = np.abs(scipy_sig.hilbert(x_filtered))
    else:
        raise ValueError(f"Invalid demodulation method '{method}'. Choose 'abs' or 'hilbert'.")

    # Multi-stage decimation down to baseband
    r1, r2 = get_demon_decimation_steps(fs, max_freq)
    x_dec1 = scipy_sig.decimate(envelope, r1, ftype="fir", zero_phase=False)
    x_base = scipy_sig.decimate(x_dec1, r2, ftype="fir", zero_phase=False)
    final_fs = (fs // r1) // r2

    # Normalization and mean centering
    x_base = x_base - np.mean(x_base)
    peak = np.max(np.abs(x_base))
    if peak > 0:
        x_base = x_base / peak

    # Baseband spectral analysis (STFT)
    hop_length = int(n_fft * (1.0 - overlap_ratio))
    f, t, Zxx = scipy_sig.stft(
        x_base,
        fs=final_fs,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
    )

    magnitude = np.abs(Zxx)

    # Discard DC / sub-hertz bins
    if discard_dc_bins > 0 and len(f) > discard_dc_bins:
        magnitude = magnitude[discard_dc_bins:, :]
        f = f[discard_dc_bins:]

    avg_spectrum = np.mean(magnitude, axis=1)

    return {
        "avg_spectrum": avg_spectrum,
        "waterfall": magnitude,
        "frequencies": f,
        "times": t,
        "final_fs": final_fs,
        "ratios": (r1, r2),
    }
