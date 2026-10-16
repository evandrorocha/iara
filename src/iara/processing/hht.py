"""
HHT (Hilbert-Huang Transform) and VMD (Variational Mode Decomposition) Processing Module

Provides adaptive, data-driven non-stationary decomposition methods for passive sonar signals:
1. EMD (Empirical Mode Decomposition)
2. Instantaneous Frequency and Amplitude via Hilbert Transform
3. Hilbert Marginal Spectrum (HMS)
4. VMD (Variational Mode Decomposition)
"""

import math
import typing
import numpy as np
import scipy.signal as scipy_sig

try:
    from PyEMD import EMD
    HAS_PYEMD = True
except ImportError:
    HAS_PYEMD = False

try:
    from vmdpy import VMD
    HAS_VMD = True
except ImportError:
    HAS_VMD = False


def compute_emd(signal: np.ndarray, max_imfs: int = 8, max_iter: int = 100) -> np.ndarray:
    """
    Decomposes 1D signal into Intrinsic Mode Functions (IMFs) via Empirical Mode Decomposition.

    Args:
        signal (np.ndarray): 1D time-domain signal.
        max_imfs (int): Maximum number of IMFs to retain.
        max_iter (int): Maximum iterations per sifting step.

    Returns:
        np.ndarray: Array of shape (n_imfs, len(signal)).
    """
    if not HAS_PYEMD:
        raise ImportError("PyEMD (EMD-signal) is required for EMD decomposition. Install with 'pip install EMD-signal'.")

    emd = EMD()
    emd.MAX_ITERATION = max_iter
    imfs = emd(signal)
    if imfs.shape[0] > max_imfs:
        imfs = imfs[:max_imfs]
    return imfs


def compute_instantaneous_properties(imfs: np.ndarray, fs: float) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Computes instantaneous amplitude and instantaneous frequency for each IMF using the Hilbert transform.

    Args:
        imfs (np.ndarray): Array of IMFs (shape: n_imfs, n_samples).
        fs (float): Sampling rate in Hz.

    Returns:
        Tuple of (amplitudes, frequencies) each of shape (n_imfs, n_samples).
    """
    dt = 1.0 / fs
    amps = []
    freqs = []

    for imf in imfs:
        analytic = scipy_sig.hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        # Instantaneous frequency: d(phase)/dt / (2*pi)
        freq = np.gradient(phase, dt) / (2.0 * np.pi)
        amps.append(amp)
        freqs.append(freq)

    return np.array(amps), np.array(freqs)


def compute_hilbert_marginal_spectrum(
    imfs: np.ndarray,
    fs: float,
    max_freq_hz: float = 50.0,
    n_bins: int = 256
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Hilbert Marginal Spectrum (HMS) by integrating instantaneous energy over time.

    Args:
        imfs (np.ndarray): Array of IMFs (shape: n_imfs, n_samples).
        fs (float): Sampling rate in Hz.
        max_freq_hz (float): Maximum frequency bin in Hz.
        n_bins (int): Number of frequency bins.

    Returns:
        Tuple of (freq_grid, marginal_spectrum).
    """
    amps, freqs = compute_instantaneous_properties(imfs, fs)
    freq_grid = np.linspace(0.1, max_freq_hz, n_bins)
    df = freq_grid[1] - freq_grid[0]
    marginal_spec = np.zeros(n_bins)

    for amp, freq in zip(amps, freqs):
        mask = (freq >= 0.1) & (freq <= max_freq_hz)
        bin_idx = np.clip(np.floor((freq[mask] - 0.1) / df).astype(int), 0, n_bins - 1)
        energy = amp[mask] ** 2
        for b, e in zip(bin_idx, energy):
            marginal_spec[b] += e

    if np.max(marginal_spec) > 0:
        marginal_spec = marginal_spec / np.max(marginal_spec)

    return freq_grid, marginal_spec


def compute_vmd_modes(
    signal: np.ndarray,
    fs: float,
    k_modes: int = 4,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7
) -> typing.Dict[str, typing.Any]:
    """
    Decomposes signal via Variational Mode Decomposition (VMD).

    Args:
        signal (np.ndarray): 1D input signal.
        fs (float): Sampling frequency in Hz.
        k_modes (int): Number of discrete modes to extract.
        alpha (float): Bandwidth constraint parameter (moderate: 1000-3000).
        tau (float): Time-step of dual ascent (0 for strict fidelity).
        tol (float): Convergence tolerance.

    Returns:
        dict: Contains 'modes' (K, N), 'central_frequencies_hz', and 'central_frequencies_rpm'.
    """
    if not HAS_VMD:
        raise ImportError("vmdpy is required for VMD decomposition. Install with 'pip install vmdpy'.")

    u, u_hat, omega = VMD(signal, alpha=alpha, tau=tau, K=k_modes, DC=0, init=1, tol=tol)
    central_freqs_hz = omega[-1] * fs
    central_freqs_rpm = central_freqs_hz * 60.0

    return {
        'modes': u,
        'central_frequencies_hz': central_freqs_hz,
        'central_frequencies_rpm': central_freqs_rpm
    }
