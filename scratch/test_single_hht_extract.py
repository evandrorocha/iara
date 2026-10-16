import os
import time
import numpy as np
import pandas as pd
import scipy.signal as scipy_sig
import scipy.io.wavfile as wavfile
from PyEMD import EMD

def process_single_hht(wav_path, output_dir, file_id, n_bins=256, max_freq_hz=1000.0, fs_target=2000):
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, f"{file_id}.pkl")
    
    fs, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    
    # Decimate to fs_target = 2000 Hz (Nyquist = 1000 Hz)
    decim_factor = int(round(fs / fs_target))
    sig_dec = scipy_sig.decimate(data, decim_factor, ftype='fir', zero_phase=True)
    actual_fs = fs / decim_factor
    
    # Normalize
    sig_dec = sig_dec - np.mean(sig_dec)
    std_val = np.std(sig_dec)
    if std_val > 0:
        sig_dec = sig_dec / std_val
        
    # Window length: 2.56s (5120 points at 2000 Hz)
    # Hop: 5.12s (like DEMON/MEL alignment, so we get ~55 windows per audio!)
    # Let's check: 282s / 5.12s = ~55 windows!
    win_len = int(2.56 * actual_fs) # 5120 samples
    hop = int(5.12 * actual_fs)    # 10240 samples
    
    n_wins = max(1, (len(sig_dec) - win_len) // hop + 1)
    
    freq_grid = np.linspace(1.0, max_freq_hz, n_bins)
    df = freq_grid[1] - freq_grid[0]
    
    window_hms = []
    window_times = []
    
    emd = EMD()
    emd.MAX_ITERATION = 50  # 50 iterations is fast and stable
    
    dt = 1.0 / actual_fs
    
    t0 = time.time()
    for w in range(n_wins):
        start = w * hop
        end = start + win_len
        if end > len(sig_dec):
            break
            
        w_sig = sig_dec[start:end]
        t_center = (start + win_len / 2.0) / actual_fs
        window_times.append(t_center)
        
        imfs = emd(w_sig)
        
        # Accumulate energy
        w_spec = np.zeros(n_bins, dtype=np.float32)
        for imf in imfs[:8]:
            analytic = scipy_sig.hilbert(imf)
            amp = np.abs(analytic)
            phase = np.unwrap(np.angle(analytic))
            inst_freq = np.gradient(phase, dt) / (2.0 * np.pi)
            
            mask = (inst_freq >= 1.0) & (inst_freq <= max_freq_hz)
            bin_idx = np.clip(np.floor((inst_freq[mask] - 1.0) / df).astype(int), 0, n_bins - 1)
            energy = (amp[mask] ** 2).astype(np.float32)
            
            for b, en in zip(bin_idx, energy):
                w_spec[b] += en
                
        # Normalize window spectrum (unit energy)
        s_max = np.max(w_spec)
        if s_max > 0:
            w_spec = w_spec / s_max
            
        window_hms.append(w_spec)
        
    window_hms = np.array(window_hms, dtype=np.float32)
    avg_spectrum = np.mean(window_hms, axis=0) if len(window_hms) > 0 else np.zeros(n_bins, dtype=np.float32)
    
    cols = [f"hht_{i}" for i in range(n_bins)]
    df_windows = pd.DataFrame(window_hms, columns=cols)
    
    data_to_save = {
        "df": df_windows,
        "times": np.array(window_times, dtype=np.float32),
        "avg_spectrum": avg_spectrum,
        "frequencies": freq_grid,
        "fs_base": actual_fs,
        "n_bins": n_bins
    }
    
    pd.to_pickle(data_to_save, pkl_path)
    elapsed = time.time() - t0
    print(f"Processed file_id {file_id}: {len(df_windows)} windows in {elapsed:.2f}s ({elapsed/len(df_windows)*1000:.1f}ms/win)")
    print(f"Saved to {pkl_path} (size: {os.path.getsize(pkl_path)} bytes)")

process_single_hht("data/iara/A/A-0001.wav", "data/iara_processed/test_hht", 1)
