"""
High-throughput parallel extraction of Pure HHT (Hilbert-Huang Transform, 256 bins)
for all 1,706 UO audio files in IARA dataset.

Output directory: data/iara_processed/hht_features/
Format per file: {file_id}.pkl containing:
  - df: DataFrame of shape (n_windows, 256)
  - times: window timestamps
  - avg_spectrum: 1D array of shape (256,)
  - frequencies: grid of 256 frequencies (1.0 to 1000.0 Hz)
  - fs_base: 2000.0 Hz
  - n_bins: 256
"""

import os
import glob
import time
import typing
import concurrent.futures
import numpy as np
import pandas as pd
import scipy.signal as scipy_sig
import scipy.io.wavfile as wavfile
from PyEMD import EMD

def process_single_hht_audio(args: typing.Tuple[str, str, int, int, float, int]):
    """
    Worker task: process a single WAV file and save its .pkl feature dictionary.
    """
    wav_path, output_dir, file_id, n_bins, max_freq_hz, fs_target = args
    pkl_path = os.path.join(output_dir, f"{file_id}.pkl")
    
    # Skip if already processed and valid
    if os.path.exists(pkl_path):
        try:
            if os.path.getsize(pkl_path) > 1000:
                return file_id, True, "cached"
        except Exception:
            pass

    try:
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
        # Hop: 5.12s (matching DEMON/MEL cadence)
        win_len = int(2.56 * actual_fs)
        hop = int(5.12 * actual_fs)
        
        n_wins = max(1, (len(sig_dec) - win_len) // hop + 1)
        
        freq_grid = np.linspace(1.0, max_freq_hz, n_bins)
        df = freq_grid[1] - freq_grid[0]
        
        window_hms = []
        window_times = []
        
        emd = EMD()
        emd.MAX_ITERATION = 30  # Fast and robust convergence
        
        dt = 1.0 / actual_fs
        
        for w in range(n_wins):
            start = w * hop
            end = start + win_len
            if end > len(sig_dec):
                break
                
            w_sig = sig_dec[start:end]
            t_center = (start + win_len / 2.0) / actual_fs
            window_times.append(t_center)
            
            imfs = emd(w_sig, max_imf=8)
            
            # Accumulate energy into frequency bins
            w_spec = np.zeros(n_bins, dtype=np.float32)
            for imf in imfs[:8]:
                analytic = scipy_sig.hilbert(imf)
                amp = np.abs(analytic)
                phase = np.unwrap(np.angle(analytic))
                inst_freq = np.gradient(phase, dt) / (2.0 * np.pi)
                
                mask = (inst_freq >= 1.0) & (inst_freq <= max_freq_hz)
                bin_idx = np.clip(np.floor((inst_freq[mask] - 1.0) / df).astype(int), 0, n_bins - 1)
                energy = (amp[mask] ** 2).astype(np.float32)
                
                np.add.at(w_spec, bin_idx, energy)
                    
            # Normalize window spectrum
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
        
        tmp_path = pkl_path + ".tmp"
        pd.to_pickle(data_to_save, tmp_path)
        os.replace(tmp_path, pkl_path)
        
        return file_id, True, f"{len(df_windows)} windows"
        
    except Exception as e:
        return file_id, False, str(e)


def main():
    print("=" * 80)
    print("EXTRAÇÃO MASSIVA DE HHT PURA (256 BINS) - 1.706 ÁUDIOS UO DO IARA")
    print("=" * 80)
    
    output_dir = "data/iara_processed/hht_features"
    os.makedirs(output_dir, exist_ok=True)
    
    n_bins = 256
    max_freq_hz = 1000.0
    fs_target = 2000
    
    # Locate all WAV files across collections A, B, C, D, E
    raw_files = []
    for col in ["A", "B", "C", "D", "E"]:
        pattern = os.path.join("data", "iara", col, "*.wav")
        files = glob.glob(pattern)
        raw_files.extend(files)
        
    raw_files = sorted(raw_files)
    print(f"Total de arquivos WAV encontrados: {len(raw_files)}")
    
    tasks = []
    for wav_path in raw_files:
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        file_id = int(basename.split("-")[-1])
        tasks.append((wav_path, output_dir, file_id, n_bins, max_freq_hz, fs_target))
        
    print(f"Diretório de saída: {output_dir}")
    print(f"Configuração: {n_bins} bins de 1.0 a {max_freq_hz:.0f} Hz (res: {max_freq_hz/n_bins:.2f} Hz/bin)")
    
    # Number of CPU workers
    cpu_count = os.cpu_count() or 4
    max_workers = min(24, cpu_count)
    print(f"Iniciando extração paralela com {max_workers} processos...")
    
    start_time = time.time()
    success_count = 0
    cached_count = 0
    error_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_hht_audio, task): task[2] for task in tasks}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            file_id = futures[future]
            try:
                fid, success, msg = future.result()
                if success:
                    success_count += 1
                    if msg == "cached":
                        cached_count += 1
                else:
                    error_count += 1
                    print(f"  [ERRO] ID {fid}: {msg}")
            except Exception as e:
                error_count += 1
                print(f"  [EXCEÇÃO] ID {file_id}: {e}")
                
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                speed = i / max(0.01, elapsed)
                eta = (len(tasks) - i) / max(0.01, speed)
                print(f"  Progresso: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%) | "
                      f"Processados: {success_count} (Cache: {cached_count}) | Erros: {error_count} | "
                      f"Velocidade: {speed:.1f} arq/s | ETA: {eta:.0f}s ({eta/60:.1f} min)")
                      
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"EXTRAÇÃO MASSIVA DE HHT CONCLUÍDA EM {total_time:.1f} segundos ({total_time/60:.2f} minutos)!")
    print(f"Sucesso: {success_count}/{len(tasks)} | Falhas: {error_count}")
    print("=" * 80)

if __name__ == "__main__":
    main()
