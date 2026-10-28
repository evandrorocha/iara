"""
High-throughput parallel extraction of 2D Hilbert-Huang spectrograms (32 time bins x 128 freq bins)
for all 1,706 UO audio files in IARA dataset.

Same signal chain as the 1D HHT (scratch/extract_all_hht.py): decimation to 2 kHz, z-scored signal,
2.56 s windows every 5.12 s, EMD with 8 IMFs (MAX_ITERATION=30), energy amp^2, 1-1000 Hz. The difference
is that the energy is binned in time inside the window (32 bins of 80 ms) instead of summed, using
iara.processing.hht.compute_hilbert_2d_spectrogram. Limits: 80 ms time bins only resolve modulations
below ~6 Hz; the first/last time bins carry EMD/Hilbert end effects; frequency bins are 7.9 Hz wide
(the 1D HHT uses 3.9 Hz).

Output directory: data/iara_processed/hht_2d_features/
Format per file: {file_id}.pkl containing:
  - spectrograms: 3D numpy array of shape (n_windows, 32, 128), float32
  - times: window timestamps
  - fs_base: 2000.0 Hz
  - n_time_bins: 32
  - n_freq_bins: 128
"""

import os
import glob
import time
import typing
import pickle
import concurrent.futures
import numpy as np
import scipy.signal as scipy_sig
import scipy.io.wavfile as wavfile
from PyEMD import EMD

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import iara.processing.hht as iara_hht  # noqa: E402

N_IMFS = 8


def process_single_hht_2d_audio(args: typing.Tuple[str, str, int, int, int, float, int]):
    """
    Worker task: process a single WAV file and save its 2D Hilbert Spectrogram .pkl.
    """
    wav_path, output_dir, file_id, n_time_bins, n_freq_bins, max_freq_hz, fs_target = args
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

        # Z-Score Normalization
        sig_dec = sig_dec - np.mean(sig_dec)
        std_val = np.std(sig_dec)
        if std_val > 0:
            sig_dec = sig_dec / std_val

        # Window length: 2.56s (5120 points at 2000 Hz)
        # Hop: 5.12s (matching DEMON/MEL cadence)
        win_len = int(2.56 * actual_fs)
        hop = int(5.12 * actual_fs)

        n_wins = max(1, (len(sig_dec) - win_len) // hop + 1)

        window_spectrograms = []
        window_times = []

        emd = EMD()
        emd.MAX_ITERATION = 30  # Fast and robust convergence

        for w in range(n_wins):
            start = w * hop
            end = start + win_len
            if end > len(sig_dec):
                break

            w_sig = sig_dec[start:end]
            t_center = (start + win_len / 2.0) / actual_fs
            window_times.append(t_center)

            # Mesmas 8 IMFs do HHT 1D (extract_all_hht.py), para que as duas features difiram só na
            # dimensão temporal dentro da janela e na resolução em frequência.
            imfs = emd(w_sig, max_imf=N_IMFS)

            # Espectrograma de Hilbert 2D (tempo x frequência): log1p + normalização pelo pico da janela
            _, _, spec_2d = iara_hht.compute_hilbert_2d_spectrogram(
                imfs[:N_IMFS], actual_fs, n_time_bins=n_time_bins, n_freq_bins=n_freq_bins,
                max_freq_hz=max_freq_hz, log_scale=True)

            window_spectrograms.append(spec_2d)

        window_spectrograms = np.array(window_spectrograms, dtype=np.float32)  # (N, n_time_bins, n_freq_bins)

        data_to_save = {
            "spectrograms": window_spectrograms,
            "times": np.array(window_times, dtype=np.float32),
            "fs_base": actual_fs,
            "n_time_bins": n_time_bins,
            "n_freq_bins": n_freq_bins,
            "max_freq_hz": max_freq_hz,
            "file_id": file_id
        }

        tmp_path = pkl_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, pkl_path)

        return file_id, True, f"{len(window_spectrograms)} windows"

    except Exception as e:
        return file_id, False, str(e)


def build_unified_cache(output_dir="data/iara_processed/hht_2d_features",
                        cache_path="data/iara_processed/cache_hht_2d.pkl"):
    """
    Consolidates individual .pkl files into a single unified cache file for fast loading.
    """
    print(f"\nConstruindo cache unificado em '{cache_path}'...")
    file_list = sorted([f for f in os.listdir(output_dir) if f.endswith(".pkl") and not f.endswith(".tmp")])
    cache = {}
    for fname in file_list:
        file_id = int(os.path.splitext(fname)[0])
        fpath = os.path.join(output_dir, fname)
        try:
            with open(fpath, "rb") as f:
                d = pickle.load(f)
                cache[file_id] = d["spectrograms"]  # (n_windows, 32, 128)
        except Exception as e:
            print(f"Erro ao ler {fpath}: {e}")

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache unificado salvo com sucesso! {len(cache)} gravações armazenadas.")
    return cache


def main():
    print("=" * 80)
    print("EXTRAÇÃO MASSIVA DE ESPECTROGRAMAS HILBERT 2D (32x128) — DATASET IARA")
    print("=" * 80)

    output_dir = "data/iara_processed/hht_2d_features"
    os.makedirs(output_dir, exist_ok=True)

    n_time_bins = 32
    n_freq_bins = 128
    max_freq_hz = 1000.0
    fs_target = 2000

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
        tasks.append((wav_path, output_dir, file_id, n_time_bins, n_freq_bins, max_freq_hz, fs_target))

    print(f"Diretório de saída: {output_dir}")
    print(f"Resolução por janela (2,56s): {n_time_bins} passos temporais x {n_freq_bins} bins de frequência")

    cpu_count = os.cpu_count() or 4
    max_workers = min(24, cpu_count)
    print(f"Iniciando extração paralela com {max_workers} processos...")

    start_time = time.time()
    success_count = 0
    cached_count = 0
    error_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_hht_2d_audio, task): task[2] for task in tasks}

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

            if i % 100 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                speed = i / max(0.01, elapsed)
                eta = (len(tasks) - i) / max(0.01, speed)
                print(f"  Progresso: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%) | "
                      f"Processados: {success_count} (Cache: {cached_count}) | Erros: {error_count} | "
                      f"Velocidade: {speed:.1f} arq/s | ETA: {eta:.0f}s ({eta/60:.1f} min)")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"EXTRAÇÃO CONCLUÍDA EM {total_time:.1f}s ({total_time/60:.2f} min)!")
    print(f"Sucesso: {success_count}/{len(tasks)} | Falhas: {error_count}")
    print("=" * 80)

    # Build cache
    build_unified_cache(output_dir)


if __name__ == "__main__":
    main()
