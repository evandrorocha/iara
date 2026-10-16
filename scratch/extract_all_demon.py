"""
High-throughput parallel extraction of DEMON features for all 1,706 UO audio files in IARA.
"""

import os
import glob
import time
import math
import typing
import concurrent.futures
import numpy as np
import pandas as pd
import scipy.signal as scipy_sig
import scipy.io.wavfile as wavfile

# Pure python prime factorization
def get_prime_factors(n: int) -> typing.List[int]:
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

def process_single_audio(args: typing.Tuple[str, str, int]):
    """
    Worker task: process a single WAV file and save its .pkl feature dictionary.
    """
    wav_path, output_dir, file_id = args
    pkl_path = os.path.join(output_dir, f"{file_id}.pkl")
    
    # Skip if already processed and valid
    if os.path.exists(pkl_path):
        try:
            # Check non-empty
            if os.path.getsize(pkl_path) > 1000:
                return file_id, True, "cached"
        except Exception:
            pass

    try:
        fs, data = wavfile.read(wav_path)
        if data.ndim > 1:
            data = data[:, 0]

        # DEMON parameters
        passband = (1000.0, 2000.0)
        max_freq = 50
        n_fft = 512
        overlap_ratio = 0.5
        discard_dc_bins = 4

        # 1. Bandpass filter in SOS format
        nyq = fs / 2.0
        f_low, f_high = passband
        wp = [f_low / nyq, f_high / nyq]
        ws = [(f_low * 0.7) / nyq, min(0.99, (f_high * 1.3) / nyq)]
        N, wc = scipy_sig.cheb2ord(wp, ws, gpass=0.5, gstop=40)
        sos = scipy_sig.cheby2(N, rs=40, Wn=wc, btype="bandpass", output="sos", analog=False)
        x_filtered = scipy_sig.sosfilt(sos, data.astype(np.float64))

        # 2. Envelope extraction (full-wave rectification)
        envelope = np.abs(x_filtered)

        # 3. Two-stage decimation to baseband (50 Hz)
        r1, r2 = get_demon_decimation_steps(fs, max_freq)
        x_dec1 = scipy_sig.decimate(envelope, r1, ftype="fir", zero_phase=False)
        x_base = scipy_sig.decimate(x_dec1, r2, ftype="fir", zero_phase=False)
        final_fs = (fs // r1) // r2

        # 4. Centralization and normalization
        x_base = x_base - np.mean(x_base)
        peak = np.max(np.abs(x_base))
        if peak > 0:
            x_base = x_base / peak

        # 5. Baseband STFT
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

        # Build DataFrame with shape (n_windows, n_frequencies)
        columns = [f"f {i}" for i in range(len(f))]
        df_windows = pd.DataFrame(magnitude.T, columns=columns)

        data_to_save = {
            "df": df_windows,
            "times": t,
            "avg_spectrum": avg_spectrum,
            "frequencies": f,
            "fs_base": final_fs,
        }

        # Save to temporary file first then atomic rename
        tmp_path = pkl_path + ".tmp"
        pd.to_pickle(data_to_save, tmp_path)
        os.replace(tmp_path, pkl_path)

        return file_id, True, f"{len(df_windows)} windows"

    except Exception as e:
        return file_id, False, str(e)


def main():
    print("=" * 70)
    print("EXTRAÇÃO MASSIVA DE FEATURES DEMON (1.706 Áudios UO)")
    print("=" * 70)

    output_dir = "data/iara_processed/demon_features"
    os.makedirs(output_dir, exist_ok=True)

    # Find all UO WAV files (Collections A, B, C, D, E)
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
        # Extract integer ID: e.g. 'A-0001' -> 1
        file_id = int(basename.split("-")[-1])
        tasks.append((wav_path, output_dir, file_id))

    print(f"Diretório de saída: {output_dir}")
    print(f"Iniciando extração paralela com ProcessPoolExecutor...")

    start_time = time.time()
    success_count = 0
    error_count = 0

    max_workers = min(16, os.cpu_count() or 4)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_audio, task): task[2] for task in tasks}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            file_id = futures[future]
            try:
                fid, success, msg = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"  [ERRO] ID {fid}: {msg}")
            except Exception as e:
                error_count += 1
                print(f"  [EXCEÇÃO] ID {file_id}: {e}")

            if i % 200 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                speed = i / max(0.01, elapsed)
                eta = (len(tasks) - i) / max(0.01, speed)
                print(f"  Progresso: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%) | "
                      f"Sucesso: {success_count} | Erros: {error_count} | "
                      f"Velocidade: {speed:.1f} arq/s | ETA: {eta:.0f}s")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"EXTRAÇÃO CONCLUÍDA EM {total_time:.1f} segundos!")
    print(f"Sucesso: {success_count}/{len(tasks)} | Falhas: {error_count}")
    print("=" * 70)

if __name__ == "__main__":
    main()
