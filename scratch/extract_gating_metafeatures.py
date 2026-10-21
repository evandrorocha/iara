"""
Extrai 7 meta-características acústicas físicas leves e interpretáveis diretamente dos arquivos .wav
para alimentar o GATING FRONT-END do sistema Gated-SVM.
Executa em paralelo via ProcessPoolExecutor em ~15-20s.
"""
import os
import glob
import time
import pickle
import concurrent.futures
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import soundfile as sf

def extract_file_metafeatures(args):
    file_id, wav_path, duration = args
    try:
        info = sf.info(wav_path)
        fs = info.samplerate
        n_frames = min(int(duration * fs), info.frames)
        # Ler do meio do áudio (região de maior aproximação / CPA)
        start_frame = max(0, int((info.frames - n_frames) / 2))
        sig, _ = sf.read(wav_path, start=start_frame, frames=n_frames, dtype='float32')
        if sig.ndim > 1:
            sig = sig[:, 0]

        # 1. Tonal Crest Factor em Baixa Frequência (< 250 Hz) -> Indicador LOFAR
        f, psd = signal.welch(sig, fs=fs, nperseg=4096)
        low_mask = (f >= 5.0) & (f <= 250.0)
        low_psd = psd[low_mask]
        crest_factor = float(np.max(low_psd) / (np.mean(low_psd) + 1e-12))
        spectral_flatness = float(stats.gmean(low_psd + 1e-12) / (np.mean(low_psd) + 1e-12))

        # 2. Modulação de Envoltória de Alta Frequência (2 kHz - 8 kHz) -> Indicador DEMON
        sos = signal.butter(4, [2000, 8000], btype='bandpass', fs=fs, output='sos')
        sig_bp = signal.sosfilt(sos, sig)
        env = np.abs(signal.hilbert(sig_bp))
        env_detrend = env - np.mean(env)
        f_env, psd_env = signal.welch(env_detrend, fs=fs, nperseg=4096)
        env_mask = (f_env >= 5.0) & (f_env <= 40.0)
        mod_index = float(np.max(psd_env[env_mask]) / (np.mean(psd_env[env_mask]) + 1e-12))

        # 3. Variação Temporal / Fluxo Espectral -> Indicador HHT
        seg_len = int(fs * 1.0)
        n_segs = max(1, len(sig) // seg_len)
        energies = [float(np.sum(sig[i*seg_len:(i+1)*seg_len]**2)) for i in range(n_segs)]
        flux_std = float(np.std(energies) / (np.mean(energies) + 1e-12))

        # 4. Razão de Energia Alta vs Baixa (High/Low Energy Ratio)
        high_mask = (f >= 2000.0) & (f <= 8000.0)
        e_high = float(np.sum(psd[high_mask]))
        e_low = float(np.sum(low_psd))
        high_low_ratio = float((e_high + 1e-12) / (e_low + 1e-12))

        # 5. RMS Energy e Zero-Crossing Rate
        rms_db = float(10 * np.log10(np.mean(sig**2) + 1e-12))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(sig)))))

        return file_id, {
            'crest_factor': crest_factor,
            'spectral_flatness': spectral_flatness,
            'mod_index': mod_index,
            'flux_std': flux_std,
            'high_low_ratio': high_low_ratio,
            'rms_db': rms_db,
            'zcr': zcr
        }
    except Exception as e:
        return file_id, None

def main():
    wav_files = glob.glob('data/iara/**/*.wav', recursive=True)
    print(f"Encontrados {len(wav_files)} arquivos .wav em data/iara.")

    tasks = []
    for p in wav_files:
        fid = int(os.path.splitext(os.path.basename(p))[0].split('-')[-1])
        tasks.append((fid, p, 10.0))

    t0 = time.time()
    results = {}
    max_workers = min(16, os.cpu_count() or 4)
    print(f"Extraindo meta-características com {max_workers} processos paralelos...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for fid, feat_dict in executor.map(extract_file_metafeatures, tasks):
            if feat_dict is not None:
                results[fid] = feat_dict

    elapsed = time.time() - t0
    print(f"Extração concluída com sucesso em {elapsed:.1f}s!")
    print(f"Total de áudios com meta-características extraídas: {len(results)}")

    out_path = 'data/iara_processed/cache_gating_metafeatures.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Salvo em '{out_path}'.")

if __name__ == '__main__':
    main()
