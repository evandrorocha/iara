import time
import numpy as np
import scipy.signal as scipy_sig
import scipy.io.wavfile as wavfile
from PyEMD import EMD

def benchmark_one_audio(wav_path):
    fs, data = wavfile.read(wav_path)
    if data.ndim > 1: data = data[:, 0]
    data = data.astype(np.float32)
    
    t0 = time.time()
    # Decimate from 48 kHz to 2000 Hz (factor 24)
    sig_dec = scipy_sig.decimate(data, 24, ftype='fir', zero_phase=True)
    actual_fs = fs / 24.0
    t_decim = time.time() - t0
    
    # 2.56s windows (5120 samples), hop = 1.28s (2560 samples)
    win_len = 5120
    hop = 2560
    n_wins = max(1, (len(sig_dec) - win_len) // hop + 1)
    
    emd = EMD()
    emd.MAX_ITERATION = 100
    
    t1 = time.time()
    for w in range(min(n_wins, 20)): # test up to 20 windows
        w_sig = sig_dec[w*hop : w*hop + win_len]
        imfs = emd(w_sig)
    t_emd = time.time() - t1
    
    return len(data)/fs, n_wins, t_decim, t_emd, t_emd / min(n_wins, 20)

for p in ['data/iara/A/A-0001.wav', 'data/iara/A/A-0002.wav', 'data/iara/A/A-0004.wav']:
    dur, n_wins, t_dec, t_emd, per_win = benchmark_one_audio(p)
    print(f"{p}: Dur={dur:.1f}s, TotalWins={n_wins}, Decim={t_dec:.3f}s, EMD(20 wins)={t_emd:.2f}s, PerWin={per_win*1000:.1f}ms")
