"""
Módulo de Implementação do Sistema Gated Front-End para Classificação Acústica Submarina (IARA).
Implementa:
1. AcousticFrontEndGating: Extração de meta-características acústicas físicas e seleção de rota.
2. GatedSVMSystem: Pipeline integrado que recebe o áudio, roteia a entrada e classifica.
"""
import os
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import soundfile as sf


class AcousticFrontEndGating:
    """
    Módulo de Gating Front-End que analisa propriedades físicas do sinal acústico
    no tempo e frequência antes da extração de representações pesadas.
    """
    def __init__(self, high_low_thresh: float = 0.8, mod_thresh: float = 1.15,
                 crest_thresh: float = 5.0, flux_thresh: float = 0.35):
        self.high_low_thresh = high_low_thresh
        self.mod_thresh = mod_thresh
        self.crest_thresh = crest_thresh
        self.flux_thresh = flux_thresh

    def extract_metafeatures(self, sig: np.ndarray, fs: float = 48000.0) -> dict:
        """
        Extrai 4 meta-características físicas do áudio em tempo real.
        """
        if sig.ndim > 1:
            sig = sig[:, 0]

        # 1. Tonal Crest Factor em Baixa Frequência (< 250 Hz) -> Indicador LOFAR
        f, psd = signal.welch(sig, fs=fs, nperseg=4096)
        low_mask = (f >= 5.0) & (f <= 250.0)
        low_psd = psd[low_mask]
        crest_factor = float(np.max(low_psd) / (np.mean(low_psd) + 1e-12))
        flatness = float(stats.gmean(low_psd + 1e-12) / (np.mean(low_psd) + 1e-12))

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

        # 4. Razão de Energia Alta vs Baixa (High/Low Ratio)
        high_mask = (f >= 2000.0) & (f <= 8000.0)
        e_high = float(np.sum(psd[high_mask]))
        e_low = float(np.sum(low_psd))
        high_low_ratio = float((e_high + 1e-12) / (e_low + 1e-12))

        return {
            'crest_factor': crest_factor,
            'spectral_flatness': flatness,
            'mod_index': mod_index,
            'flux_std': flux_std,
            'high_low_ratio': high_low_ratio
        }

    def select_route(self, sig: np.ndarray, fs: float = 48000.0) -> str:
        """
        Decide a rota de representação mais indicada (DEMON, LOFAR, HHT ou MEL).
        """
        feats = self.extract_metafeatures(sig, fs=fs)
        
        # Regra 1: Cavitação rápida de hélice (típica de lanchas Small)
        if feats['high_low_ratio'] > self.high_low_thresh and feats['mod_index'] > self.mod_thresh:
            return 'demon'
        
        # Regra 2: Raias harmônicas puras de máquinas pesadas (típica de cargueiros Large)
        if feats['crest_factor'] > self.crest_thresh and feats['high_low_ratio'] < 0.5:
            return 'lofar'
        
        # Regra 3: Transientes e não-estacionaridade brusca (manobra)
        if feats['flux_std'] > self.flux_thresh:
            return 'hht'
        
        # Regra 4: Padrão banda larga contínua
        return 'mel'


class GatedSVMSystem:
    """
    Sistema Integrado Gated-SVM:
    Recebe um áudio submarino, o Gating determina a representação ótima,
    extrai exclusivamente essa representação e o SVM correspondente emite a predição.
    """
    def __init__(self, gating_module: AcousticFrontEndGating = None, models_dict: dict = None):
        self.gating = gating_module or AcousticFrontEndGating()
        self.models = models_dict or {}

    def predict_audio(self, wav_path: str, duration: float = 10.0) -> dict:
        """
        Executa a predição completa do arquivo de áudio.
        """
        info = sf.info(wav_path)
        fs = info.samplerate
        n_frames = min(int(duration * fs), info.frames)
        start_frame = max(0, int((info.frames - n_frames) / 2))
        sig, _ = sf.read(wav_path, start=start_frame, frames=n_frames, dtype='float32')

        # 1. Gating seleciona a rota de representação
        route = self.gating.select_route(sig, fs=fs)

        # 2. Em um ambiente de produção, acionaria apenas o extrator e o SVM de 'route'
        return {
            'wav_path': wav_path,
            'selected_route': route,
            'duration_sec': duration
        }
