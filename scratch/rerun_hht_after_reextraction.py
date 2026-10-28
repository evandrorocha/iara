"""
Retreina SÓ os experimentos HHT dos scripts que também treinam outras representações, depois da
re-extração do HHT com uma configuração única de EMD (8 IMFs, MAX_ITERATION=30).

Motivo: o cache HHT anterior (backup em data/iara_processed/_backup_hht_iter_misto_20261015/) misturava
dois limites de iterações de peneiramento (as gravações extraídas antes de 01:55 de 15/10 com >= 100;
as demais com 30), e todas as gravações BACKGROUND estavam no segundo grupo. As predições antigas estão
em results/trainings/_backup_hht_iter_misto_20261015/.

Chama as mesmas funções, com os mesmos hiperparâmetros e os mesmos nomes de experimento dos scripts
originais (ResultsWriter grava em results/trainings/multirep/<exp>/). Não regrava os pickles de resumo
desses scripts (bigru_*_results.pkl, *_all_representations_predictions.pkl etc.).

Uso (no container, na raiz do repositório):
    python scratch/rerun_hht_after_reextraction.py svm2000 svm4000 mlp_seed42 cnn2d_b64 bigru bigru_mel bigru_lofar
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_writer import ResultsWriter  # noqa: E402

CACHE_HHT = "data/iara_processed/cache_hht.pkl"

SVM_PARAMS = {"model": "Nystroem(rbf)+SGDClassifier(hinge)", "representation": "HHT", "dim": 256, "C": 2.0,
              "penalty": "elasticnet", "l1_ratio": 0.15, "max_iter": 200, "early_stopping": True,
              "class_weight": "balanced", "gamma": "1/(n_features*var)", "random_state": 42, "num_classes": 4}


def load_hht():
    with open(CACHE_HHT, "rb") as f:
        return pickle.load(f)


def svm2000():
    import save_svm_all_representations as S
    S.run_svm_predictions(load_hht(), S.get_official_folds(), n_components=2000, C=2.0, writer=ResultsWriter(
        "svm_nystroem_2000_hht_C2.0_elasticnet_l1r0.15_4c", script=S.__file__,
        params={**SVM_PARAMS, "n_components": 2000}))


def svm4000():
    import analyze_small_errors_by_input as S
    S.run_svm_predictions(load_hht(), S.get_official_folds(), n_components=4000, C=2.0, writer=ResultsWriter(
        "svm_nystroem_4000_hht_C2.0_elasticnet_l1r0.15_4c", script=S.__file__,
        params={**SVM_PARAMS, "n_components": 4000}))


def mlp_seed42():
    import generate_mlp_predictions_and_analysis as S
    S.run_pytorch_mlp(load_hht(), S.get_official_folds(), writer=ResultsWriter(
        "mlp_hht_h32-16_e30_4c", script=S.__file__,
        params={"model": "PyTorchMLP", "representation": "HHT", "dim": 256, "hidden": [32, 16],
                "dropout": 0.2, "epochs": 30, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-3,
                "seed": 42, "early_stopping": False, "scaler": "StandardScaler(treino)", "num_classes": 4}))


def cnn2d_b64():
    import generate_cnn_predictions_and_analysis as S
    S.run_pytorch_cnn(load_hht(), S.get_official_folds(), epochs=20, batch_size=64, writer=ResultsWriter(
        "cnn2d_hht_w32_b64_e20_4c", script=S.__file__,
        params={"model": "PyTorch2DCNN (conv64-conv32-GAP-fc64)", "representation": "HHT", "dim": 256,
                "frame_step_s": 5.12, "image_windows": 32, "image_stride": 16, "epochs": 20, "batch_size": 64, "lr": 1e-3,
                "weight_decay": 1e-3, "seed": 42, "early_stopping": False,
                "normalization": "z-score escalar global (treino)", "num_classes": 4}))


def _bigru_mel_caches():
    """Mesmo alinhamento de eval_bigru_multiclass_5x2cv.main()."""
    with open("data/iara_processed/cache_fusion_mel_demon.pkl", "rb") as f:
        mel_cache = pickle.load(f)["mel"]
    hht_raw = load_hht()
    hht_cache, fused_mel_hht = {}, {}
    for fid in mel_cache:
        if fid in hht_raw:
            h, m = hht_raw[fid], mel_cache[fid]
            n = min(len(m), len(h))
            hht_cache[fid] = h[:n]
            fused_mel_hht[fid] = np.hstack([m[:n], h[:n]])
    return hht_cache, fused_mel_hht


def _run_bigru(module, cache, rep_tag, epochs):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for n_cls in (4, 3):
        t0 = time.time()
        summary, _ = module.train_and_eval_bigru(cache, num_classes=n_cls, epochs=epochs, device=device, rep_tag=rep_tag)
        print(f"  {rep_tag} {n_cls}c: SP {summary['sp_mean']:.2f} ± {summary['sp_std']:.2f} ({time.time() - t0:.0f}s)", flush=True)


def bigru():
    import eval_bigru_multiclass_5x2cv as S
    hht_cache, _ = _bigru_mel_caches()
    _run_bigru(S, hht_cache, "hht", epochs=20)


def bigru_mel():
    import eval_bigru_multiclass_5x2cv as S
    _, fused = _bigru_mel_caches()
    _run_bigru(S, fused, "mel5s-hht", epochs=20)


def bigru_lofar():
    """Mesmo alinhamento de eval_bigru_lofar_multiclass_5x2cv.main() (LOFAR, DEMON e HHT truncados juntos)."""
    import eval_bigru_lofar_multiclass_5x2cv as S
    with open("data/iara_processed/cache_lofar.pkl", "rb") as f:
        cache_lofar = pickle.load(f)
    with open("data/iara_processed/cache_fusion_mel_demon.pkl", "rb") as f:
        demon = pickle.load(f)["demon"]
    hht = load_hht()
    fused = {}
    for fid in cache_lofar:
        if fid in demon and fid in hht:
            n = min(len(cache_lofar[fid]), len(demon[fid]), len(hht[fid]))
            fused[fid] = np.hstack([cache_lofar[fid][:n], hht[fid][:n]])
    _run_bigru(S, fused, "lofar5s-hht", epochs=15)


JOBS = {f.__name__: f for f in (svm2000, svm4000, mlp_seed42, cnn2d_b64, bigru, bigru_mel, bigru_lofar)}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(f"\n===== {name} =====", flush=True)
        t0 = time.time()
        JOBS[name]()
        print(f"===== {name} concluído em {time.time() - t0:.0f}s =====", flush=True)
