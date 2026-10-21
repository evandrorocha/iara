"""
Exporta as predições já existentes em data/iara_processed/{rf,mlp,svm,cnn}_all_representations_predictions.pkl
para results/trainings/multirep/<exp>/ (formato do pipeline iara), para que fiquem rastreáveis sem retreino.

Limitação: os pickles legados guardam só a predição final por ARQUIVO (voto já aplicado), não por janela.
Por isso cada CSV exportado tem 1 linha por arquivo, e o config.json marca "granularity": "arquivo".
Ao rodar de novo o script gerador original, o mesmo diretório é sobrescrito com as predições por janela.

Só exporta as representações treinadas pelos scripts de scratch/. As demais (ex.: MLP MEL/LOFAR, SVM MEL/LOFAR,
CNN MEL) já vêm do pipeline oficial em results/trainings/tests/.

Uso (na raiz do repositório):
    python scratch/export_legacy_predictions.py
"""
import datetime
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_writer import ResultsWriter, load_file_predictions  # noqa: E402

OFFICIAL_SPLIT = ("results/trainings/tests/svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0/eval/fold_{f}/"
                  "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0_multiclass_trn.csv")

# (pickle, chave da representação, nome do experimento, script que gerou o pickle)
EXPORTS = [
    ("rf_all_representations_predictions.pkl", "mel", "rf_mel5s_n100_balanced_4c", "scratch/generate_rf_predictions_and_analysis.py"),
    ("rf_all_representations_predictions.pkl", "lofar", "rf_lofar5s_n100_balanced_4c", "scratch/generate_rf_predictions_and_analysis.py"),
    ("rf_all_representations_predictions.pkl", "hht", "rf_hht_n100_balanced_4c", "scratch/generate_rf_predictions_and_analysis.py"),
    ("rf_all_representations_predictions.pkl", "demon", "rf_demon_n100_balanced_4c", "scratch/generate_rf_predictions_and_analysis.py"),
    ("mlp_all_representations_predictions.pkl", "hht", "mlp_hht_h32-16_e30_4c", "scratch/generate_mlp_predictions_and_analysis.py"),
    ("mlp_all_representations_predictions.pkl", "demon", "mlp_demon_h32-16_e30_4c", "scratch/generate_mlp_predictions_and_analysis.py"),
    ("svm_all_representations_predictions.pkl", "hht", "svm_nystroem_2000_hht_C2.0_elasticnet_l1r0.15_4c", "scratch/save_svm_all_representations.py"),
    ("svm_all_representations_predictions.pkl", "demon", "svm_nystroem_2000_demon_C2.0_elasticnet_l1r0.15_4c", "scratch/save_svm_all_representations.py"),
    ("cnn_all_representations_predictions.pkl", "lofar", "cnn2d_lofar5s_w32_b64_e15_4c", "scratch/generate_cnn_predictions_and_analysis.py"),
    ("cnn_all_representations_predictions.pkl", "hht", "cnn2d_hht_w32_b64_e20_4c", "scratch/generate_cnn_predictions_and_analysis.py"),
    ("cnn_all_representations_predictions.pkl", "demon", "cnn2d_demon_w32_b64_e20_4c", "scratch/generate_cnn_predictions_and_analysis.py"),
]


def main():
    official_trn = {f: pd.read_csv(OFFICIAL_SPLIT.format(f=f)).groupby("File")["Target"].first().to_dict()
                    for f in range(10)}
    cache = {}
    for pkl_name, rep, exp_name, src_script in EXPORTS:
        pkl_path = os.path.join("data", "iara_processed", pkl_name)
        if pkl_path not in cache:
            with open(pkl_path, "rb") as f:
                cache[pkl_path] = pickle.load(f)
        data = cache[pkl_path]

        writer = ResultsWriter(exp_name, script=src_script, params={
            "granularity": "arquivo (exportado de pickle legado; voto majoritário já aplicado)",
            "legacy_source": pkl_path.replace("\\", "/"),
            "legacy_source_mtime": datetime.datetime.fromtimestamp(os.path.getmtime(pkl_path)).isoformat(timespec="seconds"),
            "legacy_key": rep,
            "exported_by": "scratch/export_legacy_predictions.py",
            "hiperparametros": f"ver {src_script} (chamada que gera '{exp_name}')",
        })
        for fold in range(10):
            fids = sorted(int(k) for k in data["targets"][fold])
            preds_fold = {int(k): int(v) for k, v in data[rep][fold].items()}
            tgts_fold = {int(k): int(v) for k, v in data["targets"][fold].items()}
            writer.save_fold(fold, fids, [tgts_fold[i] for i in fids], [preds_fold[i] for i in fids],
                             train_files=official_trn[fold])

        # Verificação: o que foi gravado reproduz exatamente o pickle
        preds, _ = load_file_predictions(writer.exp_dir)
        mismatches = sum(preds[f][int(k)] != int(v) for f in range(10) for k, v in data[rep][f].items())
        assert mismatches == 0, f"{exp_name}: {mismatches} predições divergentes"
        print(f"OK  {exp_name:<55} <- {pkl_name}['{rep}']")


if __name__ == "__main__":
    main()
