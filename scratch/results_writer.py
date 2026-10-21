"""
Grava as predições dos experimentos de scratch/ (HHT, DEMON, fusões, BiGRU, ...) em results/trainings/,
no mesmo formato usado pelo pipeline iara (results/trainings/tests/<exp>/eval/fold_k/<exp>_multiclass_test.csv),
para que todo resultado citado nos notebooks possa ser recalculado a partir das predições por janela.

Estrutura gerada:
    results/trainings/multirep/<exp_name>/
        config.json                               # script, hiperparâmetros, commit git, data
        eval/fold_k/<exp_name>_multiclass_test.csv  # File,Target,Prediction (1 linha por janela/imagem/sequência)
        eval/fold_k/<exp_name>_split.csv            # File,Target,Subset (trn/test) -> auditoria de vazamento

Uso:
    from results_writer import ResultsWriter
    writer = ResultsWriter("rf_hht_n100_4c", script=__file__, params={...})
    writer.save_fold(fold_idx, test_fids, y_test, win_preds, train_files=fold["train_files"])
"""
import datetime
import json
import os
import subprocess
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "trainings", "multirep")
CLASS_NAMES = {0: "SMALL", 1: "MEDIUM", 2: "LARGE", 3: "BACKGROUND"}


def _env_info():
    import platform
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "sklearn", "torch", "scipy"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            pass
    try:
        import torch
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


def _git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


class ResultsWriter:
    def __init__(self, exp_name: str, script: str, params: dict, base_dir: str = RESULTS_DIR):
        self.exp_name = exp_name
        self.exp_dir = os.path.join(base_dir, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        commit, dirty = _git_info()
        config = {
            "exp_name": exp_name,
            "script": os.path.relpath(os.path.abspath(script), REPO_ROOT).replace("\\", "/"),
            "params": params,
            "git_commit": commit,
            "git_dirty": dirty,
            "environment": _env_info(),
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "classes": CLASS_NAMES,
            "folds": "5x2cv oficial (mesmos arquivos de results/trainings/tests/*/eval/fold_k)",
            "file_prediction": "voto majoritário das linhas de cada File (collections.Counter.most_common; empate -> classe da 1ª janela)",
        }
        with open(os.path.join(self.exp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    def _fold_dir(self, fold_idx: int) -> str:
        d = os.path.join(self.exp_dir, "eval", f"fold_{fold_idx}")
        os.makedirs(d, exist_ok=True)
        return d

    def save_fold(self, fold_idx, file_ids, targets, predictions, train_files=None, scores=None, subset="test"):
        """Salva as predições de um fold (uma linha por janela/imagem/sequência).

        train_files: dict {file_id: target} usado no treino do fold; gravado em <exp>_split.csv
        scores: array (n_amostras, n_classes) opcional com probabilidades/logits por classe
        """
        df = pd.DataFrame({
            "File": np.asarray(file_ids, dtype=np.int64),
            "Target": np.asarray(targets, dtype=np.int64),
        })
        if scores is not None:
            scores = np.asarray(scores)
            for k in range(scores.shape[1]):
                df[f"Prediction_{k}"] = scores[:, k]
        df["Prediction"] = np.asarray(predictions, dtype=np.int64)

        fold_dir = self._fold_dir(fold_idx)
        df.to_csv(os.path.join(fold_dir, f"{self.exp_name}_multiclass_{subset}.csv"), index=False)

        if train_files is not None:
            test_files = df.groupby("File")["Target"].first()
            split = pd.concat([
                pd.DataFrame({"File": list(train_files.keys()), "Target": list(train_files.values()), "Subset": "trn"}),
                pd.DataFrame({"File": test_files.index, "Target": test_files.values, "Subset": subset}),
            ])
            overlap = set(split[split.Subset == "trn"].File) & set(test_files.index)
            assert not overlap, f"{self.exp_name} fold {fold_idx}: {len(overlap)} arquivos em treino e {subset}!"
            split.to_csv(os.path.join(fold_dir, f"{self.exp_name}_split.csv"), index=False)


def load_file_predictions(exp_dir: str, subset: str = "test", n_folds: int = 10):
    """Lê <exp_dir>/eval/fold_k/*_multiclass_<subset>.csv (formato do pipeline iara ou deste módulo)
    e devolve (preds, targets) como {fold: {file_id: classe}}, com voto majoritário por arquivo."""
    exp_name = os.path.basename(os.path.normpath(exp_dir))
    preds, targets = {}, {}
    for fold_idx in range(n_folds):
        path = os.path.join(exp_dir, "eval", f"fold_{fold_idx}", f"{exp_name}_multiclass_{subset}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        votes = defaultdict(list)
        for fid, p in zip(df["File"], df["Prediction"]):
            votes[int(fid)].append(int(p))
        # mesmo critério de src/iara/ml/models/trainer.py: Counter.most_common (empate -> 1ª janela)
        preds[fold_idx] = {fid: Counter(v).most_common(1)[0][0] for fid, v in votes.items()}
        targets[fold_idx] = {int(k): int(v) for k, v in df.groupby("File")["Target"].first().items()}
    return preds, targets
