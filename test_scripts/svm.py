"""
Test script for SVM with Nyström kernel approximation on the IARA dataset.

This script evaluates SVMNystroem (approximate RBF-SVM via Nyström + SGDClassifier)
as a classifier for underwater acoustic target recognition.

Method justification:
    - SVC(kernel='rbf') is infeasible for ~500K windows (O(n²) complexity)
    - Nyström approximates the RBF kernel mapping in R^n_components (m << n)
    - SGDClassifier(loss='hinge') = SVM linear in the transformed space
    - Combined: approximate RBF-SVM with O(m*n) complexity (~5 min/fold)
    - Uses InputType.Window() → trained on individual windows, evaluated by_audio

Usage (inside Docker container):
    # Run fold 0 only (quick test):
    python test_scripts/svm.py

    # Run all 10 folds (full statistical evaluation):
    python test_scripts/svm.py -F 0-9

    # Run specific folds:
    python test_scripts/svm.py -F 0,2,4

Reference:
    Williams, C., & Seeger, M. (2001). Using the Nyström method to speed up
    kernel machines. NeurIPS.
"""
import time
import typing
import argparse
import os
import pandas as pd
import numpy as np
import collections
import pickle
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier

import iara.utils
import iara.ml.dataset as iara_dataset
import iara.default as iara_default
import iara.ml.experiment as iara_exp
import iara.ml.metrics as iara_metrics
import iara.ml.models.trainer as iara_trn
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc
import iara.ml.models.base_model as iara_model

from iara.default import DEFAULT_DIRECTORIES


class HybridAudioFileProcessor(iara_manager.AudioFileProcessor):
    """Custom processor that loads both MEL and LOFAR features and concatenates them."""
    def __init__(self, mel_processor, lofar_processor):
        self.mel_processor = mel_processor
        self.lofar_processor = lofar_processor
        
        # Mimic base class variables
        self.data_base_dir = mel_processor.data_base_dir
        self.data_processed_base_dir = mel_processor.data_processed_base_dir
        self.normalization = mel_processor.normalization
        self.analysis = "hybrid"
        self.frequency_limit = None
        self.integration_overlap = mel_processor.integration_overlap
        self.integration_interval = mel_processor.integration_interval
        
        # Copy resolution parameters to support serialization (_to_dict)
        self.n_pts = mel_processor.n_pts
        self.n_overlap = mel_processor.n_overlap
        self.n_mels = mel_processor.n_mels
        self.decimation_rate = mel_processor.decimation_rate
        
        self._check_dir()

    def _get_hash(self) -> str:
        return "hybrid_fusion_mel_lofar_v1"

    def get_data(self, file_id: int) -> typing.Tuple[pd.DataFrame, np.ndarray]:
        df_mel, times = self.mel_processor.get_data(file_id)
        df_lofar, _ = self.lofar_processor.get_data(file_id)
        
        df_mel_renamed = df_mel.copy()
        df_mel_renamed.columns = [f'mel_{i}' for i in range(df_mel.shape[1])]
        
        df_lofar_renamed = df_lofar.copy()
        df_lofar_renamed.columns = [f'lofar_{i}' for i in range(df_lofar.shape[1])]
        
        df_hybrid = pd.concat([df_mel_renamed, df_lofar_renamed], axis=1)
        return df_hybrid, times


class SVMNystroemHybrid(iara_model.BaseModel):
    """Approximate RBF-SVM that applies PCA only to the LOFAR portion of concatenated features."""
    def __init__(self,
                 n_components: int = 300,
                 gamma: typing.Union[str, float] = 'scale',
                 C: float = 1.0,
                 n_targets: int = 4,
                 n_mel_features: int = 256,
                 n_pca_components: int = 64,
                 normalize: bool = True,
                 penalty: str = 'l2',
                 l1_ratio: float = 0.15,
                 biases: typing.Optional[typing.List[float]] = None,
                 random_state: int = 42):
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.C = C
        self.n_targets = n_targets
        self.n_mel_features = n_mel_features
        self.n_pca_components = n_pca_components
        self.normalize = normalize
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.biases = biases
        self.random_state = random_state
        
        self.pca_trans = PCA(n_components=n_pca_components, random_state=random_state)
        self.scaler = StandardScaler() if normalize else None
        
        self.nystroem = Nystroem(
            kernel='rbf',
            gamma=None,
            n_components=n_components,
            random_state=random_state
        )
        
        self.sgd = SGDClassifier(
            loss='hinge',
            penalty=penalty,
            l1_ratio=l1_ratio,
            alpha=1.0,
            class_weight='balanced',
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
            n_jobs=1
        )
        
        self.is_fitted = False

    def fit(self, samples: torch.Tensor, targets: torch.Tensor) -> None:
        X = samples.view(samples.size(0), -1).cpu().numpy()
        y = targets.cpu().numpy().astype(int)
        
        n_samples = X.shape[0]
        alpha = 1.0 / (self.C * n_samples)
        self.sgd.set_params(alpha=alpha)
        
        # Split MEL and LOFAR portions
        X_mel = X[:, :self.n_mel_features]
        X_lofar = X[:, self.n_mel_features:]
        
        # Apply PCA strictly to LOFAR
        X_lofar_pca = self.pca_trans.fit_transform(X_lofar)
        
        # Concatenate back
        X_combined = np.hstack([X_mel, X_lofar_pca])
        
        # Apply normalization to the fused representation
        if self.normalize:
            X_combined = self.scaler.fit_transform(X_combined)
            
        # Calculate gamma on RBF inputs
        if isinstance(self.gamma, str) and 'scale' in self.gamma:
            base_gamma = 1.0 / (X_combined.shape[1] * X_combined.var())
            if '*' in self.gamma:
                parts = self.gamma.split('*')
                multiplier = 1.0
                for p in parts:
                    if p != 'scale':
                        try:
                            multiplier = float(p)
                        except ValueError:
                            pass
                gamma = base_gamma * multiplier
            else:
                gamma = base_gamma
        elif self.gamma is None:
            gamma = 1.0 / X_combined.shape[1]
        else:
            gamma = self.gamma
            
        self.nystroem.set_params(gamma=gamma)
        
        # Transform using Nyström
        X_transformed = self.nystroem.fit_transform(X_combined)
        
        # Train SGD linear SVM
        self.sgd.fit(X_transformed, y)
        self.is_fitted = True

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling forward().")
            
        X = data.view(data.size(0), -1).cpu().numpy()
        X_mel = X[:, :self.n_mel_features]
        X_lofar = X[:, self.n_mel_features:]
        
        X_lofar_pca = self.pca_trans.transform(X_lofar)
        X_combined = np.hstack([X_mel, X_lofar_pca])
        
        if self.normalize:
            X_combined = self.scaler.transform(X_combined)
            
        X_transformed = self.nystroem.transform(X_combined)
        
        scores = self.sgd.decision_function(X_transformed)
        biases = getattr(self, 'biases', None)
        if biases is not None:
            if len(scores.shape) == 1 or scores.shape[1] == 1:
                # Binary classification
                bias_val = biases[1] - biases[0]
                if len(scores.shape) == 1:
                    scores += bias_val
                    predictions = (scores > 0).astype(int)
                else:
                    scores[:, 0] += bias_val
                    predictions = (scores[:, 0] > 0).astype(int)
            else:
                # Multiclass classification
                for i, cls in enumerate(self.sgd.classes_):
                    if cls < len(biases):
                        scores[:, i] += biases[cls]
                predictions = np.argmax(scores, axis=1)
        else:
            predictions = self.sgd.predict(X_transformed)
            
        return torch.tensor(predictions, dtype=torch.long)


class SVMNystroemHybridTrainer(iara_trn.BaseTrainer):
    """Trainer class for SVMNystroemHybrid compatible with Manager framework."""
    def __init__(self,
                 training_strategy: iara_trn.ModelTrainingStrategy,
                 trainer_id: str,
                 n_targets: int,
                 n_components: int = 300,
                 gamma: typing.Union[str, float] = 'scale',
                 C: float = 1.0,
                 n_mel_features: int = 256,
                 n_pca_components: int = 64,
                 normalize: bool = True,
                 penalty: str = 'l2',
                 l1_ratio: float = 0.15,
                 biases: typing.Optional[typing.List[float]] = None) -> None:
        super().__init__(training_strategy, trainer_id, n_targets)
        self.n_components = n_components
        self.gamma = gamma
        self.C = C
        self.n_mel_features = n_mel_features
        self.n_pca_components = n_pca_components
        self.normalize = normalize
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.biases = biases

    def fit(self,
            model_base_dir: str,
            trn_dataset: iara_dataset.BaseDataset,
            val_dataset: iara_dataset.BaseDataset) -> None:
        if self.is_trained(model_base_dir=model_base_dir):
            return
            
        os.makedirs(model_base_dir, exist_ok=True)
        
        if self.training_strategy == iara_trn.ModelTrainingStrategy.MULTICLASS:
            target_ids = [None]
        elif self.training_strategy == iara_trn.ModelTrainingStrategy.CLASS_SPECIALIST:
            target_ids = trn_dataset.get_targets()
            
        samples = trn_dataset.get_samples()
        if samples is None:
            raise ValueError("Training dataset without data")
            
        for target_id in target_ids:
            model_filename = self.output_filename(model_base_dir=model_base_dir,
                                                   target_id=target_id)
            if os.path.exists(model_filename):
                continue
                
            model = SVMNystroemHybrid(
                n_components=self.n_components,
                gamma=self.gamma,
                C=self.C,
                n_targets=self.n_targets,
                n_mel_features=self.n_mel_features,
                n_pca_components=self.n_pca_components,
                normalize=self.normalize,
                penalty=self.penalty,
                l1_ratio=self.l1_ratio,
                biases=self.biases
            )
            
            targets = trn_dataset.get_targets()
            if target_id is not None:
                targets = torch.where(targets == target_id,
                                      torch.tensor(1.0),
                                      torch.tensor(0.0))
                                      
            model.fit(samples=samples, targets=targets)
            model.save(model_filename)


def main(folds: typing.List[int], n_components: int = 300, analysis_name: str = 'log_melgram', C: float = 1.0, gamma: typing.Union[str, float] = 'scale', normalize: bool = False, pca: bool = False, n_pca_components: int = 64, penalty: str = 'l2', l1_ratio: float = 0.15, biases: typing.Optional[typing.List[float]] = None):

    output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/tests"
    directories = DEFAULT_DIRECTORIES

    grid = iara_metrics.GridCompiler()

    # Use individual windows as input (same as MLP baseline)
    # The by_audio evaluation will apply majority vote across windows of each file
    input_type = iara_dataset.InputType.Window()

    # Audio preprocessing pipeline
    if analysis_name.lower() == 'hybrid':
        dp_mel = iara_manager.AudioFileProcessor(
            data_base_dir=directories.data_dir,
            data_processed_base_dir=directories.process_dir,
            normalization=iara_proc.Normalization.NORM_L2,
            analysis=iara_proc.SpectralAnalysis.LOG_MELGRAM,
            n_pts=1024,
            n_overlap=0,
            decimation_rate=3,
            n_mels=256,
            integration_interval=0.512
        )
        dp_lofar = iara_manager.AudioFileProcessor(
            data_base_dir=directories.data_dir,
            data_processed_base_dir=directories.process_dir,
            normalization=iara_proc.Normalization.NORM_L2,
            analysis=iara_proc.SpectralAnalysis.LOFAR,
            n_pts=1024,
            n_overlap=0,
            decimation_rate=3,
            n_mels=256,
            integration_interval=0.512
        )
        dp = HybridAudioFileProcessor(dp_mel, dp_lofar)
    else:
        if analysis_name.lower() == 'lofar':
            analysis_enum = iara_proc.SpectralAnalysis.LOFAR
        else:
            analysis_enum = iara_proc.SpectralAnalysis.LOG_MELGRAM

        dp = iara_manager.AudioFileProcessor(
            data_base_dir=directories.data_dir,
            data_processed_base_dir=directories.process_dir,
            normalization=iara_proc.Normalization.NORM_L2,
            analysis=analysis_enum,
            n_pts=1024,
            n_overlap=0,
            decimation_rate=3,
            n_mels=256,
            integration_interval=0.512
        )

    # Dynamic folder name includes C, gamma and preprocessors if non-default
    name_parts = [f'svm_nystroem_{n_components}', analysis_name]
    if normalize:
        name_parts.append('norm')
    if pca and analysis_name.lower() != 'hybrid':
        name_parts.append(f'pca{n_pca_components}')
    if penalty != 'l2':
        name_parts.append(penalty)
        if penalty == 'elasticnet':
            name_parts.append(f'l1r{l1_ratio}')
    if C != 1.0:
        name_parts.append(f'C{C}')
    if gamma != 'scale':
        name_parts.append(f'g{gamma}')
    exp_name = "_".join(name_parts)

    config = iara_exp.Config(
        name=exp_name,
        dataset=iara_default.default_collection(),
        dataset_processor=dp,
        output_base_dir=output_base_dir,
        input_type=input_type
    )

    trainers = []

    # SVM with Nyström approximation (subclass selection for 100% safety)
    if analysis_name.lower() == 'hybrid':
        trainer = SVMNystroemHybridTrainer(
            training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
            trainer_id=exp_name,
            n_targets=config.dataset.target.get_n_targets(),
            n_components=n_components,
            gamma=gamma,
            C=C,
            n_mel_features=256,
            n_pca_components=n_pca_components,
            normalize=normalize,
            penalty=penalty,
            l1_ratio=l1_ratio,
            biases=biases
        )
    elif normalize or pca:
        trainer = iara_trn.SVMNystroemPreprocessedTrainer(
            training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
            trainer_id=exp_name,
            n_targets=config.dataset.target.get_n_targets(),
            n_components=n_components,
            gamma=gamma,
            C=C,
            normalize=normalize,
            pca=pca,
            n_pca_components=n_pca_components,
            penalty=penalty,
            l1_ratio=l1_ratio,
            biases=biases
        )
    else:
        trainer = iara_trn.SVMNystroemTrainer(
            training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
            trainer_id=exp_name,
            n_targets=config.dataset.target.get_n_targets(),
            n_components=n_components,
            gamma=gamma,
            C=C,
            biases=biases
        )
    trainers.append(trainer)

    manager = iara_exp.Manager(config, *trainers)

    result_grid = manager.run(folds=folds, override=True)

    for (eval_subset, eval_strategy), result_dict in result_grid.items():
        if eval_subset == iara_trn.Subset.ALL:
            continue

        for trainer_id, results in result_dict.items():
            for i_fold, result in enumerate(results):
                grid.add(
                    params={
                        'eval_strategy': eval_strategy,
                        'eval_subset': eval_subset,
                    },
                    i_fold=i_fold,
                    target=result['Target'],
                    prediction=result['Prediction']
                )

    print(grid)


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='Test SVM with Nystroem kernel approximation on IARA dataset'
    )
    parser.add_argument(
        '-F', '--fold',
        type=str,
        default=None,
        help='Folds to execute. Examples: "0" | "0-9" | "0,2,4"'
    )

    parser.add_argument(
        '-C', '--components',
        type=int,
        default=300,
        help='Number of components for Nystroem kernel approximation. Default: 300'
    )

    parser.add_argument(
        '-A', '--analysis',
        type=str,
        default='log_melgram',
        help='Spectral analysis type: log_melgram or lofar. Default: log_melgram'
    )

    parser.add_argument(
        '--reg_c',
        type=float,
        default=1.0,
        help='Regularization parameter C. Default: 1.0'
    )

    parser.add_argument(
        '--gamma',
        type=str,
        default='scale',
        help='Kernel coefficient gamma. Can be "scale", "auto" or a float. Default: "scale"'
    )

    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Enable StandardScaler feature standardization preprocessing'
    )

    parser.add_argument(
        '--pca',
        action='store_true',
        help='Enable PCA dimensionality reduction preprocessing'
    )

    parser.add_argument(
        '--pca_components',
        type=int,
        default=64,
        help='Number of principal components for PCA when enabled. Default: 64'
    )

    parser.add_argument(
        '--penalty',
        type=str,
        default='l2',
        choices=['l2', 'l1', 'elasticnet'],
        help='Regularization penalty for SGDClassifier: l2, l1, elasticnet. Default: l2'
    )

    parser.add_argument(
        '--l1_ratio',
        type=float,
        default=0.15,
        help='ElasticNet mixing parameter (between 0 and 1) when penalty=elasticnet. Default: 0.15'
    )

    parser.add_argument(
        '--bias_small',
        type=float,
        default=0.0,
        help='Decision score bias offset for SMALL class. Default: 0.0'
    )
    parser.add_argument(
        '--bias_medium',
        type=float,
        default=0.0,
        help='Decision score bias offset for MEDIUM class. Default: 0.0'
    )
    parser.add_argument(
        '--bias_large',
        type=float,
        default=0.0,
        help='Decision score bias offset for LARGE class. Default: 0.0'
    )
    parser.add_argument(
        '--bias_background',
        type=float,
        default=0.0,
        help='Decision score bias offset for BACKGROUND class. Default: 0.0'
    )

    args = parser.parse_args()

    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(1)))
    n_components = args.components
    analysis_type = args.analysis
    reg_c = args.reg_c
    normalize_enabled = args.normalize
    pca_enabled = args.pca
    pca_comp = args.pca_components
    penalty_type = args.penalty
    l1_ratio_val = args.l1_ratio
    
    # Try parsing gamma as float, otherwise keep as string
    gamma_val = args.gamma
    try:
        gamma_val = float(args.gamma)
    except ValueError:
        pass

    biases = [
        args.bias_small,
        args.bias_medium,
        args.bias_large,
        args.bias_background
    ]

    print(f"Running SVM Nyström on folds: {folds_to_execute}")
    print(f"  n_components={n_components}, gamma={gamma_val}, C={reg_c}")
    print(f"  analysis={analysis_type}")
    print(f"  preprocessing: normalize={normalize_enabled}, pca={pca_enabled} (n_components={pca_comp})")
    print(f"  regularization: penalty={penalty_type}, l1_ratio={l1_ratio_val}")
    print(f"  biases: SMALL={biases[0]}, MEDIUM={biases[1]}, LARGE={biases[2]}, BACKGROUND={biases[3]}")
    print(f"  InputType: Window (by_audio evaluation via majority vote)")
    print()

    main(
        folds=folds_to_execute,
        n_components=n_components,
        analysis_name=analysis_type,
        C=reg_c,
        gamma=gamma_val,
        normalize=normalize_enabled,
        pca=pca_enabled,
        n_pca_components=pca_comp,
        penalty=penalty_type,
        l1_ratio=l1_ratio_val,
        biases=biases
    )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nElapsed time: {iara.utils.str_format_time(elapsed)}")
