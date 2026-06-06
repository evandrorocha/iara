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
from sklearn.base import BaseEstimator, TransformerMixin

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
                 class_weight: typing.Union[str, dict, None] = 'balanced',
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
        self.class_weight = class_weight
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
            class_weight=class_weight,
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
                 biases: typing.Optional[typing.List[float]] = None,
                 class_weight: typing.Union[str, dict, None] = 'balanced') -> None:
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
        self.class_weight = class_weight

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
                biases=self.biases,
                class_weight=self.class_weight
            )
            
            targets = trn_dataset.get_targets()
            if target_id is not None:
                targets = torch.where(targets == target_id,
                                      torch.tensor(1.0),
                                      torch.tensor(0.0))
                                      
            model.fit(samples=samples, targets=targets)
            model.save(model_filename)


class KMeansNystroem(BaseEstimator, TransformerMixin):
    """Approximate RBF kernel mapping using MiniBatchKMeans centroids as landmarks."""
    def __init__(self,
                 kernel: str = 'rbf',
                 gamma: typing.Union[str, float] = 'scale',
                 n_components: int = 300,
                 random_state: int = 42):
        self.kernel = kernel
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.components_ = None
        self.normalization_ = None
        self.gamma_ = None

    def fit(self, X, y=None):
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics.pairwise import pairwise_kernels
        from scipy.linalg import svd

        # 1. Clustering via MiniBatchKMeans to find landmark centroids
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_components,
            random_state=self.random_state,
            batch_size=4096,
            max_iter=20,
            max_no_improvement=5,
            n_init=1,
            reassignment_ratio=0.01
        )
        kmeans.fit(X)
        self.components_ = kmeans.cluster_centers_

        # 2. Resolve Gamma
        if isinstance(self.gamma, str) and 'scale' in self.gamma:
            self.gamma_ = 1.0 / (X.shape[1] * X.var())
        elif self.gamma is None or (isinstance(self.gamma, str) and 'auto' in self.gamma):
            self.gamma_ = 1.0 / X.shape[1]
        else:
            self.gamma_ = self.gamma

        # 3. Compute K_UU matrix between centroids
        K_UU = pairwise_kernels(
            self.components_,
            metric=self.kernel,
            filter_params=True,
            gamma=self.gamma_
        )

        # 4. Compute K_UU^(-1/2) using SVD
        U, S, V = svd(K_UU, full_matrices=False)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)

        return self

    def transform(self, X):
        from sklearn.metrics.pairwise import pairwise_kernels
        K_XU = pairwise_kernels(
            X,
            self.components_,
            metric=self.kernel,
            filter_params=True,
            gamma=self.gamma_
        )
        return np.dot(K_XU, self.normalization_)


class StratifiedKMeansNystroem(KMeansNystroem):
    """KMeans Nyström landmarks allocated across classes.

    The regular KMeans landmark selection can spend most centroids on dense
    regions from majority/easier classes. This variant reserves a proportional
    number of centroids for each target class before building the same Nyström
    normalization matrix.
    """

    def __init__(self,
                 kernel: str = 'rbf',
                 gamma: typing.Union[str, float] = 'scale',
                 n_components: int = 300,
                 random_state: int = 42,
                 stratify_strategy: str = 'proportional',
                 component_weights: typing.Optional[typing.Dict[int, float]] = None):
        super().__init__(
            kernel=kernel,
            gamma=gamma,
            n_components=n_components,
            random_state=random_state
        )
        self.stratify_strategy = stratify_strategy
        self.component_weights = component_weights

    def _allocate_components(self, y: np.ndarray) -> typing.Dict[int, int]:
        labels, counts = np.unique(y, return_counts=True)
        n_classes = len(labels)
        if self.n_components < n_classes:
            raise ValueError(
                f"n_components={self.n_components} is smaller than the number "
                f"of classes with training samples ({n_classes})."
            )

        if self.stratify_strategy == 'balanced':
            ideal = np.full(n_classes, self.n_components / n_classes)
        elif self.stratify_strategy == 'weighted':
            if not self.component_weights:
                raise ValueError("component_weights must be set when stratify_strategy='weighted'.")
            weights = np.array([self.component_weights.get(int(label), 1.0) for label in labels])
            if np.any(weights <= 0):
                raise ValueError("All component weights must be positive.")
            ideal = weights / weights.sum() * self.n_components
        elif self.stratify_strategy == 'proportional':
            ideal = counts / counts.sum() * self.n_components
        else:
            raise ValueError(f"Unsupported stratify_strategy: {self.stratify_strategy}")

        allocation = np.maximum(1, np.floor(ideal).astype(int))
        allocation = np.minimum(allocation, counts)

        remaining = self.n_components - int(allocation.sum())
        while remaining > 0:
            candidates = np.where(allocation < counts)[0]
            if len(candidates) == 0:
                break
            deficits = ideal - allocation
            best = candidates[np.argmax(deficits[candidates])]
            allocation[best] += 1
            remaining -= 1

        return {int(label): int(n_clusters) for label, n_clusters in zip(labels, allocation)}

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("StratifiedKMeansNystroem requires class labels during fit().")

        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics.pairwise import pairwise_kernels
        from scipy.linalg import svd

        y = np.asarray(y).astype(int)
        allocation = self._allocate_components(y)
        centers = []

        for label in sorted(allocation):
            X_class = X[y == label]
            n_clusters = allocation[label]
            if n_clusters == 1:
                class_centers = X_class.mean(axis=0, keepdims=True)
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state + int(label),
                    batch_size=4096,
                    max_iter=20,
                    max_no_improvement=5,
                    n_init=1,
                    reassignment_ratio=0.01
                )
                kmeans.fit(X_class)
                class_centers = kmeans.cluster_centers_
            centers.append(class_centers)

        self.components_ = np.vstack(centers)
        self.components_by_class_ = allocation

        if isinstance(self.gamma, str) and 'scale' in self.gamma:
            self.gamma_ = 1.0 / (X.shape[1] * X.var())
        elif self.gamma is None or (isinstance(self.gamma, str) and 'auto' in self.gamma):
            self.gamma_ = 1.0 / X.shape[1]
        else:
            self.gamma_ = self.gamma

        K_UU = pairwise_kernels(
            self.components_,
            metric=self.kernel,
            filter_params=True,
            gamma=self.gamma_
        )

        U, S, V = svd(K_UU, full_matrices=False)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)

        return self


class SVMNystroemLocal(iara_model.BaseModel):
    """Approximate RBF-SVM with configurable Nyström landmark selection (Random vs KMeans) and RMS injection."""
    def __init__(self,
                 n_components: int = 300,
                 gamma: typing.Union[str, float] = 'scale',
                 C: float = 1.0,
                 n_targets: int = 4,
                 normalize: bool = False,
                 pca: bool = False,
                 n_pca_components: int = 64,
                 penalty: str = 'l2',
                 l1_ratio: float = 0.15,
                 biases: typing.Optional[typing.List[float]] = None,
                 class_weight: typing.Union[str, dict, None] = 'balanced',
                 rms: bool = False,
                 rms_scale: float = 1.0,
                 rms_mode: str = 'log',
                 nystroem_mode: str = 'random',
                 nystroem_stratify_strategy: str = 'proportional',
                 nystroem_component_weights: typing.Optional[typing.Dict[int, float]] = None,
                 random_state: int = 42):
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.C = C
        self.n_targets = n_targets
        self.normalize = normalize
        self.pca = pca
        self.n_pca_components = n_pca_components
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.biases = biases
        self.class_weight = class_weight
        self.rms = rms
        self.rms_scale = rms_scale
        self.rms_mode = rms_mode
        self.nystroem_mode = nystroem_mode
        self.nystroem_stratify_strategy = nystroem_stratify_strategy
        self.nystroem_component_weights = nystroem_component_weights
        self.random_state = random_state
        
        self.scaler = StandardScaler() if normalize else None
        self.pca_trans = PCA(n_components=n_pca_components, random_state=random_state) if pca else None
        self.rms_scaler = StandardScaler()
        
        if nystroem_mode == 'kmeans':
            self.nystroem = KMeansNystroem(
                kernel='rbf',
                gamma=gamma,
                n_components=n_components,
                random_state=random_state
            )
        elif nystroem_mode == 'stratified_kmeans':
            self.nystroem = StratifiedKMeansNystroem(
                kernel='rbf',
                gamma=gamma,
                n_components=n_components,
                random_state=random_state,
                stratify_strategy=nystroem_stratify_strategy,
                component_weights=nystroem_component_weights
            )
        else:
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
            class_weight=class_weight,
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
            n_jobs=1
        )
        
        self.is_fitted = False

    def _normalize_and_extract_rms(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        # Compute RMS of each raw frame
        if self.rms_mode == 'linear':
            # X is in dB (log_melgram or LOFAR). Convert to linear power.
            power_linear = 10.0 ** (X / 10.0)
            rms = np.sqrt(np.mean(power_linear, axis=1, keepdims=True))
        else:
            # Traditional log-RMS
            rms = np.sqrt(np.mean(X**2, axis=1, keepdims=True))
            
        # Replicate standard IARA Normalization.NORM_L2
        min_vals = X.min(axis=1, keepdims=True)
        max_vals = X.max(axis=1, keepdims=True)
        range_vals = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
        X_minmax = (X - min_vals) / range_vals
        l2_norms = np.linalg.norm(X_minmax, ord=2, axis=1, keepdims=True)
        l2_norms = np.where(l2_norms == 0, 1.0, l2_norms)
        X_norm = X_minmax / l2_norms
        
        return X_norm, rms

    def fit(self, samples: torch.Tensor, targets: torch.Tensor) -> None:
        X = samples.view(samples.size(0), -1).cpu().numpy()
        y = targets.cpu().numpy().astype(int)
        
        n_samples = X.shape[0]
        
        if self.rms:
            X_norm, rms = self._normalize_and_extract_rms(X)
        else:
            X_norm = X
            
        if self.normalize:
            X_norm = self.scaler.fit_transform(X_norm)
        if self.pca:
            X_norm = self.pca_trans.fit_transform(X_norm)
            
        if self.nystroem_mode == 'random':
            if isinstance(self.gamma, str) and 'scale' in self.gamma:
                base_gamma = 1.0 / (X_norm.shape[1] * X_norm.var())
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
                gamma = 1.0 / X_norm.shape[1]
            else:
                gamma = self.gamma
                
            self.nystroem.set_params(gamma=gamma)
            
        if self.nystroem_mode == 'stratified_kmeans':
            X_transformed = self.nystroem.fit_transform(X_norm, y)
        else:
            X_transformed = self.nystroem.fit_transform(X_norm)
        
        if self.rms:
            rms_scaled = self.rms_scaler.fit_transform(rms) * self.rms_scale
            X_final = np.hstack([X_transformed, rms_scaled])
        else:
            X_final = X_transformed
            
        alpha = 1.0 / (self.C * n_samples)
        self.sgd.set_params(alpha=alpha)
        self.sgd.fit(X_final, y)
        self.is_fitted = True

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling forward().")
            
        X = data.view(data.size(0), -1).cpu().numpy()
        
        if self.rms:
            X_norm, rms = self._normalize_and_extract_rms(X)
        else:
            X_norm = X
            
        if self.normalize:
            X_norm = self.scaler.transform(X_norm)
        if self.pca:
            X_norm = self.pca_trans.transform(X_norm)
            
        X_transformed = self.nystroem.transform(X_norm)
        
        if self.rms:
            rms_scaled = self.rms_scaler.transform(rms) * self.rms_scale
            X_final = np.hstack([X_transformed, rms_scaled])
        else:
            X_final = X_transformed
            
        scores = self.sgd.decision_function(X_final)
        biases = getattr(self, 'biases', None)
        if biases is not None:
            if len(scores.shape) == 1 or scores.shape[1] == 1:
                bias_val = biases[1] - biases[0]
                if len(scores.shape) == 1:
                    scores += bias_val
                    predictions = (scores > 0).astype(int)
                else:
                    scores[:, 0] += bias_val
                    predictions = (scores[:, 0] > 0).astype(int)
            else:
                for i, cls in enumerate(self.sgd.classes_):
                    if cls < len(biases):
                        scores[:, i] += biases[cls]
                predictions = np.argmax(scores, axis=1)
        else:
            predictions = self.sgd.predict(X_final)
            
        return torch.tensor(predictions, dtype=torch.long)


class SVMNystroemLocalTrainer(iara_trn.BaseTrainer):
    """Trainer class for SVMNystroemLocal compatible with Manager framework."""
    def __init__(self,
                 training_strategy: iara_trn.ModelTrainingStrategy,
                 trainer_id: str,
                 n_targets: int,
                 n_components: int = 300,
                 gamma: typing.Union[str, float] = 'scale',
                 C: float = 1.0,
                 normalize: bool = False,
                 pca: bool = False,
                 n_pca_components: int = 64,
                 penalty: str = 'l2',
                 l1_ratio: float = 0.15,
                 biases: typing.Optional[typing.List[float]] = None,
                 class_weight: typing.Union[str, dict, None] = 'balanced',
                 rms: bool = False,
                 rms_scale: float = 1.0,
                 rms_mode: str = 'log',
                 nystroem_mode: str = 'random',
                 nystroem_stratify_strategy: str = 'proportional',
                 nystroem_component_weights: typing.Optional[typing.Dict[int, float]] = None) -> None:
        super().__init__(training_strategy, trainer_id, n_targets)
        self.n_components = n_components
        self.gamma = gamma
        self.C = C
        self.normalize = normalize
        self.pca = pca
        self.n_pca_components = n_pca_components
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.biases = biases
        self.class_weight = class_weight
        self.rms = rms
        self.rms_scale = rms_scale
        self.rms_mode = rms_mode
        self.nystroem_mode = nystroem_mode
        self.nystroem_stratify_strategy = nystroem_stratify_strategy
        self.nystroem_component_weights = nystroem_component_weights

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
                
            model = SVMNystroemLocal(
                n_components=self.n_components,
                gamma=self.gamma,
                C=self.C,
                n_targets=self.n_targets,
                normalize=self.normalize,
                pca=self.pca,
                n_pca_components=self.n_pca_components,
                penalty=self.penalty,
                l1_ratio=self.l1_ratio,
                biases=self.biases,
                class_weight=self.class_weight,
                rms=self.rms,
                rms_scale=self.rms_scale,
                rms_mode=self.rms_mode,
                nystroem_mode=self.nystroem_mode,
                nystroem_stratify_strategy=self.nystroem_stratify_strategy,
                nystroem_component_weights=self.nystroem_component_weights
            )
            
            targets = trn_dataset.get_targets()
            if target_id is not None:
                targets = torch.where(targets == target_id,
                                       torch.tensor(1.0),
                                       torch.tensor(0.0))
                                       
            model.fit(samples=samples, targets=targets)
            model.save(model_filename)


class SVMNystroemCascaded(iara_model.BaseModel):
    """Wrapper that runs a multiclass SVM model and then a binary specialist SVM model.
    
    If the multiclass model predicts class 1 (MEDIUM), the specialist model re-evaluates
    the sample and can keep it as class 1 (MEDIUM) or reclassify it as class 0 (SMALL).
    """
    def __init__(self, multiclass_model: iara_model.BaseModel, specialist_model: iara_model.BaseModel, cascade_bias_small: float = 0.0):
        super().__init__()
        self.multiclass_model = multiclass_model
        self.specialist_model = specialist_model
        self.cascade_bias_small = cascade_bias_small
        self.is_fitted = True

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # 1. Get predictions from the multiclass model
        preds = self.multiclass_model.forward(data)
        
        # 2. Re-evaluate windows predicted as MEDIUM (1) using the specialist model
        medium_mask = (preds == 1)
        if medium_mask.any():
            data_med = data[medium_mask]
            
            # Temporarily inject cascade_bias_small to prioritize SMALL (0) over MEDIUM (1)
            # Binary classification in SVMNystroemLocal uses biases[1] - biases[0].
            # To prioritize class 0, we want bias_val = bias[1] - bias[0] to be negative, so we set biases = [cascade_bias_small, 0.0]
            if self.cascade_bias_small != 0.0:
                self.specialist_model.biases = [self.cascade_bias_small, 0.0]
            else:
                self.specialist_model.biases = None
                
            # Specialist model outputs 0 (SMALL) or 1 (MEDIUM)
            spec_preds = self.specialist_model.forward(data_med)
            
            # Put the specialist predictions back into the original preds tensor
            preds[medium_mask] = spec_preds
            
        return preds


class SVMNystroemCascadedTrainer(iara_trn.BaseTrainer):
    """Trainer class for SVMNystroemCascaded that trains the two stages of the cascade."""
    def __init__(self,
                 training_strategy: iara_trn.ModelTrainingStrategy,
                 trainer_id: str,
                 n_targets: int,
                 n_components: int = 4000,
                 n_components_specialist: int = 1000,
                 gamma: typing.Union[str, float] = 'scale',
                 C: float = 1.0,
                 normalize: bool = False,
                 pca: bool = False,
                 n_pca_components: int = 64,
                 penalty: str = 'l2',
                 l1_ratio: float = 0.15,
                 biases: typing.Optional[typing.List[float]] = None,
                 class_weight: typing.Union[str, dict, None] = 'balanced',
                 nystroem_mode: str = 'random',
                 nystroem_stratify_strategy: str = 'proportional',
                 nystroem_component_weights: typing.Optional[typing.Dict[int, float]] = None,
                 cascade_bias_small: float = 0.0) -> None:
        super().__init__(training_strategy, trainer_id, n_targets)
        self.n_components = n_components
        self.n_components_specialist = n_components_specialist
        self.gamma = gamma
        self.C = C
        self.normalize = normalize
        self.pca = pca
        self.n_pca_components = n_pca_components
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.biases = biases
        self.class_weight = class_weight
        self.nystroem_mode = nystroem_mode
        self.nystroem_stratify_strategy = nystroem_stratify_strategy
        self.nystroem_component_weights = nystroem_component_weights
        self.cascade_bias_small = cascade_bias_small

    def output_filename(self,
                        model_base_dir: str,
                        target_id: typing.Optional[int] = None,
                        complement: str = None,
                        extention: str = 'pkl') -> str:
        sufix = self.training_strategy.to_str(target_id=target_id)
        if complement is not None:
            sufix = f"{sufix}_{complement}"
            
        if extention == 'csv' and getattr(self, 'cascade_bias_small', 0.0) != 0.0:
            sufix = f"{sufix}_cbs{self.cascade_bias_small:.4f}".rstrip('0').rstrip('.')
            
        return os.path.join(model_base_dir, f'{str(self.trainer_id)}_{sufix}.{extention}')

    def load(self, model_base_dir: str) -> iara_model.BaseModel:
        filename = self.output_filename(model_base_dir=model_base_dir)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The model file '{filename}' does not exist. Ensure that the model is trained before evaluating.")
        model = iara_model.BaseModel.load(filename)
        model.cascade_bias_small = self.cascade_bias_small
        return model

    def fit(self,
            model_base_dir: str,
            trn_dataset: iara_dataset.BaseDataset,
            val_dataset: iara_dataset.BaseDataset) -> None:
        if self.is_trained(model_base_dir=model_base_dir):
            return
            
        os.makedirs(model_base_dir, exist_ok=True)
        
        # Cascaded training only makes sense in MULTICLASS mode
        if self.training_strategy != iara_trn.ModelTrainingStrategy.MULTICLASS:
            raise ValueError("SVMNystroemCascadedTrainer is only supported for MULTICLASS training strategy.")
            
        samples = trn_dataset.get_samples()
        if samples is None:
            raise ValueError("Training dataset without data")
        targets = trn_dataset.get_targets()
        
        # 1. Fit multiclass model (first stage)
        print(" -> Fitting first-stage multiclass model...")
        multiclass_model = SVMNystroemLocal(
            n_components=self.n_components,
            gamma=self.gamma,
            C=self.C,
            n_targets=self.n_targets,
            normalize=self.normalize,
            pca=self.pca,
            n_pca_components=self.n_pca_components,
            penalty=self.penalty,
            l1_ratio=self.l1_ratio,
            biases=self.biases,
            class_weight=self.class_weight,
            rms=False,
            nystroem_mode=self.nystroem_mode,
            nystroem_stratify_strategy=self.nystroem_stratify_strategy,
            nystroem_component_weights=self.nystroem_component_weights
        )
        multiclass_model.fit(samples=samples, targets=targets)
        
        # 2. Filter dataset for SMALL (0) and MEDIUM (1) classes to fit binary specialist (second stage)
        print(" -> Filtering training samples for second-stage binary specialist...")
        y = targets.cpu().numpy().astype(int)
        binary_mask = (y == 0) | (y == 1)
        samples_bin = samples[binary_mask]
        targets_bin = targets[binary_mask]
        
        print(f" -> Fitting second-stage binary specialist (SMALL vs MEDIUM, samples={len(samples_bin)})...")
        # For the binary model, n_targets=2
        # Filter class weights dict if it's custom
        if isinstance(self.class_weight, dict):
            specialist_class_weight = {k: v for k, v in self.class_weight.items() if k in (0, 1)}
        else:
            specialist_class_weight = self.class_weight
            
        specialist_model = SVMNystroemLocal(
            n_components=self.n_components_specialist,
            gamma=self.gamma,
            C=self.C,
            n_targets=2,
            normalize=self.normalize,
            pca=self.pca,
            n_pca_components=self.n_pca_components,
            penalty=self.penalty,
            l1_ratio=self.l1_ratio,
            biases=None,
            class_weight=specialist_class_weight,
            rms=False,
            nystroem_mode=self.nystroem_mode,
            nystroem_stratify_strategy=self.nystroem_stratify_strategy,
            nystroem_component_weights=None
        )
        specialist_model.fit(samples_bin, targets_bin)
        
        # 3. Create cascaded model and save
        cascaded_model = SVMNystroemCascaded(multiclass_model, specialist_model, self.cascade_bias_small)
        model_filename = self.output_filename(model_base_dir=model_base_dir)
        cascaded_model.save(model_filename)
        print(f" -> Cascaded model saved successfully to: {model_filename}")


class SVMNystroemHybridCascaded(iara_model.BaseModel):
    """Wrapper that runs a multiclass SVM model on MEL features and then uses specialist
    SVM models on LOFAR features to resolve confusions:
    - If predicted class is MEDIUM (1), specialist_model_medium resolves SMALL vs MEDIUM.
    - If predicted class is LARGE (2), specialist_model_large resolves SMALL vs LARGE.
    Both specialists can be the same model (backward compatible) or different ones.
    """
    def __init__(self,
                 multiclass_model: iara_model.BaseModel,
                 specialist_model_medium: iara_model.BaseModel,
                 cascade_bias_small: float = 0.0,
                 cascade_threshold: float = 0.5,
                 specialist_model_large: typing.Optional[iara_model.BaseModel] = None,
                 large_specialist_runner_up_only: bool = False,
                 multiclass_is_lofar: bool = False,
                 medium_only_routing: bool = False,
                 disable_sm_routing: bool = False):
        super().__init__()
        self.multiclass_model = multiclass_model
        self.specialist_model_medium = specialist_model_medium
        # None means: trust the multiclass prediction for LARGE (no specialist correction)
        self.specialist_model_large = specialist_model_large
        self.cascade_bias_small = cascade_bias_small
        self.cascade_threshold = cascade_threshold
        # When True, SL specialist is only called if runner-up class is SMALL (not MEDIUM/BG)
        self.large_specialist_runner_up_only = large_specialist_runner_up_only
        # When True, stage-1 multiclass uses LOFAR features (data[:, 256:]) instead of MEL
        self.multiclass_is_lofar = multiclass_is_lofar
        # When True, SM specialist is called only for MEDIUM predictions (not SMALL+MEDIUM)
        self.medium_only_routing = medium_only_routing
        # When True, SM specialist is never called (only SL specialist is active)
        self.disable_sm_routing = disable_sm_routing
        self.is_fitted = True

    def _multiclass_decision_scores(self, data: torch.Tensor) -> np.ndarray:
        """Return raw SGD decision scores (n_samples, n_classes) from the multiclass model."""
        X = data.view(data.size(0), -1).cpu().numpy()
        model = self.multiclass_model
        if hasattr(model, 'n_mel_features'):
            X_mel = X[:, :model.n_mel_features]
            X_lofar = X[:, model.n_mel_features:]
            X_lofar_pca = model.pca_trans.transform(X_lofar)
            X = np.hstack([X_mel, X_lofar_pca])
            if model.normalize and model.scaler is not None:
                X = model.scaler.transform(X)
        elif getattr(self, 'multiclass_is_lofar', False):
            X = X[:, 256:]  # LOFAR slice from hybrid input
            if getattr(model, 'normalize', False) and getattr(model, 'scaler', None) is not None:
                X = model.scaler.transform(X)
            if getattr(model, 'pca', False) and getattr(model, 'pca_trans', None) is not None:
                X = model.pca_trans.transform(X)
        else:
            X = X[:, :256]
            if getattr(model, 'normalize', False) and getattr(model, 'scaler', None) is not None:
                X = model.scaler.transform(X)
            if getattr(model, 'pca', False) and getattr(model, 'pca_trans', None) is not None:
                X = model.pca_trans.transform(X)
        X_nys = model.nystroem.transform(X)
        return model.sgd.decision_function(X_nys)  # (n_samples, n_classes)

    def _evaluate_specialist(self, data: torch.Tensor, model: iara_model.BaseModel, allowed_classes: list, bias: float, threshold: float = 0.5, fallback_class = 1) -> torch.Tensor:
        # fallback_class: int (fixed class) or np.ndarray (per-sample fallback, e.g. original multiclass preds)
        X = data.view(data.size(0), -1).cpu().numpy()

        if hasattr(model, 'n_mel_features'):
            # Hybrid specialist (SVMNystroemHybrid): data is MEL+LOFAR
            X_mel = X[:, :model.n_mel_features]
            X_lofar = X[:, model.n_mel_features:]
            X_lofar_pca = model.pca_trans.transform(X_lofar)
            X = np.hstack([X_mel, X_lofar_pca])
            if model.normalize and model.scaler is not None:
                X = model.scaler.transform(X)
        else:
            # LOFAR-only specialist (SVMNystroemLocal): data is LOFAR only
            if getattr(model, 'normalize', False) and getattr(model, 'scaler', None) is not None:
                X = model.scaler.transform(X)
            if getattr(model, 'pca', False) and getattr(model, 'pca_trans', None) is not None:
                X = model.pca_trans.transform(X)

        # Apply Nystroem
        X_transformed = model.nystroem.transform(X)

        # Apply RMS if the model has it (LOFAR-only specialist only)
        if getattr(model, 'rms', False):
            _, rms = model._normalize_and_extract_rms(data.view(data.size(0), -1).cpu().numpy())
            rms_scaled = model.rms_scaler.transform(rms) * model.rms_scale
            X_final = np.hstack([X_transformed, rms_scaled])
        else:
            X_final = X_transformed

        # Get decision function scores
        scores = model.sgd.decision_function(X_final)

        if len(scores.shape) == 1 or scores.shape[1] == 1:
            # Binary specialist: score > 0 → allowed_classes[1], else → allowed_classes[0]
            bias_val = bias if 0 in allowed_classes else 0.0
            if len(scores.shape) == 1:
                scores_binary = scores + bias_val
            else:
                scores_binary = scores[:, 0] + bias_val

            probs = 1.0 / (1.0 + np.exp(-scores_binary))
            conf = np.maximum(probs, 1.0 - probs)

            predictions_spec = (scores_binary > 0).astype(int)
            predictions_spec = np.where(predictions_spec == 0, allowed_classes[0], allowed_classes[1])

            predictions = np.where(conf >= threshold, predictions_spec, fallback_class)
        else:
            # Multiclass specialist: pick scores for the two relevant classes
            c0, c1 = allowed_classes[0], allowed_classes[1]
            bias_c0 = bias if c0 == 0 else 0.0
            bias_c1 = bias if c1 == 0 else 0.0

            score_c0 = scores[:, c0] + bias_c0
            score_c1 = scores[:, c1] + bias_c1

            max_score = np.maximum(score_c0, score_c1)
            exp_c0 = np.exp(score_c0 - max_score)
            exp_c1 = np.exp(score_c1 - max_score)
            p_c0 = exp_c0 / (exp_c0 + exp_c1)
            p_c1 = exp_c1 / (exp_c0 + exp_c1)
            conf = np.maximum(p_c0, p_c1)

            predictions_spec = np.where(score_c0 > score_c1, c0, c1)

            predictions = np.where(conf >= threshold, predictions_spec, fallback_class)

        return torch.tensor(predictions, dtype=torch.long)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data has shape (n_samples, 1280): 0-255 MEL, 256-1279 LOFAR
        if hasattr(self.multiclass_model, 'n_mel_features'):
            preds = self.multiclass_model.forward(data)
        elif getattr(self, 'multiclass_is_lofar', False):
            preds = self.multiclass_model.forward(data[:, 256:])
        else:
            preds = self.multiclass_model.forward(data[:, :256])

        # SM specialist routing
        if getattr(self, 'disable_sm_routing', False):
            medium_mask = torch.zeros(preds.shape, dtype=torch.bool)
        elif getattr(self, 'medium_only_routing', False):
            medium_mask = (preds == 1)
        else:
            medium_mask = (preds == 0) | (preds == 1)
        if medium_mask.any():
            # When specialist confidence < threshold, fall back to the multiclass prediction
            # (instead of a fixed class) so uncertain windows preserve the first-stage verdict.
            original_preds_med = preds[medium_mask].cpu().numpy()
            spec_preds_med = self._evaluate_specialist(
                data[medium_mask],
                model=self.specialist_model_medium,
                allowed_classes=[0, 1],
                bias=self.cascade_bias_small,
                threshold=getattr(self, 'cascade_threshold', 0.5),
                fallback_class=original_preds_med
            )
            preds[medium_mask] = spec_preds_med

        # LARGE (2): specialist resolves SMALL vs LARGE (only if a dedicated specialist exists)
        large_mask = (preds == 2)
        if large_mask.any() and self.specialist_model_large is not None:
            if getattr(self, 'large_specialist_runner_up_only', False):
                # Only call SL specialist for windows where SMALL is the runner-up class.
                # For windows where MEDIUM (or BG) is runner-up, trust the multiclass LARGE.
                scores = self._multiclass_decision_scores(data[large_mask])
                # scores shape: (n_large_windows, n_classes); classes order: [0,1,2,3]
                small_is_runner_up = scores[:, 0] > scores[:, 1]  # SMALL score > MEDIUM score
                sl_submask = torch.tensor(small_is_runner_up, dtype=torch.bool)
                large_indices = torch.where(large_mask)[0]
                sl_indices = large_indices[sl_submask]
                if sl_indices.numel() > 0:
                    sl_data = data[sl_indices] if hasattr(self.specialist_model_large, 'n_mel_features') \
                        else data[sl_indices, 256:]
                    spec_preds_sl = self._evaluate_specialist(
                        sl_data,
                        model=self.specialist_model_large,
                        allowed_classes=[0, 2],
                        bias=self.cascade_bias_small,
                        threshold=getattr(self, 'cascade_threshold', 0.5),
                        fallback_class=2
                    )
                    preds[sl_indices] = spec_preds_sl
            else:
                # Always call SL specialist for all LARGE-predicted windows
                large_data = data[large_mask] if hasattr(self.specialist_model_large, 'n_mel_features') \
                    else data[large_mask, 256:]
                spec_preds_large = self._evaluate_specialist(
                    large_data,
                    model=self.specialist_model_large,
                    allowed_classes=[0, 2],
                    bias=self.cascade_bias_small,
                    threshold=getattr(self, 'cascade_threshold', 0.5),
                    fallback_class=2
                )
                preds[large_mask] = spec_preds_large

        return preds



class SVMNystroemHybridCascadedTrainer(iara_trn.BaseTrainer):
    """Trainer class for SVMNystroemHybridCascaded that trains the two stages of the hybrid cascade."""
    def __init__(self,
                 training_strategy: iara_trn.ModelTrainingStrategy,
                 trainer_id: str,
                 n_targets: int,
                 n_components: int = 4000,
                 n_components_specialist: int = 3000,
                 gamma: typing.Union[str, float] = 'scale',
                 C: float = 1.0,
                 normalize: bool = False,
                 pca_specialist: bool = True,
                 n_pca_components: int = 64,
                 penalty: str = 'l2',
                 l1_ratio: float = 0.15,
                 biases: typing.Optional[typing.List[float]] = None,
                 class_weight: typing.Union[str, dict, None] = 'balanced',
                 nystroem_mode: str = 'random',
                 nystroem_stratify_strategy: str = 'proportional',
                 nystroem_component_weights: typing.Optional[typing.Dict[int, float]] = None,
                 cascade_bias_small: float = 0.0,
                 cascade_threshold: float = 0.5,
                 pretrained_specialist_dir: typing.Optional[str] = None,
                 pretrained_large_specialist_dir: typing.Optional[str] = None,
                 pretrained_multiclass_dir: typing.Optional[str] = None,
                 pretrained_lofar_multiclass_dir: typing.Optional[str] = None,
                 large_specialist_runner_up_only: bool = False,
                 medium_only_routing: bool = False,
                 disable_sm_routing: bool = False) -> None:
        super().__init__(training_strategy, trainer_id, n_targets)
        self.n_components = n_components
        self.n_components_specialist = n_components_specialist
        self.gamma = gamma
        self.C = C
        self.normalize = normalize
        self.pca_specialist = pca_specialist
        self.n_pca_components = n_pca_components
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.biases = biases
        self.class_weight = class_weight
        self.nystroem_mode = nystroem_mode
        self.nystroem_stratify_strategy = nystroem_stratify_strategy
        self.nystroem_component_weights = nystroem_component_weights
        self.cascade_bias_small = cascade_bias_small
        self.cascade_threshold = cascade_threshold
        self.pretrained_specialist_dir = pretrained_specialist_dir
        self.pretrained_large_specialist_dir = pretrained_large_specialist_dir
        self.pretrained_multiclass_dir = pretrained_multiclass_dir
        self.pretrained_lofar_multiclass_dir = pretrained_lofar_multiclass_dir
        self.large_specialist_runner_up_only = large_specialist_runner_up_only
        self.medium_only_routing = medium_only_routing
        self.disable_sm_routing = disable_sm_routing

    def output_filename(self,
                        model_base_dir: str,
                        target_id: typing.Optional[int] = None,
                        complement: str = None,
                        extention: str = 'pkl') -> str:
        sufix = self.training_strategy.to_str(target_id=target_id)
        if complement is not None:
            sufix = f"{sufix}_{complement}"
            
        if extention == 'csv':
            if getattr(self, 'cascade_bias_small', 0.0) != 0.0:
                sufix = f"{sufix}_cbs{self.cascade_bias_small:.4f}".rstrip('0').rstrip('.')
            if getattr(self, 'cascade_threshold', 0.5) != 0.5:
                sufix = f"{sufix}_ct{self.cascade_threshold:.2f}".rstrip('0').rstrip('.')
            
        return os.path.join(model_base_dir, f'{str(self.trainer_id)}_{sufix}.{extention}')

    def load(self, model_base_dir: str) -> iara_model.BaseModel:
        filename = self.output_filename(model_base_dir=model_base_dir)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The model file '{filename}' does not exist. Ensure that the model is trained before evaluating.")
        model = iara_model.BaseModel.load(filename)
        model.cascade_bias_small = self.cascade_bias_small
        model.cascade_threshold = self.cascade_threshold
        model.disable_sm_routing = self.disable_sm_routing
        model.medium_only_routing = self.medium_only_routing
        model.multiclass_is_lofar = bool(getattr(self, 'pretrained_lofar_multiclass_dir', None))
        return model

    def fit(self,
            model_base_dir: str,
            trn_dataset: iara_dataset.BaseDataset,
            val_dataset: iara_dataset.BaseDataset) -> None:
        if self.is_trained(model_base_dir=model_base_dir):
            return
            
        os.makedirs(model_base_dir, exist_ok=True)
        
        # Cascaded training only makes sense in MULTICLASS mode
        if self.training_strategy != iara_trn.ModelTrainingStrategy.MULTICLASS:
            raise ValueError("SVMNystroemHybridCascadedTrainer is only supported for MULTICLASS training strategy.")
            
        samples = trn_dataset.get_samples()
        if samples is None:
            raise ValueError("Training dataset without data")
        targets = trn_dataset.get_targets()
        
        # Split features into MEL (first 256) and LOFAR (remaining 1024)
        samples_mel = samples[:, :256]
        samples_lofar = samples[:, 256:]

        # Extract fold index from model_base_dir (needed for loading any pre-trained model)
        import re
        match = re.search(r"fold_(\d+)", model_base_dir)
        fold_idx = int(match.group(1)) if match else None

        # 1. Fit or load multiclass model (first stage)
        multiclass_is_lofar = False
        if self.pretrained_lofar_multiclass_dir and fold_idx is not None:
            lmc_model_dir = os.path.join(self.pretrained_lofar_multiclass_dir, "model", f"fold_{fold_idx}")
            if not os.path.exists(lmc_model_dir):
                lmc_model_dir = os.path.join(self.pretrained_lofar_multiclass_dir, f"fold_{fold_idx}")
            pkl_files = [f for f in os.listdir(lmc_model_dir) if f.endswith(".pkl")]
            multiclass_pkls = [f for f in pkl_files if "multiclass" in f]
            lmc_pkl = os.path.join(lmc_model_dir, multiclass_pkls[0] if multiclass_pkls else pkl_files[0])
            print(f" -> Loading pre-trained LOFAR multiclass model from: {lmc_pkl}")
            multiclass_model = iara_model.BaseModel.load(lmc_pkl)
            multiclass_is_lofar = True
        elif self.pretrained_multiclass_dir and fold_idx is not None:
            mc_model_dir = os.path.join(self.pretrained_multiclass_dir, "model", f"fold_{fold_idx}")
            if not os.path.exists(mc_model_dir):
                mc_model_dir = os.path.join(self.pretrained_multiclass_dir, f"fold_{fold_idx}")
            pkl_files = [f for f in os.listdir(mc_model_dir) if f.endswith(".pkl")]
            multiclass_pkls = [f for f in pkl_files if "multiclass" in f]
            mc_pkl = os.path.join(mc_model_dir, multiclass_pkls[0] if multiclass_pkls else pkl_files[0])
            print(f" -> Loading pre-trained hybrid multiclass model from: {mc_pkl}")
            multiclass_model = iara_model.BaseModel.load(mc_pkl)
        else:
            print(" -> Fitting first-stage multiclass model on MEL features...")
            multiclass_model = SVMNystroemLocal(
                n_components=self.n_components,
                gamma=self.gamma,
                C=self.C,
                n_targets=self.n_targets,
                normalize=self.normalize,
                pca=False,
                penalty=self.penalty,
                l1_ratio=self.l1_ratio,
                biases=self.biases,
                class_weight=self.class_weight,
                rms=False,
                nystroem_mode=self.nystroem_mode,
                nystroem_stratify_strategy=self.nystroem_stratify_strategy,
                nystroem_component_weights=self.nystroem_component_weights
            )
            multiclass_model.fit(samples=samples_mel, targets=targets)
        
        def _load_pretrained_specialist(specialist_dir, fold_idx, label):
            model_dir = os.path.join(specialist_dir, "model", f"fold_{fold_idx}")
            if not os.path.exists(model_dir):
                model_dir = os.path.join(specialist_dir, f"fold_{fold_idx}")
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Pre-trained {label} specialist folder not found: {model_dir}")
            pkl_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
            if len(pkl_files) == 1:
                pkl_path = os.path.join(model_dir, pkl_files[0])
            elif len(pkl_files) > 1:
                multiclass_files = [f for f in pkl_files if "multiclass" in f]
                pkl_path = os.path.join(model_dir, multiclass_files[0] if multiclass_files else pkl_files[0])
            else:
                raise FileNotFoundError(f"No .pkl model found in pre-trained {label} folder: {model_dir}")
            print(f" -> Loading pre-trained {label} specialist from: {pkl_path}")
            return iara_model.BaseModel.load(pkl_path)

        # 2. Load or train the MEDIUM specialist (SMALL vs MEDIUM binary)
        if self.pretrained_specialist_dir and fold_idx is not None:
            specialist_model_medium = _load_pretrained_specialist(
                self.pretrained_specialist_dir, fold_idx, "MEDIUM"
            )
        else:
            print(" -> Filtering training samples for MEDIUM binary specialist (SMALL vs MEDIUM)...")
            y = targets.cpu().numpy().astype(int)
            binary_mask = (y == 0) | (y == 1)
            samples_lofar_bin = samples_lofar[binary_mask]
            targets_bin = targets[binary_mask]

            print(f" -> Fitting MEDIUM binary specialist (samples={len(samples_lofar_bin)})...")
            if isinstance(self.class_weight, dict):
                specialist_class_weight = {k: v for k, v in self.class_weight.items() if k in (0, 1)}
            else:
                specialist_class_weight = self.class_weight

            specialist_model_medium = SVMNystroemLocal(
                n_components=self.n_components_specialist,
                gamma=self.gamma,
                C=self.C,
                n_targets=2,
                normalize=self.normalize,
                pca=self.pca_specialist,
                n_pca_components=self.n_pca_components,
                penalty=self.penalty,
                l1_ratio=self.l1_ratio,
                biases=None,
                class_weight=specialist_class_weight,
                rms=False,
                nystroem_mode=self.nystroem_mode,
                nystroem_stratify_strategy=self.nystroem_stratify_strategy,
                nystroem_component_weights=None
            )
            specialist_model_medium.fit(samples_lofar_bin, targets_bin)

        # 3. Load or reuse the LARGE specialist (SMALL vs LARGE)
        if self.pretrained_large_specialist_dir and fold_idx is not None:
            specialist_model_large = _load_pretrained_specialist(
                self.pretrained_large_specialist_dir, fold_idx, "LARGE"
            )
        else:
            # No dedicated LARGE specialist: reuse the MEDIUM specialist (same as before)
            specialist_model_large = None

        # 4. Create hybrid cascaded model and save
        cascaded_model = SVMNystroemHybridCascaded(
            multiclass_model,
            specialist_model_medium,
            self.cascade_bias_small,
            self.cascade_threshold,
            specialist_model_large=specialist_model_large,
            large_specialist_runner_up_only=self.large_specialist_runner_up_only,
            multiclass_is_lofar=multiclass_is_lofar,
            medium_only_routing=self.medium_only_routing,
            disable_sm_routing=self.disable_sm_routing
        )
        model_filename = self.output_filename(model_base_dir=model_base_dir)
        cascaded_model.save(model_filename)
        print(f" -> Hybrid cascaded model saved successfully to: {model_filename}")


class SVMNystroemWithRMS(iara_model.BaseModel):
    """Approximate RBF-SVM that extracts and injects raw RMS energy as an auxiliary feature."""
    def __init__(self,
                 n_components: int = 300,
                 gamma: typing.Union[str, float] = 'scale',
                 C: float = 1.0,
                 n_targets: int = 4,
                 normalize: bool = False,
                 pca: bool = False,
                 n_pca_components: int = 64,
                 penalty: str = 'l2',
                 l1_ratio: float = 0.15,
                 biases: typing.Optional[typing.List[float]] = None,
                 class_weight: typing.Union[str, dict, None] = 'balanced',
                 rms_scale: float = 1.0,
                 random_state: int = 42):
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.C = C
        self.n_targets = n_targets
        self.normalize = normalize
        self.pca = pca
        self.n_pca_components = n_pca_components
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.biases = biases
        self.class_weight = class_weight
        self.rms_scale = rms_scale
        self.random_state = random_state
        
        self.scaler = StandardScaler() if normalize else None
        self.pca_trans = PCA(n_components=n_pca_components, random_state=random_state) if pca else None
        self.rms_scaler = StandardScaler()
        
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
            class_weight=class_weight,
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
            n_jobs=1
        )
        
        self.is_fitted = False

    def _normalize_and_extract_rms(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        # Compute RMS of each raw frame (each row)
        rms = np.sqrt(np.mean(X**2, axis=1, keepdims=True))
        
        # Replicate standard IARA Normalization.NORM_L2
        min_vals = X.min(axis=1, keepdims=True)
        max_vals = X.max(axis=1, keepdims=True)
        range_vals = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
        X_minmax = (X - min_vals) / range_vals
        l2_norms = np.linalg.norm(X_minmax, ord=2, axis=1, keepdims=True)
        l2_norms = np.where(l2_norms == 0, 1.0, l2_norms)
        X_norm = X_minmax / l2_norms
        
        return X_norm, rms

    def fit(self, samples: torch.Tensor, targets: torch.Tensor) -> None:
        X = samples.view(samples.size(0), -1).cpu().numpy()
        y = targets.cpu().numpy().astype(int)
        
        n_samples = X.shape[0]
        
        # Extract RMS and apply L2 normalization locally
        X_norm, rms = self._normalize_and_extract_rms(X)
        
        # Apply standard preprocessing on normalized spectra
        if self.normalize:
            X_norm = self.scaler.fit_transform(X_norm)
        if self.pca:
            X_norm = self.pca_trans.fit_transform(X_norm)
            
        # Calculate gamma on RBF inputs
        if isinstance(self.gamma, str) and 'scale' in self.gamma:
            base_gamma = 1.0 / (X_norm.shape[1] * X_norm.var())
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
            gamma = 1.0 / X_norm.shape[1]
        else:
            gamma = self.gamma
            
        self.nystroem.set_params(gamma=gamma)
        
        # Transform using Nyström
        X_transformed = self.nystroem.fit_transform(X_norm)
        
        # Scale the RMS feature
        rms_scaled = self.rms_scaler.fit_transform(rms) * self.rms_scale
        
        # Concatenate RMS scaled feature to the end of Nyström features
        X_final = np.hstack([X_transformed, rms_scaled])
        
        # Train SGD linear SVM on the (m + 1) feature space
        alpha = 1.0 / (self.C * n_samples)
        self.sgd.set_params(alpha=alpha)
        self.sgd.fit(X_final, y)
        self.is_fitted = True

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling forward().")
            
        X = data.view(data.size(0), -1).cpu().numpy()
        
        X_norm, rms = self._normalize_and_extract_rms(X)
        
        if self.normalize:
            X_norm = self.scaler.transform(X_norm)
        if self.pca:
            X_norm = self.pca_trans.transform(X_norm)
            
        X_transformed = self.nystroem.transform(X_norm)
        rms_scaled = self.rms_scaler.transform(rms) * self.rms_scale
        
        X_final = np.hstack([X_transformed, rms_scaled])
        
        scores = self.sgd.decision_function(X_final)
        biases = getattr(self, 'biases', None)
        if biases is not None:
            if len(scores.shape) == 1 or scores.shape[1] == 1:
                bias_val = biases[1] - biases[0]
                if len(scores.shape) == 1:
                    scores += bias_val
                    predictions = (scores > 0).astype(int)
                else:
                    scores[:, 0] += bias_val
                    predictions = (scores[:, 0] > 0).astype(int)
            else:
                for i, cls in enumerate(self.sgd.classes_):
                    if cls < len(biases):
                        scores[:, i] += biases[cls]
                predictions = np.argmax(scores, axis=1)
        else:
            predictions = self.sgd.predict(X_final)
            
        return torch.tensor(predictions, dtype=torch.long)


class SVMNystroemWithRMSTrainer(iara_trn.BaseTrainer):
    """Trainer class for SVMNystroemWithRMS compatible with Manager framework."""
    def __init__(self,
                 training_strategy: iara_trn.ModelTrainingStrategy,
                 trainer_id: str,
                 n_targets: int,
                 n_components: int = 300,
                 gamma: typing.Union[str, float] = 'scale',
                 C: float = 1.0,
                 normalize: bool = False,
                 pca: bool = False,
                 n_pca_components: int = 64,
                 penalty: str = 'l2',
                 l1_ratio: float = 0.15,
                 biases: typing.Optional[typing.List[float]] = None,
                 class_weight: typing.Union[str, dict, None] = 'balanced',
                 rms_scale: float = 1.0) -> None:
        super().__init__(training_strategy, trainer_id, n_targets)
        self.n_components = n_components
        self.gamma = gamma
        self.C = C
        self.normalize = normalize
        self.pca = pca
        self.n_pca_components = n_pca_components
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.biases = biases
        self.class_weight = class_weight
        self.rms_scale = rms_scale

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
                
            model = SVMNystroemWithRMS(
                n_components=self.n_components,
                gamma=self.gamma,
                C=self.C,
                n_targets=self.n_targets,
                normalize=self.normalize,
                pca=self.pca,
                n_pca_components=self.n_pca_components,
                penalty=self.penalty,
                l1_ratio=self.l1_ratio,
                biases=self.biases,
                class_weight=self.class_weight,
                rms_scale=self.rms_scale
            )
            
            targets = trn_dataset.get_targets()
            if target_id is not None:
                targets = torch.where(targets == target_id,
                                       torch.tensor(1.0),
                                       torch.tensor(0.0))
                                       
            model.fit(samples=samples, targets=targets)
            model.save(model_filename)


def main(folds: typing.List[int], n_components: int = 300, analysis_name: str = 'log_melgram', C: float = 1.0, gamma: typing.Union[str, float] = 'scale', normalize: bool = False, pca: bool = False, n_pca_components: int = 64, penalty: str = 'l2', l1_ratio: float = 0.15, biases: typing.Optional[typing.List[float]] = None, class_weight: typing.Union[str, dict, None] = 'balanced', rms: bool = False, rms_scale: float = 1.0, rms_mode: str = 'log', nystroem_mode: str = 'random', nystroem_stratify_strategy: str = 'proportional', nystroem_component_weights: typing.Optional[typing.Dict[int, float]] = None, cascade: bool = False, cascade_components: int = 1000, cascade_bias_small: float = 0.0, cascade_threshold: float = 0.5, pretrained_specialist_dir: typing.Optional[str] = None, pretrained_large_specialist_dir: typing.Optional[str] = None, pretrained_multiclass_dir: typing.Optional[str] = None, pretrained_lofar_multiclass_dir: typing.Optional[str] = None, large_specialist_runner_up_only: bool = False, medium_only_routing: bool = False, disable_sm_routing: bool = False):

    output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/tests"
    directories = DEFAULT_DIRECTORIES

    grid = iara_metrics.GridCompiler()

    # Use individual windows as input (same as MLP baseline)
    # The by_audio evaluation will apply majority vote across windows of each file
    input_type = iara_dataset.InputType.Window()

    # Audio preprocessing pipeline
    norm_mode = iara_proc.Normalization.NONE if rms else iara_proc.Normalization.NORM_L2

    if analysis_name.lower() == 'hybrid':
        dp_mel = iara_manager.AudioFileProcessor(
            data_base_dir=directories.data_dir,
            data_processed_base_dir=directories.process_dir,
            normalization=norm_mode,
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
            normalization=norm_mode,
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
            normalization=norm_mode,
            analysis=analysis_enum,
            n_pts=1024,
            n_overlap=0,
            decimation_rate=3,
            n_mels=256,
            integration_interval=0.512
        )

    # Dynamic folder name includes C, gamma and preprocessors if non-default
    name_parts = [f'svm_nystroem_{n_components}', analysis_name]
    if cascade:
        if pretrained_specialist_dir:
            import re as _re
            _sm_base = os.path.basename(pretrained_specialist_dir.rstrip('/\\'))
            _sm_comp = _re.search(r'nystroem_(\d+)', _sm_base)
            _sm_pca  = _re.search(r'pca(\d+)', _sm_base)
            if _sm_comp and _sm_pca:
                spec_name = f'sm{_sm_comp.group(1)}pca{_sm_pca.group(1)}'
            else:
                spec_name = _sm_base
            name_parts.append(f'cascade_pretrained_{spec_name}_smallalso_fbmc')
            if pretrained_large_specialist_dir:
                _sl_base = os.path.basename(pretrained_large_specialist_dir.rstrip('/\\'))
                _sl_comp = _re.search(r'nystroem_(\d+)', _sl_base)
                _sl_pca  = _re.search(r'pca(\d+)', _sl_base)
                if _sl_comp and _sl_pca:
                    large_spec_name = f'sl{_sl_comp.group(1)}pca{_sl_pca.group(1)}'
                else:
                    large_spec_name = _sl_base
                name_parts.append(f'large_{large_spec_name}')
            if pretrained_lofar_multiclass_dir:
                lmc_base = os.path.basename(pretrained_lofar_multiclass_dir.rstrip('/\\'))
                lmc_comp = _re.search(r'nystroem_(\d+)', lmc_base)
                lmc_tag = 'lofarmc'
                if lmc_comp:
                    lmc_tag += lmc_comp.group(1)
                name_parts.append(lmc_tag)
            elif pretrained_multiclass_dir:
                # Use short tag: extract n_components and C from dir name to keep path short
                mc_base = os.path.basename(pretrained_multiclass_dir.rstrip('/\\'))
                mc_comp = _re.search(r'nystroem_(\d+)', mc_base)
                mc_c = _re.search(r'_C([\d.]+)', mc_base)
                mc_tag = 'hybmc'
                if mc_comp:
                    mc_tag += mc_comp.group(1)
                if mc_c:
                    mc_tag += f'C{mc_c.group(1)}'
                name_parts.append(mc_tag)
            if disable_sm_routing:
                name_parts.append('nosm')
            elif medium_only_routing:
                name_parts.append('smonly')
            if large_specialist_runner_up_only:
                name_parts.append('slro')
        else:
            name_parts.append(f'cascade_{cascade_components}')
    if rms:
        name_parts.append('rms')
        if rms_mode == 'linear':
            name_parts.append('linear')
        if rms_scale != 1.0:
            name_parts.append(f'scale{rms_scale}')
    if nystroem_mode != 'random':
        name_parts.append(nystroem_mode)
        if nystroem_mode == 'stratified_kmeans' and nystroem_stratify_strategy != 'proportional':
            name_parts.append(nystroem_stratify_strategy)
        if nystroem_component_weights:
            s_val = nystroem_component_weights.get(0, 1.0)
            m_val = nystroem_component_weights.get(1, 1.0)
            l_val = nystroem_component_weights.get(2, 1.0)
            b_val = nystroem_component_weights.get(3, 1.0)
            name_parts.append(f'ncw_S{s_val}_M{m_val}_L{l_val}_B{b_val}')
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
    
    if isinstance(class_weight, dict):
        s_val = class_weight.get(0, 1.0)
        m_val = class_weight.get(1, 1.0)
        l_val = class_weight.get(2, 1.0)
        b_val = class_weight.get(3, 1.0)
        name_parts.append(f'w_S{s_val}_M{m_val}_L{l_val}_B{b_val}')
    elif class_weight is None:
        name_parts.append('w_none')
        
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
    if cascade:
        if analysis_name.lower() == 'hybrid':
            trainer = SVMNystroemHybridCascadedTrainer(
                training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
                trainer_id=exp_name,
                n_targets=config.dataset.target.get_n_targets(),
                n_components=n_components,
                n_components_specialist=cascade_components,
                gamma=gamma,
                C=C,
                normalize=normalize,
                pca_specialist=True,
                n_pca_components=n_pca_components,
                penalty=penalty,
                l1_ratio=l1_ratio,
                biases=biases,
                class_weight=class_weight,
                nystroem_mode=nystroem_mode,
                nystroem_stratify_strategy=nystroem_stratify_strategy,
                nystroem_component_weights=nystroem_component_weights,
                cascade_bias_small=cascade_bias_small,
                cascade_threshold=cascade_threshold,
                pretrained_specialist_dir=pretrained_specialist_dir,
                pretrained_large_specialist_dir=pretrained_large_specialist_dir,
                pretrained_multiclass_dir=pretrained_multiclass_dir,
                pretrained_lofar_multiclass_dir=pretrained_lofar_multiclass_dir,
                large_specialist_runner_up_only=large_specialist_runner_up_only,
                medium_only_routing=medium_only_routing
            )
        else:
            trainer = SVMNystroemCascadedTrainer(
                training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
                trainer_id=exp_name,
                n_targets=config.dataset.target.get_n_targets(),
                n_components=n_components,
                n_components_specialist=cascade_components,
                gamma=gamma,
                C=C,
                normalize=normalize,
                pca=pca,
                n_pca_components=n_pca_components,
                penalty=penalty,
                l1_ratio=l1_ratio,
                biases=biases,
                class_weight=class_weight,
                nystroem_mode=nystroem_mode,
                nystroem_stratify_strategy=nystroem_stratify_strategy,
                nystroem_component_weights=nystroem_component_weights,
                cascade_bias_small=cascade_bias_small
            )
    elif rms or nystroem_mode != 'random' or rms_mode == 'linear':
        trainer = SVMNystroemLocalTrainer(
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
            biases=biases,
            class_weight=class_weight,
            rms=rms,
            rms_scale=rms_scale,
            rms_mode=rms_mode,
            nystroem_mode=nystroem_mode,
            nystroem_stratify_strategy=nystroem_stratify_strategy,
            nystroem_component_weights=nystroem_component_weights
        )
    elif analysis_name.lower() == 'hybrid':
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
            biases=biases,
            class_weight=class_weight
        )
    elif normalize or pca or penalty != 'l2':
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
            biases=biases,
            class_weight=class_weight
        )
    else:
        trainer = iara_trn.SVMNystroemTrainer(
            training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
            trainer_id=exp_name,
            n_targets=config.dataset.target.get_n_targets(),
            n_components=n_components,
            gamma=gamma,
            C=C,
            biases=biases,
            class_weight=class_weight
        )
    trainers.append(trainer)

    manager = iara_exp.Manager(config, *trainers)

    result_grid = manager.run(folds=folds, override=False)

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
        '--rms',
        action='store_true',
        help='Inject raw RMS energy as an auxiliary feature'
    )
    parser.add_argument(
        '--rms_scale',
        type=float,
        default=1.0,
        help='Scaling factor for the RMS feature to increase its influence. Default: 1.0'
    )
    parser.add_argument(
        '--rms_mode',
        type=str,
        default='log',
        choices=['log', 'linear'],
        help='RMS energy mode: log (traditional) or linear (relative physical energy). Default: log'
    )
    parser.add_argument(
        '--cascade',
        action='store_true',
        help='Enable two-stage cascaded SVM classifier (multiclass + specialist binary)'
    )
    parser.add_argument(
        '--cascade_components',
        type=int,
        default=1000,
        help='Number of components for the Nystroem specialist binary SVM. Default: 1000'
    )
    parser.add_argument(
        '--cascade_bias_small',
        type=float,
        default=0.0,
        help='Decision score bias offset to prioritize SMALL over MEDIUM in specialist. Default: 0.0'
    )
    parser.add_argument(
        '--cascade_threshold',
        type=float,
        default=0.5,
        help='Confidence threshold to trust specialist in cascade. Default: 0.5'
    )
    parser.add_argument(
        '--nystroem_mode',
        type=str,
        default='random',
        choices=['random', 'kmeans', 'stratified_kmeans'],
        help='Nystrom landmark selection mode: random, kmeans, or stratified_kmeans centroids. Default: random'
    )
    parser.add_argument(
        '--nystroem_stratify_strategy',
        type=str,
        default='proportional',
        choices=['proportional', 'balanced', 'weighted'],
        help='Allocation strategy for stratified_kmeans landmarks. Default: proportional'
    )
    parser.add_argument(
        '--nystroem_component_weights',
        type=str,
        default=None,
        help='Comma-separated component weights for SMALL,MEDIUM,LARGE,BACKGROUND when using weighted stratification. Example: 2,1,1,1'
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
    parser.add_argument(
        '--cw_type',
        type=str,
        default='balanced',
        choices=['balanced', 'custom', 'none'],
        help='Class weight mode: balanced, custom, or none. Default: balanced'
    )
    parser.add_argument(
        '--cw_small',
        type=float,
        default=1.0,
        help='Custom class weight for SMALL class. Only used when --cw_type=custom. Default: 1.0'
    )
    parser.add_argument(
        '--cw_medium',
        type=float,
        default=1.0,
        help='Custom class weight for MEDIUM class. Only used when --cw_type=custom. Default: 1.0'
    )
    parser.add_argument(
        '--cw_large',
        type=float,
        default=1.0,
        help='Custom class weight for LARGE class. Only used when --cw_type=custom. Default: 1.0'
    )
    parser.add_argument(
        '--cw_background',
        type=float,
        default=1.0,
        help='Custom class weight for BACKGROUND class. Only used when --cw_type=custom. Default: 1.0'
    )
    parser.add_argument(
        '--pretrained_specialist_dir',
        type=str,
        default=None,
        help='Path to pre-trained MEDIUM specialist experiment dir (used for SMALL vs MEDIUM disambiguation). If provided, the specialist will NOT be trained.'
    )
    parser.add_argument(
        '--pretrained_large_specialist_dir',
        type=str,
        default=None,
        help='Path to pre-trained LARGE specialist experiment dir (used for SMALL vs LARGE disambiguation). If not provided, the MEDIUM specialist is reused for LARGE.'
    )
    parser.add_argument(
        '--pretrained_multiclass_dir',
        type=str,
        default=None,
        help='Path to pre-trained hybrid multiclass experiment dir. If provided, the first-stage multiclass model is loaded from here instead of being trained on MEL features.'
    )
    parser.add_argument(
        '--pretrained_lofar_multiclass_dir',
        type=str,
        default=None,
        help='Path to pre-trained LOFAR multiclass experiment dir. If provided, it is used as stage-1 multiclass (LOFAR features); takes precedence over --pretrained_multiclass_dir.'
    )
    parser.add_argument(
        '--large_specialist_runner_up_only',
        action='store_true',
        default=False,
        help='Only call SL specialist when runner-up class for a LARGE-predicted window is SMALL. Windows where runner-up is MEDIUM or BG keep the LARGE prediction.'
    )
    parser.add_argument(
        '--medium_only_routing',
        action='store_true',
        default=False,
        help='Route only MEDIUM predictions through SM specialist (not SMALL+MEDIUM). SMALL predictions from stage-1 are kept as-is.'
    )
    parser.add_argument(
        '--disable_sm_routing',
        action='store_true',
        default=False,
        help='Disable SM specialist entirely. Only SL specialist is active (for LARGE predictions).'
    )

    args = parser.parse_args()

    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(1)))
    n_components = args.components
    analysis_type = args.analysis
    reg_c = args.reg_c
    normalize_enabled = args.normalize
    rms_enabled = args.rms
    rms_scale_val = args.rms_scale
    rms_mode_val = args.rms_mode
    nystroem_mode_val = args.nystroem_mode
    nystroem_stratify_strategy_val = args.nystroem_stratify_strategy
    nystroem_component_weights_val = None
    pca_enabled = args.pca
    pca_comp = args.pca_components
    penalty_type = args.penalty
    l1_ratio_val = args.l1_ratio
    cascade_enabled = args.cascade
    cascade_comp = args.cascade_components
    cascade_threshold_val = args.cascade_threshold
    pretrained_specialist_dir_val = args.pretrained_specialist_dir
    pretrained_large_specialist_dir_val = args.pretrained_large_specialist_dir
    pretrained_multiclass_dir_val = args.pretrained_multiclass_dir
    pretrained_lofar_multiclass_dir_val = args.pretrained_lofar_multiclass_dir
    large_specialist_runner_up_only_val = args.large_specialist_runner_up_only
    medium_only_routing_val = args.medium_only_routing
    disable_sm_routing_val = args.disable_sm_routing
    
    # Try parsing gamma as float, otherwise keep as string
    gamma_val = args.gamma
    try:
        gamma_val = float(args.gamma)
    except ValueError:
        pass

    if args.nystroem_component_weights is not None:
        component_weight_values = [float(v.strip()) for v in args.nystroem_component_weights.split(',')]
        if len(component_weight_values) != 4:
            raise ValueError("--nystroem_component_weights must contain 4 values: SMALL,MEDIUM,LARGE,BACKGROUND")
        nystroem_component_weights_val = {
            0: component_weight_values[0],
            1: component_weight_values[1],
            2: component_weight_values[2],
            3: component_weight_values[3]
        }
        nystroem_stratify_strategy_val = 'weighted'

    biases = [
        args.bias_small,
        args.bias_medium,
        args.bias_large,
        args.bias_background
    ]

    # Resolve class_weight argument
    cw_type = args.cw_type
    if cw_type == 'balanced':
        class_weight = 'balanced'
    elif cw_type == 'none':
        class_weight = None
    elif cw_type == 'custom':
        class_weight = {
            0: args.cw_small,
            1: args.cw_medium,
            2: args.cw_large,
            3: args.cw_background
        }

    print(f"Running SVM Nyström on folds: {folds_to_execute}")
    print(f"  n_components={n_components}, gamma={gamma_val}, C={reg_c}")
    print(f"  analysis={analysis_type}")
    print(f"  preprocessing: normalize={normalize_enabled}, rms={rms_enabled} (mode={rms_mode_val}, scale={rms_scale_val}), pca={pca_enabled} (n_components={pca_comp})")
    print(f"  nystroem: mode={nystroem_mode_val}, stratify_strategy={nystroem_stratify_strategy_val}, component_weights={nystroem_component_weights_val}")
    if cascade_enabled:
        print(f"  cascade: enabled=True, components={cascade_comp}, threshold={cascade_threshold_val}")
        if pretrained_specialist_dir_val:
            print(f"    pretrained_specialist_dir (MEDIUM)={pretrained_specialist_dir_val}")
        if pretrained_large_specialist_dir_val:
            print(f"    pretrained_large_specialist_dir (LARGE)={pretrained_large_specialist_dir_val}")
        if pretrained_multiclass_dir_val:
            print(f"    pretrained_multiclass_dir (stage-1 hybrid)={pretrained_multiclass_dir_val}")
    print(f"  regularization: penalty={penalty_type}, l1_ratio={l1_ratio_val}")
    print(f"  biases: SMALL={biases[0]}, MEDIUM={biases[1]}, LARGE={biases[2]}, BACKGROUND={biases[3]}")
    print(f"  class weights: type={cw_type}")
    if cw_type == 'custom':
        print(f"    SMALL={args.cw_small}, MEDIUM={args.cw_medium}, LARGE={args.cw_large}, BACKGROUND={args.cw_background}")
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
        biases=biases,
        class_weight=class_weight,
        rms=rms_enabled,
        rms_scale=rms_scale_val,
        rms_mode=rms_mode_val,
        nystroem_mode=nystroem_mode_val,
        nystroem_stratify_strategy=nystroem_stratify_strategy_val,
        nystroem_component_weights=nystroem_component_weights_val,
        cascade=cascade_enabled,
        cascade_components=cascade_comp,
        cascade_bias_small=args.cascade_bias_small,
        cascade_threshold=cascade_threshold_val,
        pretrained_specialist_dir=pretrained_specialist_dir_val,
        pretrained_large_specialist_dir=pretrained_large_specialist_dir_val,
        pretrained_multiclass_dir=pretrained_multiclass_dir_val,
        pretrained_lofar_multiclass_dir=pretrained_lofar_multiclass_dir_val,
        large_specialist_runner_up_only=large_specialist_runner_up_only_val,
        medium_only_routing=medium_only_routing_val,
        disable_sm_routing=disable_sm_routing_val
    )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nElapsed time: {iara.utils.str_format_time(elapsed)}")
