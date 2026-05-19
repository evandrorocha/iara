"""
Module containing SVM-based model using Nyström kernel approximation.

This module implements an approximate SVM with RBF kernel using the Nyström
method (Williams & Seeger, NeurIPS 2001) combined with a stochastic gradient
descent classifier (hinge loss). This approach makes kernel SVM scalable to
large datasets (500K+ samples) while preserving the non-linear decision boundary
of the RBF kernel.

Why Nyström + SGD instead of vanilla SVC(kernel='rbf'):
    - SVC with RBF has O(n²~n³) complexity: infeasible for 500K samples
    - Nyström approximates the RBF kernel mapping explicitly in R^m (m << n)
    - SGDClassifier with hinge loss = SVM linear in the transformed space
    - Combined result: approximate RBF-SVM with O(m*n) complexity

References:
    Williams, C., & Seeger, M. (2001). Using the Nyström method to speed up
    kernel machines. Advances in Neural Information Processing Systems.
"""
import typing

import torch
import numpy as np

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import iara.ml.models.base_model as iara_model


class SVMNystroem(iara_model.BaseModel):
    """Approximate SVM with RBF kernel using Nyström method + SGDClassifier.

    The pipeline is:
        x (input features)
            ↓ Nyström: non-linear transformation approximating RBF kernel
        z(x) ∈ R^n_components
            ↓ SGDClassifier (hinge loss): linear SVM in transformed space
        class prediction

    The decision boundary is non-linear in the original feature space because
    the Nyström transformation is non-linear.

    Attributes:
        nystroem (Nystroem): Kernel approximation transformer.
        sgd (SGDClassifier): Linear SVM trained in the transformed space.
        is_fitted (bool): Whether the model has been trained.
    """

    def __init__(self,
                 n_components: int = 300,
                 gamma: typing.Union[str, float] = 'scale',
                 C: float = 1.0,
                 n_targets: int = 4,
                 random_state: int = 42):
        """Initialize SVMNystroem model.

        Args:
            n_components (int): Number of Nyström landmark points. Controls
                the quality of the RBF approximation. More components = better
                approximation but more memory and time. Default: 300.
            gamma (Union[str, float]): RBF kernel parameter γ in
                exp(-γ||x-y||²). 'scale' uses 1/(n_features * X.var()).
                Default: 'scale'.
            C (float): Regularization parameter. Smaller values = stronger
                regularization (larger margin, more misclassifications allowed).
                Default: 1.0.
            n_targets (int): Number of output classes. Default: 4.
            random_state (int): Random seed for reproducibility. Default: 42.
        """
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.C = C
        self.n_targets = n_targets
        self.random_state = random_state

        self.nystroem = Nystroem(
            kernel='rbf',
            gamma=None,  # Will be set in fit() if 'scale'
            n_components=n_components,
            random_state=random_state
        )

        # alpha = 1 / (C * n_samples) — will be set properly during fit
        # using alpha=1.0 as placeholder, recalculated in fit()
        self.sgd = SGDClassifier(
            loss='hinge',
            alpha=1.0,
            class_weight='balanced',
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
            n_jobs=1
        )

        self.is_fitted = False

    def fit(self, samples: torch.Tensor, targets: torch.Tensor) -> None:
        """Fit the Nyström + SGD SVM pipeline.

        Step 1: Fit Nyström transformer on training data (selects landmarks).
        Step 2: Transform all samples to the approximate RBF feature space.
        Step 3: Fit SGDClassifier (hinge loss) on transformed samples.

        Args:
            samples (torch.Tensor): Training samples, shape [n_samples, n_features].
            targets (torch.Tensor): Class labels, shape [n_samples].
        """
        X = samples.view(samples.size(0), -1).cpu().numpy()
        y = targets.cpu().numpy().astype(int)

        n_samples = X.shape[0]

        # Recalculate alpha so that C = 1 / (alpha * n_samples)
        alpha = 1.0 / (self.C * n_samples)
        self.sgd.set_params(alpha=alpha)

        # Calculate gamma if 'scale' or with a multiplier
        if isinstance(self.gamma, str) and 'scale' in self.gamma:
            base_gamma = 1.0 / (X.shape[1] * X.var())
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
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma

        self.nystroem.set_params(gamma=gamma)

        # Step 1 & 2: Fit and transform with Nyström
        X_transformed = self.nystroem.fit_transform(X)

        # Step 3: Fit linear SVM (hinge loss) in transformed space
        self.sgd.fit(X_transformed, y)

        self.is_fitted = True

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Predict class labels for input samples.

        Args:
            data (torch.Tensor): Input samples, shape [n_samples, n_features].

        Returns:
            torch.Tensor: Predicted class labels, shape [n_samples].
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling forward(). "
                               "Call fit() first.")

        X = data.view(data.size(0), -1).cpu().numpy()
        X_transformed = self.nystroem.transform(X)
        predictions = self.sgd.predict(X_transformed)
        return torch.tensor(predictions, dtype=torch.long)


class SVMNystroemPreprocessed(iara_model.BaseModel):
    """Approximate SVM with RBF kernel including StandardScaler and PCA preprocessing.

    The pipeline is:
        x (input features)
            ↓ [Optional] StandardScaler (z-score scaling)
        x_scaled
            ↓ [Optional] PCA (principal component analysis dimensionality reduction)
        x_pca
            ↓ Nyström: non-linear transformation approximating RBF kernel
        z(x) ∈ R^n_components
            ↓ SGDClassifier (hinge loss): linear SVM in transformed space
        class prediction
    """

    def __init__(self,
                 n_components: int = 300,
                 gamma: typing.Union[str, float] = 'scale',
                 C: float = 1.0,
                 n_targets: int = 4,
                 normalize: bool = False,
                 pca: bool = False,
                 n_pca_components: int = 64,
                 random_state: int = 42):
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.C = C
        self.n_targets = n_targets
        self.normalize = normalize
        self.pca = pca
        self.n_pca_components = n_pca_components
        self.random_state = random_state

        self.scaler = StandardScaler() if normalize else None
        self.pca_trans = PCA(n_components=n_pca_components, random_state=random_state) if pca else None

        self.nystroem = Nystroem(
            kernel='rbf',
            gamma=None,  # Will be set in fit()
            n_components=n_components,
            random_state=random_state
        )

        self.sgd = SGDClassifier(
            loss='hinge',
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

        # 1. Apply optional normalization
        if self.normalize:
            X = self.scaler.fit_transform(X)

        # 2. Apply optional PCA reduction
        if self.pca:
            X = self.pca_trans.fit_transform(X)

        # Calculate gamma on the processed X
        if isinstance(self.gamma, str) and 'scale' in self.gamma:
            base_gamma = 1.0 / (X.shape[1] * X.var())
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
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma

        self.nystroem.set_params(gamma=gamma)

        # 3. Fit & transform with Nyström
        X_transformed = self.nystroem.fit_transform(X)

        # 4. Fit SGDClassifier
        self.sgd.fit(X_transformed, y)
        self.is_fitted = True

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling forward().")

        X = data.view(data.size(0), -1).cpu().numpy()

        if self.normalize:
            X = self.scaler.transform(X)

        if self.pca:
            X = self.pca_trans.transform(X)

        X_transformed = self.nystroem.transform(X)
        predictions = self.sgd.predict(X_transformed)
        return torch.tensor(predictions, dtype=torch.long)
