"""
Module containing a 2D CNN for 2D Hilbert-Huang spectrograms (time x frequency inside each window).

Unlike the 1D SpectralCNN (convolution along the 256-bin marginal spectrum) and the legacy
PyTorchHHTCNN (images of 32 consecutive marginal spectra, ~164 s, global (1, 1) pooling, collapsed),
this model takes the time-frequency energy distribution inside one window:
    Input: (batch, in_channels, n_time_bins, n_freq_bins)
    Default: (batch, 1, 32, 128)

AdaptiveAvgPool2d((pooled_time, pooled_freq)) keeps a coarse position in time and frequency for the
classifier. With the default feature (80 ms time bins) only modulations below ~6 Hz are resolved in time.
"""
import typing
import torch
import torch.nn as nn

import iara.ml.models.base_model as iara_model


class Hilbert2DCNN(iara_model.BaseModel):
    """
    2D CNN specialized for True 2D Hilbert-Huang Spectrograms (Time-Frequency matrices).

    Architecture:
    - Conv Block 1: Conv2d -> BatchNorm2d -> ReLU -> Dropout2d -> MaxPool2d (extracts local time-frequency edges)
    - Conv Block 2: Conv2d -> BatchNorm2d -> ReLU -> Dropout2d -> MaxPool2d (captures harmonic patterns and continuity)
    - Conv Block 3: Conv2d -> BatchNorm2d -> ReLU -> Dropout2d -> MaxPool2d (abstracts broadband signatures)
    - Topology-Preserving Pooling: AdaptiveAvgPool2d((pooled_time, pooled_freq))
    - Dense Classifier: Flatten -> Dropout -> Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear(n_targets)
    """

    def __init__(self,
                 n_targets: int = 4,
                 in_channels: int = 1,
                 n_time_bins: int = 32,
                 n_freq_bins: int = 128,
                 conv_channels: typing.List[int] = [32, 64, 128],
                 kernel_sizes: typing.List[typing.Tuple[int, int]] = [(3, 5), (3, 5), (3, 3)],
                 paddings: typing.List[typing.Tuple[int, int]] = [(1, 2), (1, 2), (1, 1)],
                 conv_dropout: float = 0.15,
                 pooled_time: int = 2,
                 pooled_freq: int = 8,
                 classification_n_neurons: int = 128,
                 classification_dropout: float = 0.35):
        super().__init__()

        self.n_targets = n_targets
        self.in_channels = in_channels
        self.n_time_bins = n_time_bins
        self.n_freq_bins = n_freq_bins
        self.pooled_time = pooled_time
        self.pooled_freq = pooled_freq

        # Stage 1: Multiscale 2D Feature Extraction
        layers = []
        in_ch = in_channels
        for out_ch, k_sz, pad in zip(conv_channels, kernel_sizes, paddings):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k_sz, padding=pad))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if conv_dropout > 0:
                layers.append(nn.Dropout2d(p=conv_dropout))
            layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
            in_ch = out_ch

        # Retain coarse time (2) and coarse frequency bands (8)
        layers.append(nn.AdaptiveAvgPool2d((pooled_time, pooled_freq)))
        self.conv_layers = nn.Sequential(*layers)

        # Stage 2: Classification Head
        in_features = in_ch * pooled_time * pooled_freq
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=classification_dropout),
            nn.Linear(in_features, classification_n_neurons),
            nn.BatchNorm1d(classification_n_neurons),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classification_dropout),
            nn.Linear(classification_n_neurons, n_targets)
        )

    def to_feature_space(self, data: torch.Tensor) -> torch.Tensor:
        """
        Passes data through convolutional backbone.
        Accepts (B, H, W) or (B, C, H, W).
        """
        if data.dim() == 3:
            data = data.unsqueeze(1)
        return self.conv_layers(data)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing classification logits.
        """
        features = self.to_feature_space(data)
        return self.classifier(features)
