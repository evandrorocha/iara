"""
Module containing a 1D CNN that convolves along the frequency axis of a single spectrum.

Designed for representations whose windows are long and sparse in time (HHT and DEMON, one
spectrum every 5.12 s), where the useful structure lies along frequency (tonal lines, harmonics)
rather than along time. It does not replace iara.ml.models.cnn.CNN (2D, used for LOFAR/MEL images).
"""
import typing
import torch

import iara.ml.models.base_model as iara_model

class SpectralCNN(iara_model.BaseModel):
    """1D CNN over frequency.

    Input: (batch, n_context, n_bins) - n_context neighbouring windows of the same recording enter as
    channels (n_context = 1 -> a single window). Output: (batch, n_targets) logits.

    Each block: Conv1d (dilated) -> BatchNorm1d -> ReLU -> Dropout -> optional MaxPool1d(2).
    The feature map is reduced to `pooled_bins` positions with AdaptiveAvgPool1d (instead of a global
    pooling) so that the classifier keeps the coarse absolute position of the energy in frequency.
    """

    def __init__(self,
                 n_bins: int,
                 n_targets: int,
                 n_context: int = 1,
                 conv_channels: typing.List[int] = [32, 64, 64, 128],
                 kernel_size: int = 7,
                 dilations: typing.List[int] = [1, 1, 2, 4],
                 pool_after: typing.List[bool] = [True, True, True, False],
                 conv_dropout: float = 0.1,
                 pooled_bins: int = 8,
                 classification_n_neurons: int = 64,
                 classification_dropout: float = 0.4):
        super().__init__()

        if not len(conv_channels) == len(dilations) == len(pool_after):
            raise ValueError("conv_channels, dilations and pool_after must have the same length")

        self.n_bins = n_bins
        self.n_context = n_context

        layers = []
        in_ch = n_context
        for out_ch, dil, pool in zip(conv_channels, dilations, pool_after):
            layers.append(torch.nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                                          dilation=dil, padding=dil * (kernel_size - 1) // 2))
            layers.append(torch.nn.BatchNorm1d(out_ch))
            layers.append(torch.nn.ReLU())
            if conv_dropout > 0:
                layers.append(torch.nn.Dropout(conv_dropout))
            if pool:
                layers.append(torch.nn.MaxPool1d(2))
            in_ch = out_ch
        layers.append(torch.nn.AdaptiveAvgPool1d(pooled_bins))
        self.conv_layers = torch.nn.Sequential(*layers)

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(classification_dropout),
            torch.nn.Linear(in_ch * pooled_bins, classification_n_neurons),
            torch.nn.ReLU(),
            torch.nn.Dropout(classification_dropout),
            torch.nn.Linear(classification_n_neurons, n_targets),
        )

    def to_feature_space(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() == 2:
            data = data.unsqueeze(1)
        return self.conv_layers(data)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.to_feature_space(data))
