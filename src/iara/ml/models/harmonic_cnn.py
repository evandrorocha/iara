"""
Module containing a harmonic-stacking CNN for DEMON (envelope modulation) spectra.

In a DEMON spectrum the propeller shows up as a shaft-rate line f0 and its blade-rate harmonics k*f0.
On a linear frequency axis the spacing between harmonics depends on f0, so a small convolution
kernel sees a different pattern for each ship. Harmonic stacking (as in the HCQT of Bittner et al.,
2017) samples the spectrum at h*f0 for several harmonics h and stacks them as channels: at position
f0, channel h holds the energy at the h-th harmonic, so the whole harmonic series is aligned in the
same position whatever f0 is. Positions where h*f0 falls outside the spectrum are filled with 0.
"""
import typing
import numpy as np
import torch

import iara.ml.models.base_model as iara_model


class HarmonicStack(torch.nn.Module):
    """(batch, n_bins) or (batch, 1, n_bins) -> (batch, len(harmonics), n_bins).

    For harmonic h, position f0 takes the maximum of the spectrum over the bins of the interval
    h*[f0 - df/2, f0 + df/2]. For h > 1 consecutive f0 positions are h bins apart in h*f0, so plain
    interpolation would skip lines narrower than h bins; the maximum over the interval does not.
    """

    def __init__(self, frequencies: typing.Iterable[float], harmonics: typing.Iterable[float]):
        super().__init__()
        f = np.asarray(frequencies, dtype=np.float64)
        df = f[1] - f[0]
        n = len(f)
        half = int(np.ceil(max(max(harmonics), 1) / 2))
        offsets = np.arange(-half, half + 1)
        idx, mask, valid = [], [], []
        for h in harmonics:
            pos = (h * f - f[0]) / df
            center = np.rint(pos).astype(np.int64)
            reach = max(h, 1) / 2
            cand = center[:, None] + offsets[None, :]
            inside = (np.abs(cand - pos[:, None]) <= reach) | (offsets[None, :] == 0)
            inside &= (cand >= 0) & (cand < n)
            idx.append(np.clip(cand, 0, n - 1))
            mask.append(inside)
            valid.append((pos >= -0.5) & (pos <= n - 0.5))
        self.register_buffer("idx", torch.tensor(np.stack(idx)))  # (H, F, K)
        self.register_buffer("mask", torch.tensor(np.stack(mask)))
        self.register_buffer("valid", torch.tensor(np.stack(valid), dtype=torch.float32))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() == 3:
            data = data[:, 0, :]
        vals = data[:, self.idx]  # (B, H, F, K)
        vals = vals.masked_fill(~self.mask, float("-inf")).amax(-1)
        return torch.nan_to_num(vals, neginf=0.0) * self.valid


class HarmonicCNN(iara_model.BaseModel):
    """Harmonic stacking followed by Conv1d blocks along the f0 axis.

    Input: (batch, n_bins) spectrum (already normalized). Output: (batch, n_targets) logits.
    Each block: Conv1d (dilated) -> BatchNorm1d -> ReLU -> Dropout -> optional MaxPool1d(2); then
    AdaptiveAvgPool1d(pooled_bins) keeps the coarse absolute position of f0 for the classifier.
    """

    def __init__(self,
                 frequencies: typing.Iterable[float],
                 n_targets: int,
                 harmonics: typing.List[float] = [0.5, 1, 2, 3, 4, 5],
                 conv_channels: typing.List[int] = [32, 64, 64],
                 kernel_size: int = 5,
                 dilations: typing.List[int] = [1, 2, 4],
                 pool_after: typing.List[bool] = [True, True, False],
                 conv_dropout: float = 0.1,
                 pooled_bins: int = 8,
                 classification_n_neurons: int = 64,
                 classification_dropout: float = 0.4):
        super().__init__()

        if not len(conv_channels) == len(dilations) == len(pool_after):
            raise ValueError("conv_channels, dilations and pool_after must have the same length")

        self.harmonics = list(harmonics)
        self.stack = HarmonicStack(frequencies, self.harmonics)

        layers = []
        in_ch = len(self.harmonics)
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
        return self.conv_layers(self.stack(data))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.to_feature_space(data))
