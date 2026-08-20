"""
2D Convolutional Recurrent Neural Network (2D-CRNN) for underwater acoustic spectrograms.
Combines 2D CNN layers (inspired by Silva et al., 2025) across Time x Frequency with a
Bi-GRU temporal sequence backbone and Temporal Attention.
"""
import typing
import torch
import torch.nn as nn
import iara.ml.models.base_model as iara_model


class TemporalAttention(nn.Module):
    """Learned attention mechanism over the temporal sequence dimension."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(16, hidden_dim // 2), 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, hidden_dim)
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)
        return context


class CRNN2D(iara_model.BaseModel):
    """
    2D-CRNN Architecture:
    1. 2D Convolutional Front-End:
       Applies Conv2D (5x5 kernels) + BatchNorm2D + ReLU + MaxPool2D over the 2D spectrogram (Time x Frequency),
       capturing harmonic lines (LOFAR) and broadband modulation (MEL) simultaneously.
    2. Frequency Projection:
       Pools across frequency to produce a rich temporal feature sequence (B, T', D_feat).
    3. Bi-GRU Recurrent Backbone:
       Models long-range temporal dependencies across the sequence.
    4. Temporal Attention Pooling:
       Focuses on the most informative time steps.
    5. Dense Classification Head:
       Supports binary specialist (n_targets=1) and multiclass (n_targets=4).
    """
    def __init__(
        self,
        input_freq_bins: int = 1024,
        conv_channels: typing.Tuple[int, ...] = (32, 64, 128),
        conv_kernel_size: int = 5,
        freq_pool_size: int = 32,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pooling: str = "attention",
        n_targets: int = 1,
        fc_hidden: typing.Optional[int] = 64,
        activation_output_layer: typing.Optional[typing.Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.input_freq_bins = input_freq_bins
        self.conv_channels = conv_channels
        self.freq_pool_size = freq_pool_size
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling.lower()
        self.n_targets = n_targets

        # 1. 2D Convolutional Layers
        conv_layers = []
        in_ch = 1
        pad = (conv_kernel_size - 1) // 2
        for out_ch in conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=conv_kernel_size, padding=pad),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Pool only in frequency to preserve temporal resolution
                nn.Dropout2d(p=dropout * 0.5)
            ])
            in_ch = out_ch

        self.conv2d_net = nn.Sequential(*conv_layers)
        
        # Adaptive pooling in frequency to standard width
        self.freq_pool = nn.AdaptiveAvgPool2d((None, freq_pool_size))
        feat_dim = conv_channels[-1] * freq_pool_size

        # 2. Recurrent Stage (Bi-GRU)
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0
        )

        gru_out_dim = gru_hidden_size * (2 if bidirectional else 1)

        # 3. Pooling Mechanism
        if self.pooling == "attention":
            self.pool_layer = TemporalAttention(gru_out_dim)
        elif self.pooling in ("mean", "max", "last"):
            self.pool_layer = None
        else:
            raise ValueError(f"Unsupported pooling: {pooling}. Choose 'attention', 'mean', 'max', 'last'.")

        # 4. Dense Head
        head_layers = []
        if fc_hidden is not None and fc_hidden > 0:
            head_layers.extend([
                nn.Linear(gru_out_dim, fc_hidden),
                nn.BatchNorm1d(fc_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden, n_targets)
            ])
        else:
            head_layers.extend([
                nn.Dropout(dropout),
                nn.Linear(gru_out_dim, n_targets)
            ])

        if activation_output_layer is not None:
            head_layers.append(activation_output_layer())

        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 1, seq_len, freq_bins) or (batch_size, seq_len, freq_bins)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, T, F)

        # 2D CNN over Spectrogram
        conv_out = self.conv2d_net(x)          # (B, C_out, T, F')
        conv_pooled = self.freq_pool(conv_out) # (B, C_out, T, 8)
        
        # Reshape to (B, T, C_out * 8)
        B, C, T, F_red = conv_pooled.shape
        feat = conv_pooled.permute(0, 2, 1, 3).contiguous().view(B, T, C * F_red)

        # Bi-GRU over Time
        gru_out, _ = self.gru(feat)            # (B, T, gru_out_dim)

        # Temporal Pooling
        if self.pooling == "attention":
            pooled = self.pool_layer(gru_out)
        elif self.pooling == "mean":
            pooled = torch.mean(gru_out, dim=1)
        elif self.pooling == "max":
            pooled = torch.max(gru_out, dim=1)[0]
        elif self.pooling == "last":
            pooled = gru_out[:, -1, :]
        else:
            pooled = torch.mean(gru_out, dim=1)

        # Classification Logits
        logits = self.classifier(pooled)       # (B, n_targets)
        return logits
