"""
Convolutional Recurrent Neural Network (CRNN) for sequential underwater acoustic data.
Combines 1D convolutional feature extraction across spectral bins with a Bi-GRU
temporal recurrence and attention pooling.
"""
import typing
import torch
import torch.nn as nn
import iara.ml.models.base_model as iara_model


class TemporalAttention(nn.Module):
    """
    Learned attention mechanism over the temporal sequence dimension.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(16, hidden_dim // 2), 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, hidden_dim)
        scores = self.attn(x)  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(x * weights, dim=1)  # (batch_size, hidden_dim)
        return context


class CRNN(iara_model.BaseModel):
    """
    CRNN architecture combining:
    1. 1D Convolutional blocks (per time frame) to extract local spectral textures
       and filter high-frequency noise.
    2. Bidirectional GRU to model long-range temporal rhythm (propeller cavitation / diesel cycles).
    3. Temporal Attention Pooling to focus on the most informative acoustic transients.
    4. Dense classification head supporting both binary specialist (n_targets=1)
       and multiclass classification (n_targets=4).
    """
    def __init__(
        self,
        input_dim: int = 288,
        conv_channels: typing.Tuple[int, ...] = (32, 64),
        conv_kernel_size: int = 5,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pooling: str = "attention",  # 'attention', 'mean', 'max', 'last'
        n_targets: int = 1,          # 1 for binary specialist, 4 for multiclass
        fc_hidden: typing.Optional[int] = 32,
        activation_output_layer: typing.Optional[typing.Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.conv_channels = conv_channels
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling.lower()
        self.n_targets = n_targets

        # 1. Convolutional Front-End (applied per spectral frame)
        conv_layers = []
        in_ch = 1
        for out_ch in conv_channels:
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=conv_kernel_size, stride=1, padding=conv_kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout * 0.5)
            ])
            in_ch = out_ch

        # Global adaptive pooling after convolutions to standard representation size
        self.conv_net = nn.Sequential(*conv_layers)
        self.conv_pool = nn.AdaptiveAvgPool1d(16)
        conv_feat_dim = conv_channels[-1] * 16

        # 2. Recurrent Stage (Bi-GRU)
        self.gru = nn.GRU(
            input_size=conv_feat_dim,
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

        # 4. Fully Connected Head
        head_layers = []
        if fc_hidden is not None and fc_hidden > 0:
            head_layers.extend([
                nn.Linear(gru_out_dim, fc_hidden),
                nn.BatchNorm1d(fc_hidden),
                nn.GELU(),
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
        # x: (batch_size, seq_len, input_dim)
        B, T, D = x.shape

        # Pass through Conv Front-End across spectral bins
        # Reshape to (B * T, 1, D)
        x_frames = x.contiguous().view(B * T, 1, D)
        conv_out = self.conv_net(x_frames)        # (B*T, conv_channels[-1], reduced_D)
        conv_pooled = self.conv_pool(conv_out)    # (B*T, conv_channels[-1], 16)
        feat = conv_pooled.view(B, T, -1)         # (B, T, conv_feat_dim)

        # Pass through Bi-GRU
        gru_out, _ = self.gru(feat)               # (B, T, gru_out_dim)

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

        # Classification Head
        logits = self.classifier(pooled)          # (B, n_targets)
        return logits
