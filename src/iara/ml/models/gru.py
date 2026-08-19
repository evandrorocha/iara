"""
Module containing Gated Recurrent Unit (GRU) based models for sequential acoustic data.
"""
import typing
import torch
import torch.nn as nn
import iara.ml.models.base_model as iara_model


class TemporalAttention(nn.Module):
    """
    Learned attention mechanism over the temporal dimension.
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


class GRU(iara_model.BaseModel):
    """
    Gated Recurrent Unit (GRU) model for underwater acoustic target classification.
    Supports binary (specialist) and multiclass classification over temporal sequences of spectral frames.
    """
    def __init__(
        self,
        input_dim: int = 256,
        hidden_size: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        pooling: str = "mean",  # 'mean', 'last', 'max', 'attention'
        n_targets: int = 1,     # 1 for binary with sigmoid/logits, or 4 for multiclass
        fc_hidden: typing.Optional[int] = 32,
        activation_output_layer: typing.Optional[typing.Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling.lower()
        self.n_targets = n_targets

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=gru_dropout,
        )

        out_hidden_dim = hidden_size * 2 if bidirectional else hidden_size

        if self.pooling == "attention":
            self.attention = TemporalAttention(out_hidden_dim)

        # Classification head
        head_layers = []
        if fc_hidden is not None and fc_hidden > 0:
            head_layers.append(nn.BatchNorm1d(out_hidden_dim))
            if dropout > 0:
                head_layers.append(nn.Dropout(dropout))
            head_layers.append(nn.Linear(out_hidden_dim, fc_hidden))
            head_layers.append(nn.ReLU())
            head_layers.append(nn.BatchNorm1d(fc_hidden))
            if dropout > 0:
                head_layers.append(nn.Dropout(dropout))
            head_layers.append(nn.Linear(fc_hidden, n_targets))
        else:
            head_layers.append(nn.BatchNorm1d(out_hidden_dim))
            if dropout > 0:
                head_layers.append(nn.Dropout(dropout))
            head_layers.append(nn.Linear(out_hidden_dim, n_targets))

        if activation_output_layer is not None:
            head_layers.append(activation_output_layer())

        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle input shapes:
        # (batch, seq_len, features)
        # or (batch, 1, seq_len, features) / (batch, channels, seq_len, features)
        # or (batch, features) -> treat as (batch, 1, features)
        if x.dim() == 4:
            # (B, 1, T, D) -> (B, T, D)
            x = x.squeeze(1)
        elif x.dim() == 2:
            # (B, D) -> (B, 1, D)
            x = x.unsqueeze(1)

        out, h_n = self.gru(x)  # out: (B, T, out_hidden_dim)

        if self.pooling == "mean":
            feat = torch.mean(out, dim=1)
        elif self.pooling == "max":
            feat, _ = torch.max(out, dim=1)
        elif self.pooling == "attention":
            feat = self.attention(out)
        elif self.pooling == "last":
            if self.bidirectional:
                feat = torch.cat([out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]], dim=1)
            else:
                feat = out[:, -1, :]
        else:
            feat = torch.mean(out, dim=1)

        logits = self.classifier(feat)
        return logits
