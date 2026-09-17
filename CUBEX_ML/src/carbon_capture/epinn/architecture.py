"""E-PINN Multi-Head Architecture for Intermediate Physical Reconstruction."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER

# The 15 deterministically-derived and physical intermediate output quantities (Section B & C)
PHYSICAL_OUTPUT_NAMES = [
    "E_actual",
    "E_baseline",
    "E_reduced",
    "E_removed",
    "E_displaced_chem",
    "E_displaced_heat",
    "E_displaced",
    "EC_hardware",
    "PE_aux",
    "PE_lifecycle",
    "L_slip",
    "L_transport",
    "L_leakage",
    "F_multiplier",
    "CC_T",
]

class TemporalEncoder(nn.Module):
    """LSTM-based temporal encoder for sequence process data."""
    def __init__(self, input_dim: int = 26, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.latent_proj = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.SiLU()
        
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (batch_size, seq_len, input_dim)
        Returns: (batch_size, input_dim) representing the encoded current latent process state.
        """
        _, (h_n, _) = self.lstm(x_seq)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        # Take the last layer's hidden state
        last_hidden = h_n[-1]
        # Project back to the backbone's expected input dimension (e.g. 26)
        latent_state = self.activation(self.latent_proj(last_hidden))
        return latent_state


class EPINNBackbone(nn.Module):
    """Shared physics-aware representation backbone."""

    def __init__(
        self,
        input_dim: int = 26,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "silu",
        dropout: float = 0.05,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        act_cls = nn.SiLU if activation == "silu" else nn.GELU

        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_d))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = h_d

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class EPINNMultiHead(nn.Module):
    """E-PINN with dedicated regression heads for all intermediate physical quantities."""

    def __init__(
        self,
        input_dim: int = 26,
        hidden_dims: Optional[List[int]] = None,
        output_names: Optional[List[str]] = None,
        activation: str = "silu",
        dropout: float = 0.05,
    ):
        super().__init__()
        self.output_names = output_names or list(PHYSICAL_OUTPUT_NAMES)
        self.backbone = EPINNBackbone(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )

        # Output projection head: predicts all 15 physical quantities
        self.head = nn.Linear(self.backbone.output_dim, len(self.output_names))

        # Optional temporal encoder
        self.temporal_encoder = TemporalEncoder(input_dim=input_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 3:
            # Temporal mode: x is (batch, seq_len, features)
            # We compress the sequence into a latent representation for the current timestep
            x = self.temporal_encoder(x)
            
        feat = self.backbone(x)
        raw_out = self.head(feat)

        out_dict = {}
        for i, name in enumerate(self.output_names):
            out_dict[name] = raw_out[:, i]

        return out_dict
