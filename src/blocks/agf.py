# src/blocks/agf.py

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerwiseProjection(nn.Module):
    """
    Projects per-layer features from input_dim -> d_model.

    Input:
        x: (B, L, S, D_in)

    Output:
        h: (B, L, S, D_model)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        use_layer_specific_projection: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_layer_specific_projection = use_layer_specific_projection

        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

        if use_layer_specific_projection:
            self.proj = nn.ModuleList(
                [nn.Linear(input_dim, d_model) for _ in range(num_layers)]
            )
        else:
            self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, S, D_in)
        """
        b, l, s, d = x.shape
        if l != self.num_layers:
            raise ValueError(f"Expected num_layers={self.num_layers}, got {l}")

        x = self.norm(x)

        if self.use_layer_specific_projection:
            outs = []
            for i in range(l):
                xi = self.proj[i](x[:, i])  # (B, S, D_model)
                outs.append(xi)
            h = torch.stack(outs, dim=1)   # (B, L, S, D_model)
        else:
            h = self.proj(x)               # (B, L, S, D_model)

        return self.dropout(h)


class AGFScoreNetwork(nn.Module):
    """
    Scores each layer at each segment.

    Inputs:
        h:       (B, L, S, D)
        context: (B, S, D)

    Output:
        scores:  (B, L, S, 1)
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.content_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.context_mlp = nn.Sequential(
            nn.Linear(d_model * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        h:       (B, L, S, D)
        context: (B, S, D)
        """
        context_exp = context.unsqueeze(1).expand_as(h)  # (B, L, S, D)

        content_score = self.content_mlp(h)              # (B, L, S, 1)

        joint = torch.cat(
            [
                h,
                context_exp,
                h - context_exp,
                h * context_exp,
            ],
            dim=-1,
        )                                                # (B, L, S, 4D)

        context_score = self.context_mlp(joint)          # (B, L, S, 1)
        scores = content_score + context_score
        return scores


class AGFResidualGate(nn.Module):
    """
    Blends fused representation with base representation.

    fused: (B, S, D)
    base:  (B, S, D)
    out:   (B, S, D)
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, fused: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        gate = self.gate_net(torch.cat([fused, base], dim=-1))
        out = gate * fused + (1.0 - gate) * base
        return out


class AGFFusion(nn.Module):
    """
    Adaptive Gating Fusion over layer dimension.

    Input:
        x: (B, L, S, D_in)

    Output:
        out:          (B, S, D_model)
        layer_weights:(B, L, S)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        fusion_hidden_dim: int | None = None,
        use_layer_specific_projection: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.fusion_hidden_dim = fusion_hidden_dim or d_model

        self.proj = LayerwiseProjection(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            use_layer_specific_projection=use_layer_specific_projection,
            dropout=dropout,
        )

        self.score_net = AGFScoreNetwork(
            d_model=d_model,
            hidden_dim=self.fusion_hidden_dim,
            dropout=dropout,
        )

        self.residual_gate = AGFResidualGate(
            d_model=d_model,
            hidden_dim=self.fusion_hidden_dim,
            dropout=dropout,
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, L, S, D_in)
        """
        h = self.proj(x)                         # (B, L, S, D_model)
        base = h.mean(dim=1)                     # (B, S, D_model)

        scores = self.score_net(h, base)         # (B, L, S, 1)
        alpha = torch.softmax(scores, dim=1)     # (B, L, S, 1)

        fused = (alpha * h).sum(dim=1)           # (B, S, D_model)
        out = self.residual_gate(fused, base)    # (B, S, D_model)
        out = self.out_norm(out)
        out = self.out_dropout(out)

        layer_weights = alpha.squeeze(-1)        # (B, L, S)
        return out, layer_weights