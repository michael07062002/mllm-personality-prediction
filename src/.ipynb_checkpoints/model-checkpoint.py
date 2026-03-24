# src/model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.blocks.agf import AGFFusion
from src.blocks.mamba2 import Mamba2Backbone, Mamba2Config


@dataclass
class PersonalityModelConfig:
    input_dim: int
    d_model: int = 512
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    num_mamba_blocks: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    num_outputs: int = 5
    num_layers: int = 1
    use_agf: bool = False
    agf_hidden_dim: int | None = None
    use_layer_specific_projection: bool = False


class SingleLayerInputProjection(nn.Module):
    """
    For x: (B, S, D_in) -> (B, S, D_model)
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalPooling(nn.Module):
    """
    Masked mean pooling over temporal dimension.
    x:    (B, S, D)
    mask: (B, S), True for valid positions
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(-1)      # (B, S, 1)
        x = x * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = x.sum(dim=1) / denom
        return pooled


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, num_outputs: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PersonalityMambaModel(nn.Module):
    """
    Unified model.

    Single-layer mode:
        input x: (B, S, D_in)

    Multi-layer mode:
        input x: (B, L, S, D_in)

    Output:
        preds: (B, 5)
    """

    def __init__(self, cfg: PersonalityModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_agf = cfg.use_agf

        if self.use_agf:
            self.fusion = AGFFusion(
                input_dim=cfg.input_dim,
                d_model=cfg.d_model,
                num_layers=cfg.num_layers,
                fusion_hidden_dim=cfg.agf_hidden_dim,
                use_layer_specific_projection=cfg.use_layer_specific_projection,
                dropout=cfg.dropout,
            )
        else:
            self.input_proj = SingleLayerInputProjection(
                input_dim=cfg.input_dim,
                d_model=cfg.d_model,
                dropout=cfg.dropout,
            )

        mamba_cfg = Mamba2Config(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            dropout=cfg.dropout,
            mlp_ratio=cfg.mlp_ratio,
        )

        self.backbone = Mamba2Backbone(
            cfg=mamba_cfg,
            num_blocks=cfg.num_mamba_blocks,
        )
        self.pool = TemporalPooling()
        self.head = RegressionHead(
            d_model=cfg.d_model,
            num_outputs=cfg.num_outputs,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        aux: Dict[str, torch.Tensor] = {}

        if self.use_agf:
            # x: (B, L, S, D_in)
            h, layer_weights = self.fusion(x)      # h: (B, S, D_model)
            aux["layer_weights"] = layer_weights
        else:
            # x: (B, S, D_in)
            h = self.input_proj(x)                 # (B, S, D_model)

        h = self.backbone(h)                       # (B, S, D_model)
        pooled = self.pool(h, mask)                # (B, D_model)
        preds = self.head(pooled)                  # (B, 5)

        aux["temporal_features"] = h
        aux["pooled_features"] = pooled
        aux["preds"] = preds
        return aux


def build_single_layer_model(
    input_dim: int,
    d_model: int = 512,
    d_state: int = 64,
    d_conv: int = 4,
    expand: int = 2,
    num_mamba_blocks: int = 4,
    mlp_ratio: float = 2.0,
    dropout: float = 0.1,
    num_outputs: int = 5,
) -> PersonalityMambaModel:
    cfg = PersonalityModelConfig(
        input_dim=input_dim,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        num_mamba_blocks=num_mamba_blocks,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        num_outputs=num_outputs,
        num_layers=1,
        use_agf=False,
    )
    return PersonalityMambaModel(cfg)


def build_multi_layer_model(
    input_dim: int,
    num_layers: int,
    d_model: int = 512,
    d_state: int = 64,
    d_conv: int = 4,
    expand: int = 2,
    num_mamba_blocks: int = 4,
    mlp_ratio: float = 2.0,
    dropout: float = 0.1,
    num_outputs: int = 5,
    agf_hidden_dim: int | None = None,
    use_layer_specific_projection: bool = False,
) -> PersonalityMambaModel:
    cfg = PersonalityModelConfig(
        input_dim=input_dim,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        num_mamba_blocks=num_mamba_blocks,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        num_outputs=num_outputs,
        num_layers=num_layers,
        use_agf=True,
        agf_hidden_dim=agf_hidden_dim,
        use_layer_specific_projection=use_layer_specific_projection,
    )
    return PersonalityMambaModel(cfg)