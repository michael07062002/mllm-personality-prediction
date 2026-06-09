from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from src.training.blocks.agf import AGFFusion
from src.training.blocks.mamba2 import Mamba2Backbone, Mamba2Config


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


def _sanitize(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class SingleLayerInputProjection(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _sanitize(x.float())
        x = self.norm(x)
        x = self.proj(x)
        x = self.dropout(x)
        return _sanitize(x)


class TemporalPooling(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = _sanitize(x)
        mask = mask.float().unsqueeze(-1)
        x = x * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = x.sum(dim=1) / denom
        return _sanitize(pooled)


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, num_outputs: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = d_model // 2
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _sanitize(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return _sanitize(x)


class PersonalityMambaModel(nn.Module):
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
        x = _sanitize(x)
        mask = mask.bool()
        if self.use_agf:
            h, layer_weights = self.fusion(x)
            h = _sanitize(h)
            aux["layer_weights"] = _sanitize(layer_weights)
        else:
            h = self.input_proj(x)
        h = self.backbone(h)
        h = _sanitize(h)
        pooled = self.pool(h, mask)
        preds = self.head(pooled)
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