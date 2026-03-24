# src/blocks/mamba2.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Mamba2Config:
    d_model: int = 512
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | str = "auto"
    dropout: float = 0.1
    bias: bool = True
    conv_bias: bool = True
    layer_norm_eps: float = 1e-5
    mlp_ratio: float = 2.0


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class CausalDepthwiseConv1d(nn.Module):
    """
    Causal depthwise conv over sequence dimension.
    Input/Output: (B, S, C)
    """

    def __init__(self, channels: int, kernel_size: int, bias: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C)
        x = x.transpose(1, 2)  # (B, C, S)
        pad = self.kernel_size - 1
        x = F.pad(x, (pad, 0))
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, S, C)
        return x


class SelectiveSSM(nn.Module):
    """
    Simplified pure PyTorch selective state-space core.

    Input:
        u:      (B, S, D_inner)
        delta:  (B, S, D_inner)
        B_t:    (B, S, D_state)
        C_t:    (B, S, D_state)

    Output:
        y:      (B, S, D_inner)
    """

    def __init__(self, d_inner: int, d_state: int) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        self.A_log = nn.Parameter(torch.randn(d_inner, d_state) * 0.02)
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        B_t: torch.Tensor,
        C_t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, d_inner = u.shape
        d_state = self.d_state

        A = -torch.exp(self.A_log)                          # (D_inner, D_state)

        x_state = torch.zeros(
            bsz, d_inner, d_state,
            device=u.device,
            dtype=u.dtype,
        )

        ys = []
        for t in range(seqlen):
            dt = F.softplus(delta[:, t])                    # (B, D_inner)
            Bt = B_t[:, t].unsqueeze(1)                     # (B, 1, D_state)
            Ct = C_t[:, t].unsqueeze(1)                     # (B, 1, D_state)
            ut = u[:, t].unsqueeze(-1)                      # (B, D_inner, 1)

            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))   # (B, D_inner, D_state)
            dB_u = dt.unsqueeze(-1) * Bt * ut                    # (B, D_inner, D_state)

            x_state = dA * x_state + dB_u
            yt = (x_state * Ct).sum(dim=-1) + self.D.unsqueeze(0) * u[:, t]
            ys.append(yt)

        y = torch.stack(ys, dim=1)                          # (B, S, D_inner)
        return y


class Mamba2Mixer(nn.Module):
    """
    Pure PyTorch approximation of a Mamba-2-style mixer.
    Designed for sequence input (B, S, D_model).
    """

    def __init__(self, cfg: Mamba2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.d_inner = cfg.expand * cfg.d_model
        self.d_state = cfg.d_state

        if cfg.dt_rank == "auto":
            self.dt_rank = max(16, cfg.d_model // 16)
        else:
            self.dt_rank = int(cfg.dt_rank)

        self.in_proj = nn.Linear(
            cfg.d_model,
            self.d_inner * 2,
            bias=cfg.bias,
        )

        self.conv = CausalDepthwiseConv1d(
            channels=self.d_inner,
            kernel_size=cfg.d_conv,
            bias=cfg.conv_bias,
        )

        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state + self.d_state,
            bias=False,
        )

        self.dt_proj = nn.Linear(
            self.dt_rank,
            self.d_inner,
            bias=True,
        )

        self.ssm = SelectiveSSM(
            d_inner=self.d_inner,
            d_state=self.d_state,
        )

        self.out_proj = nn.Linear(
            self.d_inner,
            cfg.d_model,
            bias=cfg.bias,
        )

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, D_model)
        """
        xz = self.in_proj(x)                        # (B, S, 2*D_inner)
        x_part, z_part = xz.chunk(2, dim=-1)       # each (B, S, D_inner)

        x_part = self.conv(x_part)
        x_part = F.silu(x_part)

        x_dbl = self.x_proj(x_part)                # (B, S, dt_rank + 2*d_state)
        delta_raw, B_t, C_t = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        delta = self.dt_proj(delta_raw)            # (B, S, D_inner)
        y = self.ssm(
            u=x_part,
            delta=delta,
            B_t=B_t,
            C_t=C_t,
        )                                          # (B, S, D_inner)

        y = y * F.silu(z_part)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 2.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Mamba2Block(nn.Module):
    """
    Pre-norm residual block:
        x = x + Mamba2Mixer(norm1(x))
        x = x + FFN(norm2(x))
    """

    def __init__(self, cfg: Mamba2Config) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.mixer = Mamba2Mixer(cfg)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.ffn = FeedForward(
            d_model=cfg.d_model,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Mamba2Backbone(nn.Module):
    """
    Stack of Mamba2 blocks.
    Input/Output: (B, S, D_model)
    """

    def __init__(self, cfg: Mamba2Config, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Mamba2Block(cfg) for _ in range(num_blocks)])
        self.out_norm = RMSNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x