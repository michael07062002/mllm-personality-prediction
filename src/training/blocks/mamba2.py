from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple

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
    head_dim: int = 64
    n_groups: int = 8
    residual_in_fp32: bool = True
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: Tuple[float, float] = (0.0, float("inf"))

    def __post_init__(self):
        intermediate_size = self.expand * self.d_model
        if intermediate_size % self.head_dim != 0:
            raise ValueError(
                f"expand*d_model must be divisible by head_dim, got "
                f"{intermediate_size} and head_dim={self.head_dim}"
            )
        self.num_heads = intermediate_size // self.head_dim
        if self.num_heads % self.n_groups != 0:
            raise ValueError(
                f"num_heads must be divisible by n_groups, got "
                f"{self.num_heads} and n_groups={self.n_groups}"
            )


class Mamba2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, bias: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.transpose(1, 2)  
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        x = x.transpose(1, 2) 
        return x


class Mamba2Mixer(nn.Module):
    def __init__(self, cfg: Mamba2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.d_model
        self.intermediate_size = cfg.expand * cfg.d_model
        self.ssm_state_size = cfg.d_state
        self.conv_kernel_size = cfg.d_conv
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.n_groups = cfg.n_groups
        self.time_step_limit = cfg.time_step_limit
        self.time_step_min = cfg.time_step_min
        self.time_step_max = cfg.time_step_max
        self.time_step_floor = cfg.time_step_floor
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=cfg.bias,
        )
        self.conv1d = CausalDepthwiseConv1d(
            channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            bias=cfg.conv_bias,
        )
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.D = nn.Parameter(torch.empty(self.num_heads))
        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            eps=cfg.layer_norm_eps,
        )
        self.out_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=cfg.bias,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.init_mamba2_weights()

    @torch.no_grad()
    def init_mamba2_weights(self) -> None:
        A = torch.arange(1, self.num_heads + 1, device=self.A_log.device, dtype=torch.float32)
        self.A_log.copy_(torch.log(A))
        self.D.fill_(1.0)
        dt = torch.exp(
            torch.rand(self.num_heads, device=self.dt_bias.device, dtype=torch.float32)
            * (math.log(self.time_step_max) - math.log(self.time_step_min))
            + math.log(self.time_step_min)
        ).clamp(min=self.time_step_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.copy_(inv_dt)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        hidden_states = torch.nan_to_num(hidden_states)
        projected_states = self.in_proj(hidden_states)
        gate, hidden_states_B_C, dt = torch.split(
            projected_states,
            [self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )
        hidden_states_B_C = self.conv1d(hidden_states_B_C)
        hidden_states_B_C = F.silu(hidden_states_B_C)
        hidden_states_ssm, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
        x = hidden_states_ssm.reshape(batch_size, seq_len, self.num_heads, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).float()
        repeat_factor = self.num_heads // self.n_groups
        B = B.repeat_interleave(repeat_factor, dim=2)
        C = C.repeat_interleave(repeat_factor, dim=2)
        A = -torch.exp(self.A_log.float())                      
        D = self.D.float()                                          
        state = torch.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            self.ssm_state_size,
            device=x.device,
            dtype=torch.float32,
        )
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]                                        
            dt_t = dt[:, t].unsqueeze(-1).expand(-1, -1, self.head_dim) 
            B_t = B[:, t]                                            
            C_t = C[:, t]                                           
            dA = torch.exp(dt_t[..., None] * A.view(1, self.num_heads, 1, 1))
            dB = dt_t[..., None] * B_t[:, :, None, :]
            state = state * dA + dB * x_t[..., None]
            y_t = torch.einsum("bhpn,bhn->bhp", state, C_t) + x_t * D.view(1, self.num_heads, 1)
            outputs.append(y_t.reshape(batch_size, -1))
        y = torch.stack(outputs, dim=1)                               
        y = self.norm(y, gate)
        y = self.out_proj(y.to(dtype))
        y = self.dropout(y)
        y = torch.nan_to_num(y)
        return y


class Mamba2Block(nn.Module):
    def __init__(self, cfg: Mamba2Config) -> None:
        super().__init__()
        self.residual_in_fp32 = cfg.residual_in_fp32
        self.norm = Mamba2RMSNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.mixer = Mamba2Mixer(cfg)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Mamba2Backbone(nn.Module):
    def __init__(self, cfg: Mamba2Config, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Mamba2Block(cfg) for _ in range(num_blocks)])
        self.out_norm = Mamba2RMSNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        x = torch.nan_to_num(x)
        return x