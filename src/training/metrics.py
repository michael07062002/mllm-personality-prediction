from __future__ import annotations
from typing import Dict, List
import torch


BIG5_NAMES = ["O", "C", "E", "A", "N"]
VA_NAMES = ["valence", "arousal"]
TRAIT_NAMES = BIG5_NAMES


def infer_target_names(num_outputs: int) -> List[str]:
    if num_outputs == 5:
        return BIG5_NAMES
    if num_outputs == 2:
        return VA_NAMES
    return [f"target_{i}" for i in range(num_outputs)]


def concordance_correlation_coefficient(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    var_true = torch.var(y_true, unbiased=False)
    var_pred = torch.var(y_pred, unbiased=False)
    cov = torch.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2.0 * cov) / (
        var_true + var_pred + (mean_true - mean_pred) ** 2 + eps
    )
    return ccc


def ccc_per_target(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    target_names: List[str] | None = None,
) -> Dict[str, float]:
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got {y_true.shape=} and {y_pred.shape=}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape=} vs {y_pred.shape=}")
    num_outputs = int(y_true.shape[1])
    target_names = target_names or infer_target_names(num_outputs)
    if num_outputs != len(target_names):
        raise ValueError(
            f"Expected {len(target_names)} targets, got shape {y_true.shape}"
        )
    out: Dict[str, float] = {}
    for i, name in enumerate(target_names):
        out[f"ccc_{name}"] = float(
            concordance_correlation_coefficient(y_true[:, i], y_pred[:, i]).item()
        )
    return out


def mean_ccc(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    target_names: List[str] | None = None,
) -> float:
    metrics = ccc_per_target(y_true, y_pred, target_names=target_names)
    values = list(metrics.values())
    return float(sum(values) / len(values))


def build_ccc_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    target_names: List[str] | None = None,
) -> Dict[str, float]:
    metrics = ccc_per_target(y_true, y_pred, target_names=target_names)
    metrics["mean_ccc"] = float(sum(metrics.values()) / len(metrics))
    return metrics


def ccc_per_trait(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    trait_names: List[str] | None = None,
) -> Dict[str, float]:
    return ccc_per_target(y_true, y_pred, target_names=trait_names)