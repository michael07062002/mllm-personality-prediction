from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(preds, targets)


def ccc_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_true = y_true.float()
    y_pred = y_pred.float()
    mean_true = torch.mean(y_true, dim=0)
    mean_pred = torch.mean(y_pred, dim=0)
    var_true = torch.var(y_true, dim=0, unbiased=False)
    var_pred = torch.var(y_pred, dim=0, unbiased=False)
    cov = torch.mean((y_true - mean_true) * (y_pred - mean_pred), dim=0)
    ccc = (2.0 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + eps)
    return ccc


def ccc_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    ccc = ccc_torch(targets, preds)
    return 1.0 - ccc.mean()


def combined_loss(preds: torch.Tensor, targets: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
    return mse_loss(preds, targets) + alpha * ccc_loss(preds, targets)


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.2) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return combined_loss(preds, targets, alpha=self.alpha)


