from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)


def get_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def detach_to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
    }


def format_experiment_name(layers: List[int]) -> str:
    if len(layers) == 1:
        return f"layer_{int(layers[0]):02d}"
    layer_str = "-".join(f"{int(x):02d}" for x in layers)
    return f"fusion_{layer_str}"


def build_output_dirs(
    outputs_root: str | Path,
    model_name: str,
    layers: List[int],
) -> Dict[str, Path]:
    outputs_root = Path(outputs_root)
    exp_name = format_experiment_name(layers)
    model_dir = ensure_dir(outputs_root / model_name)
    exp_dir = ensure_dir(model_dir / exp_name)

    dirs = {
        "model_dir": model_dir,
        "experiment_dir": exp_dir,
        "checkpoints_dir": ensure_dir(exp_dir / "checkpoints"),
        "plots_dir": ensure_dir(exp_dir / "plots"),
        "logs_dir": ensure_dir(exp_dir / "logs"),
        "predictions_dir": ensure_dir(exp_dir / "predictions"),
    }
    return dirs


def save_training_history(history: Dict[str, List[float]], path: str | Path) -> None:
    save_json(history, path)


def plot_training_curves(history: Dict[str, List[float]], save_dir: str | Path) -> None:
    save_dir = ensure_dir(save_dir)

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    if "train_loss" in history and "test_loss" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_loss"], label="train_loss")
        plt.plot(epochs, history["test_loss"], label="test_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train/Test Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "loss_curve.png", dpi=200)
        plt.close()

    if "train_mean_ccc" in history and "test_mean_ccc" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_mean_ccc"], label="train_mean_ccc")
        plt.plot(epochs, history["test_mean_ccc"], label="test_mean_ccc")
        plt.xlabel("Epoch")
        plt.ylabel("CCC")
        plt.title("Train/Test Mean CCC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "mean_ccc_curve.png", dpi=200)
        plt.close()

    trait_names = ["O", "C", "E", "A", "N"]
    for split in ["train", "test"]:
        keys = [f"{split}_ccc_{t}" for t in trait_names]
        if all(k in history for k in keys):
            plt.figure(figsize=(8, 5))
            for k in keys:
                plt.plot(epochs, history[k], label=k)
            plt.xlabel("Epoch")
            plt.ylabel("CCC")
            plt.title(f"{split.capitalize()} CCC by Trait")
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_dir / f"{split}_ccc_traits.png", dpi=200)
            plt.close()