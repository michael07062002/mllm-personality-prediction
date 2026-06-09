from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.training.dataset import (
    build_multi_layer_dataset,
    build_single_layer_dataset,
    pad_multi_layer_batch,
    pad_single_layer_batch,
)
from src.training.model import build_multi_layer_model, build_single_layer_model
from src.training.trainer import Trainer
from src.training.utils import get_device, set_seed


DATASET_EXPERIMENT_ROOTS = {
    "first_impressions": Path("experiments") / "first_impressions_dlsp",
    "affwild2_va": Path("experiments") / "affwild2_va_video_dlsp",
}


DEFAULT_SEED = 42


DATASET_TEST_SPLITS = {
    "first_impressions": "test",
    "affwild2_va": "val",
}


BIG5_COLUMNS = ["O", "C", "E", "A", "N"]
VA_COLUMNS = ["valence", "arousal"]


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Train config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return data


def parse_model_name(model_name: str) -> Tuple[str, str]:
    """
    Expected:
        first_impressions/qwen25_vl_3b
        affwild2_va/internvl3_2b
    """
    parts = str(model_name).strip().split("/")

    if len(parts) != 2:
        raise ValueError(
            "data.model_name must have format '<dataset>/<model>', for example:\n"
            "  first_impressions/qwen25_vl_3b\n"
            "  affwild2_va/internvl3_2b"
        )

    dataset_name, backbone_name = parts[0].strip(), parts[1].strip()

    if dataset_name not in DATASET_EXPERIMENT_ROOTS:
        raise ValueError(
            f"Unknown dataset in data.model_name: {dataset_name}. "
            f"Available: {list(DATASET_EXPERIMENT_ROOTS.keys())}"
        )

    if not backbone_name:
        raise ValueError(f"Empty model name in data.model_name={model_name}")

    return dataset_name, backbone_name


def build_auto_run_name(layers: list[int]) -> str:
    if len(layers) == 1:
        return f"layer_{layers[0]}"

    return "fusion_" + "-".join(str(x) for x in layers)


def resolve_feature_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg["data"]

    model_name = str(data_cfg["model_name"]).strip()
    dataset_name, backbone_name = parse_model_name(model_name)

    experiment_root = DATASET_EXPERIMENT_ROOTS[dataset_name]
    feature_root = experiment_root / backbone_name / "layer_features"

    train_index_csv = feature_root / "train" / "index.csv"
    test_split = DATASET_TEST_SPLITS[dataset_name]
    test_index_csv = feature_root / test_split / "index.csv"

    if not train_index_csv.exists():
        raise FileNotFoundError(
            f"Train index not found: {train_index_csv}\n"
            "Run feature extraction first, for example:\n"
            f"  python run_features.py --dataset {dataset_name} --model {backbone_name}"
        )

    if not test_index_csv.exists():
        raise FileNotFoundError(
            f"Test/val index not found: {test_index_csv}\n"
            "Run feature extraction first, for example:\n"
            f"  python run_features.py --dataset {dataset_name} --model {backbone_name}"
        )

    data_cfg["dataset_name"] = dataset_name
    data_cfg["backbone_name"] = backbone_name
    data_cfg["feature_root"] = str(feature_root)
    data_cfg["train_index_csv"] = str(train_index_csv)
    data_cfg["test_index_csv"] = str(test_index_csv)
    data_cfg["test_split"] = test_split

    return cfg


def resolve_layers(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg["data"]

    if "layers" not in data_cfg:
        raise ValueError("Missing required config field: data.layers")

    layers = [int(x) for x in data_cfg["layers"]]

    if len(layers) == 0:
        raise ValueError("data.layers must be non-empty")

    data_cfg["layers"] = layers
    data_cfg["run_name"] = build_auto_run_name(layers)

    return cfg


def _find_layer_dir(split_dir: Path, layer: int) -> Path:
    candidates = [
        split_dir / f"layer_{int(layer):02d}",
        split_dir / f"layer_{int(layer)}",
    ]

    for path in candidates:
        if path.exists() and path.is_dir():
            return path

    raise FileNotFoundError(
        "Cannot find layer directory. Tried:\n"
        + "\n".join(str(path) for path in candidates)
    )


def infer_input_dim_from_features(index_csv: str | Path, layer: int) -> int:
    index_csv = Path(index_csv)
    split_dir = index_csv.parent

    layer_dir = _find_layer_dir(split_dir, layer)
    files = sorted(layer_dir.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in {layer_dir}")

    for path in files:
        try:
            with np.load(path, allow_pickle=True) as data:
                if "X" not in data:
                    continue

                X = data["X"]

            if X.ndim != 2:
                raise ValueError(
                    f"Expected X shape (S, D), got {X.shape} in {path}"
                )

            return int(X.shape[-1])

        except Exception as e:
            print(f"[warn] failed to read feature file {path}: {e}")

    raise RuntimeError(f"Could not infer input_dim from files in {layer_dir}")


def resolve_input_dim(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg["data"]
    layers = [int(x) for x in data_cfg["layers"]]

    input_dims = {}

    for layer in layers:
        dim = infer_input_dim_from_features(
            index_csv=data_cfg["train_index_csv"],
            layer=layer,
        )
        input_dims[layer] = dim

    unique_dims = sorted(set(input_dims.values()))

    if len(unique_dims) != 1:
        raise ValueError(
            "Selected layers have different feature dimensions: "
            f"{input_dims}. This model expects one shared input_dim."
        )

    data_cfg["input_dim"] = int(unique_dims[0])

    print(f"[auto] data.input_dim = {data_cfg['input_dim']} from layers {input_dims}")

    return cfg


def resolve_num_outputs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg["data"]
    index_csv = Path(data_cfg["train_index_csv"])

    if not index_csv.exists():
        raise FileNotFoundError(f"Cannot infer num_outputs: missing {index_csv}")

    df = pd.read_csv(index_csv)

    if all(col in df.columns for col in BIG5_COLUMNS):
        data_cfg["num_outputs"] = 5
        data_cfg["target_columns"] = BIG5_COLUMNS
        target_name = "Big Five"

    elif all(col in df.columns for col in VA_COLUMNS):
        data_cfg["num_outputs"] = 2
        data_cfg["target_columns"] = VA_COLUMNS
        target_name = "Aff-Wild2 VA"

    else:
        raise ValueError(
            "Could not infer num_outputs from index.csv columns.\n"
            f"index_csv: {index_csv}\n"
            f"columns: {list(df.columns)}\n"
            f"Expected either {BIG5_COLUMNS} or {VA_COLUMNS}."
        )

    print(f"[auto] data.num_outputs = {data_cfg['num_outputs']} ({target_name})")

    return cfg


def resolve_auto_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "data" not in cfg:
        raise ValueError("Missing required section: data")

    cfg = resolve_layers(cfg)
    cfg = resolve_feature_paths(cfg)
    cfg = resolve_input_dim(cfg)
    cfg = resolve_num_outputs(cfg)

    return cfg


def build_dataloaders(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    layers = [int(x) for x in data_cfg["layers"]]

    if len(layers) == 1:
        train_dataset = build_single_layer_dataset(
            index_csv=data_cfg["train_index_csv"],
            layer=layers[0],
        )
        test_dataset = build_single_layer_dataset(
            index_csv=data_cfg["test_index_csv"],
            layer=layers[0],
        )
        collate_fn = pad_single_layer_batch

    else:
        train_dataset = build_multi_layer_dataset(
            index_csv=data_cfg["train_index_csv"],
            layers=layers,
        )
        test_dataset = build_multi_layer_dataset(
            index_csv=data_cfg["test_index_csv"],
            layers=layers,
        )
        collate_fn = pad_multi_layer_batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader


def build_model(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    layers = [int(x) for x in data_cfg["layers"]]

    if len(layers) == 1:
        return build_single_layer_model(
            input_dim=int(data_cfg["input_dim"]),
            d_model=int(model_cfg["d_model"]),
            d_state=int(model_cfg["d_state"]),
            d_conv=int(model_cfg["d_conv"]),
            expand=int(model_cfg["expand"]),
            num_mamba_blocks=int(model_cfg["num_mamba_blocks"]),
            mlp_ratio=float(model_cfg["mlp_ratio"]),
            dropout=float(model_cfg["dropout"]),
            num_outputs=int(data_cfg["num_outputs"]),
        )

    return build_multi_layer_model(
        input_dim=int(data_cfg["input_dim"]),
        num_layers=len(layers),
        d_model=int(model_cfg["d_model"]),
        d_state=int(model_cfg["d_state"]),
        d_conv=int(model_cfg["d_conv"]),
        expand=int(model_cfg["expand"]),
        num_mamba_blocks=int(model_cfg["num_mamba_blocks"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        dropout=float(model_cfg["dropout"]),
        num_outputs=int(data_cfg["num_outputs"]),
        agf_hidden_dim=int(model_cfg["agf_hidden_dim"]),
    )


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    train_cfg = cfg["train"]

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    scheduler_cfg = cfg["scheduler"]

    if not bool(scheduler_cfg["use"]):
        return None

    name = str(scheduler_cfg["name"]).lower()

    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg["mode"]),
            factor=float(scheduler_cfg["factor"]),
            patience=int(scheduler_cfg["patience"]),
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_cfg['name']}")


def maybe_compile_model(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.nn.Module:
    use_compile = bool(cfg.get("compile", False))

    if not use_compile:
        return model

    if not hasattr(torch, "compile"):
        return model

    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"[warn] torch.compile failed, using eager mode: {e}")

    return model


def print_run_info(cfg: Dict[str, Any], device: torch.device) -> None:
    data_cfg = cfg["data"]
    layers = [int(x) for x in data_cfg["layers"]]

    mode = "single_layer" if len(layers) == 1 else "multi_layer_fusion"

    print("=" * 80)
    print("RUN CONFIG")
    print(f"model_name:       {data_cfg['model_name']}")
    print(f"dataset_name:     {data_cfg['dataset_name']}")
    print(f"backbone_name:    {data_cfg['backbone_name']}")
    print(f"run_name:         {data_cfg['run_name']}")
    print(f"mode:             {mode}")
    print(f"layers:           {layers}")
    print(f"feature_root:     {data_cfg['feature_root']}")
    print(f"train_index_csv:  {data_cfg['train_index_csv']}")
    print(f"test_index_csv:   {data_cfg['test_index_csv']}")
    print(f"input_dim:        {data_cfg['input_dim']}")
    print(f"num_outputs:      {data_cfg['num_outputs']}")
    print(f"target_columns:   {data_cfg['target_columns']}")
    print(f"device:           {device}")
    print(f"batch_size:       {cfg['train']['batch_size']}")
    print(f"num_epochs:       {cfg['train']['num_epochs']}")
    print(f"lr:               {cfg['train']['lr']}")
    print(f"outputs_root:     {cfg['outputs']['root']}")
    print("=" * 80)


def main(config_path: str = "config/train.yaml") -> None:
    cfg = load_config(config_path)
    cfg = resolve_auto_config(cfg)

    set_seed(DEFAULT_SEED)
    device = get_device(str(cfg["device"]))

    print_run_info(cfg, device)

    train_loader, test_loader = build_dataloaders(cfg)

    model = build_model(cfg)
    model = maybe_compile_model(cfg, model)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        outputs_root=cfg["outputs"]["root"],
        model_name=cfg["data"]["model_name"],
        layers=[int(x) for x in cfg["data"]["layers"]],
        max_grad_norm=float(cfg["train"]["max_grad_norm"]),
    )

    trainer.save_run_config(cfg)
    trainer.save_model_info(
        extra={
            "device": str(device),
            "dataset_name": cfg["data"]["dataset_name"],
            "backbone_name": cfg["data"]["backbone_name"],
            "feature_root": cfg["data"]["feature_root"],
            "train_index_csv": cfg["data"]["train_index_csv"],
            "test_index_csv": cfg["data"]["test_index_csv"],
            "train_size": len(train_loader.dataset),
            "test_size": len(test_loader.dataset),
            "run_name": cfg["data"]["run_name"],
            "input_dim": cfg["data"]["input_dim"],
            "num_outputs": cfg["data"]["num_outputs"],
            "target_columns": cfg["data"]["target_columns"],
        }
    )

    trainer.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=int(cfg["train"]["num_epochs"]),
        save_every_epoch_predictions=bool(cfg["train"]["save_every_epoch_predictions"]),
    )


if __name__ == "__main__":
    main()