# src/dataset.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TRAIT_COLUMNS = ["O", "C", "E", "A", "N"]


def _normalize_layer_name(layer: int) -> str:
    return f"layer_{int(layer):02d}"


def _to_abs_path(index_csv: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (index_csv.parent / path).resolve()


def _find_path_column(df: pd.DataFrame) -> str:
    candidates = [
        "file_path",
        "path",
        "npz_path",
        "feature_path",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"index.csv must contain one of path columns: {candidates}. "
        f"Found columns: {list(df.columns)}"
    )


def _validate_required_columns(df: pd.DataFrame) -> None:
    required = {"video_id", "layer", *TRAIT_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"index.csv is missing required columns: {sorted(missing)}")


def load_index(index_csv: str | Path) -> pd.DataFrame:
    index_csv = Path(index_csv)
    if not index_csv.exists():
        raise FileNotFoundError(f"index.csv not found: {index_csv}")

    df = pd.read_csv(index_csv)
    if len(df) == 0:
        raise ValueError(f"index.csv is empty: {index_csv}")

    _validate_required_columns(df)

    path_col = _find_path_column(df)
    df = df.copy()
    df["layer"] = df["layer"].astype(int)
    df["file_path"] = df[path_col].map(lambda x: str(_to_abs_path(index_csv, str(x))))
    df["video_id"] = df["video_id"].astype(str)

    for col in TRAIT_COLUMNS:
        df[col] = df[col].astype(np.float32)

    return df


def load_npz_x(npz_path: str | Path) -> np.ndarray:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Feature file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if "X" not in data:
            raise KeyError(f"'X' not found in {npz_path}")
        x = data["X"]

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected X with shape (S, D), got {x.shape} in {npz_path}")
    return x


@dataclass
class SingleLayerSample:
    x: torch.Tensor        # (S, D)
    y: torch.Tensor        # (5,)
    mask: torch.Tensor     # (S,)
    video_id: str
    layer: int


@dataclass
class MultiLayerSample:
    x: torch.Tensor        # (L, S, D)
    y: torch.Tensor        # (5,)
    mask: torch.Tensor     # (S,)
    video_id: str
    layers: List[int]


class SingleLayerFeatureDataset(Dataset):
    """
    One sample = one video from one selected layer.
    Returns:
        x:    (S, D)
        y:    (5,) -> [O, C, E, A, N]
        mask: (S,)
    """

    def __init__(
        self,
        index_csv: str | Path,
        layer: int,
    ) -> None:
        super().__init__()
        self.index_csv = Path(index_csv)
        self.layer = int(layer)

        df = load_index(self.index_csv)
        df = df[df["layer"] == self.layer].copy().reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No rows found for layer={self.layer} in {self.index_csv}")

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        x = load_npz_x(row["file_path"])
        s = x.shape[0]

        y = row[TRAIT_COLUMNS].to_numpy(dtype=np.float32)
        mask = np.ones(s, dtype=np.bool_)

        return {
            "x": torch.from_numpy(x),                     # (S, D)
            "y": torch.from_numpy(y),                     # (5,)
            "mask": torch.from_numpy(mask),               # (S,)
            "video_id": str(row["video_id"]),
            "layer": int(row["layer"]),
        }


class MultiLayerFeatureDataset(Dataset):
    """
    One sample = one video with multiple selected layers.
    Uses the shared split-level index.csv and groups rows by video_id.

    Returns:
        x:    (L, S, D)
        y:    (5,)
        mask: (S,)
    """

    def __init__(
        self,
        index_csv: str | Path,
        layers: Sequence[int],
    ) -> None:
        super().__init__()
        self.index_csv = Path(index_csv)
        self.layers = [int(x) for x in layers]

        if len(self.layers) == 0:
            raise ValueError("layers must be non-empty")

        df = load_index(self.index_csv)
        df = df[df["layer"].isin(self.layers)].copy()

        if len(df) == 0:
            raise ValueError(f"No rows found for layers={self.layers} in {self.index_csv}")

        grouped: Dict[str, Dict[int, Dict]] = {}
        for _, row in df.iterrows():
            video_id = str(row["video_id"])
            layer = int(row["layer"])
            grouped.setdefault(video_id, {})
            grouped[video_id][layer] = {
                "file_path": row["file_path"],
                "y": row[TRAIT_COLUMNS].to_numpy(dtype=np.float32),
            }

        samples: List[Dict] = []
        for video_id, layer_map in grouped.items():
            if not all(layer in layer_map for layer in self.layers):
                continue

            samples.append(
                {
                    "video_id": video_id,
                    "paths": [layer_map[layer]["file_path"] for layer in self.layers],
                    "y": layer_map[self.layers[0]]["y"],
                    "layers": list(self.layers),
                }
            )

        if len(samples) == 0:
            raise ValueError(
                f"No complete multi-layer samples found for layers={self.layers} "
                f"in {self.index_csv}"
            )

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        xs = [load_npz_x(path) for path in sample["paths"]]
        seg_lens = [x.shape[0] for x in xs]
        feat_dims = [x.shape[1] for x in xs]

        if len(set(seg_lens)) != 1:
            raise ValueError(
                f"All layers for one video must have same S. "
                f"Got seg_lens={seg_lens} for video_id={sample['video_id']}"
            )

        if len(set(feat_dims)) != 1:
            raise ValueError(
                f"All layers for one video must have same D. "
                f"Got feat_dims={feat_dims} for video_id={sample['video_id']}"
            )

        x = np.stack(xs, axis=0).astype(np.float32)   # (L, S, D)
        s = x.shape[1]
        mask = np.ones(s, dtype=np.bool_)

        return {
            "x": torch.from_numpy(x),                      # (L, S, D)
            "y": torch.from_numpy(sample["y"]),            # (5,)
            "mask": torch.from_numpy(mask),                # (S,)
            "video_id": sample["video_id"],
            "layers": sample["layers"],
        }


def pad_single_layer_batch(batch: List[Dict]) -> Dict:
    if len(batch) == 0:
        raise ValueError("Empty batch")

    bsz = len(batch)
    max_s = max(item["x"].shape[0] for item in batch)
    d = batch[0]["x"].shape[1]

    x_pad = torch.zeros((bsz, max_s, d), dtype=torch.float32)
    mask_pad = torch.zeros((bsz, max_s), dtype=torch.bool)
    y = torch.zeros((bsz, 5), dtype=torch.float32)

    video_ids: List[str] = []
    layers: List[int] = []

    for i, item in enumerate(batch):
        s = item["x"].shape[0]
        x_pad[i, :s] = item["x"]
        mask_pad[i, :s] = item["mask"]
        y[i] = item["y"]
        video_ids.append(item["video_id"])
        layers.append(int(item["layer"]))

    return {
        "x": x_pad,               # (B, S, D)
        "mask": mask_pad,         # (B, S)
        "y": y,                   # (B, 5)
        "video_id": video_ids,
        "layer": layers,
    }


def pad_multi_layer_batch(batch: List[Dict]) -> Dict:
    if len(batch) == 0:
        raise ValueError("Empty batch")

    bsz = len(batch)
    l = batch[0]["x"].shape[0]
    max_s = max(item["x"].shape[1] for item in batch)
    d = batch[0]["x"].shape[2]

    x_pad = torch.zeros((bsz, l, max_s, d), dtype=torch.float32)
    mask_pad = torch.zeros((bsz, max_s), dtype=torch.bool)
    y = torch.zeros((bsz, 5), dtype=torch.float32)

    video_ids: List[str] = []
    layers = batch[0]["layers"]

    for i, item in enumerate(batch):
        s = item["x"].shape[1]
        x_pad[i, :, :s] = item["x"]
        mask_pad[i, :s] = item["mask"]
        y[i] = item["y"]
        video_ids.append(item["video_id"])

    return {
        "x": x_pad,               # (B, L, S, D)
        "mask": mask_pad,         # (B, S)
        "y": y,                   # (B, 5)
        "video_id": video_ids,
        "layers": layers,
    }


def build_single_layer_dataset(index_csv: str | Path, layer: int) -> SingleLayerFeatureDataset:
    return SingleLayerFeatureDataset(index_csv=index_csv, layer=layer)


def build_multi_layer_dataset(index_csv: str | Path, layers: Sequence[int]) -> MultiLayerFeatureDataset:
    return MultiLayerFeatureDataset(index_csv=index_csv, layers=layers)