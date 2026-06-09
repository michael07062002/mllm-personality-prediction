from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


BIG5_COLUMNS = ["O", "C", "E", "A", "N"]
VA_COLUMNS = ["valence", "arousal"]

TRAIT_COLUMNS = BIG5_COLUMNS


def infer_target_columns(df: pd.DataFrame) -> List[str]:
    if all(c in df.columns for c in BIG5_COLUMNS):
        return BIG5_COLUMNS
    if all(c in df.columns for c in VA_COLUMNS):
        return VA_COLUMNS
    raise ValueError(
        "Could not infer target columns from index.csv.\n"
        f"Columns: {list(df.columns)}\n"
        f"Expected either {BIG5_COLUMNS} or {VA_COLUMNS}."
    )


def load_index(index_csv: str | Path) -> pd.DataFrame:
    index_csv = Path(index_csv)
    if not index_csv.exists():
        raise FileNotFoundError(f"index.csv not found: {index_csv}")
    df = pd.read_csv(index_csv)
    if len(df) == 0:
        raise ValueError(f"index.csv is empty: {index_csv}")
    if "video_id" not in df.columns:
        raise ValueError(f"index.csv is missing required column: video_id")
    target_columns = infer_target_columns(df)
    df = df.copy()
    df["video_id"] = df["video_id"].astype(str)
    for col in target_columns:
        df[col] = df[col].astype(np.float32)
    return df.reset_index(drop=True)


def _find_layer_dir(split_dir: Path, layer: int) -> Path:
    candidates = [
        split_dir / f"layer_{int(layer):02d}",
        split_dir / f"layer_{int(layer)}",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(
        "Cannot find layer directory. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


def build_npz_path_from_video_id(index_csv: str | Path, layer: int, video_id: str) -> Path:
    index_csv = Path(index_csv)
    split_dir = index_csv.parent
    layer_dir = _find_layer_dir(split_dir, int(layer))
    stem = Path(str(video_id)).stem
    return layer_dir / f"{stem}.npz"


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
    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    video_id: str
    layer: int


@dataclass
class MultiLayerSample:
    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    video_id: str
    layers: List[int]


class SingleLayerFeatureDataset(Dataset):
    def __init__(
        self,
        index_csv: str | Path,
        layer: int,
    ) -> None:
        super().__init__()
        self.index_csv = Path(index_csv)
        self.layer = int(layer)
        df = load_index(self.index_csv)
        self.target_columns = infer_target_columns(df)
        valid_rows = []
        for _, row in df.iterrows():
            npz_path = build_npz_path_from_video_id(
                index_csv=self.index_csv,
                layer=self.layer,
                video_id=row["video_id"],
            )
            if npz_path.exists():
                valid_rows.append(row)
        if len(valid_rows) == 0:
            raise ValueError(
                f"No valid samples found for layer={self.layer} "
                f"using features under {self.index_csv.parent}"
            )
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        npz_path = build_npz_path_from_video_id(
            index_csv=self.index_csv,
            layer=self.layer,
            video_id=row["video_id"],
        )
        x = load_npz_x(npz_path)
        s = x.shape[0]
        y = row[self.target_columns].to_numpy(dtype=np.float32)
        mask = np.ones(s, dtype=np.bool_)
        return {
            "x": torch.from_numpy(x),              
            "y": torch.from_numpy(y),          
            "mask": torch.from_numpy(mask),    
            "video_id": str(row["video_id"]),
            "layer": self.layer,
            "target_names": list(self.target_columns),
        }


class MultiLayerFeatureDataset(Dataset):
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
        self.target_columns = infer_target_columns(df)
        samples: List[Dict] = []
        for _, row in df.iterrows():
            video_id = str(row["video_id"])
            paths = [
                str(build_npz_path_from_video_id(self.index_csv, layer, video_id))
                for layer in self.layers
            ]
            if not all(Path(p).exists() for p in paths):
                continue
            samples.append(
                {
                    "video_id": video_id,
                    "paths": paths,
                    "y": row[self.target_columns].to_numpy(dtype=np.float32),
                    "layers": list(self.layers),
                    "target_names": list(self.target_columns),
                }
            )

        if len(samples) == 0:
            raise ValueError(
                f"No complete multi-layer samples found for layers={self.layers} "
                f"under {self.index_csv.parent}"
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

        x = np.stack(xs, axis=0).astype(np.float32) 
        s = x.shape[1]
        mask = np.ones(s, dtype=np.bool_)

        return {
            "x": torch.from_numpy(x),              
            "y": torch.from_numpy(sample["y"]),      
            "mask": torch.from_numpy(mask),      
            "video_id": sample["video_id"],
            "layers": sample["layers"],
            "target_names": sample["target_names"],
        }


def pad_single_layer_batch(batch: List[Dict]) -> Dict:
    if len(batch) == 0:
        raise ValueError("Empty batch")
    bsz = len(batch)
    max_s = max(item["x"].shape[0] for item in batch)
    d = batch[0]["x"].shape[1]
    y_dim = batch[0]["y"].shape[0]
    x_pad = torch.zeros((bsz, max_s, d), dtype=torch.float32)
    mask_pad = torch.zeros((bsz, max_s), dtype=torch.bool)
    y = torch.zeros((bsz, y_dim), dtype=torch.float32)
    video_ids: List[str] = []
    layers: List[int] = []
    target_names = list(batch[0].get("target_names", []))
    for i, item in enumerate(batch):
        s = item["x"].shape[0]
        x_pad[i, :s] = item["x"]
        mask_pad[i, :s] = item["mask"]
        y[i] = item["y"]
        video_ids.append(item["video_id"])
        layers.append(int(item["layer"]))

    return {
        "x": x_pad,                
        "mask": mask_pad,            
        "y": y,                       
        "video_id": video_ids,
        "layer": layers,
        "target_names": target_names,
    }


def pad_multi_layer_batch(batch: List[Dict]) -> Dict:
    if len(batch) == 0:
        raise ValueError("Empty batch")
    bsz = len(batch)
    l = batch[0]["x"].shape[0]
    max_s = max(item["x"].shape[1] for item in batch)
    d = batch[0]["x"].shape[2]
    y_dim = batch[0]["y"].shape[0]
    x_pad = torch.zeros((bsz, l, max_s, d), dtype=torch.float32)
    mask_pad = torch.zeros((bsz, max_s), dtype=torch.bool)
    y = torch.zeros((bsz, y_dim), dtype=torch.float32)
    video_ids: List[str] = []
    layers = batch[0]["layers"]
    target_names = list(batch[0].get("target_names", []))
    for i, item in enumerate(batch):
        s = item["x"].shape[1]
        x_pad[i, :, :s] = item["x"]
        mask_pad[i, :s] = item["mask"]
        y[i] = item["y"]
        video_ids.append(item["video_id"])

    return {
        "x": x_pad,                 
        "mask": mask_pad,         
        "y": y,                   
        "video_id": video_ids,
        "layers": layers,
        "target_names": target_names,
    }


def build_single_layer_dataset(index_csv: str | Path, layer: int) -> SingleLayerFeatureDataset:
    return SingleLayerFeatureDataset(index_csv=index_csv, layer=layer)


def build_multi_layer_dataset(index_csv: str | Path, layers: Sequence[int]) -> MultiLayerFeatureDataset:
    return MultiLayerFeatureDataset(index_csv=index_csv, layers=layers)