from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.extraction.annotations import AffWild2VAAnnotations, FirstImpressionsAnnotations
from src.extraction.defaults import MODEL_SPECS, PROMPTS
from src.extraction.extractor import GenericVLHiProbeExtractor
from src.extraction.paths import resolve_dataset_paths
from src.extraction.video import PaperVideoSegmenter, VideoFileMap
from src.features.defaults import FEATURE_DEFAULTS


BIG5_COLUMNS = ["O", "C", "E", "A", "N"]
VA_COLUMNS = ["valence", "arousal"]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Feature config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return data


def get_required_name(cfg: Dict[str, Any], key: str) -> str:
    value = cfg.get(key)

    if not value:
        raise ValueError(f"Missing required config field: {key}")

    return str(value).strip()


def resolve_model_id(model_name: str) -> str:
    model_name = str(model_name).strip()

    if model_name not in MODEL_SPECS:
        raise KeyError(
            f"Unknown model name: {model_name}. "
            f"Available: {list(MODEL_SPECS.keys())}"
        )

    return MODEL_SPECS[model_name]


def read_layers_from_dlsp_json(dataset_name: str, model_name: str, paths: Dict[str, Path]) -> List[int]:
    top_layers_json = paths["experiment_root"] / model_name / "analysis" / "top_layers.json"

    if not top_layers_json.exists():
        raise FileNotFoundError(
            f"Could not find DLSP top layers: {top_layers_json}\n"
            "Run python run_dlsp.py first or pass layers manually."
        )

    with open(top_layers_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "top_layers" not in data:
        raise KeyError(f"'top_layers' not found in {top_layers_json}")

    return [int(x) for x in data["top_layers"]]


def resolve_layers(cfg: Dict[str, Any], dataset_name: str, model_name: str, paths: Dict[str, Path]) -> List[int]:
    layers = cfg.get("layers", "auto")

    if isinstance(layers, str):
        if layers.strip().lower() != "auto":
            raise ValueError("config field 'layers' must be 'auto' or a list of integers")

        return read_layers_from_dlsp_json(
            dataset_name=dataset_name,
            model_name=model_name,
            paths=paths,
        )

    if isinstance(layers, list):
        if len(layers) == 0:
            raise ValueError("layers list is empty")

        return [int(x) for x in layers]

    raise TypeError("config field 'layers' must be 'auto' or a list of integers")


def make_layer_feature_dirs(
    experiment_root: Path,
    model_name: str,
    split_name: str,
    selected_layers: List[int],
) -> Dict[str, Any]:
    base_dir = experiment_root / model_name / "layer_features" / split_name
    base_dir.mkdir(parents=True, exist_ok=True)

    layer_dirs = {}

    for layer in selected_layers:
        layer_dir = base_dir / f"layer_{int(layer):02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_dirs[int(layer)] = layer_dir

    return {
        "base_dir": base_dir,
        "layer_dirs": layer_dirs,
        "index_csv": base_dir / "index.csv",
        "meta_json": base_dir / "meta.json",
    }


def build_first_impressions_split_frames(paths: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    train_ann = FirstImpressionsAnnotations(paths["annot_train_path"])
    train_ann.load()
    train_df = train_ann.build_dataframe().copy()
    train_df["split"] = "train"

    test_ann = FirstImpressionsAnnotations(paths["annot_test_path"])
    test_ann.load()
    test_df = test_ann.build_dataframe().copy()
    test_df["split"] = "test"

    train_map = VideoFileMap(paths["train_video_root"])
    train_map.build()

    test_map = VideoFileMap(paths["test_video_root"])
    test_map.build()

    train_df = (
        train_df[train_df["video_id"].isin(train_map.file_map.keys())]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        test_df[test_df["video_id"].isin(test_map.file_map.keys())]
        .copy()
        .reset_index(drop=True)
    )

    return {
        "train": {
            "df": train_df,
            "file_map": train_map,
        },
        "test": {
            "df": test_df,
            "file_map": test_map,
        },
    }


def build_affwild2_split_frames(paths: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    ann = AffWild2VAAnnotations(
        annot_root=paths["annot_root"],
        group_target="valence",
        label_scale="zero_one",
    )

    df = ann.build_dataframe().copy()

    file_map = VideoFileMap(paths["video_root"])
    file_map.build()

    out = {}

    for split_name in ["train", "val"]:
        split_df = (
            df[df["split"] == split_name]
            .copy()
            .reset_index(drop=True)
        )

        split_df["resolved_path"] = split_df["video_id"].apply(lambda x: file_map.get(x))
        split_df = (
            split_df[split_df["resolved_path"].notna()]
            .copy()
            .drop(columns=["resolved_path"])
            .reset_index(drop=True)
        )

        out[split_name] = {
            "df": split_df,
            "file_map": file_map,
        }

    return out


def build_split_frames(dataset_name: str, paths: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    if dataset_name == "first_impressions":
        return build_first_impressions_split_frames(paths)

    if dataset_name == "affwild2_va":
        return build_affwild2_split_frames(paths)

    raise ValueError(f"Unknown dataset name: {dataset_name}")


def extract_selected_layers_per_segment(
    extractor: GenericVLHiProbeExtractor,
    frames,
    meta: Dict,
    prompt: str,
    selected_layers: List[int],
) -> Dict[int, np.ndarray]:
    full = extractor._extract_hidden_video_pooled_single(
        frames=frames,
        meta=meta,
        prompt=prompt,
    )  # (L, D)

    out = {}

    for layer in selected_layers:
        layer = int(layer)

        if layer < 0 or layer >= full.shape[0]:
            raise IndexError(
                f"Layer {layer} out of range for model with {full.shape[0]} layers."
            )

        out[layer] = full[layer].astype(np.float32)

    return out


def get_target_columns(df: pd.DataFrame) -> List[str]:
    if all(col in df.columns for col in BIG5_COLUMNS):
        return BIG5_COLUMNS

    if all(col in df.columns for col in VA_COLUMNS):
        return VA_COLUMNS

    raise ValueError(
        f"Cannot infer target columns. Columns: {list(df.columns)}"
    )


def build_index_row(row: pd.Series, split_name: str, path: str, n_segments: int) -> Dict[str, Any]:
    target_columns = get_target_columns(pd.DataFrame([row]))

    out = {
        "video_id": str(row["video_id"]),
        "path": path,
        "split": split_name,
        "P": float(row["P"]),
        "n_segments": int(n_segments),
    }

    for col in target_columns:
        out[col] = float(row[col])

    for optional_col in [
        "raw_valence",
        "raw_arousal",
        "n_frames",
        "n_valid_frames",
        "n_tracks",
        "base_id",
        "segment_id",
    ]:
        if optional_col in row.index:
            value = row[optional_col]

            if pd.isna(value):
                out[optional_col] = ""
            elif isinstance(value, (np.integer, int)):
                out[optional_col] = int(value)
            elif isinstance(value, (np.floating, float)):
                out[optional_col] = float(value)
            else:
                out[optional_col] = str(value)

    return out


def save_layer_npz(
    out_file: Path,
    X_layer: np.ndarray,
    row: pd.Series,
    layer: int,
) -> None:
    payload = {
        "X": X_layer.astype(np.float32),
        "video_id": str(row["video_id"]),
        "layer": np.int64(layer),
        "P": np.float32(row["P"]),
    }

    if all(col in row.index for col in BIG5_COLUMNS):
        for col in BIG5_COLUMNS:
            payload[col] = np.float32(row[col])

    if all(col in row.index for col in VA_COLUMNS):
        for col in VA_COLUMNS:
            payload[col] = np.float32(row[col])

    np.savez_compressed(out_file, **payload)


def run_split_feature_extraction(
    dataset_name: str,
    model_name: str,
    model_id: str,
    split_name: str,
    df: pd.DataFrame,
    file_map: VideoFileMap,
    segmenter: PaperVideoSegmenter,
    extractor: GenericVLHiProbeExtractor,
    selected_layers: List[int],
    experiment_root: Path,
    prompt: str,
    max_segments,
) -> Dict[str, Any]:
    dirs = make_layer_feature_dirs(
        experiment_root=experiment_root,
        model_name=model_name,
        split_name=split_name,
        selected_layers=selected_layers,
    )

    index_rows = []

    skipped_no_path = 0
    skipped_video = 0
    skipped_all_segments = 0
    skipped_segment_errors = 0

    print("=" * 80)
    print("RUN FEATURE EXTRACTION")
    print(f"dataset: {dataset_name}")
    print(f"model:   {model_name}")
    print(f"split:   {split_name}")
    print(f"layers:  {selected_layers}")
    print("=" * 80)

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc=f"{model_name}/{split_name}")
    ):
        video_id = str(row["video_id"])
        path = file_map.get(video_id)

        if not path:
            skipped_no_path += 1
            continue

        try:
            segments, meta = segmenter.load_segments(
                path,
                max_segments=max_segments,
            )
        except Exception as e:
            skipped_video += 1
            print(f"[warn] failed video {video_id}: {e}")

            if i % 10 == 0:
                gc.collect()
                extractor._empty_cache()

            continue

        per_layer_segment_features = {
            int(layer): []
            for layer in selected_layers
        }

        try:
            for frames in segments:
                try:
                    segment_output = extract_selected_layers_per_segment(
                        extractor=extractor,
                        frames=frames,
                        meta=meta,
                        prompt=prompt,
                        selected_layers=selected_layers,
                    )

                    for layer in selected_layers:
                        x = segment_output[int(layer)]

                        if np.isfinite(x).all():
                            per_layer_segment_features[int(layer)].append(x)
                        else:
                            skipped_segment_errors += 1

                except Exception as e:
                    skipped_segment_errors += 1
                    print(f"[warn] failed segment {video_id}: {e}")
                    continue

            n_valid_segments = min(
                len(per_layer_segment_features[int(layer)])
                for layer in selected_layers
            )

            if n_valid_segments <= 0:
                skipped_all_segments += 1
                continue

            for layer in selected_layers:
                layer = int(layer)
                feats = per_layer_segment_features[layer][:n_valid_segments]
                X_layer = np.stack(feats, axis=0).astype(np.float32)  # (S, D)

                out_file = dirs["layer_dirs"][layer] / f"{Path(video_id).stem}.npz"

                save_layer_npz(
                    out_file=out_file,
                    X_layer=X_layer,
                    row=row,
                    layer=layer,
                )

            index_rows.append(
                build_index_row(
                    row=row,
                    split_name=split_name,
                    path=path,
                    n_segments=n_valid_segments,
                )
            )

        finally:
            try:
                del segments
            except Exception:
                pass

            try:
                del per_layer_segment_features
            except Exception:
                pass

            if i % 10 == 0:
                gc.collect()
                extractor._empty_cache()

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(dirs["index_csv"], index=False)

    meta = {
        "dataset": dataset_name,
        "model_name": model_name,
        "model_id": model_id,
        "split": split_name,
        "selected_layers": [int(x) for x in selected_layers],
        "segment_len": int(segmenter.segment_len),
        "k_frames": int(segmenter.k_frames),
        "resize_short_side": segmenter.resize_short_side,
        "max_segments": max_segments,
        "n_videos_processed": int(len(index_df)),
        "skipped_no_path": int(skipped_no_path),
        "skipped_video": int(skipped_video),
        "skipped_all_segments": int(skipped_all_segments),
        "skipped_segment_errors": int(skipped_segment_errors),
        "index_csv": str(dirs["index_csv"]),
        "layer_dirs": {
            str(k): str(v)
            for k, v in dirs["layer_dirs"].items()
        },
        "note": (
            "Per-layer segment-level features. "
            "Each .npz stores X with shape (S, D) for one video and one selected layer."
        ),
    }

    with open(dirs["meta_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved index:", dirs["index_csv"])
    print("Saved meta: ", dirs["meta_json"])
    print("Processed:  ", len(index_df))

    return {
        "split": split_name,
        "index_csv": str(dirs["index_csv"]),
        "meta_json": str(dirs["meta_json"]),
        "n_videos_processed": int(len(index_df)),
    }


def run_feature_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_name = get_required_name(cfg, "dataset")
    model_name = get_required_name(cfg, "model")
    device = str(cfg.get("device", "auto"))

    if dataset_name not in FEATURE_DEFAULTS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(FEATURE_DEFAULTS.keys())}"
        )

    paths = resolve_dataset_paths(dataset_name)
    model_id = resolve_model_id(model_name)

    selected_layers = resolve_layers(
        cfg=cfg,
        dataset_name=dataset_name,
        model_name=model_name,
        paths=paths,
    )

    defaults = FEATURE_DEFAULTS[dataset_name]
    segmentation = defaults["segmentation"]

    split_frames = build_split_frames(
        dataset_name=dataset_name,
        paths=paths,
    )

    segmenter = PaperVideoSegmenter(
        segment_len=int(segmentation["segment_len"]),
        k_frames=int(segmentation["k_frames"]),
        resize_short_side=segmentation["resize_short_side"],
    )

    extractor = GenericVLHiProbeExtractor(
        model_id=model_id,
        device=device,
    )

    split_results = {}

    for split_name in defaults["splits"]:
        if split_name not in split_frames:
            raise KeyError(f"Missing split data for split={split_name}")

        split_result = run_split_feature_extraction(
            dataset_name=dataset_name,
            model_name=model_name,
            model_id=model_id,
            split_name=split_name,
            df=split_frames[split_name]["df"],
            file_map=split_frames[split_name]["file_map"],
            segmenter=segmenter,
            extractor=extractor,
            selected_layers=selected_layers,
            experiment_root=paths["experiment_root"],
            prompt=PROMPTS[dataset_name],
            max_segments=segmentation["max_segments"],
        )

        split_results[split_name] = split_result

    return {
        "dataset": dataset_name,
        "model": model_name,
        "model_id": model_id,
        "layers": selected_layers,
        "output_root": str(paths["experiment_root"] / model_name / "layer_features"),
        "splits": split_results,
    }