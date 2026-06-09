from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.extraction.annotations import AffWild2VAAnnotations, FirstImpressionsAnnotations
from src.extraction.defaults import DATASET_DEFAULTS, MODEL_SPECS, PROMPTS
from src.extraction.dlsp import select_top_layers
from src.extraction.extractor import GenericVLHiProbeExtractor
from src.extraction.paths import resolve_dataset_paths
from src.extraction.video import PaperVideoSegmenter, VideoFileMap


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DLSP config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    return data


def resolve_model_id(model_name: str) -> str:
    model_name = str(model_name).strip()
    if model_name not in MODEL_SPECS:
        raise KeyError(
            f"Unknown model name: {model_name}. "
            f"Available: {list(MODEL_SPECS.keys())}"
        )
    return MODEL_SPECS[model_name]


def get_required_name(cfg: Dict[str, Any], key: str) -> str:
    value = cfg.get(key)
    if not value:
        raise ValueError(f"Missing required config field: {key}")
    return str(value).strip()


def build_subset(dataset_name: str, paths: Dict[str, Path]):
    defaults = DATASET_DEFAULTS[dataset_name]
    sampling = defaults["sampling"]
    q = float(sampling["quantile_q"])
    n_total = int(sampling["n_total"])
    seed = int(sampling["seed"])
    if dataset_name == "first_impressions":
        ann = FirstImpressionsAnnotations(paths["annot_path"])
        ann.load()
        ann.build_dataframe()
        return ann.pick_subset(
            q=q,
            n_total=n_total,
            seed=seed,
            unique_base=bool(sampling["unique_base"]),
            out_csv=None,
        )
    if dataset_name == "affwild2_va":
        annotation_cfg = defaults["annotation"]
        ann = AffWild2VAAnnotations(
            annot_root=paths["annot_root"],
            group_target=str(annotation_cfg["group_target"]),
            label_scale=str(annotation_cfg["label_scale"]),
        )
        ann.build_dataframe()
        return ann.pick_subset(
            q=q,
            n_total=n_total,
            seed=seed,
            split_filter=annotation_cfg["subset_splits"],
            out_csv=None,
        )
    raise ValueError(f"Unknown dataset name: {dataset_name}")


def run_dlsp_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_name = get_required_name(cfg, "dataset")
    model_name = get_required_name(cfg, "model")
    device = str(cfg.get("device", "auto"))
    if dataset_name not in DATASET_DEFAULTS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_DEFAULTS.keys())}"
        )
    paths = resolve_dataset_paths(dataset_name)
    model_id = resolve_model_id(model_name)
    defaults = DATASET_DEFAULTS[dataset_name]
    segmentation = defaults["segmentation"]
    dlsp = defaults["dlsp"]
    experiment_root = paths["experiment_root"]
    out_json = experiment_root / model_name / "analysis" / "top_layers.json"
    subset = build_subset(
        dataset_name=dataset_name,
        paths=paths,
    )
    video_map = VideoFileMap(paths["video_root"])
    video_map.build()
    segmenter = PaperVideoSegmenter(
        segment_len=int(segmentation["segment_len"]),
        k_frames=int(segmentation["k_frames"]),
        resize_short_side=segmentation["resize_short_side"],
    )
    extractor = GenericVLHiProbeExtractor(
        model_id=model_id,
        device=device,
    )
    extraction_result = extractor.extract_dataset(
        df=subset,
        file_map=video_map,
        segmenter=segmenter,
        prompt=PROMPTS[dataset_name],
        max_segments=segmentation["max_segments"],
    )
    result = select_top_layers(
        X=extraction_result["X"],
        g=extraction_result["g"],
        out_json=out_json,
        bins=int(dlsp["bins"]),
        eps=float(dlsp["eps"]),
        entropy_topk=int(dlsp["entropy_topk"]),
        topk_layers=int(dlsp["topk_layers"]),
        filter_bad=bool(dlsp["filter_bad"]),
    )
    return {
        "best_layer": result["best_layer"],
        "top_layers": result["top_layers"],
        "out_json": str(out_json),
    }