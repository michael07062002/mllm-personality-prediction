from __future__ import annotations

from pathlib import Path
from typing import Dict


def get_project_root() -> Path:
    current = Path.cwd().resolve()
    for path in [current, *current.parents]:
        if (path / "src").exists() and (path / "config").exists():
            return path
    raise RuntimeError(
        "Could not find project root. "
        "Run scripts from repository root or keep src/ and config/ in the project."
    )


def get_data_root() -> Path:
    project_root = get_project_root()
    data_root = project_root / "data"

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_root}\n"
            "Expected all datasets under repo/data/."
        )

    return data_root


def _check_dir(path: Path, name: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _check_file(path: Path, name: str) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _resolve_first_impressions() -> Dict[str, Path]:
    data_root = get_data_root()

    dataset_root = data_root / "FirstImpressionsV2"
    annotations_root = dataset_root / "annotations"

    train_video_root = dataset_root / "data" / "train"
    test_video_root = dataset_root / "data" / "test"

    annot_train_path = annotations_root / "annotation_training.pkl"
    annot_test_path = annotations_root / "annotation_test.pkl"

    _check_dir(dataset_root, "FirstImpressionsV2 dataset root")
    _check_dir(annotations_root, "FirstImpressionsV2 annotations directory")
    _check_dir(train_video_root, "FirstImpressionsV2 train video directory")
    _check_dir(test_video_root, "FirstImpressionsV2 test video directory")

    _check_file(annot_train_path, "FirstImpressionsV2 train annotation")
    _check_file(annot_test_path, "FirstImpressionsV2 test annotation")

    return {
        "annot_path": annot_train_path,
        "video_root": train_video_root,
        "annot_train_path": annot_train_path,
        "annot_test_path": annot_test_path,
        "train_video_root": train_video_root,
        "test_video_root": test_video_root,
        "experiment_root": Path("experiments") / "first_impressions_dlsp",
    }


def _resolve_affwild2_va() -> Dict[str, Path]:
    data_root = get_data_root()
    dataset_root = data_root / "AffWild2_CVPR-26"
    video_root = dataset_root / "videos"
    annot_root = dataset_root / "Annotations" / "VA_Estimation_Challenge"
    train_annot_root = annot_root / "Train_Set"
    val_annot_root = annot_root / "Validation_Set"
    _check_dir(dataset_root, "Aff-Wild2 dataset root")
    _check_dir(video_root, "Aff-Wild2 video directory")
    _check_dir(annot_root, "Aff-Wild2 VA annotation directory")
    _check_dir(train_annot_root, "Aff-Wild2 train annotation directory")
    _check_dir(val_annot_root, "Aff-Wild2 validation annotation directory")
    return {
        "annot_root": annot_root,
        "video_root": video_root,
        "experiment_root": Path("experiments") / "affwild2_va_video_dlsp",
    }


def resolve_dataset_paths(dataset_name: str) -> Dict[str, Path]:
    dataset_name = str(dataset_name).strip().lower()
    if dataset_name == "first_impressions":
        return _resolve_first_impressions()
    if dataset_name == "affwild2_va":
        return _resolve_affwild2_va()
    raise ValueError(
        f"Unknown dataset name: {dataset_name}. "
        "Expected first_impressions or affwild2_va."
    )