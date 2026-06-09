from __future__ import annotations

import argparse
from pathlib import Path

from src.features.pipeline import load_yaml, run_feature_pipeline


def main(config_path: str = "config/features.yaml") -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Path to feature extraction config. Default: config/features.yaml",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["first_impressions", "affwild2_va"],
        help="Override dataset from config.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "qwen25_vl_3b",
            "qwen3_vl_2b",
            "internvl3_2b",
            "smolvlm_256m",
            "smolvlm_500m",
        ],
        help="Override model from config.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "mps", "cuda", "cpu"],
        help="Override device from config.",
    )

    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help=(
            "Override layers. Use comma-separated list, e.g. 21,19,18,36. "
            "Use auto to read analysis/top_layers.json."
        ),
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    if args.dataset is not None:
        cfg["dataset"] = args.dataset

    if args.model is not None:
        cfg["model"] = args.model

    if args.device is not None:
        cfg["device"] = args.device

    if args.layers is not None:
        if args.layers.strip().lower() == "auto":
            cfg["layers"] = "auto"
        else:
            cfg["layers"] = [
                int(x.strip())
                for x in args.layers.split(",")
                if x.strip()
            ]

    result = run_feature_pipeline(cfg)

    print("=" * 80)
    print("FEATURE EXTRACTION DONE")
    print(f"Dataset: {result['dataset']}")
    print(f"Model:   {result['model']}")
    print(f"Layers:  {result['layers']}")
    print(f"Output:  {result['output_root']}")
    print("=" * 80)


if __name__ == "__main__":
    main()