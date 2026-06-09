from __future__ import annotations

import argparse
from pathlib import Path

from src.extraction.pipeline import load_yaml, run_dlsp_pipeline


def main(config_path: str = "config/dlsp.yaml") -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Path to DLSP config. Default: config/dlsp.yaml",
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

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    if args.dataset is not None:
        cfg["dataset"] = args.dataset

    if args.model is not None:
        cfg["model"] = args.model

    if args.device is not None:
        cfg["device"] = args.device

    result = run_dlsp_pipeline(cfg)

    print("=" * 80)
    print("DLSP DONE")
    print(f"Best layer: {result['best_layer']}")
    print(f"Top layers: {result['top_layers']}")
    print(f"Saved: {result['out_json']}")
    print("=" * 80)


if __name__ == "__main__":
    main()