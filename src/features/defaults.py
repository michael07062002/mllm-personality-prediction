from __future__ import annotations


FEATURE_DEFAULTS = {
    "first_impressions": {
        "splits": ["train", "test"],
        "segmentation": {
            "segment_len": 128,
            "k_frames": 4,
            "resize_short_side": 224,
            "max_segments": None,
        },
    },
    "affwild2_va": {
        "splits": ["train", "val"],
        "segmentation": {
            "segment_len": 1334,
            "k_frames": 6,
            "resize_short_side": 224,
            "max_segments": None,
        },
    },
}