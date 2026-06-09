from __future__ import annotations


MODEL_SPECS = {
    "qwen25_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen3_vl_2b": "Qwen/Qwen3-VL-2B-Instruct",
    "internvl3_2b": "OpenGVLab/InternVL3-2B-hf",
    "smolvlm_256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "smolvlm_500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
}


PROMPTS = {
    "first_impressions": (
        "Analyze the person in this video and evaluate their Big Five personality traits (OCEAN). "
        "Return JSON only: "
        "{\"openness\":0-1, \"conscientiousness\":0-1, \"extraversion\":0-1, "
        "\"agreeableness\":0-1, \"neuroticism\":0-1}."
    ),
    "affwild2_va": (
        "Analyze the visible facial affect of the person in this video. "
        "Focus on emotional valence and arousal. "
        "Return JSON only: "
        "{\"valence\":0_to_1, \"arousal\":0_to_1}."
    ),
}


DATASET_DEFAULTS = {
    "first_impressions": {
        "sampling": {
            "n_total": 160,
            "quantile_q": 0.30,
            "seed": 42,
            "unique_base": True,
        },
        "segmentation": {
            "segment_len": 64,
            "k_frames": 6,
            "resize_short_side": 336,
            "max_segments": None,
        },
        "dlsp": {
            "bins": 30,
            "eps": 1e-6,
            "entropy_topk": 512,
            "topk_layers": 3,
            "filter_bad": True,
        },
    },
    "affwild2_va": {
        "annotation": {
            "group_target": "valence",
            "label_scale": "zero_one",
            "subset_splits": ["train"],
        },
        "sampling": {
            "n_total": 160,
            "quantile_q": 0.30,
            "seed": 42,
            "unique_base": False,
        },
        "segmentation": {
            "segment_len": 1334,
            "k_frames": 6,
            "resize_short_side": 224,
            "max_segments": None,
        },
        "dlsp": {
            "bins": 30,
            "eps": 1e-6,
            "entropy_topk": 512,
            "topk_layers": 3,
            "filter_bad": True,
        },
    },
}