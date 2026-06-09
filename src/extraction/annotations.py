from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


class FirstImpressionsAnnotations:
    TRAITS = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]
    COLMAP = {
        "openness": "O",
        "conscientiousness": "C",
        "extraversion": "E",
        "agreeableness": "A",
        "neuroticism": "N",
    }
    TARGET_COLUMNS = ["O", "C", "E", "A", "N"]

    def __init__(self, annot_path: str | Path) -> None:
        self.annot_path = Path(annot_path)
        self.ann = {}
        self.df: Optional[pd.DataFrame] = None

    
    def load(self) -> None:
        if not self.annot_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annot_path}")
        with open(self.annot_path, "rb") as f:
            try:
                self.ann = pickle.load(f, encoding="latin1")
            except Exception:
                f.seek(0)
                self.ann = pickle.load(f)

    
    def build_dataframe(self) -> pd.DataFrame:
        if not self.ann:
            self.load()
        keys = None
        for trait in self.TRAITS:
            if trait not in self.ann:
                raise KeyError(f"Missing trait in annotation file: {trait}")
            trait_keys = set(self.ann[trait].keys())
            keys = trait_keys if keys is None else keys & trait_keys
        video_ids = sorted(keys or [])
        if not video_ids:
            raise RuntimeError("No common video ids found across Big Five traits.")
        df = pd.DataFrame({"video_id": video_ids})
        for trait in self.TRAITS:
            col = self.COLMAP[trait]
            df[col] = (
                pd.Series(self.ann[trait])
                .reindex(video_ids)
                .astype("float32")
                .to_numpy()
            )
        df["P"] = df[self.TARGET_COLUMNS].mean(axis=1).astype("float32")
        parsed = df["video_id"].str.extract(r"^(?P<base>.+?)\.(?P<seg>\d+)\.mp4$")
        df["base_id"] = parsed["base"]
        df["segment_id"] = pd.to_numeric(parsed["seg"], errors="coerce")
        self.df = df.reset_index(drop=True)
        return self.df

    def pick_subset(self,
        q: float = 0.30,
        n_total: int = 160,
        seed: int = 42,
        unique_base: bool = True,
        out_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        if self.df is None:
            self.build_dataframe()
        df = self.df.copy()
        n_side = int(n_total) // 2
        low_thr = float(df["P"].quantile(q))
        high_thr = float(df["P"].quantile(1.0 - q))
        low_df = df[df["P"] <= low_thr].copy()
        high_df = df[df["P"] >= high_thr].copy()
        if unique_base and "base_id" in df.columns:
            low_unique = (
                low_df.sample(frac=1.0, random_state=seed)
                .drop_duplicates("base_id")
            )
            high_unique = (
                high_df.sample(frac=1.0, random_state=seed + 1)
                .drop_duplicates("base_id")
            )
            if len(low_unique) >= n_side and len(high_unique) >= n_side:
                low_df = low_unique
                high_df = high_unique
        if len(low_df) < n_side or len(high_df) < n_side:
            raise RuntimeError(
                f"Not enough low/high samples: low={len(low_df)}, high={len(high_df)}, "
                f"needed_each={n_side}."
            )
        low_pick = low_df.sample(n=n_side, random_state=seed).copy()
        high_pick = high_df.sample(n=n_side, random_state=seed + 1).copy()
        low_pick["g"] = 0
        high_pick["g"] = 1
        picked = pd.concat([low_pick, high_pick], axis=0).reset_index(drop=True)
        if out_csv is not None:
            out_csv = Path(out_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            picked.to_csv(out_csv, index=False)
        return picked


class AffWild2VAAnnotations:
    SPLIT_DIRS = {
        "train": "Train_Set",
        "val": "Validation_Set",
    }

    TARGET_COLUMNS = ["valence", "arousal"]

    def __init__(
        self,
        annot_root: str | Path,
        group_target: str = "valence",
        label_scale: str = "zero_one",
        valid_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.annot_root = Path(annot_root)
        self.group_target = group_target
        self.label_scale = label_scale
        self.valid_range = valid_range
        self.df: Optional[pd.DataFrame] = None
        self.track_df: Optional[pd.DataFrame] = None

        if self.label_scale not in {"raw", "zero_one"}:
            raise ValueError("label_scale must be either 'raw' or 'zero_one'")

    @staticmethod
    def normalize_track_id(track_id: str) -> str:
        return re.sub(r"_(left|right)$", "", str(track_id))

    @staticmethod
    def to_01(x: float) -> float:
        return float(np.clip((float(x) + 1.0) / 2.0, 0.0, 1.0))

    def _scale_label(self, x: float) -> float:
        if self.label_scale == "raw":
            return float(x)
        return self.to_01(x)

    def _read_va_txt(self, path: Path) -> tuple[pd.DataFrame, int, int]:
        raw = pd.read_csv(path)
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        if "valence" not in raw.columns or "arousal" not in raw.columns:
            raise ValueError(f"Bad VA columns in {path}: {raw.columns.tolist()}")
        df = raw[["valence", "arousal"]].copy()
        df["valence"] = pd.to_numeric(df["valence"], errors="coerce")
        df["arousal"] = pd.to_numeric(df["arousal"], errors="coerce")
        n_frames = int(len(df))
        lo, hi = self.valid_range
        valid = (
            np.isfinite(df["valence"].to_numpy())
            & np.isfinite(df["arousal"].to_numpy())
            & (df["valence"].to_numpy() >= lo)
            & (df["valence"].to_numpy() <= hi)
            & (df["arousal"].to_numpy() >= lo)
            & (df["arousal"].to_numpy() <= hi)
        )
        df_valid = df.loc[valid].reset_index(drop=True)
        return df_valid, n_frames, int(len(df_valid))

    
    def _compute_group_scalar(self, df: pd.DataFrame) -> pd.Series:
        if self.group_target == "valence":
            return df["valence"]
        if self.group_target == "arousal":
            return df["arousal"]
        if self.group_target == "affect_mean":
            return df[["valence", "arousal"]].mean(axis=1)
        if self.group_target == "affect_norm":
            return np.sqrt(df["valence"] ** 2 + df["arousal"] ** 2)
        raise ValueError(
            f"Unknown group_target={self.group_target}. "
            "Use one of: valence, arousal, affect_mean, affect_norm."
        )

    
    def build_dataframe(self) -> pd.DataFrame:
        if not self.annot_root.exists():
            raise FileNotFoundError(f"Annotation root not found: {self.annot_root}")
        track_rows = []
        for split, dirname in self.SPLIT_DIRS.items():
            split_dir = self.annot_root / dirname
            if not split_dir.exists():
                raise FileNotFoundError(f"Split dir not found: {split_dir}")
            txt_files = sorted(split_dir.glob("*.txt"))
            for txt_path in tqdm(txt_files, desc=f"Reading VA annotations: {split}"):
                track_id = txt_path.stem
                video_id = self.normalize_track_id(track_id)
                try:
                    df_valid, n_frames, n_valid = self._read_va_txt(txt_path)
                except Exception as e:
                    print(f"[warn] failed to read annotation file {txt_path}: {e}")
                    continue
                if n_valid <= 0:
                    continue
                track_rows.append(
                    {
                        "split": split,
                        "track_id": track_id,
                        "video_id": video_id,
                        "txt_path": str(txt_path),
                        "n_frames": int(n_frames),
                        "n_valid_frames": int(n_valid),
                        "raw_valence_sum": float(df_valid["valence"].sum()),
                        "raw_arousal_sum": float(df_valid["arousal"].sum()),
                    }
                )
        if not track_rows:
            raise RuntimeError(f"No valid VA annotation files found under {self.annot_root}")
        track_df = pd.DataFrame(track_rows)
        self.track_df = track_df.copy()
        rows = []
        for (split, video_id), gdf in track_df.groupby(["split", "video_id"], sort=True):
            n_valid = int(gdf["n_valid_frames"].sum())
            n_frames = int(gdf["n_frames"].sum())
            raw_valence = float(gdf["raw_valence_sum"].sum() / max(n_valid, 1))
            raw_arousal = float(gdf["raw_arousal_sum"].sum() / max(n_valid, 1))
            rows.append(
                {
                    "video_id": video_id,
                    "split": split,
                    "valence": self._scale_label(raw_valence),
                    "arousal": self._scale_label(raw_arousal),
                    "raw_valence": raw_valence,
                    "raw_arousal": raw_arousal,
                    "n_frames": n_frames,
                    "n_valid_frames": n_valid,
                    "n_tracks": int(len(gdf)),
                    "track_ids": "|".join(gdf["track_id"].astype(str).tolist()),
                    "txt_paths": "|".join(gdf["txt_path"].astype(str).tolist()),
                }
            )
        df = pd.DataFrame(rows).sort_values(["split", "video_id"]).reset_index(drop=True)
        df["P"] = self._compute_group_scalar(df).astype("float32")
        self.df = df
        return df

    
    def pick_subset(
        self,
        q: float = 0.30,
        n_total: int | None = 160,
        seed: int = 42,
        split_filter: Optional[Iterable[str]] = ("train",),
        out_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        if self.df is None:
            self.build_dataframe()
        df = self.df.copy()
        if split_filter is not None:
            split_filter = list(split_filter)
            df = df[df["split"].isin(split_filter)].copy()
        if len(df) == 0:
            raise RuntimeError(f"No rows after split_filter={split_filter}")
        low_thr = float(df["P"].quantile(q))
        high_thr = float(df["P"].quantile(1.0 - q))
        low_df = df[df["P"] <= low_thr].copy()
        high_df = df[df["P"] >= high_thr].copy()
        if n_total is None:
            n_side = min(len(low_df), len(high_df))
        else:
            n_side = min(int(n_total) // 2, len(low_df), len(high_df))
        if n_side <= 0:
            raise RuntimeError(
                f"Empty low/high group: low={len(low_df)}, high={len(high_df)}."
            )
        low_pick = low_df.sample(n=n_side, random_state=seed).copy()
        high_pick = high_df.sample(n=n_side, random_state=seed + 1).copy()
        low_pick["g"] = 0
        high_pick["g"] = 1
        picked = pd.concat([low_pick, high_pick], axis=0).reset_index(drop=True)
        if out_csv is not None:
            out_csv = Path(out_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            picked.to_csv(out_csv, index=False)
        return picked