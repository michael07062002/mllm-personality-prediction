from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image


class VideoFileMap:
    def __init__(
        self,
        dataset_root: str | Path,
        exts: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.exts = tuple(e.lower() for e in exts)
        self.file_map: Dict[str, str] = {}
        self.stem_map: Dict[str, str] = {}

    
    def build(self) -> Dict[str, str]:
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
        file_map: Dict[str, str] = {}
        stem_map: Dict[str, str] = {}
        dup_file = 0
        dup_stem = 0
        for root, _, files in os.walk(self.dataset_root):
            for filename in files:
                ext = Path(filename).suffix.lower()
                if ext not in self.exts:
                    continue
                full = str(Path(root) / filename)
                stem = Path(filename).stem
                if filename in file_map:
                    dup_file += 1
                else:
                    file_map[filename] = full
                if stem in stem_map:
                    dup_stem += 1
                else:
                    stem_map[stem] = full
        if not file_map:
            raise RuntimeError(f"No video files found under: {self.dataset_root}")
        self.file_map = file_map
        self.stem_map = stem_map
        print(f"Total video files found: {len(self.file_map)}")
        print(f"Unique stems found:      {len(self.stem_map)}")
        print(f"Duplicate filenames:     {dup_file}")
        print(f"Duplicate stems:         {dup_stem}")
        return self.file_map

    def get(self, video_id: str) -> Optional[str]:
        video_id = str(video_id)
        if video_id in self.file_map:
            return self.file_map[video_id]
        stem = Path(video_id).stem
        if stem in self.stem_map:
            return self.stem_map[stem]
        for ext in self.exts:
            candidate = video_id + ext
            if candidate in self.file_map:
                return self.file_map[candidate]
        return None


class PaperVideoSegmenter:
    def __init__(
        self,
        segment_len: int = 64,
        k_frames: int = 6,
        resize_short_side: Optional[int] = None,
    ) -> None:
        self.segment_len = int(segment_len)
        self.k_frames = int(k_frames)
        self.resize_short_side = resize_short_side
        if self.segment_len <= 0:
            raise ValueError("segment_len must be > 0")
        if self.k_frames <= 0:
            raise ValueError("k_frames must be > 0")
        self.fixed_stride = max(1, self.segment_len // self.k_frames)

    
    def _maybe_resize(self, img: Image.Image) -> Image.Image:
        if self.resize_short_side is None:
            return img
        w, h = img.size
        short = min(w, h)
        if short <= self.resize_short_side:
            return img
        scale = self.resize_short_side / float(short)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return img.resize((new_w, new_h), Image.BICUBIC)

    
    def _read_frame(self, cap: cv2.VideoCapture, frame_idx: int) -> Optional[Image.Image]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        return self._maybe_resize(img)

    
    def _sample_keyframes_in_segment(
        self,
        cap: cv2.VideoCapture,
        start: int,
        end: int,
    ) -> List[Image.Image]:
        length = max(0, end - start)
        if length <= 0:
            return []
        if length >= self.k_frames:
            if length >= self.segment_len:
                stride = self.fixed_stride
                idxs = [start + i * stride for i in range(self.k_frames)]
            else:
                idxs = np.linspace(start, end - 1, num=self.k_frames, dtype=int).tolist()
        else:
            idxs = np.linspace(start, end - 1, num=self.k_frames, dtype=int).tolist()
        idxs = [min(max(start, int(x)), end - 1) for x in idxs]
        frames: List[Image.Image] = []
        for frame_idx in idxs:
            img = self._read_frame(cap, frame_idx)
            if img is not None:
                frames.append(img)
        if not frames:
            return []
        while len(frames) < self.k_frames:
            frames.append(frames[-1])
        return frames[: self.k_frames]

    
    def load_segments(
        self,
        video_path: str | Path,
        max_segments: Optional[int] = None,
    ) -> tuple[List[List[Image.Image]], Dict]:
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            fps = 30.0
        meta = {
            "fps": fps,
            "width": width,
            "height": height,
            "num_frames": num_frames,
        }
        if num_frames <= 0:
            cap.release()
            raise RuntimeError(f"Empty video: {video_path}")
        total_segments = int(np.ceil(num_frames / self.segment_len))
        if max_segments is not None:
            total_segments = min(total_segments, int(max_segments))
        segments: List[List[Image.Image]] = []
        for segment_idx in range(total_segments):
            start = segment_idx * self.segment_len
            end = min((segment_idx + 1) * self.segment_len, num_frames)
            frames = self._sample_keyframes_in_segment(cap, start, end)
            if frames:
                segments.append(frames)
        cap.release()
        if not segments:
            raise RuntimeError(f"No segments extracted: {video_path}")
        return segments, meta