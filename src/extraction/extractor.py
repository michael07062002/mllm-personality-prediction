from __future__ import annotations

import gc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor


class GenericVLHiProbeExtractor:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = model_id
        self.is_smolvlm = "smolvlm" in model_id.lower()
        self.device = self._select_device(device)
        if self.device.type in ("cuda", "mps"):
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        print(f"Loading model: {model_id}")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.dtype}")
        self.model = self._load_model(model_id)
        self.processor = self._load_processor(model_id)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self._VIS_START_CANDS = [
            "<|vision_start|>",
            "<|image_start|>",
            "<|video_start|>",
            "<|im_start|>",
            "<im_start>",
            "<image_start>",
            "<video_start>",
        ]

        self._VIS_END_CANDS = [
            "<|vision_end|>",
            "<|image_end|>",
            "<|video_end|>",
            "<|im_end|>",
            "<im_end>",
            "<image_end>",
            "<video_end>",
        ]

        self._VIS_TOKEN_SUBSTRS = [
            "vision",
            "image",
            "video",
            "img",
        ]

    def _select_device(self, device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_processor(self, model_id: str):
        if self.is_smolvlm:
            return AutoProcessor.from_pretrained(
                model_id,
                size={"longest_edge": 224},
            )
        return AutoProcessor.from_pretrained(model_id, use_fast=False)

    def _load_model(self, model_id: str):
        mid = model_id.lower()
        if "smolvlm" in mid:
            return AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                _attn_implementation="eager",
            ).to(self.device).eval()
        if "qwen2.5-vl" in mid:
            cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
            return cls.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device).eval()
        if "qwen3-vl" in mid:
            cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
            return cls.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device).eval()

        if "internvl" in mid:
            cls = getattr(transformers, "AutoModelForImageTextToText", None)
            return cls.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device).eval()
        raise ValueError(f"Unsupported model_id: {model_id}")

    def _empty_cache(self) -> None:
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    def _resolve_path(
        self,
        file_map: Union[Dict[str, str], object],
        video_id: str,
    ) -> Optional[str]:
        if hasattr(file_map, "get") and callable(getattr(file_map, "get")):
            return file_map.get(video_id)
        if isinstance(file_map, dict):
            return file_map.get(video_id)
        return None

    def _build_messages(
        self,
        frames: List[Image.Image],
        meta: Dict,
        prompt: str,
    ) -> List[Dict]:
        if self.is_smolvlm:
            content = [{"type": "image"} for _ in frames]
            content.append({"type": "text", "text": prompt})
            return [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        meta_for_model = {
            "fps": float(meta.get("fps", 0.0)),
            "width": int(meta.get("width", 0)),
            "height": int(meta.get("height", 0)),
        }
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "video_metadata": meta_for_model,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

    def _move_to_device(self, inputs: Dict) -> Dict:
        out = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                out[key] = value.to(self.device)
            else:
                out[key] = value
        return out

    def _safe_token_id(self, token: str) -> Optional[int]:
        if self.tokenizer is None:
            return None
        try:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None:
                return None
            token_id = int(token_id)
        except Exception:
            return None
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        unk_token = getattr(self.tokenizer, "unk_token", None)
        if unk_id is not None and token_id == int(unk_id) and token != unk_token:
            return None
        return token_id

    def _find_spans_between(
        self,
        ids_1d: np.ndarray,
        start_id: int,
        end_id: int,
    ) -> List[Tuple[int, int]]:
        starts = np.where(ids_1d == start_id)[0].tolist()
        ends = np.where(ids_1d == end_id)[0].tolist()
        if not starts or not ends:
            return []
        spans = []
        j = 0
        for start in starts:
            while j < len(ends) and ends[j] <= start:
                j += 1
            if j < len(ends):
                end = ends[j]
                if end > start + 1:
                    spans.append((start + 1, end))
        return spans

    def _collect_special_tokens(self) -> List[str]:
        if self.tokenizer is None:
            return []
        tokens = []
        additional = getattr(self.tokenizer, "additional_special_tokens", None)
        if additional:
            tokens.extend(list(additional))
        token_map = getattr(self.tokenizer, "special_tokens_map_extended", None)
        if isinstance(token_map, dict):
            for value in token_map.values():
                if isinstance(value, str):
                    tokens.append(value)
                elif isinstance(value, (list, tuple)):
                    tokens.extend(list(value))
        out = []
        seen = set()
        for token in tokens:
            if isinstance(token, str) and token not in seen:
                out.append(token)
                seen.add(token)
        return out

    def _collect_visual_token_ids(self) -> List[int]:
        token_ids: List[int] = []
        objects = [
            self.processor,
            self.tokenizer,
            getattr(self.model, "config", None),
        ]
        attrs = [
            "image_token_id",
            "video_token_id",
            "vision_token_id",
            "image_pad_token_id",
            "video_pad_token_id",
            "vision_pad_token_id",
        ]
        for obj in objects:
            if obj is None:
                continue
            for attr in attrs:
                value = getattr(obj, attr, None)
                if value is None:
                    continue
                if isinstance(value, int):
                    token_ids.append(int(value))
                elif isinstance(value, str):
                    token_id = self._safe_token_id(value)
                    if token_id is not None:
                        token_ids.append(int(token_id))
        if self.is_smolvlm:
            for token in [
                "<image>",
                "<fake_token_around_image>",
                "<image_token>",
                "<global-img>",
                "<img>",
            ]:
                token_id = self._safe_token_id(token)
                if token_id is not None:
                    token_ids.append(token_id)
        for token in self._collect_special_tokens():
            token_l = token.lower()
            if any(substr in token_l for substr in self._VIS_TOKEN_SUBSTRS):
                token_id = self._safe_token_id(token)
                if token_id is not None:
                    token_ids.append(token_id)
        return sorted(set(int(x) for x in token_ids))

    def _visual_token_positions(self, input_ids: torch.Tensor) -> np.ndarray:
        ids = input_ids[0].detach().cpu().numpy().astype(np.int64)
        start_ids = [self._safe_token_id(s) for s in self._VIS_START_CANDS]
        end_ids = [self._safe_token_id(s) for s in self._VIS_END_CANDS]
        start_ids = [x for x in start_ids if x is not None]
        end_ids = [x for x in end_ids if x is not None]
        spans: List[Tuple[int, int]] = []
        for start_id in start_ids:
            for end_id in end_ids:
                spans.extend(self._find_spans_between(ids, start_id, end_id))
        if spans:
            positions = []
            for start, end in spans:
                positions.extend(list(range(start, end)))
            positions = np.array(sorted(set(positions)), dtype=np.int64)
            if positions.size > 0:
                return positions
        visual_token_ids = self._collect_visual_token_ids()
        if visual_token_ids:
            mask = np.isin(ids, np.array(visual_token_ids, dtype=np.int64))
            positions = np.where(mask)[0].astype(np.int64)
            if positions.size > 0:
                return positions
        raise RuntimeError(
            "Could not find visual token positions. "
            "This run is intentionally stopped instead of falling back to text/non-pad tokens."
        )

    def _prepare_inputs(
        self,
        frames: List[Image.Image],
        meta: Dict,
        prompt: str,
    ) -> Dict:
        messages = self._build_messages(frames, meta, prompt)
        if self.is_smolvlm:
            prompt_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=prompt_text,
                images=frames,
                return_tensors="pt",
            )
        else:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        inputs.pop("token_type_ids", None)
        return self._move_to_device(inputs)

    def _forward_hidden_states(self, inputs: Dict):
        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            raise RuntimeError("No input_ids in processor output.")
        with torch.inference_mode():
            out = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "Model forward did not return hidden_states. "
                "No encoder/decoder fallback is used in this pipeline."
            )
        return tuple(hidden_states), input_ids

    def _extract_hidden_video_pooled_single(
        self,
        frames: List[Image.Image],
        meta: Dict,
        prompt: str,
    ) -> np.ndarray:
        inputs = self._prepare_inputs(frames, meta, prompt)
        try:
            hidden_states, input_ids = self._forward_hidden_states(inputs)
            positions = self._visual_token_positions(input_ids)
            vectors = []
            for h in hidden_states:
                pooled = h[0, positions, :].mean(dim=0)
                vectors.append(pooled.detach().float().cpu().numpy())
            return np.stack(vectors, axis=0).astype(np.float32)
        finally:
            try:
                del inputs
            except Exception:
                pass

    def extract_dataset(
        self,
        df: pd.DataFrame,
        file_map: Union[Dict[str, str], object],
        segmenter: object,
        prompt: str,
        max_segments: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        ok_ids: List[str] = []
        X_list: List[np.ndarray] = []
        g_list: List[int] = []
        skipped_no_path = 0
        skipped_video = 0
        skipped_all_segments = 0
        skipped_segment_errors = 0
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"Extracting {self.model_id}")):
            video_id = str(row["video_id"])
            path = self._resolve_path(file_map, video_id)
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
                    self._empty_cache()
                continue
            segment_features = []
            try:
                for frames in segments:
                    try:
                        features = self._extract_hidden_video_pooled_single(
                            frames=frames,
                            meta=meta,
                            prompt=prompt,
                        )
                        if np.isfinite(features).all():
                            segment_features.append(features)
                        else:
                            skipped_segment_errors += 1
                    except Exception as e:
                        skipped_segment_errors += 1
                        print(f"[warn] failed segment {video_id}: {e}")
                        continue
                if not segment_features:
                    skipped_all_segments += 1
                    continue
                segment_features = np.stack(segment_features, axis=0)       
                video_features = segment_features.mean(axis=0).astype(np.float32)  
                ok_ids.append(video_id)
                X_list.append(video_features)
                g_list.append(int(row["g"]))
            finally:
                try:
                    del segments
                except Exception:
                    pass

                try:
                    del segment_features
                except Exception:
                    pass
                if i % 10 == 0:
                    gc.collect()
                    self._empty_cache()
        if not X_list:
            raise RuntimeError(
                "No videos processed. "
                f"skipped_no_path={skipped_no_path}, "
                f"skipped_video={skipped_video}, "
                f"skipped_all_segments={skipped_all_segments}, "
                f"skipped_segment_errors={skipped_segment_errors}"
            )
        X = np.stack(X_list, axis=0).astype(np.float32)
        g = np.asarray(g_list, dtype=np.int64)
        return {
            "X": X,
            "g": g,
            "video_ids": np.array(ok_ids, dtype=object),
        }