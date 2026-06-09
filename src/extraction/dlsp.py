from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


class DLSPAnalyzer:
    def __init__(
        self,
        bins: int = 30,
        eps: float = 1e-6,
        entropy_topk: int = 512,
        topk_layers: int = 3,
    ) -> None:
        self.bins = int(bins)
        self.eps = float(eps)
        self.entropy_topk = int(entropy_topk)
        self.topk_layers = int(topk_layers)

    def _kl(
        self,
        m0: np.ndarray,
        v0: np.ndarray,
        m1: np.ndarray,
        v1: np.ndarray,
    ) -> np.ndarray:
        v0 = np.maximum(v0, self.eps)
        v1 = np.maximum(v1, self.eps)
        return 0.5 * (
            np.log(v1 / v0)
            + (v0 + (m0 - m1) ** 2) / v1
            - 1.0
        )

    def _entropy_1d(self, x: np.ndarray) -> float:
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if np.isclose(xmin, xmax):
            return 0.0
        hist, _ = np.histogram(x, bins=self.bins, range=(xmin, xmax))
        p = hist.astype(np.float64)
        total = p.sum()
        if total <= 0:
            return 0.0
        p = p / total
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - x.mean()) / (x.std(ddof=0) + 1e-8)

    def compute_metrics(self, X: np.ndarray, g: np.ndarray) -> pd.DataFrame:
        X = np.asarray(X, dtype=np.float32)
        g = np.asarray(g, dtype=np.int64)
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, L, D), got {X.shape}")
        groups = np.unique(g)
        if len(groups) != 2:
            raise ValueError(f"DLSP expects exactly 2 groups, got {groups}")
        low = X[g == 0]
        high = X[g == 1]
        if len(low) < 2 or len(high) < 2:
            raise ValueError(f"Too few samples for DLSP: low={len(low)}, high={len(high)}")
        _, num_layers, dim = X.shape
        entropy_k = min(self.entropy_topk, dim)
        rows = []
        for layer_idx in range(num_layers):
            X0 = low[:, layer_idx, :]
            X1 = high[:, layer_idx, :]
            m0 = X0.mean(axis=0)
            v0 = X0.var(axis=0)
            m1 = X1.mean(axis=0)
            v1 = X1.var(axis=0)
            kl = float(self._kl(m0, v0, m1, v1).mean())
            ldr = float(((m0 - m1) ** 2 / (v0 + v1 + self.eps)).mean())
            diff = np.abs(m0 - m1)
            top_dims = np.argsort(-diff)[:entropy_k]
            entropy = float(
                np.mean([self._entropy_1d(X[:, layer_idx, d]) for d in top_dims])
            )
            rows.append(
                {
                    "layer": int(layer_idx),
                    "KL": kl,
                    "LDR": ldr,
                    "Entropy": entropy,
                }
            )
        df = pd.DataFrame(rows)
        df["KL_z"] = self._zscore(df["KL"].to_numpy())
        df["LDR_z"] = self._zscore(df["LDR"].to_numpy())
        df["Entropy_z"] = self._zscore(df["Entropy"].to_numpy())
        df["S"] = df["KL_z"] + df["LDR_z"] + df["Entropy_z"]
        return df

    def rank_layers(self, X: np.ndarray, g: np.ndarray) -> Dict:
        df = self.compute_metrics(X, g)
        df = df.loc[
            (df["layer"] >= 3)
            & (df["layer"] < df["layer"].max())
        ].copy()
        if len(df) == 0:
            raise RuntimeError("No layers left for DLSP ranking.")
        top_layers = (
            df.nlargest(self.topk_layers, "S")["layer"]
            .astype(int)
            .tolist()
        )
        best_layer = int(df.loc[df["S"].idxmax(), "layer"])
        return {
            "best_layer": best_layer,
            "top_layers": top_layers,
        }


def select_top_layers(
    X: np.ndarray,
    g: np.ndarray,
    out_json: str | Path,
    bins: int = 30,
    eps: float = 1e-6,
    entropy_topk: int = 512,
    topk_layers: int = 3,
    filter_bad: bool = True,
) -> Dict:
    X = np.asarray(X, dtype=np.float32)
    g = np.asarray(g, dtype=np.int64)
    if filter_bad:
        good_mask = np.isfinite(X).all(axis=(1, 2))
        X = X[good_mask]
        g = g[good_mask]
    analyzer = DLSPAnalyzer(
        bins=bins,
        eps=eps,
        entropy_topk=entropy_topk,
        topk_layers=topk_layers,
    )
    result = analyzer.rank_layers(X, g)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_layer": int(result["best_layer"]),
        "top_layers": [int(x) for x in result["top_layers"]],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload