# Re-run the multi-preset optimizer (environment reset safety)
import os

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /path/to/project/src
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
# ============== Configuration =================
presets = {
    "speed": {
        "query_time_ms": ("min", 0.5),
        "build_time_s": ("min", 0.15),
        "index_size_mb": ("min", 0.10),
        "label_precision_at_k_micro": ("max", 0.2),
        "vector_recall_at_k": ("max", 0.05),
    },
    "balanced": {
        "query_time_ms": ("min", 0.2),
        "build_time_s": ("min", 0.15),
        "index_size_mb": ("min", 0.15),
        "label_precision_at_k_micro": ("max", 0.20),
        "vector_recall_at_k": ("max", 0.25),
    },
    "quality": {
        "label_precision_at_k_micro": ("max", 0.35),
        "vector_recall_at_k": ("max", 0.5),
        "query_time_ms": ("min", 0.10),
        "index_size_mb": ("min", 0.03),
        "build_time_s": ("min", 0.02),
    },
}



@dataclass
class Objective:
    direction: str  # "min" or "max"
    weight: float

def _normalize_series(s: pd.Series, method: str = "minmax") -> pd.Series:
    if method == "zscore":
        mu, std = s.mean(), s.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(s)), index=s.index)
        z = (s - mu) / std
        return (np.tanh(z) + 1) / 2
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

def _build_score(df: pd.DataFrame, objective_specs: Dict[str, Tuple[str, float]], normalize: str):
    score_parts = {}
    for col, (direction, weight) in objective_specs.items():
        norm = _normalize_series(df[col].astype(float), method=normalize)
        if direction.lower() == "min":
            norm = 1 - norm
        score_parts[col] = weight * norm
        df[f"norm_{col}"] = norm
    total_weight = sum(w for _, w in objective_specs.values()) or 1.0
    df["score"] = sum(score_parts.values()) / total_weight
    return df

def optimize_params(
    df: pd.DataFrame,
    param_cols: List[str],
    objective_specs: Dict[str, Tuple[str, float]],
    normalize: str = "minmax",
    filters: Optional[Dict[str, List]] = None,
    top_n: int = 3,
):
    data = df.copy()
    if filters:
        for col, allowed in filters.items():
            data = data[data[col].isin(allowed)]
    needed_cols = set(param_cols) | set(objective_specs.keys())
    data = data.dropna(subset=list(needed_cols)).copy()
    data = _build_score(data, objective_specs, normalize)
    rank_cols = param_cols + list(objective_specs.keys()) + [f"norm_{c}" for c in objective_specs.keys()] + ["score"]
    ranking = data[rank_cols].sort_values("score", ascending=False).reset_index(drop=True)
    return ranking.head(top_n), ranking

def optimize_with_presets(
    csv_path: str,
    param_cols: List[str] = ["pca_dim", "nlist", "nprobe"],
    normalize: str = "minmax",
    filters: Optional[Dict[str, List]] = None,
    top_n_each: int = 3,
    presets: Optional[Dict[str, Dict[str, Tuple[str, float]]]] = None,
):
    df = pd.read_csv(csv_path)
    if presets is None:
        presets = {
            "speed_first": {
                "query_time_ms": ("min", 0.5),
                "build_time_s": ("min", 0.3),
                "index_size_mb": ("min", 0.1),
                "label_precision_at_k_micro": ("max", 0.05),
                "vector_recall_at_k": ("max", 0.05),
            },
            "memory_first": {
                "index_size_mb": ("min", 0.6),
                "build_time_s": ("min", 0.15),
                "query_time_ms": ("min", 0.15),
                "label_precision_at_k_micro": ("max", 0.05),
                "vector_recall_at_k": ("max", 0.05),
            },
            "quality_first": {
                "label_precision_at_k_micro": ("max", 0.45),
                "vector_recall_at_k": ("max", 0.35),
                "query_time_ms": ("min", 0.10),
                "index_size_mb": ("min", 0.05),
                "build_time_s": ("min", 0.05),
            },
        }
    results = {}
    for name, spec in presets.items():
        top_df, full_rank = optimize_params(
            df=df,
            param_cols=param_cols,
            objective_specs=spec,
            normalize=normalize,
            filters=filters,
            top_n=top_n_each,
        )
        results[name] = top_df
        print(f"\n=== {name} ===")
        print(top_df[param_cols + ['score']])
    return results

# Execute on your file
if __name__ == "__main__":
    results = optimize_with_presets(
        os.path.join(PROJECT_DIR, 'data', "ivf_FMNIST_query_metrics.csv"),
        param_cols=["pca_dim", "nlist", "nprobe"],
        normalize="minmax",  # or "zscore"
        filters=None,  #  {"k":[20]}
        top_n_each=3,
        presets=presets
    )

