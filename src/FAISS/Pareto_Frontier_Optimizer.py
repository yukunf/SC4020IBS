# Re-run the multi-preset optimizer (environment reset safety) + Pareto frontier
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

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

# ========= Scoring & Normalization ============
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
        return (np.tanh(z) + 1) / 2  # squashed to [0,1]
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

def _build_score(df: pd.DataFrame, objective_specs: Dict[str, Tuple[str, float]], normalize: str):
    score_parts = {}
    df = df.copy()
    for col, (direction, weight) in objective_specs.items():
        if col not in df.columns:
            raise KeyError(f"Objective column '{col}' not found in DataFrame.")
        norm = _normalize_series(df[col].astype(float), method=normalize)
        if direction.lower() == "min":
            norm = 1 - norm  # smaller is better -> higher normalized reward
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
            if col not in data.columns:
                raise KeyError(f"Filter column '{col}' not in DataFrame.")
            data = data[data[col].isin(allowed)]
    needed_cols = set(param_cols) | set(objective_specs.keys())
    data = data.dropna(subset=list(needed_cols)).copy()
    data = _build_score(data, objective_specs, normalize)
    rank_cols = param_cols + list(objective_specs.keys()) + [f"norm_{c}" for c in objective_specs.keys()] + ["score"]
    ranking = data[rank_cols].sort_values("score", ascending=False).reset_index(drop=True)
    return ranking.head(top_n), ranking

# ========= Pareto Frontier (Fast Non-Dominated Sort) ============
def _to_minimization_matrix(df: pd.DataFrame, objs: Dict[str, Tuple[str, float]]) -> np.ndarray:
    """
    Convert mixed min/max objectives to a uniform minimization matrix.
    For 'max' objectives, we multiply by -1 so that 'greater is better' becomes 'smaller is better'.
    """
    X = []
    for col, (direction, _) in objs.items():
        vals = df[col].astype(float).to_numpy()
        if direction.lower() == "max":
            vals = -vals
        X.append(vals)
    return np.vstack(X).T  # shape (N, M)

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Returns True if point a Pareto-dominates point b under minimization:
    a is no worse in all dims and strictly better in at least one.
    """
    return np.all(a <= b) and np.any(a < b)

def _fast_non_dominated_sort(X: np.ndarray) -> List[List[int]]:
    """
    NSGA-II style fast non-dominated sorting.
    X: (N, M) minimization matrix
    Returns list of fronts (each a list of indices), front[0] is the Pareto frontier.
    """
    N = X.shape[0]
    S = [[] for _ in range(N)]   # whom i dominates
    n = np.zeros(N, dtype=int)   # domination count
    fronts: List[List[int]] = [[]]

    for p in range(N):
        Sp = []
        np_count = 0
        for q in range(N):
            if p == q:
                continue
            if _dominates(X[p], X[q]):
                Sp.append(q)
            elif _dominates(X[q], X[p]):
                np_count += 1
        S[p] = Sp
        n[p] = np_count
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts

def _crowding_distance(X_front: np.ndarray) -> np.ndarray:
    """
    Crowding distance for one front (minimization). Larger = more isolated.
    """
    N, M = X_front.shape
    if N == 0:
        return np.array([])
    dist = np.zeros(N, dtype=float)
    for m in range(M):
        order = np.argsort(X_front[:, m])
        vals = X_front[order, m]
        vmin, vmax = vals[0], vals[-1]
        # Boundary points get inf distance to preserve extremes
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        denom = vmax - vmin
        if denom == 0:
            continue
        for i in range(1, N - 1):
            dist[order[i]] += (vals[i + 1] - vals[i - 1]) / denom
    return dist

def compute_pareto_frontier(
    df: pd.DataFrame,
    objectives: Dict[str, Tuple[str, float]],
    return_all_fronts: bool = False
) -> Tuple[pd.DataFrame, Optional[List[pd.DataFrame]]]:
    """
    Label df with:
      - pareto_rank (0 = frontier)
      - is_pareto (True for rank 0)
      - dominates (how many points this row dominates)
      - crowding_distance (only defined within its front)
    Returns (labeled_df, all_fronts_as_dfs or None)
    """
    if len(objectives) == 0:
        raise ValueError("Objectives cannot be empty for Pareto computation.")
    for col in objectives.keys():
        if col not in df.columns:
            raise KeyError(f"Objective column '{col}' not found in DataFrame.")

    # Construct minimization matrix and run sorting
    X = _to_minimization_matrix(df, objectives)
    fronts = _fast_non_dominated_sort(X)

    # Prepare labels
    pareto_rank = np.full(len(df), fill_value=-1, dtype=int)
    crowd = np.zeros(len(df), dtype=float)
    dominates_counts = np.zeros(len(df), dtype=int)

    # Compute crowding distance per front and domination counts
    for r, idxs in enumerate(fronts):
        pareto_rank[idxs] = r
        cf = _crowding_distance(X[idxs])
        crowd[idxs] = cf

    # Dominates count (costly O(N^2), but datasets are usually modest)
    for i in range(len(df)):
        dominates_counts[i] = int(np.sum([_dominates(X[i], X[j]) for j in range(len(df)) if i != j]))

    out = df.copy()
    out["pareto_rank"] = pareto_rank
    out["is_pareto"] = out["pareto_rank"] == 0
    out["crowding_distance"] = crowd
    out["dominates"] = dominates_counts

    if not return_all_fronts:
        return out, None
    fronts_dfs = [out.iloc[idxs].copy() for idxs in fronts]
    return out, fronts_dfs

# ========= High-level API ============
def optimize_with_presets(
    csv_path: str,
    param_cols: List[str] = ["pca_dim", "nlist", "nprobe"],
    normalize: str = "minmax",
    filters: Optional[Dict[str, List]] = None,
    top_n_each: int = 3,
    presets: Optional[Dict[str, Dict[str, Tuple[str, float]]]] = None,
    also_show_pareto: bool = True,
):
    df = pd.read_csv(csv_path)
    if presets is None:
        presets = {
            "speed_first": {
                "query_time_ms": ("min", 0.4),
                "build_time_s": ("min", 0.4),
                "index_size_mb": ("min", 0.1),
                "label_precision_at_k_micro": ("max", 0.05),
                "vector_recall_at_k": ("max", 0.05),
            },
            "memory_first": {
                "index_size_mb": ("min", 0.5),
                "build_time_s": ("min", 0.2),
                "query_time_ms": ("min", 0.15),
                "label_precision_at_k_micro": ("max", 0.1),
                "vector_recall_at_k": ("max", 0.05),
            },
            "quality_balanced": {
                "label_precision_at_k_micro": ("max", 0.15),
                "vector_recall_at_k": ("max", 0.25),
                "query_time_ms": ("min", 0.25),
                "index_size_mb": ("min", 0.25),
                "build_time_s": ("min", 0.1),
            },
        }

    results = {}
    for name, spec in presets.items():
        print(f"\n=== {name} :: weighted-score top {top_n_each} ===")
        top_df, full_rank = optimize_params(
            df=df,
            param_cols=param_cols,
            objective_specs=spec,
            normalize=normalize,
            filters=filters,
            top_n=top_n_each,
        )
        results[name] = top_df

        # Print the compact view
        cols_to_show = param_cols + list(spec.keys()) + ["score"]
        print(top_df[cols_to_show].to_string(index=False))

        if also_show_pareto:
            # Compute Pareto on the (filtered) space used for this preset
            _, filtered_all = optimize_params(
                df=df,
                param_cols=param_cols,
                objective_specs=spec,
                normalize=normalize,
                filters=filters,
                top_n=len(df)  # get all rows post-filter for a complete Pareto set
            )
            labeled, _ = compute_pareto_frontier(filtered_all, spec, return_all_fronts=False)
            pareto_subset = (
                labeled[labeled["is_pareto"]]
                .sort_values(["pareto_rank", "crowding_distance"], ascending=[True, False])
                .head(top_n_each)
            )
            print(f"\n--- {name} :: Pareto frontier suggestions (top {top_n_each} by crowding distance) ---")
            pareto_cols = param_cols + list(spec.keys()) + ["pareto_rank", "crowding_distance", "dominates"]
            print(pareto_subset[pareto_cols].to_string(index=False))

    return results

# ========= Example run ============
if __name__ == "__main__":
    CSV = os.path.join(PROJECT_DIR, 'data', "ivf_gallery_query_metrics.csv")
    results = optimize_with_presets(
        CSV,
        param_cols=["pca_dim", "nlist", "nprobe"],
        normalize="minmax",   # or "zscore"
        filters=None,         # e.g. {"k":[20]}
        top_n_each=3,
        also_show_pareto=True
    )

    # Optional: Global Pareto across a canonical “balanced” trade-off space
    canonical_objectives = {
        "query_time_ms": ("min", 0.25),
        "build_time_s": ("min", 0.15),
        "index_size_mb": ("min", 0.15),
        "label_precision_at_k_micro": ("max", 0.25),
        "vector_recall_at_k": ("max", 0.20),
    }
    try:
        df_all = pd.read_csv(CSV)
        labeled, fronts = compute_pareto_frontier(df_all.dropna(subset=list(canonical_objectives.keys())).copy(),
                                                  canonical_objectives,
                                                  return_all_fronts=True)
        frontier0 = labeled[labeled["pareto_rank"] == 0]
        print(f"\n=== Global Pareto (canonical) :: {len(frontier0)} configs on rank-0 ===")
        show_cols = ["pca_dim", "nlist", "nprobe"] + list(canonical_objectives.keys()) + ["dominates", "crowding_distance"]
        print(frontier0[show_cols].sort_values("crowding_distance", ascending=False).to_string(index=False))
    except Exception as e:
        print(f"[Global Pareto skipped] {e}")
