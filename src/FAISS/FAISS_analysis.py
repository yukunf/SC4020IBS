
import time, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, faiss
from typing import Union

from src.FAISS import FAISS_NeighborSearch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /path/to/project/src
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

INSHOP_VECTOR_GALLERY = os.path.join(PROJECT_DIR, "data", "inshop_clip_vectors_gallery.npy")
INSHOP_VECTOR_QUERY = os.path.join(PROJECT_DIR, "data", "inshop_clip_vectors_query.npy")
INSHOP_LABEL_GALLERY_NPY = os.path.join(PROJECT_DIR, "data", "inshop_clip_labels_gallery.npy")
INSHOP_LABEL_QUERY_NPY = os.path.join(PROJECT_DIR, "data", "inshop_clip_labels_query.npy")

FMNIST_VECTOR_GALLERY = os.path.join(PROJECT_DIR, "data", "FMNIST_gallery_vectors.npy")
FMNIST_LABEL_GALLERY = os.path.join(PROJECT_DIR, "data", "FMNIST_gallery_labels.npy")
FMNIST_VECTOR_QUERY = os.path.join(PROJECT_DIR, "data", "FMNIST_query_vectors.npy")
FMNIST_LABEL_QUERY = os.path.join(PROJECT_DIR, "data", "FMNIST_query_labels.npy")
#=================== IVF Index Factory ==================
def create_ivf_index(
    vectors_path: str = FAISS_NeighborSearch.VECTORS_PATH,
    index_path: str = FAISS_NeighborSearch.INDEX_PATH,
    nlist: int = 1024,
    nprobe: int = 8,
    pca_dim: int = None,
    metric: str = "L2",
):
    """
    This create an index from the previous file we have, only giving 3 hyper parameters, in favor of grid search.

    Parameters:
        vectors_path : path of the vectors file
        nlist        : Inverted Buskets
        nprobe       : Nums of buskets to visit
        pca_dim      : PCA dimension
        metric       : "L2" or "IP", we use L2 distance

    Return:
        index.
    """
    if pca_dim is not None:
        index_factory_str = f"PCA{pca_dim},IVF{nlist},Flat"
    else:
        index_factory_str = f"IVF{nlist},Flat"

    base_dir = os.path.dirname(vectors_path)
    # index_name = f"ivf_{'pca'+str(pca_dim)+'_' if pca_dim else ''}nlist{nlist}.index"
    # index_path = os.path.join(base_dir, index_name)

    index = FAISS_NeighborSearch.load_or_create_index(
        vectors_path=vectors_path,
        index_path=index_path,
        model_name="fashion-clip",
        index_factory_str=index_factory_str,
        metric=metric,
        force_rebuild=True,
    )

    try:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
    except Exception:
        if hasattr(index, "index"):
            faiss.ParameterSpace().set_index_parameter(index.index, "nprobe", nprobe)

    print(f"IVF Index created: {index_factory_str} | nprobe={nprobe}")
    print(f"Saved at: {index_path}")
    print(f"Total vectors: {index.ntotal}")
    return index


def create_index_hnsw(
    vectors_path: str = FAISS_NeighborSearch.VECTORS_PATH,
    index_path: str = FAISS_NeighborSearch.INDEX_PATH,
    M: int = 32,
    efConstruction: int = 200,
    efSearch: int = 64,
    metric: str = "L2",
):
    """
    Create an HNSW index (no PCA) using FAISS factory.
    Only exposes M, efConstruction, efSearch for grid search.

    Params:
        vectors_path   : path to npy vectors (float32)
        index_path     : where to save the index
        M              : number of neighbors per node
        efConstruction : build-time exploration width
        efSearch       : query-time exploration width
        metric         : "L2" or "IP"
    """
    index_factory_str = f"HNSW{int(M)},Flat"

    index = FAISS_NeighborSearch.load_or_create_index(
        vectors_path=vectors_path,
        index_path=index_path,
        model_name="fashion-clip",
        index_factory_str=index_factory_str,
        metric=metric,
        force_rebuild=True,  # ensure efConstruction takes effect
    )

    try:
        if hasattr(index, "hnsw"):
            index.hnsw.efConstruction = int(efConstruction)
    except Exception:
        pass

    try:
        faiss.ParameterSpace().set_index_parameter(index, "efSearch", int(efSearch))
    except Exception:
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = int(efSearch)

    print(f"HNSW Index created: {index_factory_str} | M={M}, efC={efConstruction}, efS={efSearch}")
    print(f"Saved at: {index_path}")
    print(f"Total vectors: {index.ntotal}")
    return index


# =================== Stratified Sample Util function ===================
def _sample_query_ids(labels: np.ndarray, n_per_label: int = 2):
    ids = []
    for lb in np.unique(labels):
        idxs = np.where(labels == lb)[0]
        if len(idxs) == 0:
            continue
        take = min(n_per_label, len(idxs))
        ids.extend(np.random.choice(idxs, size=take, replace=False).tolist())
    return ids




# ========================= Plot Code =========================
def _annotate_cells(ax, data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if isinstance(v, (int, float)) and not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black")

import math

def _facet_by_pca(df: pd.DataFrame, value_col: str, title_prefix: str, cmap="viridis"):
    pcas = sorted(df["pca_dim"].unique())
    cols = min(4, max(1, len(pcas)))
    rows = math.ceil(len(pcas) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4.0*rows))
    axes = np.array(axes, ndmin=2)

    used = 0
    for i, p in enumerate(pcas):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        sub = df[df["pca_dim"] == p].groupby(["nprobe", "nlist"], as_index=False)[value_col].mean()
        piv = sub.pivot_table(index="nprobe", columns="nlist", values=value_col, aggfunc="mean").sort_index().sort_index(axis=1)
        im = ax.imshow(piv.values, aspect="auto", cmap=cmap)
        _annotate_cells(ax, piv.values)
        ax.set_title(f"PCA={int(p) if p!=0 else 'None'}")
        ax.set_xlabel("nlist"); ax.set_ylabel("nprobe")
        ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels(piv.columns.astype(int))
        ax.set_yticks(np.arange(piv.shape[0])); ax.set_yticklabels(piv.index.astype(int))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        used += 1

    for j in range(used, rows*cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"{title_prefix} — (faceted by PCA)")
    fig.tight_layout()
    return fig


def plot_ivf_build_and_size_panels(df: pd.DataFrame,dataset_title: str):
    agg = df.groupby(["pca_dim", "nlist"], as_index=False)[["build_time_s", "index_size_mb"]].mean()

    piv_build = agg.pivot_table(index="pca_dim", columns="nlist", values="build_time_s", aggfunc="mean").sort_index().sort_index(axis=1)
    piv_size  = agg.pivot_table(index="pca_dim", columns="nlist", values="index_size_mb", aggfunc="mean").sort_index().sort_index(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cmap = "viridis"

    im1 = axes[0].imshow(piv_build.values, aspect="auto", cmap=cmap)
    _annotate_cells(axes[0], piv_build.values)
    axes[0].set_title("Build Time (s)")
    axes[0].set_xlabel("nlist"); axes[0].set_ylabel("PCA dim")
    axes[0].set_xticks(np.arange(piv_build.shape[1])); axes[0].set_xticklabels(piv_build.columns.astype(int))
    axes[0].set_yticks(np.arange(piv_build.shape[0])); axes[0].set_yticklabels(piv_build.index.astype(int))
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(piv_size.values, aspect="auto", cmap=cmap)
    _annotate_cells(axes[1], piv_size.values)
    axes[1].set_title("Index Size (MB)")
    axes[1].set_xlabel("nlist"); axes[1].set_ylabel("PCA dim")
    axes[1].set_xticks(np.arange(piv_size.shape[1])); axes[1].set_xticklabels(piv_size.columns.astype(int))
    axes[1].set_yticks(np.arange(piv_size.shape[0])); axes[1].set_yticklabels(piv_size.index.astype(int))
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"Dataset:{dataset_title} IVF — PCA × nlist")
    fig.tight_layout()
    return fig



def plot_ivf_querytime_by_pca(df: pd.DataFrame,dataset_title: str):
    return _facet_by_pca(df, value_col="query_time_ms", title_prefix=f"Data:{dataset_title} Query Time (ms)", cmap="coolwarm")

def plot_ivf_vector_recall_by_pca(df: pd.DataFrame, dataset_title:str, k_val: int = 20):
    fig = _facet_by_pca(df, value_col="vector_recall_at_k", title_prefix=f"Data:{dataset_title} Vector Recall@{k_val}", cmap="viridis")
    return fig

def plot_ivf_label_precision_by_pca(df: pd.DataFrame, dataset_title:str, k_val: int = 20):
    fig = _facet_by_pca(df, value_col="label_precision_at_k_micro", title_prefix=f"Data:{dataset_title} Label Precision@{k_val} (micro)", cmap="plasma")
    return fig

def plot_ivf_label_recall_by_pca(df: pd.DataFrame, dataset_title:str, k_val: int = 20):
    fig = _facet_by_pca(df, value_col="label_recall_at_k_micro", title_prefix=f"Data:{dataset_title} Label Recall@{k_val} (micro)", cmap="viridis")
    return fig

def plot_ivf_macro_for_label(df_label: pd.DataFrame, label_id: int, k_val: int = 20):
    sub = df_label[df_label["label"] == int(label_id)].copy()
    if sub.empty:
        print(f"[warn] no data for label={label_id}")
        return None

    # 两个子图：precision_macro / recall_macro（PCA 分面：行列为 nprobe × nlist 的拼图）
    def _facet(value_col: str, title: str, cmap="viridis"):
        pcas = sorted(sub["pca_dim"].unique())
        cols = min(4, max(1, len(pcas)))
        rows = math.ceil(len(pcas) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4.0*rows))
        axes = np.array(axes, ndmin=2)

        used = 0
        for i, p in enumerate(pcas):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ss = sub[sub["pca_dim"] == p].groupby(["nprobe", "nlist"], as_index=False)[value_col].mean()
            piv = ss.pivot_table(index="nprobe", columns="nlist", values=value_col, aggfunc="mean").sort_index().sort_index(axis=1)
            im = ax.imshow(piv.values, aspect="auto", cmap=cmap)
            _annotate_cells(ax, piv.values)
            ax.set_title(f"PCA={int(p) if p!=0 else 'None'}")
            ax.set_xlabel("nlist"); ax.set_ylabel("nprobe")
            ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels(piv.columns.astype(int))
            ax.set_yticks(np.arange(piv.shape[0])); ax.set_yticklabels(piv.index.astype(int))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            used += 1

        for j in range(used, rows*cols):
            r, c = divmod(j, cols)
            axes[r, c].axis("off")

        fig.suptitle(f"{title} (macro) — label={label_id}, Recall@{k_val}")
        fig.tight_layout()
        return fig

    fig1 = _facet("precision_at_k_macro", "Label Precision")
    fig2 = _facet("recall_at_k_macro", "Label Recall")
    return fig1, fig2


def plot_hnsw_build_and_size(df: pd.DataFrame):
    # 对同一(M, efConstruction)聚合（对 efSearch 取均值）
    agg = df.groupby(["M", "efConstruction"], as_index=False)[["build_time_s", "index_size_mb"]].mean()

    piv_build = agg.pivot_table(index="efConstruction", columns="M", values="build_time_s", aggfunc="mean")
    piv_build = piv_build.sort_index().sort_index(axis=1)
    piv_size  = agg.pivot_table(index="efConstruction", columns="M", values="index_size_mb", aggfunc="mean")
    piv_size  = piv_size.sort_index().sort_index(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cmap = "viridis"

    im1 = axes[0].imshow(piv_build.values, aspect="auto", cmap=cmap)
    _annotate_cells(axes[0], piv_build.values)
    axes[0].set_title("Build Time (s)")
    axes[0].set_xlabel("M"); axes[0].set_ylabel("efConstruction")
    axes[0].set_xticks(np.arange(piv_build.shape[1])); axes[0].set_xticklabels(piv_build.columns.astype(int))
    axes[0].set_yticks(np.arange(piv_build.shape[0])); axes[0].set_yticklabels(piv_build.index.astype(int))
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(piv_size.values, aspect="auto", cmap=cmap)
    _annotate_cells(axes[1], piv_size.values)
    axes[1].set_title("Index Size (MB)")
    axes[1].set_xlabel("M"); axes[1].set_ylabel("efConstruction")
    axes[1].set_xticks(np.arange(piv_size.shape[1])); axes[1].set_xticklabels(piv_size.columns.astype(int))
    axes[1].set_yticks(np.arange(piv_size.shape[0])); axes[1].set_yticklabels(piv_size.index.astype(int))
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("HNSW — (efConstruction × M)")
    fig.tight_layout()
    return fig
def plot_hnsw_querytime(df: pd.DataFrame):
    agg = df.groupby(["M", "efSearch"], as_index=False)["query_time_ms"].mean()
    piv = agg.pivot_table(index="efSearch", columns="M", values="query_time_ms", aggfunc="mean")
    piv = piv.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(piv.values, aspect="auto", cmap="coolwarm")
    _annotate_cells(ax, piv.values)
    ax.set_title("Query Time (ms)")
    ax.set_xlabel("M"); ax.set_ylabel("efSearch")
    ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels(piv.columns.astype(int))
    ax.set_yticks(np.arange(piv.shape[0])); ax.set_yticklabels(piv.index.astype(int))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("HNSW — (efSearch × M)")
    fig.tight_layout()
    return fig
def plot_hnsw_recall(df: pd.DataFrame, k_val: int = 20):
    agg = df.groupby(["M", "efSearch"], as_index=False)["recall_at_k"].mean()
    piv = agg.pivot_table(index="efSearch", columns="M", values="recall_at_k", aggfunc="mean")
    piv = piv.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    _annotate_cells(ax, piv.values)
    ax.set_title(f"Recall@{k_val}")
    ax.set_xlabel("M"); ax.set_ylabel("efSearch")
    ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels(piv.columns.astype(int))
    ax.set_yticks(np.arange(piv.shape[0])); ax.set_yticklabels(piv.index.astype(int))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("HNSW — (efSearch × M)")
    fig.tight_layout()
    return fig


#================== Grid Search ========================

def grid_search_ivf(

    nlist_param,              # [256, 512, 1024]
    nprobe_param,             # [1, 4, 8, 16, 32]
    pca_dim_param,            # [None, 64, 128, 256]
    indexBuilder=create_ivf_index,
    k: int = 20,
    n_per_label=None,
    save_csv=None,
    gallery_vector_path = FMNIST_VECTOR_GALLERY,
    gallery_label_path = FMNIST_LABEL_GALLERY,
    query_vector_path = FMNIST_VECTOR_QUERY,
    query_label_path = FMNIST_LABEL_QUERY

):

    # ---------- 1) Load ----------
    xb_gallery = np.load(gallery_vector_path).astype("float32", order="C")
    y_gallery  = np.load(gallery_label_path).astype(int)
    x_query    = np.load(query_vector_path).astype("float32", order="C")
    y_query    = np.load(query_label_path).astype(int)

    # Optional Startified Sampling
    if isinstance(n_per_label, int) and n_per_label > 0:
        q_keep = []
        for lb in np.unique(y_query):
            idxs = np.where(y_query == lb)[0]
            if len(idxs) == 0:
                continue
            take = min(n_per_label, len(idxs))
            picks = np.random.choice(idxs, size=take, replace=False)
            q_keep.extend(picks.tolist())
        q_keep = np.array(sorted(q_keep), dtype=int)
        xq, yq = x_query[q_keep], y_query[q_keep]
    else:
        xq, yq = x_query, y_query

    nq, d = xq.shape

    # ---------- 2) （Flat on Gallery） ----------
    temp_dir = os.path.join(os.path.dirname(gallery_vector_path), "temp_faiss")
    os.makedirs(temp_dir, exist_ok=True)
    baseline_path = os.path.join(temp_dir, "gallery_flat.index")

    index_flat = FAISS_NeighborSearch.load_or_create_index(
        vectors_path=gallery_vector_path,
        index_path=baseline_path,
        model_name="fashion-clip",
        index_factory_str="Flat",
        metric="L2",
        force_rebuild=True,
    )


    Dg, Ig = index_flat.search(xq, int(k))  # (nq, k) —— gallery

    # Count label
    _, counts = np.unique(y_gallery, return_counts=True)
    label_card = dict(zip(np.unique(y_gallery), counts))

    RESULTS = []
    PER_LABEL_ROWS = []

    # ---------- 3) Grid Search ----------
    for pca_dim in pca_dim_param:
        for nlist in nlist_param:
            for nprobe in nprobe_param:

                idx_path = os.path.join(temp_dir, "ivf.index")

                t0 = time.perf_counter()
                index = indexBuilder(
                    vectors_path=gallery_vector_path,
                    index_path=idx_path,
                    nlist=int(nlist),
                    nprobe=int(nprobe),
                    pca_dim=None if (pca_dim in [None, 0]) else int(pca_dim),
                    metric="L2",
                )
                build_time_s = time.perf_counter() - t0

                # File Size
                if os.path.exists(idx_path):
                    faiss.write_index(index, idx_path)  # ensure persisted
                    index = faiss.read_index(idx_path)  # reload for consistency
                    index_size_mb = os.path.getsize(idx_path) / (1024**2)
                else:
                    index_size_mb = np.nan

                # 3.2 Batch Query
                t1 = time.perf_counter()
                Da, Ia = index.search(xq, int(k))  # (nq, k) —— gallery
                qtime = time.perf_counter() - t1
                avg_query_ms = (qtime / max(1, nq)) * 1000.0

                # ---------- 4) Recall@k（ANN recall）for vector ----------
                overlaps = []
                for i in range(nq):
                    overlaps.append(len(set(Ia[i]).intersection(set(Ig[i]))) / float(k))
                vector_recall_at_k = float(np.mean(overlaps)) if overlaps else 0.0

                # ---------- 5) Precision@k / Recall@k for label----------
                # per-query precision@k
                # per-query recall@k
                per_query_prec = []
                per_query_reca = []
                for i in range(nq):
                    neigh_labels = y_gallery[Ia[i]]
                    hits = np.sum(neigh_labels == yq[i])
                    per_query_prec.append(hits / float(k))
                    denom = label_card.get(int(yq[i]), 0)
                    per_query_reca.append((hits / float(denom)) if denom > 0 else 0.0)

                label_precision_at_k_micro = float(np.mean(per_query_prec)) if per_query_prec else 0.0
                label_recall_at_k_micro    = float(np.mean(per_query_reca)) if per_query_reca else 0.0

                # 同时统计每个 label 的 macro 指标（按该类的 query 求均值）
                for lb in np.unique(yq):
                    mask = (yq == lb)
                    if not np.any(mask):
                        continue
                    Ia_lb = Ia[mask]
                    # precision@k for this label
                    precs = []
                    recas = []
                    denom = label_card.get(int(lb), 0)
                    for row in Ia_lb:
                        hits = np.sum(y_gallery[row] == lb)
                        precs.append(hits / float(k))
                        recas.append((hits / float(denom)) if denom > 0 else 0.0)
                    PER_LABEL_ROWS.append({
                        "pca_dim": int(pca_dim) if pca_dim not in [None, 0] else 0,
                        "nlist": int(nlist),
                        "nprobe": int(nprobe),
                        "k": int(k),
                        "label": int(lb),
                        "precision_at_k_macro": float(np.mean(precs)) if precs else 0.0,
                        "recall_at_k_macro": float(np.mean(recas)) if recas else 0.0,
                    })

                # ---------- 6) 记录一行整体结果 ----------
                RESULTS.append({
                    "pca_dim": int(pca_dim) if pca_dim not in [None, 0] else 0,
                    "nlist": int(nlist),
                    "nprobe": int(nprobe),
                    "k": int(k),
                    "build_time_s": round(build_time_s, 4),
                    "query_time_ms": round(avg_query_ms, 3),
                    "index_size_mb": round(index_size_mb, 2) if not np.isnan(index_size_mb) else np.nan,
                    "vector_recall_at_k": round(vector_recall_at_k, 4),
                    "label_precision_at_k_micro": round(label_precision_at_k_micro, 4),
                    "label_recall_at_k_micro": round(label_recall_at_k_micro, 4),
                    "n_query": int(nq),
                })

    df = pd.DataFrame(RESULTS)
    df_label = pd.DataFrame(PER_LABEL_ROWS)

    # 若指定保存路径，也把 per-label 一起保存
    if save_csv:
        base = os.path.splitext(save_csv)[0]
        df.to_csv(save_csv, index=False)
        if not df_label.empty:
            df_label.to_csv(base + "_per_label.csv", index=False)
        print(f"Saved overall to: {save_csv}")
        if not df_label.empty:
            print(f"Saved per-label to: {base + '_per_label.csv'}")

    # 返回两个 DataFrame：整体 & 每类
    return df, df_label

def grid_search_hnsw(
    M_param,                 # [16, 32, 48, 64]
    efC_param,               # [100, 200, 300]
    efS_param,               # [32, 64, 128, 256]
    indexBuilder=create_index_hnsw,
    k: int = 20,
    n_per_label: int = 2,
    save_csv: Union[str, bool] = False,
):
    """
    Grid-search for HNSW over (M, efConstruction, efSearch).
    Measures: build_time_s, query_time_ms (k), recall_at_k, index_size_mb.
    Returns: DataFrame.
    """
    labels = np.load(FAISS_NeighborSearch.IDS_PATH)
    xb = np.load(FAISS_NeighborSearch.VECTORS_PATH).astype("float32", order="C")

    baseline_path = os.path.join(os.path.dirname(FAISS_NeighborSearch.VECTORS_PATH), "baseline_flat.index")
    index_flat = FAISS_NeighborSearch.load_or_create_index(
        vectors_path=FAISS_NeighborSearch.VECTORS_PATH,
        index_path=baseline_path,
        model_name="fashion-clip",
        index_factory_str="Flat",
        metric="L2",
        force_rebuild=False,
    )

    qids = _sample_query_ids(labels, n_per_label=max(1, int(n_per_label)))
    if not qids:
        raise RuntimeError("No query ids sampled.")

    temp_dir = os.path.join(os.path.dirname(FAISS_NeighborSearch.VECTORS_PATH), "temp_faiss")
    os.makedirs(temp_dir, exist_ok=True)

    RESULTS = []

    for M in M_param:
        for efC in efC_param:
            for efS in efS_param:
                idx_path = os.path.join(temp_dir, "hnsw.index")

                t0 = time.perf_counter()
                index = indexBuilder(
                    vectors_path=FAISS_NeighborSearch.VECTORS_PATH,
                    index_path=idx_path,
                    M=int(M),
                    efConstruction=int(efC),
                    efSearch=int(efS),
                    metric="L2",
                )
                build_time_s = time.perf_counter() - t0

                faiss.write_index(index, idx_path)
                index = faiss.read_index(idx_path)
                index_size_mb = os.path.getsize(idx_path) / (1024**2)

                t1 = time.perf_counter()
                recall_sum = 0.0
                for qid in qids:
                    q = index_flat.reconstruct(int(qid)).reshape(1, -1)
                    Di, Ii = index.search(q, int(k))
                    recall_sum += float((labels[Ii[0]] == labels[qid]).mean())
                qtime = time.perf_counter() - t1
                avg_query_ms = (qtime / len(qids)) * 1000.0
                avg_recall = recall_sum / len(qids)

                RESULTS.append({
                    "M": int(M),
                    "efConstruction": int(efC),
                    "efSearch": int(efS),
                    "k": int(k),
                    "build_time_s": round(build_time_s, 4),
                    "query_time_ms": round(avg_query_ms, 3),
                    "index_size_mb": round(index_size_mb, 2),
                    "recall_at_k": round(avg_recall, 4),
                })

    df = pd.DataFrame(RESULTS)
    if save_csv:
        csv_path = save_csv if isinstance(save_csv, str) else os.path.join(temp_dir, "hnsw_grid_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    return df






if __name__ == "__main__":
    nlist = [256, 512, 1024]
    nprobe = [1, 4, 8, 16, 32]
    pca_dim = [None, 64, 128, 256,512,1024]
    K = 20


    df, df_label = grid_search_ivf(
        nlist_param=nlist,
        nprobe_param=nprobe,
        pca_dim_param=pca_dim,
        indexBuilder=create_ivf_index,
        k=K,
        n_per_label=None,
        save_csv=os.path.join(PROJECT_DIR,'data',"ivf_FMNIST_query_metrics.csv"),
    )

    # === 只输出聚合后的大图 ===
    plot_ivf_build_and_size_panels(df,'FMNIST')
    plot_ivf_querytime_by_pca(df,'FMNIST')
    plot_ivf_vector_recall_by_pca(df, 'FMNIST', k_val=K)
    plot_ivf_label_precision_by_pca(df,'FMNIST', k_val=K)
    plot_ivf_label_recall_by_pca(df, 'FMNIST',k_val=K)


    plt.show()

#Test for INSHOP Test data
    nlist = [32, 64, 128, 256]
    """ normally from 1~4 times of sqrt(N), FAISS prompts warning if centroids
    is larger than 4sqrt(N) ( approximately)
    """
    nprobe = [1, 4, 8, 16, 32]
    # Relatively fixed
    pca_dim = [None, 64, 128, 256, 512, 1024]
    K = 20

    df, df_label = grid_search_ivf(
        nlist_param=nlist,
        nprobe_param=nprobe,
        pca_dim_param=pca_dim,
        indexBuilder=create_ivf_index,
        k=K,
        n_per_label=None,
        save_csv=os.path.join(PROJECT_DIR, 'data', "ivf_INSHOP_query_metrics.csv"),
        gallery_vector_path=INSHOP_VECTOR_GALLERY,
        gallery_label_path=INSHOP_LABEL_GALLERY_NPY,
        query_vector_path=INSHOP_VECTOR_QUERY,
        query_label_path=INSHOP_LABEL_QUERY_NPY
    )

    # === 只输出聚合后的大图 ===
    plot_ivf_build_and_size_panels(df,'INSHOP')
    plot_ivf_querytime_by_pca(df,'INSHOP')
    plot_ivf_vector_recall_by_pca(df, 'INSHOP', k_val=K)
    plot_ivf_label_precision_by_pca(df,'INSHOP', k_val=K)
    plot_ivf_label_recall_by_pca(df, 'INSHOP',k_val=K)

    plt.show()

    M_list = [16, 32, 48, 64]
    efC_list = [100, 200, 300]
    efS_list = [32, 64, 128, 256]
    K = 20

    df_hnsw = grid_search_hnsw(
        M_param=M_list,
        efC_param=efC_list,
        efS_param=efS_list,
        indexBuilder=create_index_hnsw,
        k=K,
        n_per_label=2,
        save_csv=False,
    )
    print(df_hnsw.head())

    fig1 = plot_hnsw_build_and_size(df_hnsw)  # 两面板：Build / Size，维度= efConstruction × M
    fig2 = plot_hnsw_querytime(df_hnsw)  # 单图：Query Time，维度= efSearch × M（avg over efConstruction）
    fig3 = plot_hnsw_recall(df_hnsw, k_val=20)  # 单图：Recall，维度= efSearch × M（avg over efConstruction）
    plt.show()






