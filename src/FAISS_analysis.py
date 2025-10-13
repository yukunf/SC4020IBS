import FAISS_NeighborSearch

import time, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, faiss
import math
from typing import Optional, Union

from src.FAISS_NeighborSearch import PROJECT_DIR


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
        force_rebuild=False,
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
    """在热力图上标注数字"""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if isinstance(val, (int, float)) and not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")


def plot_build_and_size_panels(df: pd.DataFrame):
    piv_build = df.pivot_table(index="pca_dim", columns="nlist", values="build_time_s", aggfunc="mean")
    piv_size  = df.pivot_table(index="pca_dim", columns="nlist", values="index_size_mb", aggfunc="mean")

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

    fig.suptitle("Build Time & Index Size vs PCA × nlist", fontsize=12)
    fig.tight_layout()
    return fig  # ✅ 返回 fig


def plot_querytime_faceted_by_pca(df: pd.DataFrame):
    """
    Facet by PCA dim.
    Each subplot: heatmap of query_time_ms with rows=nprobe, cols=nlist.
    """
    pcas = sorted(df["pca_dim"].unique())  # 0 代表 None
    import math
    cols = min(4, max(1, len(pcas)))
    rows = math.ceil(len(pcas) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.2*cols, 3.8*rows))
    axes = np.array(axes, ndmin=2)
    cmap = "coolwarm"

    used = 0
    for i, p in enumerate(pcas):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        sub = df[df["pca_dim"] == p]
        # rows=nprobe, cols=nlist
        piv = sub.pivot_table(index="nprobe", columns="nlist", values="query_time_ms", aggfunc="mean")
        piv = piv.sort_index(axis=0).sort_index(axis=1)

        im = ax.imshow(piv.values, aspect="auto", cmap=cmap)
        # 数字标注
        for ii in range(piv.shape[0]):
            for jj in range(piv.shape[1]):
                val = piv.values[ii, jj]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    ax.text(jj, ii, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")

        p_label = "None" if int(p) == 0 else str(int(p))
        ax.set_title(f"PCA={p_label}")
        ax.set_xlabel("nlist"); ax.set_ylabel("nprobe")
        ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels(piv.columns.astype(int))
        ax.set_yticks(np.arange(piv.shape[0])); ax.set_yticklabels(piv.index.astype(int))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        used += 1

    # 隐藏多余子图
    total = rows * cols
    for j in range(used, total):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.suptitle("Query Time (ms): nprobe × nlist (faceted by PCA dim)", fontsize=12)
    fig.tight_layout()
    return fig



def plot_recall_faceted(df: pd.DataFrame, k_val: int):
    nprobes = sorted(df["nprobe"].unique())
    cols = min(4, max(1, len(nprobes)))
    rows = math.ceil(len(nprobes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2*cols, 3.8*rows))
    axes = np.array(axes, ndmin=2)
    cmap = "viridis"

    for i, npb in enumerate(nprobes):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        sub = df[df["nprobe"] == npb]
        piv = sub.pivot_table(index="pca_dim", columns="nlist", values="recall_at_k", aggfunc="mean")
        im = ax.imshow(piv.values, aspect="auto", cmap=cmap)
        _annotate_cells(ax, piv.values)
        ax.set_title(f"nprobe={npb}")
        ax.set_xlabel("nlist"); ax.set_ylabel("PCA dim")
        ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels(piv.columns.astype(int))
        ax.set_yticks(np.arange(piv.shape[0])); ax.set_yticklabels(piv.index.astype(int))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(len(nprobes), rows*cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"Recall@{k_val} vs PCA × nlist (faceted by nprobe)", fontsize=12)
    fig.tight_layout()
    return fig  # ✅ 返回 fig

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
    indexBuilder=create_ivf_index,    # pass builder function in
    k: int = 20,              # ✅ 单一 k 值
    n_per_label=2,
    save_csv=None
):
    """
    Perform a grid search on (nlist, nprobe, pca_dim)
      - build_time_s, query_time_ms(k), recall_at_k, index_size_mb
    Returns：DataFrame
    """
    # --- Building data and ground truth ---
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

    # 预计算基线近邻（取到 k 即可）
    BASE_NEI = {}
    for qid in qids:
        q = index_flat.reconstruct(int(qid)).reshape(1, -1)
        Df, If = index_flat.search(q, int(k))
        BASE_NEI[qid] = If[0]

    RESULTS = []

    # --- Grid Search ---
    for pca_dim in pca_dim_param:
        for nlist in nlist_param:
            for nprobe in nprobe_param:
                # 构建&计时
                t0 = time.perf_counter()
                temp_dir = os.path.join(os.path.dirname(FAISS_NeighborSearch.VECTORS_PATH), "temp_faiss")
                os.makedirs(temp_dir, exist_ok=True)  # Create temp folder
                idx_name = f"ivf_{'pca' + str(pca_dim) + '_' if pca_dim not in [None, 0] else ''}nlist{nlist}.index"
                idx_path = os.path.join(temp_dir, idx_name)

                index = indexBuilder(
                    vectors_path=FAISS_NeighborSearch.VECTORS_PATH,
                    index_path=idx_path,
                    nlist=int(nlist),
                    nprobe=int(nprobe),
                    pca_dim=None if (pca_dim in [None, 0]) else int(pca_dim),
                    metric="L2",
                )
                build_time_s = time.perf_counter() - t0

                # 索引大小
                index_size_mb = (os.path.getsize(idx_path) / (1024**2)) if os.path.exists(idx_path) else np.nan

                # 查询一次（固定 k）
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
                    "pca_dim": int(pca_dim) if pca_dim not in [None, 0] else 0,
                    "nlist": int(nlist),
                    "nprobe": int(nprobe),
                    "k": int(k),
                    "build_time_s": round(build_time_s, 4),
                    "query_time_ms": round(avg_query_ms, 3),
                    "index_size_mb": round(index_size_mb, 2) if not np.isnan(index_size_mb) else np.nan,
                    "recall_at_k": round(avg_recall, 4),
                })

    df = pd.DataFrame(RESULTS)
    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"Saved: {save_csv}")
    return df

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
    nlist = [256, 512, 1024, 2048]  # Based on the range of sqrt(N) to 4sqrt(N): 256 ~ 1024
    nprobe = [1, 4, 8, 16, 32]
    pca_dim = [None, 64, 128, 256, 512]
    K = 20  # 单一 k

    df = grid_search_ivf(
        nlist_param=nlist,
        nprobe_param=nprobe,
        pca_dim_param=pca_dim,
        indexBuilder=create_ivf_index,
        k=K,
        n_per_label=2,
        save_csv=False,
    )

    save_dir = os.path.join(PROJECT_DIR,"data/FAISSresults/plots")
    os.makedirs(save_dir, exist_ok=True)
    fig1 = plot_build_and_size_panels(df)
    fig1.savefig(os.path.join(save_dir,"buildTime_and_size.png"), dpi=300, bbox_inches="tight")

    fig2 = plot_querytime_faceted_by_pca(df)
    fig2.savefig(os.path.join(save_dir, "query_time.png"), dpi=300, bbox_inches="tight")
    fig3 = plot_recall_faceted(df, K)
    fig3.savefig(os.path.join(save_dir, "recall.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # M_list = [16, 32, 48, 64]
    # efC_list = [100, 200, 300]
    # efS_list = [32, 64, 128, 256]
    # K = 20
    #
    # df_hnsw = grid_search_hnsw(
    #     M_param=M_list,
    #     efC_param=efC_list,
    #     efS_param=efS_list,
    #     indexBuilder=create_index_hnsw,
    #     k=K,
    #     n_per_label=2,
    #     save_csv=False,
    # )
    # print(df_hnsw.head())
    #
    # fig1 = plot_hnsw_build_and_size(df_hnsw)  # 两面板：Build / Size，维度= efConstruction × M
    # fig2 = plot_hnsw_querytime(df_hnsw)  # 单图：Query Time，维度= efSearch × M（avg over efConstruction）
    # fig3 = plot_hnsw_recall(df_hnsw, k_val=20)  # 单图：Recall，维度= efSearch × M（avg over efConstruction）
    # plt.show()






