import json
import math
import os
import time

import faiss
import hashlib
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Torch和FAISS会双开OpenMP
#from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /path/to/project/src
MODEL_NAME = 'patrickjohncyh/fashion-clip'
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # /path/to/project

# Config dataset
VECTORS_PATH = os.path.join(PROJECT_DIR, "data", "fmnist_resnet50_vectors.npy")
IDS_PATH = os.path.join(PROJECT_DIR, "data", "fmnist_resnet50_labels.npy")
INDEX_PATH = os.path.join(PROJECT_DIR, "data", "fmnist_resnet50.index")

INSHOP_VECTOR_GALLERY = os.path.join(PROJECT_DIR, "data", "inshop_clip_vectors_gallery.npy")
INSHOP_VECTOR_QUERY = os.path.join(PROJECT_DIR, "data", "inshop_clip_vectors_query.npy")
INSHOP_LABEL_GALLERY_NPY = os.path.join(PROJECT_DIR, "data", "inshop_clip_labels_gallery.npy")
INSHOP_LABEL_QUERY_NPY = os.path.join(PROJECT_DIR, "data", "inshop_clip_labels_query.npy")


FMNIST_VECTOR_GALLERY = os.path.join(PROJECT_DIR, "data", "FMNIST_gallery_vectors.npy")
FMNIST_LABEL_GALLERY = os.path.join(PROJECT_DIR, "data", "FMNIST_gallery_labels.npy")
FMNIST_VECTOR_QUERY = os.path.join(PROJECT_DIR, "data", "FMNIST_query_vectors.npy")
FMNIST_LABEL_QUERY = os.path.join(PROJECT_DIR, "data", "FMNIST_query_labels.npy")


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# def load_clip_model(model_name=MODEL_NAME):
#     device = get_device()
#     model = CLIPModel.from_pretrained(model_name).to(device).eval()
#     processor = CLIPProcessor.from_pretrained(model_name)
#     return model, processor, device


def encode_image(path, model, processor, device, normalize=True):
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs).detach().cpu().numpy().astype("float32")
    if normalize:
        # 余弦检索常用：单位化
        feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    return feats  # shape: (1, D)


def encode_images(paths, model, processor, device, batch_size=64, normalize=True):
    embs = []
    for i in range(0, len(paths), batch_size):
        batch_imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch_size]]
        with torch.no_grad():
            inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
            feats = model.get_image_features(**inputs).detach().cpu().numpy().astype("float32")
        embs.append(feats)
    if not embs:
        return np.zeros((0, model.config.projection_dim), dtype="float32")
    embs = np.vstack(embs)
    if normalize:
        embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return embs  # shape: (N, D)


# Tool function for md5
def md5sum(path, chunk_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_or_build_index(vectors_path=VECTORS_PATH, index_path=INDEX_PATH):
    """加载已保存的 faiss 索引，无则构建"""
    if os.path.exists(index_path):
        print(f"Loading existing index from {index_path}")
        index = faiss.read_index(index_path)
    else:
        print("Index not found, building a new one...")
        gallery = np.load(vectors_path).astype("float32")
        dim = gallery.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(gallery)
        faiss.write_index(index, index_path)
        print(f"New index built and saved to {index_path}")
    return index

def load_or_create_index(
    vectors_path=VECTORS_PATH,
    index_path=INDEX_PATH,
    model_name="FAISS-fashion",
    index_factory_str="Flat",
    metric="L2",  # or "IP"
    force_rebuild=False,
):
    """
    Create index by FAISS factory, can change the index type by changing strings for the factory function.
      index_factory_str :
        "Flat"
        "IVF1024,Flat"
        "IVF1024,PQ64"
        "HNSW32"
        "PCA256,Flat"
    """
    meta_path = index_path + ".meta.json"


    if os.path.exists(index_path) and not force_rebuild:
        # 载入 meta 验证
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

        # 若 meta 匹配，直接加载
        if (
            meta.get("model_name") == model_name
            and meta.get("index_type") == index_factory_str
            and meta.get("vectors_md5") == md5sum(vectors_path)
        ):
            print(f"Loading existing index from {index_path}")
            return faiss.read_index(index_path)

        print("Meta changed — rebuilding index...")


    print("Building new index:", index_factory_str)
    xb = np.load(vectors_path).astype("float32", order="C")
    d = xb.shape[1]
    print("Vectors dimension:", d)
    metric_type = faiss.METRIC_L2 if metric.upper() == "L2" else faiss.METRIC_INNER_PRODUCT

    index = faiss.index_factory(d, index_factory_str, metric_type)

    # 若索引需要训练（如 IVF, PQ）
    if index.is_trained:
        print("Index already trained.")
    else:
        ntrain = min(len(xb), 50000)
        print(f"Training index on {ntrain} vectors...")
        index.train(xb[:ntrain])

    # 添加数据
    index.add(xb)
    faiss.write_index(index, index_path)

    # -------------------------------
    # 写 meta 文件
    # -------------------------------
    meta = {
        "model_name": model_name,
        "index_type": index_factory_str,
        "metric": metric,
        "dim": d,
        "vectors_path": vectors_path,
        "vectors_md5": md5sum(vectors_path),
        "index_md5": md5sum(index_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Index and meta saved to {index_path}")

    return index

# def query_neighbors(index, queries, k=5):
#     # 保证 numpy float32
#     if isinstance(queries, torch.Tensor):
#         queries = queries.detach().cpu().numpy()
#     queries = queries.astype("float32")
#
#     # 保证二维 (n, d)
#     if queries.ndim == 1:
#         queries = queries[None, :]  # 加 batch 维
#
#     # Sanity check
#     assert queries.shape[1] == index.d, \
#         f"query 维度({queries.shape[1]}) != index 维度({index.d})"
#
#     # 搜索
#     distances, indices = index.search(queries, k)
#     return distances, indices


def evaluate_index_factory(
    index_factory_str: str,
    x_gallery: np.ndarray,
    y_gallery: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
    *,
    k: int = 50,
    metric: str = "L2",
    train_size_cap: int = 50000,
    index_path: str = None,
    params: dict = None
):
    """
    使用 FAISS factory 构建并评测索引。
    增加：vector_recall@k（与 Flat 基线重合度）
    """
    x_gallery = np.asarray(x_gallery, dtype="float32", order="C")
    x_query   = np.asarray(x_query,   dtype="float32", order="C")
    y_gallery = np.array(list(map(str, y_gallery)))
    y_query   = np.array(list(map(str, y_query)))

    d = x_gallery.shape[1]
    nq = len(x_query)
    k = int(min(k, len(x_gallery)))
    params = params or {}

    metric_type = faiss.METRIC_L2 if metric.upper() == "L2" else faiss.METRIC_INNER_PRODUCT

    # --- 1. 构建索引 + 计时 ---
    t0 = time.perf_counter()
    index = faiss.index_factory(d, index_factory_str, metric_type)
    if not index.is_trained:
        ntrain = min(len(x_gallery), train_size_cap)
        index.train(x_gallery[:ntrain])
    index.add(x_gallery)
    build_time_s = time.perf_counter() - t0

    # 设置检索参数
    if params:
        try:
            ps = faiss.ParameterSpace()
            for k_param, v_param in params.items():
                ps.set_index_parameter(index, k_param, v_param)
        except Exception:
            pass

    # --- 2. 保存并统计体积 ---
    if index_path is not None:
        faiss.write_index(index, index_path)
        index = faiss.read_index(index_path)
        index_size_mb = os.path.getsize(index_path) / (1024.0 ** 2)
    else:
        index_size_mb = float("nan")

    # --- 3. Flat 基线（用于向量 recall）---
    flat = faiss.IndexFlatL2(d)
    flat.add(x_gallery)
    _, baseline_I = flat.search(x_query, k)

    # --- 4. 批量查询 ---
    t1 = time.perf_counter()
    D, I = index.search(x_query, k)
    time_milli = ((time.perf_counter() - t1) * 1000.0)
    avg_query_ms =  time_milli / max(1, nq)
    qps = max(1,nq) / time_milli * 1000.0


    # --- 5. 计算指标 ---
    # 向量命中率（vector recall）
    overlaps = [len(set(I[i]).intersection(set(baseline_I[i]))) / float(k) for i in range(nq)]
    vector_recall_at_k = float(np.mean(overlaps)) if overlaps else 0.0

    # 标签命中率（Acc@1/10/50）
    topk_labels = y_gallery[I]
    def acc_at_k(K):
        K = min(K, k)
        hits = (topk_labels[:, :K] == y_query[:, None]).any(axis=1)
        return float(hits.mean()) if nq > 0 else 0.0

    acc1, acc10, acc50 = acc_at_k(1), acc_at_k(10), acc_at_k(50)

    # --- 6. 打印结果 ---
    param_str = ", ".join(f"{kk}={vv}" for kk, vv in params.items()) or "-"
    print(
        f"[{index_factory_str:12s}] Params: {param_str:18s} | "
        f"Acc@1={acc1:.4f}  Acc@10={acc10:.4f}  Acc@50={acc50:.4f}  "
        f"VecRecall={vector_recall_at_k:.4f} | "
        f"Build={build_time_s:.3f}s  Query={avg_query_ms:.3f}ms Query/sec={qps:.1f}  "
        f"Size={index_size_mb:.2f}MB  n_query={nq}"
    )

    return {
        "index_type": index_factory_str,
        "params": params,
        "k_eval": k,
        "acc@1": acc1,
        "acc@10": acc10,
        "acc@50": acc50,
        "vector_recall@k": vector_recall_at_k,
        "build_time_s": build_time_s,
        "avg_query_ms": avg_query_ms,
        "query_per_s": qps,
        "index_size_mb": index_size_mb,
        "n_query": nq,
    }

if __name__ == "__main__":
    Xg = np.load(INSHOP_VECTOR_GALLERY).astype("float32", order="C")
    Lg = np.load(INSHOP_LABEL_GALLERY_NPY)
    Xq = np.load(INSHOP_VECTOR_QUERY).astype("float32", order="C")
    Lq = np.load(INSHOP_LABEL_QUERY_NPY)

    test_list = [
        ("Flat", None),
        ("PCA64,Flat", None),
        ("PCA256,Flat", None),
        ("IVF1024,Flat", {"nprobe": 16}),
        ("IVF1024,PQ64", {"nprobe": 16}),
        ("HNSW32", {"efSearch": 64}),
    ]

    out_dir = os.path.join(PROJECT_DIR, "data", "tmp_faiss_bench")
    os.makedirs(out_dir, exist_ok=True)

    for name, params in test_list:
        idx_path = os.path.join(out_dir, f"{name.replace(',', '_')}.index")
        evaluate_index_factory(
            index_factory_str=name,
            x_gallery=Xg,
            y_gallery=Lg,
            x_query=Xq,
            y_query=Lq,
            k=50,
            metric="L2",
            index_path=idx_path,
            params=params
        )
