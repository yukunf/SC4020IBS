import json
import os
import faiss
import hashlib
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Torch和FAISS会双开OpenMP
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /path/to/project/src
MODEL_NAME = 'patrickjohncyh/fashion-clip'
PROJECT_DIR = os.path.dirname(BASE_DIR)  # /path/to/project

# Config dataset
VECTORS_PATH = os.path.join(PROJECT_DIR, "data", "fmnist_resnet50_vectors.npy")
IDS_PATH = os.path.join(PROJECT_DIR, "data", "fmnist_resnet50_labels.npy")
INDEX_PATH = os.path.join(PROJECT_DIR, "data", "fmnist_resnet50.index")


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_clip_model(model_name=MODEL_NAME):
    device = get_device()
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


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
            print(f"✅ Loading existing index from {index_path}")
            return faiss.read_index(index_path)

        print("⚠️ Meta changed — rebuilding index...")


    print("🔧 Building new index:", index_factory_str)
    xb = np.load(vectors_path).astype("float32", order="C")
    d = xb.shape[1]
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
    print(f"✅ Index and meta saved to {index_path}")

    return index

def query_neighbors(index, queries, k=5):
    # 保证 numpy float32
    if isinstance(queries, torch.Tensor):
        queries = queries.detach().cpu().numpy()
    queries = queries.astype("float32")

    # 保证二维 (n, d)
    if queries.ndim == 1:
        queries = queries[None, :]  # 加 batch 维

    # Sanity check
    assert queries.shape[1] == index.d, \
        f"query 维度({queries.shape[1]}) != index 维度({index.d})"

    # 搜索
    distances, indices = index.search(queries, k)
    return distances, indices
if __name__ == "__main__":
    labels = np.load(IDS_PATH)

    # -----------------------------
    # 1️⃣ 基础 Flat 索引
    # -----------------------------
    index_flat = load_or_create_index(
        vectors_path=VECTORS_PATH,
        index_path=INDEX_PATH,
        model_name="fashion-clip",
        index_factory_str="Flat",
        metric="L2"
    )

    print("Index total vectors:", index_flat.ntotal)

    ntotal = index_flat.ntotal
    rand_idx = np.random.randint(0, ntotal)
    query_vec = index_flat.reconstruct(rand_idx).reshape(1, -1)

    k = 10
    D_flat, I_flat = index_flat.search(query_vec, k)

    print(f"\n🔍 [Flat] Random query vector ID: {rand_idx} (label={labels[rand_idx]})")
    print(f"Top-{k} nearest neighbor IDs:", I_flat[0])
    print(f"Corresponding labels:", labels[I_flat[0]])
    print(f"Distances:", D_flat[0])

    # -----------------------------
    # 简单 recall@k 检验 (Flat 理论=1.0)
    # -----------------------------
    recall_flat = np.mean([
        len(set(I_flat[i, :k]) & set(I_flat[i, :k])) / k
        for i in range(I_flat.shape[0])
    ])
    print(f"[Flat] Recall@{k}: {recall_flat:.3f}")

    # -----------------------------
    # 2️⃣ PCA 降维版索引测试
    # -----------------------------
    INDEX_PATH_PCA = INDEX_PATH.replace(".index", "_pca.index")
    index_pca = load_or_create_index(
        vectors_path=VECTORS_PATH,
        index_path=INDEX_PATH_PCA,
        model_name="fashion-clip",
        index_factory_str="PCA256,Flat",   # ← 用 factory 构建 PCA 降维索引
        metric="L2"
    )

    D_pca, I_pca = index_pca.search(query_vec, k)

    print(f"\n🔍 [PCA] Random query vector ID: {rand_idx} (label={labels[rand_idx]})")
    print(f"Top-{k} nearest neighbor IDs:", I_pca[0])
    print(f"Corresponding labels:", labels[I_pca[0]])
    print(f"Distances:", D_pca[0])

    # -----------------------------
    # 与 Flat 的 Recall 对比
    # -----------------------------
    recall_pca = np.mean([
        len(set(labels[I_pca[i, :k]]) & set(labels[I_flat[i, :k]]))
        / len(set(labels[I_flat[i, :k]]))
        for i in range(I_pca.shape[0])
    ])
    print(f"[PCA] Label-level Recall@{k} vs Flat ground truth: {recall_pca:.3f}")

    # -----------------------------
    # 结果总结
    # -----------------------------
    print("\n📊 Summary:")
    print(f"Flat Recall@{k}: {recall_flat:.3f}")
    print(f"PCA Recall@{k}:  {recall_pca:.3f}")

