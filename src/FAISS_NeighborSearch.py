import os
import faiss
import numpy as np
import torch

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

# 只加载一次，后续重复复用
def load_clip_model(model_name=MODEL_NAME):
    device = get_device()
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device

# 编码单张图片 -> (1, D)
def encode_image(path, model, processor, device, normalize=True):
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs).detach().cpu().numpy().astype("float32")
    if normalize:
        # 余弦检索常用：单位化
        feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    return feats  # shape: (1, D)

# 批量编码多张图片 -> (N, D)
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





def load_or_build_index(vectors_path=VECTORS_PATH, index_path=INDEX_PATH):
    """加载已保存的 faiss 索引，如果不存在则构建并保存"""
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

    index = load_or_build_index()
    print("Index total vectors:", index.ntotal)

