import os, json, numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /path/to/project/src
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

INSHOP_VECTOR_GALLERY = os.path.join(PROJECT_DIR, "data", "inshop_clip_vectors_gallery.npy")
INSHOP_VECTOR_QUERY = os.path.join(PROJECT_DIR, "data", "inshop_clip_vectors_query.npy")
INSHOP_VECTOR_GALLERY_IDS = os.path.join(PROJECT_DIR, "data", "inshop_clip_ids_gallery.json")
INSHOP_VECTOR_GALLERY_IDS_NPY = os.path.join(PROJECT_DIR, "data", "inshop_clip_labels_gallery.npy")
INSHOP_VECTOR_QUERY_IDS = os.path.join(PROJECT_DIR, "data", "inshop_clip_ids_query.json")
INSHOP_VECTOR_QUERY_IDS_NPY = os.path.join(PROJECT_DIR, "data", "inshop_clip_labels_query.npy")

VECTORS_PATH = os.path.join(PROJECT_DIR, "data", "fmnist_resnet50_vectors.npy")
IDS_PATH = os.path.join(PROJECT_DIR, "data", "fmnist_resnet50_labels.npy")
INDEX_PATH = os.path.join(PROJECT_DIR, "data", "fmnist_resnet50.index")


def stratified_split_and_save_npy(
    vectors_path: str,
    labels_path: str,
    test_ratio: float = 0.2,
    seed: int = 42,
    min_class_size: int = 2,
):
    """
    按 label 分层划分 vectors + labels，并各自保存为 npy 文件。

    Parameters
    ----------
    vectors_path : str
        原始向量文件路径 (N, D)
    labels_path : str
        标签文件路径 (N,)
    out_dir : str
        输出文件夹路径（将创建）
    test_ratio : float
        测试集比例（默认 0.2）
    seed : int
        随机种子
    min_class_size : int
        最小类样本数，太小则全划为 gallery

    输出文件
    ----------

            gallery_vectors.npy
            gallery_labels.npy
            query_vectors.npy
            query_labels.npy
    """

    vectors = np.load(vectors_path).astype("float32")
    labels = np.load(labels_path).astype(int)

    N = len(labels)
    assert len(vectors) == N, "Length doesn't match"

    np.random.seed(seed)
    gallery_idx, query_idx = [], []

    # === 按 label 分层 ===
    for lb in np.unique(labels):
        idxs = np.where(labels == lb)[0]
        if len(idxs) < min_class_size:
            gallery_idx.extend(idxs)
            continue
        np.random.shuffle(idxs)
        cut = max(1, int(len(idxs) * (1 - test_ratio)))
        gallery_idx.extend(idxs[:cut])
        query_idx.extend(idxs[cut:])

    gallery_idx = np.array(sorted(gallery_idx), dtype=int)
    query_idx = np.array(sorted(query_idx), dtype=int)

    # === 划分数据 ===
    vectors_gallery = vectors[gallery_idx]
    vectors_query = vectors[query_idx]
    labels_gallery = labels[gallery_idx]
    labels_query = labels[query_idx]

    # === 保存 ===
    np.save(os.path.join(PROJECT_DIR,"data", "FMNIST_gallery_vectors.npy"), vectors_gallery)
    np.save(os.path.join(PROJECT_DIR,"data",  "FMNIST_gallery_labels.npy"), labels_gallery)
    np.save(os.path.join(PROJECT_DIR,"data",  "FMNIST_query_vectors.npy"), vectors_query)
    np.save(os.path.join(PROJECT_DIR,"data",  "FMNIST_query_labels.npy"), labels_query)

    print(f"[split] Total={N}, Gallery={len(gallery_idx)}, Query={len(query_idx)} ({test_ratio*100:.1f}% test)")
    print(f"[save] Files saved to: data")
    return {
        "gallery_idx": gallery_idx,
        "query_idx": query_idx,
        "gallery_vectors": vectors_gallery.shape,
        "query_vectors": vectors_query.shape
    }

if __name__ == "__main__":
    # Here to convert label data from json to npy
    label = []
    with open(INSHOP_VECTOR_GALLERY_IDS, "r") as f:
        js = json.load(f)
        for item in js:
            label.append(item['label'])
    print(label)
    np.save(INSHOP_VECTOR_GALLERY_IDS_NPY, np.array(label))

    label = []
    with open(INSHOP_VECTOR_QUERY_IDS, "r") as f:
        js = json.load(f)
        for item in js:
            label.append(item['label'])
    print(label)
    np.save(INSHOP_VECTOR_QUERY_IDS_NPY, np.array(label))


    # Split FMNIST Data

    stratified_split_and_save_npy(VECTORS_PATH,IDS_PATH,0.2,42,2)

