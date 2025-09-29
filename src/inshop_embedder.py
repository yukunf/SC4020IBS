import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np
import os
import json

# --- 配置 ---
METADATA_FILE = 'data/deepfashion_metadata_final.csv'
MODEL_NAME = 'patrickjohncyh/fashion-clip' # 使用 Fashion-CLIP 模型
BATCH_SIZE = 64

# --- 输出文件路径 ---
OUTPUT_DIR = 'data'

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def embed_images(image_paths, model, processor, device):
    embeddings = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Embedding Images"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        try:
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = processor(text=None, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            
            embeddings.append(image_features.cpu().numpy())
        except Exception as e:
            print(f"Error processing batch starting with {batch_paths[0]}: {e}")
            # 添加一个与模型输出维度匹配的零向量或选择跳过
            # 这里我们选择跳过，但要注意这可能导致后续处理问题
            continue
        
    if not embeddings:
        return np.array([])
    return np.vstack(embeddings)

def main():
    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading CLIP model: {MODEL_NAME}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    print(f"Loading metadata from {METADATA_FILE}...")
    if not os.path.exists(METADATA_FILE):
        print(f"FATAL: Metadata file not found at '{METADATA_FILE}'. Please run the data preparation script first.")
        return
        
    df = pd.read_csv(METADATA_FILE)

    status_map = {
        'train': 'train',
        'val': 'query',
        'test': 'gallery'
    }

    # --- 按 evaluation_status ('train', 'val', 'test') 分别处理 ---
    for status in ['train', 'val', 'test']:
        print(f"\nProcessing '{status}' images...")
        
        df_subset = df[df['evaluation_status'] == status].copy()
        
        if df_subset.empty:
            print(f"No '{status}' images found in metadata. Skipping.")
            continue

        image_paths = df_subset['full_path'].tolist()
        existent_paths = [p for p in image_paths if os.path.exists(p)]
        print(f"Found {len(df_subset)} entries, {len(existent_paths)} images exist on disk.")

        if not existent_paths:
            print("No image files to process.")
            continue

        df_subset = df_subset[df_subset['full_path'].isin(existent_paths)]

        vectors = embed_images(existent_paths, model, processor, device)
        
        if vectors.size == 0:
            print(f"No vectors were generated for '{status}'. Skipping file save.")
            continue

        output_status = status_map[status]
        # 定义输出文件路径
        output_vectors_path = os.path.join(OUTPUT_DIR, f'inshop_clip_vectors_{output_status}.npy')
        output_ids_path = os.path.join(OUTPUT_DIR, f'inshop_clip_ids_{output_status}.json')

        np.save(output_vectors_path, vectors)
        df_subset[['item_id', 'full_path', 'gender', 'category']].to_json(output_ids_path, orient='records', lines=True)
        
        print(f"Saved {len(vectors)} vectors to {output_vectors_path}")
        print(f"Saved {len(df_subset)} IDs to {output_ids_path}")

    print("\n--- Feature Extraction Complete ---")

if __name__ == '__main__':
    main()