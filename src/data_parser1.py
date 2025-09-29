import os
import pandas as pd
from collections import defaultdict

# 定义基础目录
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
data1_dir = os.path.join(base_dir, 'data1')
img_dir = os.path.join(data1_dir, 'img')
eval_dir = os.path.join(data_dir, 'Eval')

partition_file_path = os.path.join(eval_dir, 'list_eval_partition.txt')
output_csv_path = os.path.join(data_dir, 'deepfashion_metadata_final.csv')

def create_deepfashion_metadata():
    """为DeepFashion In-shop数据集创建详细的元数据文件。"""
    # 1. 读取分区文件，创建 item_id 到 evaluation_status 的映射
    print(f"Reading partition file from {partition_file_path}...")
    try:
        partition_df = pd.read_csv(partition_file_path, sep=r'\s+', engine='python', skiprows=2,
                                   names=['image_name', 'item_id', 'evaluation_status'])
        item_status_map = partition_df.set_index('item_id')['evaluation_status'].to_dict()
        print(f"Loaded status for {len(item_status_map)} unique item_ids.")
    except Exception as e:
        print(f"Error reading partition file: {e}")
        return

    # 2. 遍历图片目录，提取信息
    metadata = []
    print(f"Walking through {img_dir} to gather image metadata...")
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                try:
                    # 从路径中解析 gender, category, item_id
                    relative_path = os.path.relpath(root, img_dir)
                    path_parts = relative_path.split(os.sep)
                    
                    if len(path_parts) >= 3 and path_parts[2].startswith('id_'):
                        gender = path_parts[0]
                        category = path_parts[1]
                        item_id = path_parts[2]
                        
                        # 获取评估状态
                        evaluation_status = item_status_map.get(item_id, 'unknown')
                        
                        metadata.append({
                            'full_path': full_path,
                            'item_id': item_id,
                            'gender': gender,
                            'category': category,
                            'evaluation_status': evaluation_status
                        })
                except Exception as e:
                    print(f"Could not process file {full_path}: {e}")

    if not metadata:
        print("No metadata was generated. Please check paths and file formats.")
        return

    # 3. 保存到CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_csv_path, index=False)
    print(f"\nSuccessfully created metadata file with {len(metadata_df)} entries.")
    print(f"Saved to {output_csv_path}")
    print("\n--- Sample of the new metadata ---")
    print(metadata_df.head())

    # 验证评估集的完整性
    if not metadata_df.empty:
        print("\n--- Verifying dataset split ---")
        print("Images per set:")
        print(metadata_df['evaluation_status'].value_counts())
        print("\nUnique item_ids per set:")
        print(metadata_df.groupby('evaluation_status')['item_id'].nunique())

if __name__ == '__main__':
    create_deepfashion_metadata()
