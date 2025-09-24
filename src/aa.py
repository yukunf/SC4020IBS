import numpy as np

# 加载特征向量和标签
vectors = np.load('data/fmnist_resnet50_vectors.npy')
labels = np.load('data/fmnist_resnet50_labels.npy')

# 打印形状以确认
print('Vectors shape:', vectors.shape)
print('Labels shape:', labels.shape)

# 现在你们可以使用'vectors'和'labels'来进行暴力搜索、LSH或Faiss的构建和评测了
# 例如，对于任务B，可以将'vectors'作为搜索库
# 对于任务C，可以将'vectors'添加到Faiss索引中