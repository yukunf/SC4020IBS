# src/embedder.py
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision import models, transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm
import os
import argparse

def get_device():
    """Checks for and returns the best available computing device."""
    if torch.backends.mps.is_available():
        print("MPS device found. Using Apple GPU.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("CUDA device found. Using NVIDIA GPU.")
        return torch.device("cuda")
    else:
        print("No GPU found. Using CPU.")
        return torch.device("cpu")

def main(args):
    """Main function to run the feature extraction process."""
    device = get_device()
    data_root = args.data_root
    output_dir = args.output_dir
    batch_size = args.batch_size
    model_name = "resnet50" # Hardcode to resnet50 for now

    vectors_path = os.path.join(output_dir, f'fmnist_{model_name}_vectors.npy')
    labels_path = os.path.join(output_dir, f'fmnist_{model_name}_labels.npy')

    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Data Loading & Transformation for ResNet-50 ---
    # ResNet-50 expects 3-channel (RGB), 224x224 images, and specific normalization
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # Convert grayscale to 3-channel
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading Fashion-MNIST dataset with ResNet-50 transformations...")
    train_dataset = FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root=data_root, train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([train_dataset, test_dataset])
    data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Dataset loaded. Total images: {len(full_dataset)}")

    # --- 3. Model Loading (ResNet-50) ---
    print(f"Loading pre-trained {model_name} model...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Remove the final fully-connected layer to get features
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    print("Model loaded and configured for feature extraction.")

    # --- 4. Feature Extraction ---
    all_feature_vectors = []
    all_labels = []

    print("\nStarting feature extraction...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Extracting {model_name} features"):
            images = images.to(device)
            feature_vectors = model(images)
            all_feature_vectors.append(feature_vectors.cpu().numpy())
            all_labels.append(labels.numpy())

    # --- 5. Save Results ---
    print("\nFeature extraction complete. Concatenating and saving results...")
    final_vectors = np.concatenate(all_feature_vectors, axis=0).astype('float32')
    final_labels = np.concatenate(all_labels, axis=0)

    # No normalization needed here as we will do it in the validator/evaluator if required
    np.save(vectors_path, final_vectors)
    np.save(labels_path, final_labels)

    print(f"Successfully saved!\n - Vectors: {vectors_path} (Shape: {final_vectors.shape})\n - Labels:  {labels_path} (Shape: {final_labels.shape})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract image features using a specified model.")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory for the dataset.")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save the output vectors and labels.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing.")
    
    args = parser.parse_args()
    main(args)