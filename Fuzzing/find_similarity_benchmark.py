import os
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt

# Device configuration
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Dataset path configuration
dataset_path = "../dataset_split/test"
folders_to_test = [str(i) for i in range(58)]  # folders 0-57

# Create directory for saving plots and CSVs
save_dir = "similarities"
os.makedirs(save_dir, exist_ok=True)

# Function to compute similarity
def compute_similarity(img_path1, img_path2):
    image1 = preprocess(Image.open(img_path1)).unsqueeze(0).to(device)
    image2 = preprocess(Image.open(img_path2)).unsqueeze(0).to(device)

    with torch.no_grad():
        feature1 = model.encode_image(image1)
        feature2 = model.encode_image(image2)

    similarity = torch.nn.functional.cosine_similarity(feature1, feature2).item()
    return similarity

# Main function
def calculate_similarity_for_folder(folder_path):
    basename = os.path.basename(folder_path)
    images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.lower().endswith(('png', 'jpg', 'jpeg'))]

    if len(images) <= 1:
        print(f"Folder {basename} skipped due to insufficient images.")
        return

    similarities = []
    for img1, img2 in tqdm(itertools.product(images, images), total=len(images)**2, desc=f"Folder {basename}"):
        sim = compute_similarity(img1, img2)
        similarities.append(sim)

    similarities_array = np.array(similarities)
    mean_sim = np.mean(similarities_array)
    std_sim = np.std(similarities_array)

    # Save raw similarity values to CSV
    df_values = pd.DataFrame({'similarity': similarities_array})
    values_csv_path = os.path.join(save_dir, f"folder_{basename}_values.csv")
    df_values.to_csv(values_csv_path, index=False)

    # Save summary (mean and std) to CSV
    df_summary = pd.DataFrame({
        'folder': [basename],
        'mean_similarity': [mean_sim],
        'std_similarity': [std_sim]
    })
    summary_csv_path = os.path.join(save_dir, f"folder_{basename}_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(similarities_array, bins=50, alpha=0.75)
    plt.title(f"Semantic Similarity Distribution - Folder {basename}")
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.figtext(0.5, -0.05, f"Folder {basename} - Mean Similarity: {mean_sim:.4f}, Std: {std_sim:.4f}", ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(save_dir, f"folder_{basename}_similarity.png")
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    for folder in folders_to_test:
        folder_path = os.path.join(dataset_path, folder)
        print(f"Processing folder: {folder}")
        calculate_similarity_for_folder(folder_path)

    print("Benchmarking completed.")
