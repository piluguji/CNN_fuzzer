import clip
import torch
from PIL import Image

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model, preprocess = clip.load("ViT-B/32", device=device)

image1 = preprocess(Image.open("../dataset_split/test/0/000_0019.png")).unsqueeze(0).to(device)
image2 = preprocess(Image.open("../mud_pic.png")).unsqueeze(0).to(device)

with torch.no_grad():
    feature1 = model.encode_image(image1)
    feature2 = model.encode_image(image2)

cos_sim = torch.nn.functional.cosine_similarity(feature1, feature2)
print(f"CLIP Semantic Similarity: {cos_sim.item():.4f}")
