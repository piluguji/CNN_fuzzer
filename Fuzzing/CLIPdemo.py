import random
from PIL import Image, ImageEnhance, ImageDraw
import clip
import torch

def mutate_image(image):
    """
    Apply mutations to the input image, including:
      1. Adjusting brightness
      2. Adjusting contrast
      3. Adjusting saturation (color intensity)
      4. Adding occlusion (a random black rectangle)
    Returns the mutated image.
    """
    # Adjust brightness
    brightness_factor = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Brightness(image)
    mutated = enhancer.enhance(brightness_factor)

    # Adjust contrast
    contrast_factor = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Contrast(mutated)
    mutated = enhancer.enhance(contrast_factor)

    # Adjust saturation
    saturation_factor = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Color(mutated)
    mutated = enhancer.enhance(saturation_factor)

    # Add multiple small semi-transparent occlusions
    width, height = mutated.size
    draw = ImageDraw.Draw(mutated, mode="RGBA")

    num_blocks = random.randint(50, 150)  # number of small blocks
    for _ in range(num_blocks):
        block_width = int(width * random.uniform(0.01, 0.03))
        block_height = int(height * random.uniform(0.01, 0.03))
        block_x = random.randint(0, width - block_width)
        block_y = random.randint(0, height - block_height)

        # Draw semi-transparent gray rectangles (RGBA: (R,G,B,A))
        alpha = random.randint(80, 150)  # transparency: 0=fully transparent, 255=opaque
        draw.rectangle(
            [block_x, block_y, block_x + block_width, block_y + block_height],
            fill=(128, 128, 128, alpha)
        )

    # Convert back to RGB (removing alpha for saving/display)
    mutated = mutated.convert("RGB")

    return mutated

if __name__ == '__main__':
    # Choose device: MPS (Apple Silicon) or CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load original image
    image_path = "../dataset_split/test/0/000_0019.png"  # replace with your image path
    original_image = Image.open(image_path).convert("RGB")

    # Apply mutation
    mutated_image = mutate_image(original_image)
    mutated_image.save("mutated_image.png")  # Optional: Save for reference

    # Preprocess images for CLIP
    image1 = preprocess(original_image).unsqueeze(0).to(device)
    image2 = preprocess(mutated_image).unsqueeze(0).to(device)

    # Compute semantic similarity
    with torch.no_grad():
        feature1 = model.encode_image(image1)
        feature2 = model.encode_image(image2)

    cos_sim = torch.nn.functional.cosine_similarity(feature1, feature2)
    print(f"CLIP Semantic Similarity: {cos_sim.item():.4f}")