# traffic_sign_fuzzer.py
"""
Adversarial fuzzing script for traffic‑sign CNN classifier.

Workflow per image (folder 0 in dataset_split/test):
1. Verify original prediction is correct. Record baseline confidence.
2. For each perturbation type (brightness, contrast, saturation, occlusion):
   a. Iteratively increase magnitude (DFS‑like) while CLIP semantic similarity ≥ 0.88.
   b. If prediction flips to any other class and similarity ≥ 0.88 ⇒ save as adversarial case and
      break to next perturbation type.
   c. If prediction unchanged but confidence drops ⇒ continue increasing magnitude.
   d. If similarity < 0.88 ⇒ stop current perturbation type.

All discovered adversarial images are saved under:
    adversarial_cases/0/

Requirements:
    pip install pillow numpy tqdm tensorflow keras torch clip-anytorch
"""

import os
import random
import itertools
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageDraw

import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.models import load_model

import torch
import clip

# ------------------------ Configuration ------------------------
DATASET_DIR = Path("../dataset_split/test/0")          # folder to fuzz
MODEL_PATH = "../traffic_sign_model.keras"              # trained classifier
SAVE_ROOT = Path("adversarial_cases")                # root for saving cases

SIM_THRESHOLD = 0.88                                 # CLIP similarity gate
MAX_STEPS = 10                                       # max magnitude increments per type

# perturbation magnitudes for each step (small → large)
BRIGHTNESS_STEPS = np.linspace(1.05, 0.4, MAX_STEPS)  # >1 brightens, <1 darkens
CONTRAST_STEPS   = np.linspace(1.05, 0.4, MAX_STEPS)
SATURATION_STEPS = np.linspace(1.05, 0.4, MAX_STEPS)
OCCLUSION_STEPS  = np.linspace(0.01, 0.08, MAX_STEPS)  # fraction of img area per block

# Make sure save directory exists
(SAVE_ROOT / "0").mkdir(parents=True, exist_ok=True)

# ------------------------ Load models -------------------------
print("Loading CNN model …")
cnn_model = load_model(MODEL_PATH)

print("Loading CLIP …")
device = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# -------------------- Utility functions ----------------------

def keras_predict(img: Image.Image):
    """Return (predicted_class, confidence) for a 32×32 RGB image."""
    arr = kimage.img_to_array(img.resize((32, 32))) / 255.0
    arr = np.expand_dims(arr, 0)
    preds = cnn_model.predict(arr, verbose=0)
    return np.argmax(preds), float(np.max(preds))

def clip_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Compute cosine similarity between two images using CLIP."""
    with torch.no_grad():
        t1 = clip_preprocess(img1).unsqueeze(0).to(device)
        t2 = clip_preprocess(img2).unsqueeze(0).to(device)
        f1 = clip_model.encode_image(t1)
        f2 = clip_model.encode_image(t2)
        sim = torch.nn.functional.cosine_similarity(f1, f2).item()
    return sim

# ------------------- Perturbation helpers --------------------

def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)

def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)

def adjust_saturation(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(factor)

def add_occlusion(img: Image.Image, frac: float, blocks: int = 50) -> Image.Image:
    """Add many semi‑transparent grey blocks covering ~frac of total area (each)."""
    w, h = img.size
    mutated = img.convert("RGBA")
    draw = ImageDraw.Draw(mutated, "RGBA")
    for _ in range(blocks):
        bw = int(w * random.uniform(frac * 0.8, frac * 1.2))
        bh = int(h * random.uniform(frac * 0.8, frac * 1.2))
        bx = random.randint(0, w - bw)
        by = random.randint(0, h - bh)
        alpha = random.randint(80, 150)
        draw.rectangle([bx, by, bx + bw, by + bh], fill=(128, 128, 128, alpha))
    return mutated.convert("RGB")

PERTURBATION_TYPES = [
    ("brightness", adjust_brightness, BRIGHTNESS_STEPS),
    ("contrast",   adjust_contrast,   CONTRAST_STEPS),
    ("saturation", adjust_saturation, SATURATION_STEPS),
    ("occlusion",  add_occlusion,    OCCLUSION_STEPS),
]

# ----------------------- DFS fuzzing -------------------------

def dfs_search(img_path: Path):
    original = Image.open(img_path).convert("RGB")
    orig_class, orig_conf = keras_predict(original)

    if orig_class != 0:  # model already wrong; skip
        print(f"[SKIP] {img_path.name}: prediction incorrect ({orig_class})")
        return

    print(f"[OK]   {img_path.name}: baseline conf={orig_conf:.3f}")

    for p_name, p_func, magnitudes in PERTURBATION_TYPES:
        print(f"  Trying perturbation: {p_name}")
        for step, mag in enumerate(magnitudes, 1):
            mutated = p_func(original, mag)
            sim = clip_similarity(original, mutated)
            if sim < SIM_THRESHOLD:
                print(f"    step {step}: similarity {sim:.3f} < {SIM_THRESHOLD}, stop {p_name}")
                break  # move to next perturbation type

            pred_class, pred_conf = keras_predict(mutated)
            print(f"    step {step}: class={pred_class}, conf={pred_conf:.3f}, sim={sim:.3f}")

            if pred_class != orig_class:  # adversarial found!
                save_dir = SAVE_ROOT / "0"
                save_path = save_dir / f"{img_path.stem}_{p_name}_s{step}_sim{sim:.3f}.png"
                mutated.save(save_path)
                print(f"      >>> Adversarial saved to {save_path}")
                break  # start next perturbation type (DFS sibling)
            # else same class; if confidence dropped, continue; else if conf ↑ maybe still continue

# --------------------------- Main ----------------------------

def main():
    images = sorted(p for p in DATASET_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    for img_path in images:
        dfs_search(img_path)

    print("\nFuzzing completed.")

if __name__ == "__main__":
    main()
