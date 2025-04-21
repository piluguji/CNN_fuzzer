# traffic_sign_fuzzer.py
"""
Greedy DFS‑like adversarial fuzzing for a traffic‑sign CNN classifier over **all** 58
folders (0‑57) in `../dataset_split/test/`.

For every image in each folder:
1. Check the classifier’s prediction matches the folder label (ground truth).
2. Apply perturbation types (`brightness`, `contrast`, `saturation`, `occlusion`) one by
   one. For each type explore `MAX_STEPS = 15` magnitudes and keep the *best* candidate
   (label flipped & highest similarity, or label same & lowest confidence).
3. Accumulate chosen perturbations, then move to the next type (greedy chaining).
4. Whenever label flips and CLIP similarity ≥ 0.88, save the adversarial image to
   `adversarial_cases/<folder>/` with a filename encoding used magnitudes.
"""

import os
import random
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
DATASET_ROOT = Path("../dataset_split/test")          # root containing 0‑57 folders
MODEL_PATH   = "../traffic_sign_model.keras"           # trained classifier
SAVE_ROOT    = Path("adversarial_cases")               # where to store cases

SIM_THRESHOLD = 0.88
MAX_STEPS     = 15

BRIGHTNESS_STEPS = np.linspace(1.05, 0.4, MAX_STEPS)
CONTRAST_STEPS   = np.linspace(1.05, 0.4, MAX_STEPS)
SATURATION_STEPS = np.linspace(1.05, 0.4, MAX_STEPS)
OCCLUSION_STEPS  = np.linspace(0.01, 0.08, MAX_STEPS)

PERTURBATION_TYPES = [
    ("brightness", lambda img, m: ImageEnhance.Brightness(img).enhance(m), BRIGHTNESS_STEPS),
    ("contrast",   lambda img, m: ImageEnhance.Contrast(img).enhance(m),   CONTRAST_STEPS),
    ("saturation", lambda img, m: ImageEnhance.Color(img).enhance(m),      SATURATION_STEPS),
    ("occlusion",  None, OCCLUSION_STEPS),  # handled specially
]

# ------------------------ Model loading ------------------------
print("Loading CNN model …")
cnn_model = load_model(MODEL_PATH)

print("Loading CLIP model …")
device = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# ---------------------- Helper functions ----------------------

def keras_predict(img: Image.Image):
    arr = kimage.img_to_array(img.resize((32, 32))) / 255.0
    preds = cnn_model.predict(np.expand_dims(arr, 0), verbose=0)
    return int(np.argmax(preds)), float(np.max(preds))

def clip_similarity(a: Image.Image, b: Image.Image) -> float:
    with torch.no_grad():
        t1 = clip_preprocess(a).unsqueeze(0).to(device)
        t2 = clip_preprocess(b).unsqueeze(0).to(device)
        f1 = clip_model.encode_image(t1)
        f2 = clip_model.encode_image(t2)
        return torch.nn.functional.cosine_similarity(f1, f2).item()

def add_occlusion(img: Image.Image, frac: float, blocks: int = 50):
    w, h = img.size
    out = img.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    for _ in range(blocks):
        bw, bh = int(w * random.uniform(frac*0.8, frac*1.2)), int(h * random.uniform(frac*0.8, frac*1.2))
        bx, by = random.randint(0, w-bw), random.randint(0, h-bh)
        alpha = random.randint(80, 150)
        draw.rectangle([bx, by, bx + bw, by + bh], fill=(128, 128, 128, alpha))
    return out.convert("RGB")

# replace None with actual function reference now that add_occlusion exists
PERTURBATION_TYPES[3] = ("occlusion", add_occlusion, OCCLUSION_STEPS)

# ------------------------ Core logic --------------------------

def pick_best_variant(original, current, p_func, mags, true_label):
    best_img = None
    best_mag = None
    best_sim = -1.0
    best_conf = 1e9
    adversarial = False

    for m in mags:
        cand = p_func(current, m)
        sim = clip_similarity(original, cand)
        if sim < SIM_THRESHOLD:
            break
        pred, conf = keras_predict(cand)
        if pred != true_label:
            adversarial = True
            if sim > best_sim:
                best_img, best_mag, best_sim = cand, m, sim
        else:
            if not adversarial and conf < best_conf:
                best_img, best_mag, best_conf, best_sim = cand, m, conf, sim
    return best_img, best_mag, adversarial


def greedy_fuzz(img_path: Path, true_label: int, save_dir: Path):
    original = Image.open(img_path).convert("RGB")
    pred_label, base_conf = keras_predict(original)
    if pred_label != true_label:
        print(f"[SKIP] {img_path}: baseline pred {pred_label} ≠ label {true_label}")
        return

    current = original
    used = []  # (name, mag, adv_flag)

    for name, func, mags in PERTURBATION_TYPES:
        best_img, best_mag, adv = pick_best_variant(original, current, func, mags, true_label)
        if best_img is None:
            break  # similarity fell below threshold
        current = best_img
        used.append((name, best_mag, adv))
        if adv:
            mags_str = "_".join(f"{n}{m:.3f}" for n, m, _ in used)
            save_path = save_dir / f"{img_path.stem}_{mags_str}_adv.png"
            current.save(save_path)
            print(f"  [ADV] saved {save_path.relative_to(SAVE_ROOT)}")

# ---------------------------- Main ----------------------------

def main():
    for folder_dir in sorted(DATASET_ROOT.iterdir(), key=lambda p: int(p.name)):
        if not folder_dir.is_dir():
            continue
        label = int(folder_dir.name)
        folder_save = SAVE_ROOT / folder_dir.name
        folder_save.mkdir(parents=True, exist_ok=True)

        images = sorted(p for p in folder_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        for img in tqdm(images, desc=f"Folder {label}"):
            greedy_fuzz(img, label, folder_save)

    print("\nFuzzing finished.")

if __name__ == "__main__":
    main()
