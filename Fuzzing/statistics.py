import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Increase default font size for readability
rcParams.update({'font.size': 10})

# -------------------  Paths -------------------
TEST_ROOT = Path("../dataset_split/test")     # test set root (0‑57 folders)
ADV_ROOT  = Path("adversarial_cases")         # generated adversarial examples

# Supported perturbation keywords
PKEYS = ["brightness", "contrast", "saturation", "occlusion"]
pkey_pattern = re.compile(r"(brightness|contrast|saturation|occlusion)")

# ------------- 1. Count test images -------------
test_imgs = [p for p in TEST_ROOT.rglob("*")
             if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
total_test = len(test_imgs)

# ------------- 2. Gather adversarial images -------------
adv_imgs = [p for p in ADV_ROOT.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
total_adv = len(adv_imgs)

# Deduplicate by original filename (portion before first perturbation keyword)
orig_stems = set()
for p in adv_imgs:
    stem = p.stem
    m = pkey_pattern.search(stem)
    orig_stems.add(stem[:m.start()].rstrip("_") if m else stem)
unique_adv = len(orig_stems)

# ------------- 3. Adversarial‑rate bar chart -------------
rates = {
    "All Adv Images":      total_adv  / total_test * 100 if total_test else 0,
    "Unique Adv / Original": unique_adv / total_test * 100 if total_test else 0
}
plt.figure(figsize=(6,4))
plt.bar(rates.keys(), rates.values(), color="#5b9bd5")
plt.title("Adversarial Case Rates (%)")
plt.ylabel("Percentage of Test Images")
plt.ylim(0, max(rates.values()) * 1.2 if rates else 1)
for i, v in enumerate(rates.values()):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
plt.show()

# ------------- 4. Perturbation counts -------------
perturb_counts = {k: 0 for k in PKEYS}
for p in adv_imgs:
    stem = p.stem
    for key in PKEYS:
        if key in stem:
            perturb_counts[key] += 1

plt.figure(figsize=(6,4))
plt.bar(perturb_counts.keys(), perturb_counts.values(), color="#ed7d31")
plt.title("Perturbation Occurrences in Adversarial Cases")
plt.ylabel("Count")
for i, v in enumerate(perturb_counts.values()):
    plt.text(i, v + 0.5, str(v), ha='center')
plt.show()

# ------------- 5. Perturbation pie chart -------------
total_mentions = sum(perturb_counts.values())
if total_mentions:
    plt.figure(figsize=(6,6))
    plt.pie(perturb_counts.values(),
            labels=perturb_counts.keys(),
            autopct="%1.1f%%",
            startangle=90)
    plt.title("Perturbation Type Distribution")
    plt.axis('equal')
    plt.show()
else:
    print("No perturbation keywords found in adversarial filenames.")

# ------------- 6. Coverage (%) per perturbation -------------
coverage_pct = {k: (v / total_adv * 100) if total_adv else 0
                for k, v in perturb_counts.items()}

plt.figure(figsize=(6,4))
plt.bar(coverage_pct.keys(), coverage_pct.values(), color="#70ad47")
plt.title("Perturbation Coverage Across Adversarial Cases (%)")
plt.ylabel("Percentage of Adversarial Images")
plt.ylim(0, max(coverage_pct.values()) * 1.2 if total_adv else 1)
for i, v in enumerate(coverage_pct.values()):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
plt.show()

# ------------- 7. Console summary -------------
print(f"Total test images           : {total_test}")
print(f"Total adversarial images    : {total_adv}")
print(f"Unique adversarial originals: {unique_adv}")
print("Coverage per perturbation (percentage of adversarial images):")
for k, v in coverage_pct.items():
    print(f"  {k:<10}: {v:.2f}%")