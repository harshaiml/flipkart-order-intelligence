"""
Part 2 Task 8 -- Export real sample images as actual .png files.
FashionMNIST stores data as raw IDX binary; we pick real test-split images
covering different classes and write them out as .png files that Part 3's
classify_product_image tool will point at.
"""
import os
from torchvision import datasets
from PIL import Image

os.makedirs("data/sample_images", exist_ok=True)

test_set = datasets.FashionMNIST(root="./data", train=False, download=False)
CLASS_NAMES = [
    "Tshirt-top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankleboot",
]

# Pick one real test image per class (first occurrence), covering all 10 classes
seen = {}
for idx in range(len(test_set)):
    img, label = test_set[idx]
    if label not in seen:
        seen[label] = (idx, img)
    if len(seen) == 10:
        break

for label, (idx, img) in sorted(seen.items()):
    fname = f"data/sample_images/{label:02d}_{CLASS_NAMES[label]}.png"
    img.save(fname)
    print(f"Saved {fname} (test-split index {idx}, true label={CLASS_NAMES[label]})")

print(f"\nTotal sample images exported: {len(seen)}")
